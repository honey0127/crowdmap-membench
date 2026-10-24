// replay.cpp — 같은 할당/반납 일정(schedule.csv)을 실시간으로 재생해 낭비 W와 할당 지연 A를 잰다
//
// 빌드:  g++ -O2 -std=c++17 -pthread -o replay replay.cpp -ldl
// 실행:  ./replay schedule.csv glibc --header > results.csv
//        MALLOC_CONF="background_thread:true,dirty_decay_ms:1000,muzzy_decay_ms:0,oversize_threshold:0"
//          LD_PRELOAD=<libjemalloc.so.2> ./replay schedule.csv je_1000 >> results.csv   (위 두 줄을 한 줄로)
//        ./replay --probe     (지금 로드된 할당기와 감쇠 설정만 출력하고 끝)
// 옵션:  --sample-ms 10   --series out.csv(B·D 시계열)   --rep N(반복 번호)   --max-s X(앞 X초만)   --header
//
// 면접 대기장 비유: 같은 손님 명단(일정)을 똑같이 다시 돌리고, 의자 관리 규칙(할당기·감쇠 설정)만 바꾼다.
//   B(t) = 펴 놓은 의자     = 프로세스 RSS − 시작 시 RSS
//   D(t) = 실제로 앉은 손님 = 살아 있는 버퍼 바이트(페이지 단위로 올림)
//   W    = ∫(B − D)dt      = 빈 의자 × 시간 (MB·s).   waste_mean_MB = W ÷ 측정 시간
//   A    = 손님 1명의 의자 마련 시간 = malloc + 페이지마다 첫 쓰기 (본문 수신을 대신함)
//   F    = 반납 시간 = free
// 단순화: 할당과 반납을 한 작업 스레드에서 일정대로 실행한다. 할당기의 보존·반납 동작을 재는 용도다.
//
// jemalloc 5.3.0 주의(소스·실측 확인):
//   - 8MiB 이상 요청은 기본값으로 전용 arena(감쇠 0)로 가서 free 즉시 반납된다 → 감쇠 손잡이가 듣지 않는다.
//     MALLOC_CONF에 oversize_threshold:0 을 넣어 끈다.
//   - 그래도 arena 0(메인 스레드)은 초기화 순서 때문에 8MiB 문턱이 남는다 → 재생은 별도 스레드에서 하고,
//     oversize를 끈 경우 그 스레드의 arena 문턱도 mallctl로 끈다(아래 je_bind_arena).
#include <algorithm>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
#include <thread>
#include <vector>
#include <dlfcn.h>
#include <fcntl.h>
#include <gnu/libc-version.h>
#include <sys/resource.h>
#include <sys/types.h>
#include <unistd.h>

struct Req { double ta, tf; size_t sz; };
struct Ev { double t; uint32_t i; uint8_t alloc; };
struct Sample { float t, B, D; };

typedef int (*mallctl_fn)(const char*, void*, size_t*, void*, size_t);
static mallctl_fn g_mc = nullptr;
static size_t g_page = 4096;
static int g_statm = -1;

static inline uint64_t now_ns() {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}
static void sleep_until_ns(uint64_t t) {
    timespec ts{(time_t)(t / 1000000000ull), (long)(t % 1000000000ull)};
    while (clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &ts, nullptr) == EINTR) {}
}
static uint64_t rss_bytes() {  // /proc/self/statm 두 번째 값(resident pages)
    char buf[160];
    ssize_t n = pread(g_statm, buf, sizeof buf - 1, 0);
    if (n <= 0) return 0;
    buf[n] = 0;
    char* e = nullptr;
    strtoull(buf, &e, 10);
    return strtoull(e, nullptr, 10) * g_page;
}
static double pct(const std::vector<double>& v, double p) {  // v는 정렬돼 있어야 한다
    if (v.empty()) return 0;
    size_t i = (size_t)std::max(0.0, std::ceil(p / 100.0 * (double)v.size()) - 1.0);
    return v[std::min(i, v.size() - 1)];
}

struct AllocInfo {
    std::string name = "glibc", dirty = "NA", muzzy = "NA", bg = "NA", oversize = "NA";
    size_t opt_oversize = 0;
};
static AllocInfo detect_allocator() {
    AllocInfo ai;
    g_mc = (mallctl_fn)dlsym(RTLD_DEFAULT, "mallctl");
    if (!g_mc) {
        ai.name = std::string("glibc-") + gnu_get_libc_version();
        return ai;
    }
    ai.name = "jemalloc";
    const char* ver = nullptr;
    size_t len = sizeof ver;
    if (g_mc("version", &ver, &len, nullptr, 0) == 0 && ver) {
        std::string s = ver;
        ai.name += "-" + s.substr(0, s.find('-'));
    }
    ssize_t v;
    len = sizeof v;
    if (g_mc("opt.dirty_decay_ms", &v, &len, nullptr, 0) == 0) ai.dirty = std::to_string((long)v);
    len = sizeof v;
    if (g_mc("opt.muzzy_decay_ms", &v, &len, nullptr, 0) == 0) ai.muzzy = std::to_string((long)v);
    bool b = false;
    len = sizeof b;
    if (g_mc("background_thread", &b, &len, nullptr, 0) == 0) ai.bg = b ? "true" : "false";
    len = sizeof ai.opt_oversize;
    g_mc("opt.oversize_threshold", &ai.opt_oversize, &len, nullptr, 0);
    return ai;
}

// 호출한 스레드가 쓰는 jemalloc arena의 oversize 문턱을 확인한다.
// oversize를 끈 설정(opt.oversize_threshold=0)이면 arena 문턱도 끈다(arena 0 초기화 순서 문제 대응).
static std::string je_bind_arena(const AllocInfo& ai) {
    if (!g_mc) return "NA";
    void* w = malloc(64);  // arena에 묶이게 한 번 할당
    free(w);
    unsigned arena = 0;
    size_t len = sizeof arena;
    if (g_mc("thread.arena", &arena, &len, nullptr, 0) != 0) return "?";
    char name[64];
    snprintf(name, sizeof name, "arena.%u.oversize_threshold", arena);
    size_t thr = 0;
    len = sizeof thr;
    if (g_mc(name, &thr, &len, nullptr, 0) != 0) return "?";
    const size_t kOff = (size_t)1 << 50;
    if (ai.opt_oversize == 0 && thr < kOff) {
        size_t big = (size_t)1 << 60;
        if (g_mc(name, nullptr, nullptr, &big, sizeof big) == 0) thr = big;
    }
    if (thr >= kOff) return "off";
    return std::to_string(thr >> 20) + "MiB";
}

int main(int argc, char** argv) {
    std::string path, label = "run", series_path;
    int sample_ms = 10, rep = 1;
    double max_s = 0;
    bool header = false, probe = false;
    std::vector<std::string> pos;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--sample-ms" && i + 1 < argc) sample_ms = std::max(1, atoi(argv[++i]));
        else if (a == "--series" && i + 1 < argc) series_path = argv[++i];
        else if (a == "--rep" && i + 1 < argc) rep = atoi(argv[++i]);
        else if (a == "--max-s" && i + 1 < argc) max_s = atof(argv[++i]);
        else if (a == "--header") header = true;
        else if (a == "--probe") probe = true;
        else if (!a.empty() && a[0] == '-') {
            fprintf(stderr, "unknown option %s\n", a.c_str());
            return 2;
        } else pos.push_back(a);
    }
    g_page = (size_t)sysconf(_SC_PAGESIZE);
    AllocInfo ai = detect_allocator();
    if (probe) {
        std::thread([&] { ai.oversize = je_bind_arena(ai); }).join();
        printf("allocator=%s dirty_decay_ms=%s muzzy_decay_ms=%s background_thread=%s oversize=%s page=%zu\n",
               ai.name.c_str(), ai.dirty.c_str(), ai.muzzy.c_str(), ai.bg.c_str(), ai.oversize.c_str(), g_page);
        return 0;
    }
    if (pos.empty()) {
        fprintf(stderr, "usage: %s schedule.csv LABEL [--header] [--rep N] [--sample-ms 10] [--series f.csv] [--max-s X]\n", argv[0]);
        return 2;
    }
    path = pos[0];
    if (pos.size() > 1) label = pos[1];

    // ---- 일정 읽기: t_alloc_s,t_free_s,size_bytes (머리줄·주석 줄은 건너뜀)
    FILE* f = fopen(path.c_str(), "r");
    if (!f) { perror(path.c_str()); return 1; }
    std::vector<Req> req;
    char line[512];
    while (fgets(line, sizeof line, f)) {
        char* p = line;
        while (*p == ' ') ++p;
        if (!(isdigit((unsigned char)*p) || *p == '.')) continue;
        char* e = nullptr;
        double ta = strtod(p, &e);
        if (*e != ',') continue;
        double tf = strtod(e + 1, &e);
        if (*e != ',') continue;
        double sz = strtod(e + 1, &e);
        if (sz < 1 || tf < ta) continue;
        if (max_s > 0 && ta > max_s) continue;
        req.push_back({ta, tf, (size_t)sz});
    }
    fclose(f);
    if (req.empty()) { fprintf(stderr, "no rows in %s\n", path.c_str()); return 1; }

    std::vector<Ev> ev;
    ev.reserve(req.size() * 2);
    for (uint32_t i = 0; i < req.size(); ++i) {
        ev.push_back({req[i].ta, i, 1});
        ev.push_back({req[i].tf, i, 0});
    }
    std::sort(ev.begin(), ev.end(), [](const Ev& a, const Ev& b) {
        return a.t < b.t || (a.t == b.t && a.alloc < b.alloc);  // 같은 시각이면 반납 먼저
    });
    const double t_end = max_s > 0 ? max_s : ev.back().t;

    // ---- 측정용 배열은 미리 만들어 0으로 채운다(측정 중 RSS가 늘지 않게)
    const size_t n = req.size();
    std::vector<char*> ptr(n, nullptr);
    std::vector<double> A(n, 0.0), F(n, 0.0);
    const size_t nsamp = (size_t)(t_end * 1000.0 / sample_ms) + 4;
    std::vector<Sample> S(nsamp, Sample{0, 0, 0});

    g_statm = open("/proc/self/statm", O_RDONLY);
    if (g_statm < 0) { perror("/proc/self/statm"); return 1; }

    std::atomic<uint64_t> live{0};
    std::atomic<int> ready{0};
    std::atomic<uint64_t> t0_atomic{0};
    size_t na = 0, nf = 0, late = 0, ns = 0;
    double lag_max = 0;

    // ---- 작업 스레드: 할당기 arena에 묶은 뒤, 시작 신호를 받으면 일정을 재생한다
    std::thread worker([&] {
        ai.oversize = je_bind_arena(ai);
        ready.store(1);
        uint64_t t0;
        while ((t0 = t0_atomic.load()) == 0) usleep(1000);
        for (const Ev& e : ev) {
            if (e.t > t_end) break;
            const uint64_t target = t0 + (uint64_t)(e.t * 1e9);
            sleep_until_ns(target);
            const uint64_t t1 = now_ns();
            const double lag_ms = (double)(t1 - target) / 1e6;
            lag_max = std::max(lag_max, lag_ms);
            if (lag_ms > 5.0) ++late;
            const size_t sz = req[e.i].sz;
            const size_t rounded = (sz + g_page - 1) / g_page * g_page;
            if (e.alloc) {
                const uint64_t a0 = now_ns();
                char* p = (char*)malloc(sz);
                if (!p) { fprintf(stderr, "malloc(%zu) failed\n", sz); fflush(stderr); _exit(1); }
                volatile char* v = p;
                for (size_t off = 0; off < sz; off += g_page) v[off] = 1;  // 본문이 들어온 셈
                const uint64_t a1 = now_ns();
                ptr[e.i] = p;
                A[na++] = (double)(a1 - a0) / 1e3;  // µs
                live.fetch_add(rounded, std::memory_order_relaxed);
            } else if (ptr[e.i]) {
                live.fetch_sub(rounded, std::memory_order_relaxed);
                const uint64_t f0 = now_ns();
                free(ptr[e.i]);
                const uint64_t f1 = now_ns();
                ptr[e.i] = nullptr;
                F[nf++] = (double)(f1 - f0) / 1e3;
            }
        }
    });
    while (!ready.load()) usleep(1000);

    struct rusage ru0, ru1;
    usleep(300 * 1000);  // 할당기 초기화가 가라앉을 때까지
    const uint64_t base = rss_bytes();
    getrusage(RUSAGE_SELF, &ru0);
    const uint64_t t0 = now_ns() + 200 * 1000000ull;
    const uint64_t tend_ns = t0 + (uint64_t)(t_end * 1e9);
    const double dt = sample_ms / 1000.0;
    t0_atomic.store(t0);

    // ---- 메인 스레드 = 표본기: sample_ms마다 B(RSS 증가분)와 D(살아 있는 바이트)를 기록
    {
        uint64_t next = t0;
        const uint64_t step = (uint64_t)sample_ms * 1000000ull;
        while (true) {
            next += step;
            if (next > tend_ns || ns >= nsamp) break;
            sleep_until_ns(next);
            double B = (double)rss_bytes() - (double)base;
            double D = (double)live.load(std::memory_order_relaxed);
            S[ns++] = Sample{(float)((next - t0) / 1e9), (float)B, (float)D};
        }
    }
    worker.join();
    getrusage(RUSAGE_SELF, &ru1);

    // ---- 집계
    const double MB = 1024.0 * 1024.0;
    double W = 0, Bsum = 0, Dsum = 0, Bmax = 0, Dmax = 0;
    for (size_t k = 0; k < ns; ++k) {
        W += (S[k].B - S[k].D) * dt;
        Bsum += S[k].B;
        Dsum += S[k].D;
        Bmax = std::max(Bmax, (double)S[k].B);
        Dmax = std::max(Dmax, (double)S[k].D);
    }
    const double dur = ns * dt;
    A.resize(na);
    F.resize(nf);
    double Asum = 0;
    for (double x : A) Asum += x;
    std::sort(A.begin(), A.end());
    std::sort(F.begin(), F.end());
    const long minflt = ru1.ru_minflt - ru0.ru_minflt;
    const double cpu_ms = ((ru1.ru_utime.tv_sec - ru0.ru_utime.tv_sec) + (ru1.ru_stime.tv_sec - ru0.ru_stime.tv_sec)) * 1e3 +
                          ((ru1.ru_utime.tv_usec - ru0.ru_utime.tv_usec) + (ru1.ru_stime.tv_usec - ru0.ru_stime.tv_usec)) / 1e3;

    if (header)
        printf("label,rep,allocator,dirty_decay_ms,bg_thread,je_oversize,n,dur_s,A_p50_us,A_p99_us,A_max_us,A_sum_ms,"
               "F_p99_us,W_MBs,waste_mean_MB,B_mean_MB,B_max_MB,D_mean_MB,D_max_MB,minflt,cpu_ms,lag_max_ms,late_events\n");
    printf("%s,%d,%s,%s,%s,%s,%zu,%.1f,%.1f,%.1f,%.1f,%.2f,%.1f,%.1f,%.2f,%.2f,%.2f,%.2f,%.2f,%ld,%.0f,%.2f,%zu\n",
           label.c_str(), rep, ai.name.c_str(), ai.dirty.c_str(), ai.bg.c_str(), ai.oversize.c_str(), na, dur,
           pct(A, 50), pct(A, 99), A.empty() ? 0.0 : A.back(), Asum / 1e3, pct(F, 99), W / MB,
           ns ? W / MB / dur : 0.0, ns ? Bsum / ns / MB : 0.0, Bmax / MB, ns ? Dsum / ns / MB : 0.0, Dmax / MB,
           minflt, cpu_ms, lag_max, late);
    fflush(stdout);
    if (lag_max > 50.0)
        fprintf(stderr, "[경고] %s: 재생이 일정보다 최대 %.1fms 늦었다. 이 회차는 다시 돌린다.\n", label.c_str(), lag_max);

    if (!series_path.empty()) {
        if (FILE* o = fopen(series_path.c_str(), "w")) {
            fprintf(o, "t_s,B_MB,D_MB\n");
            for (size_t k = 0; k < ns; ++k) fprintf(o, "%.3f,%.3f,%.3f\n", S[k].t, S[k].B / MB, S[k].D / MB);
            fclose(o);
        }
    }
    return 0;
}
