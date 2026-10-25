// chunkpool.cpp — 3단계 측정: 청크 풀에서 반납·재확보의 실제 CPU 비용(실효 T*)과 정책 결정 재생
//
// 빌드:  g++ -O2 -std=c++17 -o chunkpool chunkpool.cpp
//
// 1) 실효 T* (반납 후 다시 쓰는 비용 / 붙잡아 두는 비용)
//    ./chunkpool tstar [--chunk 1M,2M] [--method dontneed,free,remap] [--thp default,on,off] [--batch 1,16]
//                      [--region 512M] [--reps 300] [--price-ratio 9.107] [--cpu K]
//    - 청크 batch개(연속)를 반납하고 바로 다시 쓴다(페이지마다 첫 쓰기). 반납·재확보 각각의 스레드 CPU 시간, 벽시계 시간,
//      페이지 폴트, 반납 직후 RSS 감소 비율을 잰다.
//    - 실효 T* = (반납 + 재확보 CPU 초/MiB) × (vCPU 1시간 가격 ÷ GB 1시간 가격) × 1024.
//      기본 가격 비율 9.107은 AWS Fargate(리눅스 x86, us-east-1) $0.04048/vCPU-시 ÷ $0.004445/GB-시다(계산, 가격은 확인 필요).
//    - method: dontneed = madvise(MADV_DONTNEED), free = madvise(MADV_FREE, 게으른 반납: 메모리 압박이 없으면 RSS가 안 준다),
//              remap = 같은 주소에 새 익명 매핑(mmap MAP_FIXED)
// 2) 정책 결정 재생
//    ./chunkpool replay --in dec.csv [--method dontneed] [--thp default] [--check-every 2000] [--cpu K]
//    - export_decisions.py가 만든 사건 목록(C: 처음 쓰기, R: 반납, F: 반납한 청크를 다시 쓰기)을 시간 순으로 실행한다.
//    - 사건마다 스레드 CPU 시간과 페이지 폴트를 재고, 장부상 상주 청크 수와 실제 RSS(/proc/self/statm)를 주기적으로 맞춰 본다.
//    - 시간은 빨리 감는다(기다리지 않는다). 낭비(붙잡은 MiB·초)는 장부 값이고, 실제 메모리가 줄었는지는 RSS 대조로 확인한다.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#include <sched.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <time.h>
#include <unistd.h>

static size_t PAGE = 4096;
static const size_t MiB = 1024 * 1024;

static double tcpu() {
    timespec ts;
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}
static double twall() {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}
static long minflt() {
    rusage ru;
    getrusage(RUSAGE_THREAD, &ru);
    return ru.ru_minflt;
}
static size_t rss_bytes() {
    FILE* f = fopen("/proc/self/statm", "r");
    if (!f) return 0;
    unsigned long size = 0, res = 0;
    if (fscanf(f, "%lu %lu", &size, &res) != 2) res = 0;
    fclose(f);
    return res * PAGE;
}
static void touch(char* p, size_t n) {  // 페이지마다 1바이트 쓰기(반납된 페이지면 폴트 + 0 채우기)
    volatile char* v = p;
    for (size_t off = 0; off < n; off += PAGE) v[off] = 1;
}
static size_t parse_size(const std::string& s) {
    char* e = nullptr;
    double v = strtod(s.c_str(), &e);
    if (e && *e) {
        if (*e == 'K' || *e == 'k') v *= 1024.0;
        else if (*e == 'M' || *e == 'm') v *= 1024.0 * 1024.0;
        else if (*e == 'G' || *e == 'g') v *= 1024.0 * 1024.0 * 1024.0;
    }
    return (size_t)v;
}
static std::vector<std::string> split(const std::string& s, char c = ',') {
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string x;
    while (std::getline(ss, x, c))
        if (!x.empty()) out.push_back(x);
    return out;
}

enum Method { DONTNEED, FREE, REMAP };
static Method parse_method(const std::string& m) {
    if (m == "dontneed") return DONTNEED;
    if (m == "free") return FREE;
    if (m == "remap") return REMAP;
    fprintf(stderr, "알 수 없는 method: %s\n", m.c_str());
    exit(2);
}
static int parse_thp(const std::string& t) { return t == "on" ? 1 : t == "off" ? -1 : 0; }
static void apply_thp(char* p, size_t n, int thp) {
    if (thp > 0) madvise(p, n, MADV_HUGEPAGE);
    if (thp < 0) madvise(p, n, MADV_NOHUGEPAGE);
}

struct Region {
    char* base = nullptr;
    char* raw = nullptr;
    size_t len = 0, raw_len = 0;
};
static Region map_region(size_t len, int thp) {  // 2 MiB 경계에 맞춘 익명 매핑
    Region r;
    r.raw_len = len + 2 * MiB;
    r.raw = (char*)mmap(nullptr, r.raw_len, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (r.raw == MAP_FAILED) {
        perror("mmap");
        exit(1);
    }
    uintptr_t a = ((uintptr_t)r.raw + 2 * MiB - 1) & ~(uintptr_t)(2 * MiB - 1);
    r.base = (char*)a;
    r.len = len;
    apply_thp(r.base, r.len, thp);
    return r;
}
static void release(char* p, size_t n, Method m, int thp) {
    if (m == DONTNEED) {
        if (madvise(p, n, MADV_DONTNEED) != 0) perror("madvise(DONTNEED)");
    } else if (m == FREE) {
        if (madvise(p, n, MADV_FREE) != 0) perror("madvise(FREE)");
    } else {
        void* q = mmap(p, n, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED | MAP_NORESERVE, -1, 0);
        if (q == MAP_FAILED) perror("mmap(MAP_FIXED)");
        apply_thp(p, n, thp);
    }
}
static std::string thp_system() {
    char buf[128] = "?";
    if (FILE* f = fopen("/sys/kernel/mm/transparent_hugepage/enabled", "r")) {
        if (!fgets(buf, sizeof buf, f)) strcpy(buf, "?");
        fclose(f);
        buf[strcspn(buf, "\n")] = 0;
    }
    return buf;
}
static double median(std::vector<double> v) {
    if (v.empty()) return NAN;
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}
static double pct(std::vector<double> v, double p) {
    if (v.empty()) return NAN;
    std::sort(v.begin(), v.end());
    size_t i = (size_t)std::max(0.0, std::ceil(p / 100.0 * (double)v.size()) - 1.0);
    return v[std::min(i, v.size() - 1)];
}

static int cmd_tstar(int argc, char** argv) {
    std::string chunks = "1M,2M", methods = "dontneed,free,remap", thps = "default,on,off", batches = "1,16";
    size_t region = 512 * MiB;
    long reps = 300;
    double ratio = 9.107;
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() { return std::string(i + 1 < argc ? argv[++i] : ""); };
        if (a == "--chunk") chunks = next();
        else if (a == "--method") methods = next();
        else if (a == "--thp") thps = next();
        else if (a == "--batch") batches = next();
        else if (a == "--region") region = parse_size(next());
        else if (a == "--reps") reps = atol(next().c_str());
        else if (a == "--price-ratio") ratio = atof(next().c_str());
        else if (a == "--cpu") {
            cpu_set_t set;
            CPU_ZERO(&set);
            CPU_SET(atoi(next().c_str()), &set);
            if (sched_setaffinity(0, sizeof set, &set) != 0) perror("sched_setaffinity");
        } else {
            fprintf(stderr, "알 수 없는 옵션: %s\n", a.c_str());
            return 2;
        }
    }
    printf("# page=%zu thp_system=\"%s\" region=%zu reps=%ld price_ratio=%.4g\n", PAGE, thp_system().c_str(), region, reps, ratio);
    printf("chunk_bytes,method,thp,batch,reps,release_cpu_us_per_MiB,refault_cpu_us_per_MiB,refault_wall_us_per_MiB_p50,"
           "refault_wall_us_per_MiB_p99,faults_per_MiB,rss_drop_frac,tstar_eff_s\n");
    std::mt19937_64 rng(12345);
    for (auto& cs : split(chunks)) {
        size_t chunk = parse_size(cs);
        for (auto& ts : split(thps)) {
            int thp = parse_thp(ts);
            for (auto& ms : split(methods)) {
                Method m = parse_method(ms);
                for (auto& bs : split(batches)) {
                    size_t batch = (size_t)atol(bs.c_str());
                    size_t n_chunks = region / chunk;
                    if (batch == 0 || batch > n_chunks) continue;
                    Region r = map_region(n_chunks * chunk, thp);
                    touch(r.base, r.len);
                    std::uniform_int_distribution<size_t> pick(0, n_chunks - batch);
                    std::vector<double> rel, ref, refw, flt, drop;
                    double mib = (double)(batch * chunk) / MiB;
                    for (long k = 0; k < reps; ++k) {
                        char* p = r.base + pick(rng) * chunk;
                        size_t n = batch * chunk;
                        size_t rss0 = rss_bytes();
                        double c0 = tcpu();
                        release(p, n, m, thp);
                        double c1 = tcpu();
                        size_t rss1 = rss_bytes();
                        long f1 = minflt();
                        double w1 = twall(), c2 = tcpu();
                        touch(p, n);
                        double c3 = tcpu(), w2 = twall();
                        long f2 = minflt();
                        rel.push_back((c1 - c0) * 1e6 / mib);
                        ref.push_back((c3 - c2) * 1e6 / mib);
                        refw.push_back((w2 - w1) * 1e6 / mib);
                        flt.push_back((double)(f2 - f1) / mib);
                        drop.push_back(rss0 > rss1 ? (double)(rss0 - rss1) / (double)n : 0.0);
                    }
                    double rel_m = median(rel), ref_m = median(ref);
                    double tstar = (rel_m + ref_m) * 1e-6 * ratio * 1024.0;
                    printf("%zu,%s,%s,%zu,%ld,%.2f,%.2f,%.2f,%.2f,%.1f,%.3f,%.3f\n", chunk, ms.c_str(), ts.c_str(), batch, reps,
                           rel_m, ref_m, median(refw), pct(refw, 99), median(flt), median(drop), tstar);
                    fflush(stdout);
                    munmap(r.raw, r.raw_len);
                }
            }
        }
    }
    return 0;
}

struct Ev {
    char type;
    double t;
    size_t lo, n;
    int tag;
};

static int cmd_replay(int argc, char** argv) {
    std::string in, ms = "dontneed", ts = "default";
    long check_every = 2000;
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() { return std::string(i + 1 < argc ? argv[++i] : ""); };
        if (a == "--in") in = next();
        else if (a == "--method") ms = next();
        else if (a == "--thp") ts = next();
        else if (a == "--check-every") check_every = atol(next().c_str());
        else if (a == "--cpu") {
            cpu_set_t set;
            CPU_ZERO(&set);
            CPU_SET(atoi(next().c_str()), &set);
            if (sched_setaffinity(0, sizeof set, &set) != 0) perror("sched_setaffinity");
        } else {
            fprintf(stderr, "알 수 없는 옵션: %s\n", a.c_str());
            return 2;
        }
    }
    std::ifstream f(in);
    if (!f) {
        fprintf(stderr, "파일을 열 수 없다: %s\n", in.c_str());
        return 1;
    }
    Method m = parse_method(ms);
    int thp = parse_thp(ts);
    size_t chunk = MiB, n_chunks = 0;
    std::string policy = "?", line;
    std::vector<Ev> ev;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        if (line[0] == '#') {
            for (auto& kv : split(line.substr(1), ' ')) {
                auto eq = kv.find('=');
                if (eq == std::string::npos) continue;
                std::string k = kv.substr(0, eq), v = kv.substr(eq + 1);
                if (k == "chunk_bytes") chunk = (size_t)atoll(v.c_str());
                else if (k == "n_chunks") n_chunks = (size_t)atoll(v.c_str());
                else if (k == "policy") policy = v;
            }
            continue;
        }
        if (line.rfind("type", 0) == 0) continue;
        auto c = split(line);
        if (c.size() < 5) continue;
        ev.push_back({c[0][0], atof(c[1].c_str()), (size_t)atoll(c[2].c_str()), (size_t)atoll(c[3].c_str()), atoi(c[4].c_str())});
    }
    if (n_chunks == 0) {
        fprintf(stderr, "머리줄에 n_chunks가 없다\n");
        return 1;
    }
    Region r = map_region(n_chunks * chunk, thp);
    size_t base_rss = rss_bytes();
    // 태그별 누적: [0] 준비 구간, [1] 평가 구간
    double cold_cpu = 0, rel_cpu[2] = {0, 0}, ref_cpu[2] = {0, 0};
    long rel_ops[2] = {0, 0}, ref_ops[2] = {0, 0}, ref_flt[2] = {0, 0}, rel_flt[2] = {0, 0};
    double rel_mib[2] = {0, 0}, ref_mib[2] = {0, 0}, cold_mib = 0;
    long resident = 0, checks = 0;
    double dev_max = 0, dev_sum = 0, w0 = twall();
    for (size_t i = 0; i < ev.size(); ++i) {
        const Ev& e = ev[i];
        char* p = r.base + e.lo * chunk;
        size_t n = e.n * chunk;
        int tg = e.tag ? 1 : 0;
        if (e.lo + e.n > n_chunks) {
            fprintf(stderr, "범위 밖 사건: %zu+%zu > %zu\n", e.lo, e.n, n_chunks);
            return 1;
        }
        long f0 = minflt();
        double c0 = tcpu();
        if (e.type == 'C') {
            touch(p, n);
            cold_cpu += tcpu() - c0;
            cold_mib += (double)n / MiB;
            resident += (long)e.n;
        } else if (e.type == 'R') {
            release(p, n, m, thp);
            rel_cpu[tg] += tcpu() - c0;
            rel_flt[tg] += minflt() - f0;
            rel_ops[tg] += 1;
            rel_mib[tg] += (double)n / MiB;
            resident -= (long)e.n;
        } else if (e.type == 'F') {
            touch(p, n);
            ref_cpu[tg] += tcpu() - c0;
            ref_flt[tg] += minflt() - f0;
            ref_ops[tg] += 1;
            ref_mib[tg] += (double)n / MiB;
            resident += (long)e.n;
        }
        if (check_every > 0 && (long)(i % (size_t)check_every) == 0) {
            size_t now = rss_bytes();
            double rss = (double)(now > base_rss ? now - base_rss : 0) / MiB;
            double dev = rss - (double)resident * (double)chunk / MiB;
            dev_max = std::max(dev_max, std::fabs(dev));
            dev_sum += dev;
            checks++;
        }
    }
    double wall = twall() - w0;
    printf("policy,method,thp,chunk_bytes,events,cold_MiB,cold_cpu_s,rel_ops,rel_MiB,rel_cpu_s,rel_faults,ref_ops,ref_MiB,"
           "ref_cpu_s,ref_faults,rel_ops_warm,ref_MiB_warm,rss_dev_max_MiB,rss_dev_mean_MiB,wall_s\n");
    printf("%s,%s,%s,%zu,%zu,%.1f,%.4f,%ld,%.1f,%.4f,%ld,%ld,%.1f,%.4f,%ld,%ld,%.1f,%.1f,%.1f,%.2f\n", policy.c_str(), ms.c_str(),
           ts.c_str(), chunk, ev.size(), cold_mib, cold_cpu, rel_ops[1], rel_mib[1], rel_cpu[1], rel_flt[1], ref_ops[1],
           ref_mib[1], ref_cpu[1], ref_flt[1], rel_ops[0], ref_mib[0], dev_max, checks ? dev_sum / checks : 0.0, wall);
    munmap(r.raw, r.raw_len);
    return 0;
}

int main(int argc, char** argv) {
    PAGE = (size_t)sysconf(_SC_PAGESIZE);
    if (argc < 2) {
        fprintf(stderr, "사용: %s tstar|replay [옵션]\n", argv[0]);
        return 2;
    }
    std::string mode = argv[1];
    if (mode == "tstar") return cmd_tstar(argc, argv);
    if (mode == "replay") return cmd_replay(argc, argv);
    fprintf(stderr, "알 수 없는 모드: %s\n", mode.c_str());
    return 2;
}
