// alloc_bench.cpp — 본문 버퍼 크기별 "새 메모리 확보 지연 L" 측정기
//
// 빌드:  g++ -O2 -std=c++17 -o alloc_bench alloc_bench.cpp -ldl
// 실행:  ./alloc_bench --sizes 1M,4M                         > L.csv
//        LD_PRELOAD=<libjemalloc.so.2> ./alloc_bench --sizes 1M --no-header >> L.csv
// 옵션:  --sizes 64K,256K,1M,4M,16M,64M  --reps N(기본: 크기별 자동)  --cpu K(코어 고정)
//        --thp default|on|off  (os_* 모드에 madvise(MADV_HUGEPAGE / MADV_NOHUGEPAGE) 적용)
//        --no-header           (CSV 머리줄 생략, 크기별로 나눠 돌릴 때)
//
// 모드 (면접 대기장 비유: 버퍼 = 의자)
//   os_fault     : 창고에서 새 의자를 꺼내 하나씩 펴는 시간 = mmap + 페이지마다 첫 쓰기(페이지 폴트 + 0 채우기)
//   os_populate  : 창고가 한 번에 펴서 주는 시간           = mmap(MAP_POPULATE)
//   os_unmap     : 창고에 돌려주는 시간                    = munmap
//   warm_touch   : 이미 펴 둔 의자에 앉는 시간 (기준선)     = 이미 쓴 버퍼에 페이지마다 쓰기
//   malloc_first : 할당기(glibc/jemalloc)에 처음 맡겼을 때  = 첫 malloc + 페이지마다 첫 쓰기
//   malloc_touch : 같은 크기를 반복해서 맡겼을 때           = malloc + 쓰기 (할당기가 재사용하면 warm에 가까움)
//   free         : 할당기에 돌려주는 시간
//   memset_warm  : (참고) 본문을 실제로 채우는 시간         = 펴 둔 버퍼 전체 memset
//
//   L(크기) ≈ os_fault − warm_touch   (같은 크기, p50·p99 각각)
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <dlfcn.h>
#include <gnu/libc-version.h>
#include <sched.h>
#include <sys/mman.h>
#include <unistd.h>

using clk = std::chrono::steady_clock;
static inline double us(clk::time_point a, clk::time_point b) {
    return std::chrono::duration<double, std::micro>(b - a).count();
}

static size_t g_page = 4096;
static int g_thp = 0;  // 0: 시스템 기본, 1: MADV_HUGEPAGE, -1: MADV_NOHUGEPAGE

static void touch_pages(char* p, size_t n) {  // 페이지마다 1바이트 쓰기 → 첫 쓰기면 페이지 폴트
    volatile char* v = p;
    for (size_t off = 0; off < n; off += g_page) v[off] = 1;
}

static void apply_thp(void* p, size_t n) {
    if (g_thp > 0) madvise(p, n, MADV_HUGEPAGE);
    if (g_thp < 0) madvise(p, n, MADV_NOHUGEPAGE);
}

static size_t parse_size(const std::string& s) {
    char* e = nullptr;
    double v = strtod(s.c_str(), &e);
    switch (e && *e ? *e : ' ') {
        case 'K': case 'k': v *= 1024.0; break;
        case 'M': case 'm': v *= 1024.0 * 1024.0; break;
        case 'G': case 'g': v *= 1024.0 * 1024.0 * 1024.0; break;
        default: break;
    }
    return (size_t)v;
}

static std::string g_alloc = "glibc";
static std::string g_thp_name = "default";

static void report(const char* mode, size_t size, std::vector<double>& v) {
    if (v.empty()) return;
    std::sort(v.begin(), v.end());
    auto pct = [&](double p) {  // nearest-rank
        size_t i = (size_t)std::max(0.0, std::ceil(p / 100.0 * (double)v.size()) - 1.0);
        return v[std::min(i, v.size() - 1)];
    };
    double sum = 0;
    for (double x : v) sum += x;
    double mib = (double)size / (1024.0 * 1024.0);
    printf("%s,%s,%s,%zu,%zu,%.2f,%.2f,%.2f,%.2f,%.2f\n", g_alloc.c_str(), g_thp_name.c_str(), mode, size,
           v.size(), pct(50), pct(99), v.back(), sum / (double)v.size(), pct(50) / mib);
    fflush(stdout);
}

int main(int argc, char** argv) {
    std::vector<size_t> sizes;
    std::string sizes_arg = "64K,256K,1M,4M,16M,64M";
    long reps_arg = 0;
    int cpu = -1;
    bool header = true;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--sizes" && i + 1 < argc) sizes_arg = argv[++i];
        else if (a == "--reps" && i + 1 < argc) reps_arg = atol(argv[++i]);
        else if (a == "--cpu" && i + 1 < argc) cpu = atoi(argv[++i]);
        else if (a == "--no-header") header = false;
        else if (a == "--thp" && i + 1 < argc) {
            g_thp_name = argv[++i];
            g_thp = g_thp_name == "on" ? 1 : g_thp_name == "off" ? -1 : 0;
        } else {
            fprintf(stderr, "usage: %s [--sizes 64K,1M,...] [--reps N] [--cpu K] [--thp default|on|off] [--no-header]\n", argv[0]);
            return 2;
        }
    }
    for (size_t p = 0, q; p <= sizes_arg.size(); p = q + 1) {
        q = sizes_arg.find(',', p);
        if (q == std::string::npos) q = sizes_arg.size();
        if (q > p) sizes.push_back(parse_size(sizes_arg.substr(p, q - p)));
    }
    g_page = (size_t)sysconf(_SC_PAGESIZE);
    if (cpu >= 0) {
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(cpu, &set);
        if (sched_setaffinity(0, sizeof set, &set) != 0) perror("sched_setaffinity");
    }

    // 지금 쓰는 할당기 확인 (LD_PRELOAD가 실패하면 조용히 glibc로 돈다 → 여기서 잡는다)
    typedef int (*mallctl_fn)(const char*, void*, size_t*, void*, size_t);
    if (auto mc = (mallctl_fn)dlsym(RTLD_DEFAULT, "mallctl")) {
        const char* ver = nullptr;
        size_t len = sizeof ver;
        g_alloc = "jemalloc";
        if (mc("version", &ver, &len, nullptr, 0) == 0 && ver) {
            std::string s = ver;
            g_alloc += "-" + s.substr(0, s.find('-'));
        }
        // MALLOC_CONF에 oversize_threshold:0을 줬다면 메인 스레드의 arena 0 문턱도 끈다
        // (jemalloc 5.3.0은 arena 0을 설정 적용 전에 만들어 8MiB 문턱이 남는다)
        size_t opt_os = 1;
        len = sizeof opt_os;
        if (mc("opt.oversize_threshold", &opt_os, &len, nullptr, 0) == 0 && opt_os == 0) {
            void* w = malloc(64);
            free(w);
            unsigned arena = 0;
            len = sizeof arena;
            if (mc("thread.arena", &arena, &len, nullptr, 0) == 0) {
                char name[64];
                snprintf(name, sizeof name, "arena.%u.oversize_threshold", arena);
                size_t big = (size_t)1 << 60;
                mc(name, nullptr, nullptr, &big, sizeof big);
            }
        }
    } else {
        g_alloc = std::string("glibc-") + gnu_get_libc_version();
    }

    if (header) {
        char thp[64] = "?";
        if (FILE* f = fopen("/sys/kernel/mm/transparent_hugepage/enabled", "r")) {
            if (!fgets(thp, sizeof thp, f)) strcpy(thp, "?");
            fclose(f);
            thp[strcspn(thp, "\n")] = 0;
        }
        printf("# page_size=%zu thp_system=\"%s\"\n", g_page, thp);
        printf("alloc,thp,mode,size_bytes,reps,p50_us,p99_us,max_us,mean_us,p50_us_per_MiB\n");
    }

    for (size_t size : sizes) {
        long reps = reps_arg > 0 ? reps_arg
                                 : (long)std::clamp<size_t>((size_t(1) << 31) / std::max<size_t>(size, 1), 50, 2000);
        std::vector<double> a, b;
        a.reserve(reps);
        b.reserve(reps);

        // 1) os_fault + os_unmap
        for (long r = 0; r < reps; ++r) {
            auto t0 = clk::now();
            char* p = (char*)mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            if (p == MAP_FAILED) { perror("mmap"); return 1; }
            apply_thp(p, size);
            touch_pages(p, size);
            auto t1 = clk::now();
            munmap(p, size);
            auto t2 = clk::now();
            a.push_back(us(t0, t1));
            b.push_back(us(t1, t2));
        }
        report("os_fault", size, a);
        report("os_unmap", size, b);
        a.clear();

        // 2) os_populate
        for (long r = 0; r < reps; ++r) {
            auto t0 = clk::now();
            char* p = (char*)mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE, -1, 0);
            if (p == MAP_FAILED) { perror("mmap"); return 1; }
            auto t1 = clk::now();
            munmap(p, size);
            a.push_back(us(t0, t1));
        }
        report("os_populate", size, a);
        a.clear();

        // 3) warm_touch, memset_warm (이미 펴 둔 의자)
        {
            char* p = (char*)mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            if (p == MAP_FAILED) { perror("mmap"); return 1; }
            apply_thp(p, size);
            memset(p, 0, size);
            for (long r = 0; r < reps; ++r) {
                auto t0 = clk::now();
                touch_pages(p, size);
                auto t1 = clk::now();
                a.push_back(us(t0, t1));
            }
            report("warm_touch", size, a);
            a.clear();
            long mreps = std::min<long>(reps, 200);
            for (long r = 0; r < mreps; ++r) {
                auto t0 = clk::now();
                memset(p, (int)(r & 0xff), size);
                asm volatile("" : : "r"(p) : "memory");
                auto t1 = clk::now();
                a.push_back(us(t0, t1));
            }
            report("memset_warm", size, a);
            a.clear();
            munmap(p, size);
        }

        // 4) malloc_first / malloc_touch / free (지금 로드된 할당기)
        b.clear();
        std::vector<double> first;
        for (long r = 0; r < reps; ++r) {
            auto t0 = clk::now();
            char* p = (char*)malloc(size);
            if (!p) { perror("malloc"); return 1; }
            touch_pages(p, size);
            auto t1 = clk::now();
            free(p);
            auto t2 = clk::now();
            (r == 0 ? first : a).push_back(us(t0, t1));
            b.push_back(us(t1, t2));
        }
        report("malloc_first", size, first);
        report("malloc_touch", size, a);
        report("free", size, b);
    }
    return 0;
}
