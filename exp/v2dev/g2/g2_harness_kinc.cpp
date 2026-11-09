/* g2_harness_kinc.cpp — g2_harness.cpp(SHA-256 51dc713e…)의 사후 최적화판.
 * 바꾼 것: online_kappa_T → online_kappa_T_inc (κ 보정 증분 갱신). 그 밖의 정책·후보·선택 규칙·타이머 경계는 같다.
 * 추가 모드: equiv(원본과 단계별 κ·T·C3 결과 비트 동등성 검증), kbench(③만 원본/증분 CPU 비교).
 * 원본 g2_harness.cpp는 바꾸지 않았다. 공식 G2 결과(−845.072 ms)는 원본으로만 보고한다.
 *
 * 원 설명: G2 C++ 대응 구현 (G2W3 형식)
 *
 * 용도:
 *   parity  : Python 기준선과의 수치 동등성 검증
 *   measure : CPU_rule / CPU_C3 경로 처리 시간 측정
 *
 * 측정 경계 (G2W3):
 *   타이머 밖: 파일 I/O, 메모리 적재, 출력 파일 쓰기
 *   CPU_rule 타이머 안:
 *     Ledger::build, compute_grp_r(bin_ids+unique-remap),
 *     online_group_T(base+rule), run_fallback3, 비용 계산
 *   CPU_C3 타이머 안 (CPU_rule 공통 연산 포함):
 *     compute_p1b_features(pos/D0/mD → x0,x1,x2),
 *     compute_Tk(encode+lookup+κ-scale),
 *     online_kappa_T, fallback, 비용 계산
 *
 * 빌드: bash exp/v2dev/g2/g2_build.sh
 *
 * 사용:
 *   ./g2_harness parity  <window_dir> <model_path> <result_dir>
 *   ./g2_harness measure <window_dir> <model_path> [--dev-test] --out-json <path>
 */
#include <algorithm>
#include <cassert>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <numeric>
#include <string>
#include <time.h>
#include <vector>

namespace fs = std::filesystem;

// ---------------------------------------------------------------- 형식 상수
static const uint8_t MAGIC_WIN[4] = {'G', '2', 'W', '3'};
static const uint8_t MAGIC_MOD[4] = {'G', '2', 'M', 'B'};
static const int K_SIZE   = 9;
static const double KGRID[K_SIZE] = {0.25, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0, 2.8, 4.0};
static const int INITIAL_K = 4;      // KGRID[4] = 1.0
static const double MIN_FRAC = 0.02;
static const int32_t MIN_N   = 20;
static const double NaN = std::numeric_limits<double>::quiet_NaN();
static const double MiB = 1024.0 * 1024.0;
static const int TABLE_DIM = 65;

// ---------------------------------------------------------------- 이진 헬퍼
static double le64f(const uint8_t* p) {
    uint64_t v = 0;
    for (int i = 0; i < 8; i++) v |= ((uint64_t)p[i] << (8*i));
    double d; std::memcpy(&d, &v, 8);
    return d;
}
static uint32_t le32(const uint8_t* p) {
    return (uint32_t)p[0]|((uint32_t)p[1]<<8)|((uint32_t)p[2]<<16)|((uint32_t)p[3]<<24);
}

// ---------------------------------------------------------------- P1b 모델
struct P1bModel {
    std::vector<double> cuts[3];
    int n_cuts[3];
    std::vector<double> knots;
    std::vector<uint8_t> table;
    int sentinel;
};

P1bModel read_model(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { perror(path.c_str()); std::exit(1); }
    f.seekg(0, std::ios::end); size_t sz = f.tellg(); f.seekg(0);
    std::vector<uint8_t> raw(sz);
    f.read(reinterpret_cast<char*>(raw.data()), sz);

    if (std::memcmp(raw.data(), MAGIC_MOD, 4) != 0) {
        fprintf(stderr, "모델 매직 불일치: %s\n", path.c_str()); std::exit(1);
    }
    size_t off = 4;
    auto rd32 = [&]() -> uint32_t { uint32_t v = le32(raw.data()+off); off += 4; return v; };

    uint32_t nc0 = rd32(), nc1 = rd32(), nc2 = rd32(), nk = rd32(), sentinel = rd32();

    P1bModel m;
    m.sentinel = (int)sentinel;
    for (int fi = 0; fi < 3; fi++) {
        uint32_t nc = (fi==0)?nc0:(fi==1)?nc1:nc2;
        m.n_cuts[fi] = (int)nc;
        m.cuts[fi].resize(nc);
        std::memcpy(m.cuts[fi].data(), raw.data()+off, nc*8); off += nc*8;
    }
    m.knots.resize(nk);
    std::memcpy(m.knots.data(), raw.data()+off, nk*8); off += nk*8;
    int tsz = TABLE_DIM*TABLE_DIM*TABLE_DIM;
    m.table.resize(tsz);
    std::memcpy(m.table.data(), raw.data()+off, tsz);

    return m;
}

// ---------------------------------------------------------------- 창 데이터 (G2W3)
struct Window {
    int n, G_r, n_grid, rule_n_rf;
    double b, cap, W_cal, U, W_fb, mD;

    std::vector<double>  g, h, ts;
    std::vector<uint8_t> r, dec_mask, te_mask;
    std::vector<double>  pos, D0;             // 원시 슬라이스 속성 (bytes)
    std::vector<double>  X_rule;              // n × rule_n_rf, row-major (NaN→-1e9)
    std::vector<std::vector<double>> rule_edges; // 특징별 분위 경계
    std::vector<double>  grid;

    // Python 기준선 (parity 검증용)
    std::vector<double> py_T_base, py_T_rule, py_T_rb;
    std::vector<double> py_T_ai, py_T_ai_fb, py_T_c3;
    double py_R, py_H;
};

Window read_window(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { perror(path.c_str()); std::exit(1); }
    f.seekg(0, std::ios::end); size_t sz = f.tellg(); f.seekg(0);
    std::vector<uint8_t> raw(sz);
    f.read(reinterpret_cast<char*>(raw.data()), sz);

    size_t off = 0;
    // 헤더: magic(4)+n(4)+G_r(4)+n_grid(4)+rule_n_rf(4)+b(8)+cap(8)+W_cal(8)+U(8)+W_fb(8)+mD(8) = 68
    if (std::memcmp(raw.data(), MAGIC_WIN, 4) != 0) {
        fprintf(stderr, "창 매직 불일치: %s\n", path.c_str()); std::exit(1);
    }
    off = 4;
    auto rd32  = [&]() -> uint32_t { uint32_t v=le32(raw.data()+off); off+=4; return v; };
    auto rd64f = [&]() -> double   { double   v=le64f(raw.data()+off); off+=8; return v; };

    Window w;
    w.n        = (int)rd32();
    w.G_r      = (int)rd32();
    w.n_grid   = (int)rd32();
    w.rule_n_rf= (int)rd32();
    w.b     = rd64f();
    w.cap   = rd64f();
    w.W_cal = rd64f();
    w.U     = rd64f();
    w.W_fb  = rd64f();
    w.mD    = rd64f();

    int n = w.n, n_grid = w.n_grid, n_rf = w.rule_n_rf;

    auto rdF64 = [&](int cnt) {
        std::vector<double> v(cnt);
        std::memcpy(v.data(), raw.data()+off, cnt*8); off += cnt*8;
        return v;
    };
    auto rdU8 = [&](int cnt) {
        std::vector<uint8_t> v(cnt);
        std::memcpy(v.data(), raw.data()+off, cnt); off += cnt;
        return v;
    };

    w.g        = rdF64(n);
    w.h        = rdF64(n);
    w.r        = rdU8(n);
    w.ts       = rdF64(n);
    w.dec_mask = rdU8(n);
    w.te_mask  = rdU8(n);
    w.pos      = rdF64(n);
    w.D0       = rdF64(n);
    w.X_rule   = rdF64(n * n_rf);    // row-major

    // rule_ne[n_rf] (uint32)
    std::vector<uint32_t> rule_ne(n_rf);
    std::memcpy(rule_ne.data(), raw.data()+off, n_rf*4); off += n_rf*4;

    // rule_edges_flat
    uint32_t ne_total = 0;
    for (int i = 0; i < n_rf; i++) ne_total += rule_ne[i];
    std::vector<double> edges_flat(ne_total);
    std::memcpy(edges_flat.data(), raw.data()+off, ne_total*8); off += ne_total*8;

    // 특징별 edges 분리
    w.rule_edges.resize(n_rf);
    int ei = 0;
    for (int rf = 0; rf < n_rf; rf++) {
        int ne = (int)rule_ne[rf];
        w.rule_edges[rf].assign(edges_flat.begin()+ei, edges_flat.begin()+ei+ne);
        ei += ne;
    }

    w.grid = rdF64(n_grid);

    // Python 기준선
    w.py_T_base   = rdF64(n);
    w.py_T_rule   = rdF64(n);
    w.py_T_rb     = rdF64(n);
    w.py_T_ai     = rdF64(n);
    w.py_T_ai_fb  = rdF64(n);
    w.py_T_c3     = rdF64(n);
    w.py_R = le64f(raw.data()+off); off += 8;
    w.py_H = le64f(raw.data()+off); off += 8;

    return w;
}

// ---------------------------------------------------------------- 그림자 장부
struct Ledger {
    std::vector<int>    idx;
    std::vector<double> tss;

    void build(int n, const double* ts) {
        idx.resize(n);
        std::iota(idx.begin(), idx.end(), 0);
        std::stable_sort(idx.begin(), idx.end(), [&](int a, int b){ return ts[a] < ts[b]; });
        tss.resize(n);
        for (int i = 0; i < n; i++) tss[i] = ts[idx[i]];
    }

    std::vector<int> window(double lo_t, double hi_t) const {
        auto lo = std::lower_bound(tss.begin(), tss.end(), lo_t);
        auto hi = std::lower_bound(tss.begin(), tss.end(), hi_t);
        std::vector<int> result;
        result.reserve(hi - lo);
        for (auto it = lo; it != hi; ++it)
            result.push_back(idx[it - tss.begin()]);
        return result;
    }
};

static std::vector<double> observed(
    const std::vector<int>& J, double t, double cap,
    const double* ts, const double* g, const uint8_t* r)
{
    std::vector<double> g_obs(J.size());
    for (size_t i = 0; i < J.size(); i++) {
        int j = J[i];
        double age = t - ts[j];
        bool done = r[j] && g[j] <= age;
        if (done) {
            g_obs[i] = g[j];
        } else {
            double mx = std::max(age, cap);
            g_obs[i] = mx * (1.0 + 1e-9) + 1e-9;
        }
    }
    return g_obs;
}

// ---------------------------------------------------------------- grp_r 계산 (C++ 타이머 안)
// Python: np.unique(gid_all, return_inverse=True) + upper_bound 경계 탐색
static std::vector<int32_t> compute_grp_r(
    int n, int n_rf,
    const double* X_rule,
    const std::vector<std::vector<double>>& rule_edges,
    int expected_G_r)
{
    std::vector<int64_t> gid(n, 0);
    for (int rf = 0; rf < n_rf; rf++) {
        const auto& edges = rule_edges[rf];
        for (int j = 0; j < n; j++) {
            double x = X_rule[j * n_rf + rf];
            // searchsorted(side="right") = upper_bound
            int bid = (int)(std::upper_bound(edges.begin(), edges.end(), x) - edges.begin());
            gid[j] = gid[j] * 8 + bid;
        }
    }

    // unique-remap: np.unique(gid_all, return_inverse=True)
    std::vector<int64_t> uniq_gid = gid;
    std::sort(uniq_gid.begin(), uniq_gid.end());
    uniq_gid.erase(std::unique(uniq_gid.begin(), uniq_gid.end()), uniq_gid.end());
    int G_r_computed = (int)uniq_gid.size();
    if (G_r_computed != expected_G_r) {
        fprintf(stderr, "[경고] G_r 불일치: 계산=%d 기대=%d\n", G_r_computed, expected_G_r);
    }

    std::vector<int32_t> grp_r(n);
    for (int j = 0; j < n; j++) {
        grp_r[j] = (int)(std::lower_bound(uniq_gid.begin(), uniq_gid.end(), gid[j])
                         - uniq_gid.begin());
    }
    return grp_r;
}

// ---------------------------------------------------------------- P1b 특징 계산 (C++ 타이머 안)
// pos/D0/mD → x0=pos/MiB/mD, x1=D0/MiB/mD, x2=(pos+D0)/MiB/mD
static std::vector<double> compute_p1b_features(
    int n, const double* pos, const double* D0, double mD)
{
    std::vector<double> X(n * 3);
    for (int j = 0; j < n; j++) {
        double p = pos[j], d = D0[j];
        X[j*3+0] = p / MiB / mD;           // 높이
        X[j*3+1] = d / MiB / mD;           // 사용량
        X[j*3+2] = (p + d) / MiB / mD;     // 수준
    }
    return X;
}

// ---------------------------------------------------------------- online_group_T (범용)
// G=1, grp=nullptr → base path; G=G_r, grp=grp_r → rule path
static std::vector<double> online_group_T(
    int n, int G, const int32_t* grp,
    const double* g, const double* h, const double* ts,
    const uint8_t* r, const uint8_t* dec_mask,
    int n_grid, const double* grid,
    double b, double cap, double W_cal, double U,
    const Ledger& led)
{
    std::vector<double> T(n, NaN);

    std::vector<int> dec;
    dec.reserve(n);
    for (int i = 0; i < n; i++) if (dec_mask[i]) dec.push_back(i);
    if (dec.empty()) return T;

    double t0 = ts[dec[0]], t1 = ts[dec[0]];
    for (int i : dec) { t0 = std::min(t0, ts[i]); t1 = std::max(t1, ts[i]); }

    std::vector<int> prev(G, 0);

    int n_steps = (int)std::max(0.0, std::ceil((t1 + U - t0) / U));

    for (int ki = 0; ki < n_steps; ki++) {
        double u = t0 + ki * U;

        auto D_all = led.window(u, u + U);
        std::vector<int> D;
        D.reserve(D_all.size());
        for (int j : D_all) if (dec_mask[j]) D.push_back(j);
        if (D.empty()) continue;

        auto J = led.window(u - cap - W_cal, u - cap);

        if (!J.empty()) {
            auto g_obs = observed(J, u, cap, ts, g, r);
            int nJ = (int)J.size();

            std::vector<double>  wg(G, 0.0);
            std::vector<int32_t> ng(G, 0);
            double w_total = 0.0;
            for (int i = 0; i < nJ; i++) {
                int j = J[i];
                int gj = (G == 1) ? 0 : (int)grp[j];
                wg[gj] += h[j]; ng[gj]++; w_total += h[j];
            }

            std::vector<double> C_sum(n_grid, 0.0);
            std::vector<double> S(n_grid * G, 0.0);

            for (int i = 0; i < nJ; i++) {
                int j = J[i];
                int gj = (G == 1) ? 0 : (int)grp[j];
                double hj = h[j]; double go = g_obs[i];
                for (int ig = 0; ig < n_grid; ig++) {
                    double tv = grid[ig];
                    double c = hj * std::min(go, tv) / b + hj * (go > tv ? 1.0 : 0.0);
                    C_sum[ig]       += c;
                    S[ig * G + gj]  += c;
                }
            }

            int ip = (int)(std::min_element(C_sum.begin(), C_sum.end()) - C_sum.begin());

            std::vector<int> table_g(G, ip);
            for (int gj = 0; gj < G; gj++) {
                if (wg[gj] < MIN_FRAC * w_total || ng[gj] < MIN_N) {
                    table_g[gj] = ip;
                } else {
                    double best_val = std::numeric_limits<double>::infinity();
                    int    best_ig  = ip;
                    for (int ig = 0; ig < n_grid; ig++) {
                        if (S[ig * G + gj] < best_val) {
                            best_val = S[ig * G + gj];
                            best_ig  = ig;
                        }
                    }
                    table_g[gj] = best_ig;
                }
            }
            prev = table_g;
        }

        for (int j : D) {
            int gj = (G == 1) ? 0 : (int)grp[j];
            T[j] = grid[prev[gj]];
        }
    }
    return T;
}

// ---------------------------------------------------------------- P1b encode + Tk 계산
static inline int encode_one(double x, const double* cuts, int n_cuts, int sentinel) {
    if (std::isnan(x)) return sentinel;
    return (int)(std::upper_bound(cuts, cuts + n_cuts, x) - cuts);
}

static std::vector<double> compute_Tk(
    int n, const double* X_p1b, const P1bModel& model, double cap)
{
    std::vector<double> Tk(K_SIZE * n);
    const int s = model.sentinel;
    for (int j = 0; j < n; j++) {
        int c0 = encode_one(X_p1b[j*3+0], model.cuts[0].data(), model.n_cuts[0], s);
        int c1 = encode_one(X_p1b[j*3+1], model.cuts[1].data(), model.n_cuts[1], s);
        int c2 = encode_one(X_p1b[j*3+2], model.cuts[2].data(), model.n_cuts[2], s);
        int idx_lut = (int)model.table[c0 * TABLE_DIM * TABLE_DIM + c1 * TABLE_DIM + c2];
        double rec = model.knots[idx_lut];
        for (int k = 0; k < K_SIZE; k++) {
            Tk[k * n + j] = std::min(rec * KGRID[k], cap);
        }
    }
    return Tk;
}

// ---------------------------------------------------------------- online_kappa_T
static std::vector<double> online_kappa_T(
    int n, const double* Tk,
    const double* g, const double* h, const double* ts,
    const uint8_t* r, const uint8_t* dec_mask,
    double b, double cap, double W_cal, double U,
    const Ledger& led)
{
    std::vector<double> T(n, NaN);

    std::vector<int> dec;
    dec.reserve(n);
    for (int i = 0; i < n; i++) if (dec_mask[i]) dec.push_back(i);
    if (dec.empty()) return T;

    double t0 = ts[dec[0]], t1 = ts[dec[0]];
    for (int i : dec) { t0 = std::min(t0, ts[i]); t1 = std::max(t1, ts[i]); }

    int k_prev = INITIAL_K;
    int n_steps = (int)std::max(0.0, std::ceil((t1 + U - t0) / U));

    for (int ki = 0; ki < n_steps; ki++) {
        double u = t0 + ki * U;

        auto D_all = led.window(u, u + U);
        std::vector<int> D;
        D.reserve(D_all.size());
        for (int j : D_all) if (dec_mask[j]) D.push_back(j);
        if (D.empty()) continue;

        auto J = led.window(u - cap - W_cal, u - cap);

        if (!J.empty()) {
            auto g_obs = observed(J, u, cap, ts, g, r);
            int nJ = (int)J.size();

            double min_cost = std::numeric_limits<double>::infinity();
            int best_k = k_prev;

            for (int k = 0; k < K_SIZE; k++) {
                double cost_k = 0.0;
                for (int i = 0; i < nJ; i++) {
                    int j = J[i];
                    double go  = g_obs[i];
                    double hj  = h[j];
                    double tkj = Tk[k * n + j];
                    cost_k += hj * std::min(go, tkj) / b + hj * (go > tkj ? 1.0 : 0.0);
                }
                if (cost_k < min_cost) { min_cost = cost_k; best_k = k; }
            }
            k_prev = best_k;
        }

        for (int j : D) T[j] = Tk[k_prev * n + j];
    }
    return T;
}


// ---------------------------------------------------------------- online_kappa_T_inc (증분판, 사후 최적화)
// 원본 online_kappa_T와 같은 κ 선택을 증분 갱신으로 계산한다.
//
// 불변 조건 (원본 코드에서 유도):
//   J = {j : u-cap-W_cal <= ts_j < u-cap} 이므로 age = u - ts_j > cap.
//   Tk[k,j] = min(rec_j*KGRID[k], cap) <= cap.
//   (a) r_j && g_j <= cap  → 들어올 때 이미 done, g_obs = g_j 로 고정.
//   (b) 그 밖              → g_obs > cap >= Tk (재사용 전이든 g_j > cap 재사용 후든)
//                            → cost = h_j*Tk/b + h_j, age와 무관.
//   따라서 청크 j의 후보별 기여값 c_jk는 J에 머무는 동안 바뀌지 않는다.
//   → 매 단계 J 전체를 다시 도는 대신, 들어오는 청크의 c_jk를 더하고 나가는 청크의 c_jk를 뺀다.
//
// 정확성 보호:
//   덧셈 순서가 달라 누적 합이 원본과 비트 단위로 같지 않을 수 있다.
//   최소값과 두 번째 값의 차가 tol = KINC_TIE_REL × |최소값| 이하이면
//   그 단계만 원본과 같은 순서·같은 식으로 J를 다시 계산한다(정확 경로).
//   → 증분 합 오차가 tol보다 작은 한 선택된 k는 원본과 같다. equiv 모드가 이를 420창에서 확인한다.
static const double KINC_TIE_REL = 1e-9;

struct KincStats {
    int64_t steps = 0;          // J가 비지 않은 κ 갱신 단계
    int64_t exact_steps = 0;    // 근접 동률로 정확 경로를 쓴 단계
    int64_t resets = 0;         // 창이 통째로 넘어가 합을 0으로 되돌린 횟수
    int64_t adds = 0, removes = 0;
    // CHECK 모드 전용
    int64_t contrib_mismatch = 0;   // 저장된 c_jk ≠ 그 단계에서 새로 계산한 c_jk
    int64_t k_mismatch = 0;         // 증분 argmin ≠ 원본 argmin
    double  max_rel_drift = 0.0;    // max_k |S_inc - S_exact| / S_exact
};

static inline double kinc_gobs(int j, double u, double cap,
                               const double* ts, const double* g, const uint8_t* r) {
    // observed()와 같은 식
    double age = u - ts[j];
    bool done = r[j] && g[j] <= age;
    if (done) return g[j];
    double mx = std::max(age, cap);
    return mx * (1.0 + 1e-9) + 1e-9;
}

// 원본과 같은 순서(J = 원장 위치 lo..hi-1)·같은 식으로 후보별 비용을 계산한다.
static inline void kinc_exact(double* out, int n, int lo, int hi, const Ledger& led,
                              const double* Tk, const double* g, const double* h,
                              const double* ts, const uint8_t* r,
                              double b, double cap, double u) {
    int nJ = hi - lo;
    std::vector<double> g_obs(nJ);
    for (int i = 0; i < nJ; i++) g_obs[i] = kinc_gobs(led.idx[lo + i], u, cap, ts, g, r);
    for (int k = 0; k < K_SIZE; k++) {
        double cost_k = 0.0;
        for (int i = 0; i < nJ; i++) {
            int j = led.idx[lo + i];
            double go = g_obs[i], hj = h[j], tkj = Tk[k * n + j];
            cost_k += hj * std::min(go, tkj) / b + hj * (go > tkj ? 1.0 : 0.0);
        }
        out[k] = cost_k;
    }
}

template <bool CHECK>
static std::vector<double> online_kappa_T_inc(
    int n, const double* Tk,
    const double* g, const double* h, const double* ts,
    const uint8_t* r, const uint8_t* dec_mask,
    double b, double cap, double W_cal, double U,
    const Ledger& led, KincStats* st = nullptr,
    std::vector<int>* k_trace = nullptr)
{
    std::vector<double> T(n, NaN);

    std::vector<int> dec;
    dec.reserve(n);
    for (int i = 0; i < n; i++) if (dec_mask[i]) dec.push_back(i);
    if (dec.empty()) return T;

    double t0 = ts[dec[0]], t1 = ts[dec[0]];
    for (int i : dec) { t0 = std::min(t0, ts[i]); t1 = std::max(t1, ts[i]); }

    // 원장 위치별 기여값 c[pos*K + k] (들어올 때 한 번 계산)
    std::unique_ptr<double[]> C(new double[(size_t)n * K_SIZE]);
    double S[K_SIZE] = {0};
    int lo = 0, hi = 0;           // 현재 J = 원장 위치 [lo, hi)
    const double* tss = led.tss.data();
    const int nl = (int)led.tss.size();

    int k_prev = INITIAL_K;
    int n_steps = (int)std::max(0.0, std::ceil((t1 + U - t0) / U));

    // D(결정 대상) 구간도 단조 이동하므로 포인터로 훑는다 (원본 led.window(u, u+U)와 같은 집합·순서)
    int d_lo = 0, d_hi = 0;

    for (int ki = 0; ki < n_steps; ki++) {
        double u = t0 + ki * U;

        // D = window(u, u+U) ∩ dec_mask — lower_bound 의미와 같게 이동
        while (d_lo < nl && tss[d_lo] < u) d_lo++;
        if (d_hi < d_lo) d_hi = d_lo;
        while (d_hi < nl && tss[d_hi] < u + U) d_hi++;
        bool anyD = false;
        for (int p = d_lo; p < d_hi; p++) if (dec_mask[led.idx[p]]) { anyD = true; break; }
        if (!anyD) continue;

        // J = window(u - cap - W_cal, u - cap)
        const double jlo_t = u - cap - W_cal, jhi_t = u - cap;
        int new_lo = lo;
        while (new_lo < nl && tss[new_lo] < jlo_t) new_lo++;
        if (new_lo >= hi) {
            // 이전 J와 겹치지 않는다 → 합을 정확히 0으로 되돌린다
            if (hi > lo && st) st->resets++;
            for (int k = 0; k < K_SIZE; k++) S[k] = 0.0;
            lo = hi = new_lo;
        } else {
            for (int p = lo; p < new_lo; p++) {        // 나가는 청크
                const double* c = &C[(size_t)p * K_SIZE];
                for (int k = 0; k < K_SIZE; k++) S[k] -= c[k];
            }
            if (st) st->removes += new_lo - lo;
            lo = new_lo;
        }
        int new_hi = hi;
        while (new_hi < nl && tss[new_hi] < jhi_t) new_hi++;
        for (int p = hi; p < new_hi; p++) {            // 들어오는 청크
            int j = led.idx[p];
            double go = kinc_gobs(j, u, cap, ts, g, r);
            double hj = h[j];
            double* c = &C[(size_t)p * K_SIZE];
            for (int k = 0; k < K_SIZE; k++) {
                double tkj = Tk[k * n + j];
                c[k] = hj * std::min(go, tkj) / b + hj * (go > tkj ? 1.0 : 0.0);
                S[k] += c[k];
            }
        }
        if (st) st->adds += new_hi - hi;
        hi = new_hi;

        if (hi > lo) {
            if (st) st->steps++;
            // 증분 합의 argmin과 두 번째 값
            int best = 0; double m1 = S[0], m2 = std::numeric_limits<double>::infinity();
            for (int k = 1; k < K_SIZE; k++) {
                if (S[k] < m1) { m2 = m1; m1 = S[k]; best = k; }
                else if (S[k] < m2) m2 = S[k];
            }
            bool near_tie = !(m2 - m1 > KINC_TIE_REL * std::fabs(m1));

            double Ex[K_SIZE];
            if (near_tie || CHECK) kinc_exact(Ex, n, lo, hi, led, Tk, g, h, ts, r, b, cap, u);

            int k_sel = best;
            if (near_tie) {
                if (st) st->exact_steps++;
                // 원본 선택 규칙: k 오름차순, 엄격히 작을 때만 교체
                double mc = std::numeric_limits<double>::infinity(); int bk = k_prev;
                for (int k = 0; k < K_SIZE; k++) if (Ex[k] < mc) { mc = Ex[k]; bk = k; }
                k_sel = bk;
            }

            if (CHECK && st) {
                double mc = std::numeric_limits<double>::infinity(); int bk = k_prev;
                for (int k = 0; k < K_SIZE; k++) {
                    if (Ex[k] < mc) { mc = Ex[k]; bk = k; }
                    if (Ex[k] > 0) st->max_rel_drift = std::max(st->max_rel_drift,
                                        std::fabs(S[k] - Ex[k]) / Ex[k]);
                }
                if (bk != k_sel) st->k_mismatch++;
                // 불변 조건 직접 확인: 저장된 기여값 == 이 단계에서 새로 계산한 값 (비트 단위)
                for (int p = lo; p < hi; p++) {
                    int j = led.idx[p];
                    double go = kinc_gobs(j, u, cap, ts, g, r);
                    const double* c = &C[(size_t)p * K_SIZE];
                    for (int k = 0; k < K_SIZE; k++) {
                        double tkj = Tk[k * n + j];
                        double cf = h[j] * std::min(go, tkj) / b + h[j] * (go > tkj ? 1.0 : 0.0);
                        if (std::memcmp(&cf, &c[k], 8) != 0) { st->contrib_mismatch++; }
                    }
                }
            }
            k_prev = k_sel;
            if (k_trace) k_trace->push_back(k_prev);
        }

        for (int p = d_lo; p < d_hi; p++) {
            int j = led.idx[p];
            if (dec_mask[j]) T[j] = Tk[k_prev * n + j];
        }
    }
    return T;
}

// 원본 online_kappa_T의 단계별 k 기록판 (equiv 모드 전용, 계산은 원본과 같다)
static std::vector<int> online_kappa_trace_orig(
    int n, const double* Tk,
    const double* g, const double* h, const double* ts,
    const uint8_t* r, const uint8_t* dec_mask,
    double b, double cap, double W_cal, double U, const Ledger& led)
{
    std::vector<int> tr;
    std::vector<int> dec;
    for (int i = 0; i < n; i++) if (dec_mask[i]) dec.push_back(i);
    if (dec.empty()) return tr;
    double t0 = ts[dec[0]], t1 = ts[dec[0]];
    for (int i : dec) { t0 = std::min(t0, ts[i]); t1 = std::max(t1, ts[i]); }
    int k_prev = INITIAL_K;
    int n_steps = (int)std::max(0.0, std::ceil((t1 + U - t0) / U));
    for (int ki = 0; ki < n_steps; ki++) {
        double u = t0 + ki * U;
        auto D_all = led.window(u, u + U);
        bool anyD = false;
        for (int j : D_all) if (dec_mask[j]) { anyD = true; break; }
        if (!anyD) continue;
        auto J = led.window(u - cap - W_cal, u - cap);
        if (J.empty()) continue;
        auto g_obs = observed(J, u, cap, ts, g, r);
        int nJ = (int)J.size();
        double min_cost = std::numeric_limits<double>::infinity();
        int best_k = k_prev;
        for (int k = 0; k < K_SIZE; k++) {
            double cost_k = 0.0;
            for (int i = 0; i < nJ; i++) {
                int j = J[i];
                double go = g_obs[i], hj = h[j], tkj = Tk[k * n + j];
                cost_k += hj * std::min(go, tkj) / b + hj * (go > tkj ? 1.0 : 0.0);
            }
            if (cost_k < min_cost) { min_cost = cost_k; best_k = k; }
        }
        k_prev = best_k;
        tr.push_back(k_prev);
    }
    return tr;
}

// ---------------------------------------------------------------- ledger_diff_events / run_fallback3
struct Events { std::vector<double> t, a; };

static Events ledger_diff_events(
    const std::vector<int>& m_idx,
    const double* ts, const double* g, const double* h, const uint8_t* r,
    const double* Ta, const double* Tb, double b)
{
    int nm = (int)m_idx.size();

    std::vector<double> t_aw(nm), a_aw(nm);
    std::vector<double> t_ar, a_ar;

    for (int i = 0; i < nm; i++) {
        int j = m_idx[i];
        double hold = std::min(g[j], Ta[j]);
        t_aw[i] = ts[j] + hold;
        a_aw[i] = h[j] * hold / b;
        if (r[j] && g[j] > Ta[j]) {
            t_ar.push_back(ts[j] + g[j]);
            a_ar.push_back(h[j]);
        }
    }

    std::vector<double> t_bw(nm), a_bw(nm);
    std::vector<double> t_br, a_br;

    for (int i = 0; i < nm; i++) {
        int j = m_idx[i];
        double hold = std::min(g[j], Tb[j]);
        t_bw[i] = ts[j] + hold;
        a_bw[i] = -h[j] * hold / b;
        if (r[j] && g[j] > Tb[j]) {
            t_br.push_back(ts[j] + g[j]);
            a_br.push_back(-h[j]);
        }
    }

    Events ev;
    ev.t.reserve(2*nm + (int)t_ar.size() + (int)t_br.size());
    ev.a.reserve(ev.t.capacity());
    for (auto& v : {t_aw, t_ar, t_bw, t_br}) ev.t.insert(ev.t.end(), v.begin(), v.end());
    for (auto& v : {a_aw, a_ar, a_bw, a_br}) ev.a.insert(ev.a.end(), v.begin(), v.end());
    return ev;
}

struct FallbackResult {
    std::vector<double> T;
    std::vector<bool>   mode;
    std::vector<int>    idx;
};

static FallbackResult run_fallback3(
    int n,
    const double* g, const double* h, const double* ts, const uint8_t* r,
    const std::vector<double>& T_ai_arr,
    const std::vector<double>& T_b_arr,
    const uint8_t* dec_mask,
    double b, double W)
{
    const double* Ta = T_ai_arr.data();
    const double* Tb = T_b_arr.data();

    std::vector<int> idx;
    idx.reserve(n);
    for (int i = 0; i < n; i++) if (dec_mask[i]) idx.push_back(i);
    std::stable_sort(idx.begin(), idx.end(), [&](int a, int b2){ return ts[a] < ts[b2]; });

    int m_dec = (int)idx.size();

    std::vector<int> m_idx;
    m_idx.reserve(m_dec);
    for (int j : idx) if (Ta[j] != Tb[j]) m_idx.push_back(j);

    auto ev = ledger_diff_events(m_idx, ts, g, h, r, Ta, Tb, b);

    int n_ev = (int)ev.t.size();
    std::vector<int> ev_ord(n_ev);
    std::iota(ev_ord.begin(), ev_ord.end(), 0);
    std::stable_sort(ev_ord.begin(), ev_ord.end(),
                     [&](int a, int b2){ return ev.t[a] < ev.t[b2]; });

    std::vector<double> et(n_ev), ea(n_ev);
    for (int i = 0; i < n_ev; i++) {
        et[i] = ev.t[ev_ord[i]];
        ea[i] = ev.a[ev_ord[i]];
    }

    std::vector<double> cum(n_ev + 1, 0.0);
    for (int i = 0; i < n_ev; i++) cum[i+1] = cum[i] + ea[i];

    std::vector<bool> mode(m_dec);
    for (int i = 0; i < m_dec; i++) {
        double td = ts[idx[i]];
        int hi = (int)(std::upper_bound(et.begin(), et.end(), td) - et.begin());
        int lo = (int)(std::upper_bound(et.begin(), et.end(), td - W) - et.begin());
        mode[i] = (cum[hi] - cum[lo]) > 0.0;
    }

    FallbackResult res;
    res.T = T_ai_arr;
    for (int i = 0; i < m_dec; i++) {
        int j = idx[i];
        res.T[j] = mode[i] ? Tb[j] : Ta[j];
    }
    res.mode = std::move(mode);
    res.idx  = std::move(idx);
    return res;
}

// ---------------------------------------------------------------- 비용 계산
static std::pair<double,double> compute_costs(
    const std::vector<double>& T,
    const double* g, const double* h, const uint8_t* r,
    const uint8_t* te_mask, int n, double b)
{
    double R = 0.0, H = 0.0;
    for (int i = 0; i < n; i++) {
        if (!te_mask[i]) continue;
        double tv = T[i];
        H += h[i] * std::min(g[i], tv);
        if (r[i] && g[i] > tv) R += h[i];
    }
    return {R, H};
}

// ---------------------------------------------------------------- 결과 쓰기 (parity 모드)
static void write_results(const std::string& path, int n,
    const std::vector<double>& T_base,
    const std::vector<double>& T_rule,
    const std::vector<double>& T_rb,
    const std::vector<double>& T_ai,
    const std::vector<double>& T_ai_fb,
    const std::vector<double>& T_c3,
    double R, double H)
{
    std::ofstream f(path, std::ios::binary);
    if (!f) { perror(path.c_str()); std::exit(1); }
    uint32_t hdr[2] = {(uint32_t)n, 0};
    f.write(reinterpret_cast<const char*>(hdr), 8);
    for (const auto& v : {T_base, T_rule, T_rb, T_ai, T_ai_fb, T_c3}) {
        f.write(reinterpret_cast<const char*>(v.data()), n * 8);
    }
    double rh[2] = {R, H};
    f.write(reinterpret_cast<const char*>(rh), 16);
}

// ---------------------------------------------------------------- 한 창 처리 (parity 모드)
static void process_window_parity(
    const Window& W, const P1bModel& model, const std::string& out_path)
{
    Ledger led;
    led.build(W.n, W.ts.data());

    // grp_r 계산 (bin_ids + unique-remap)
    auto grp_r = compute_grp_r(W.n, W.rule_n_rf, W.X_rule.data(), W.rule_edges, W.G_r);

    // T_base: G=1
    auto T_base = online_group_T(
        W.n, 1, nullptr,
        W.g.data(), W.h.data(), W.ts.data(),
        W.r.data(), W.dec_mask.data(),
        W.n_grid, W.grid.data(),
        W.b, W.cap, W.W_cal, W.U, led);

    // T_rule: G=G_r, grp=grp_r (C++에서 계산)
    auto T_rule = online_group_T(
        W.n, W.G_r, grp_r.data(),
        W.g.data(), W.h.data(), W.ts.data(),
        W.r.data(), W.dec_mask.data(),
        W.n_grid, W.grid.data(),
        W.b, W.cap, W.W_cal, W.U, led);

    auto fb_rb = run_fallback3(W.n,
        W.g.data(), W.h.data(), W.ts.data(), W.r.data(),
        T_rule, T_base, W.dec_mask.data(), W.b, W.W_fb);

    // P1b 특징 계산 (pos/D0/mD → x0,x1,x2)
    auto X_p1b = compute_p1b_features(W.n, W.pos.data(), W.D0.data(), W.mD);

    // Tk: encode + lookup + κ-scale
    auto Tk = compute_Tk(W.n, X_p1b.data(), model, W.cap);

    auto T_ai = online_kappa_T_inc<false>(
        W.n, Tk.data(),
        W.g.data(), W.h.data(), W.ts.data(),
        W.r.data(), W.dec_mask.data(),
        W.b, W.cap, W.W_cal, W.U, led);

    auto fb_aifb = run_fallback3(W.n,
        W.g.data(), W.h.data(), W.ts.data(), W.r.data(),
        T_ai, T_base, W.dec_mask.data(), W.b, W.W_fb);

    auto fb_c3 = run_fallback3(W.n,
        W.g.data(), W.h.data(), W.ts.data(), W.r.data(),
        fb_aifb.T, fb_rb.T, W.dec_mask.data(), W.b, W.W_fb);

    auto [R_bytes, H_bytes] = compute_costs(fb_c3.T,
        W.g.data(), W.h.data(), W.r.data(), W.te_mask.data(), W.n, W.b);

    write_results(out_path, W.n,
        T_base, T_rule, fb_rb.T,
        T_ai, fb_aifb.T, fb_c3.T,
        R_bytes / MiB, H_bytes / MiB);
}

// ---------------------------------------------------------------- CPU 타이머
static inline int64_t cpu_ns() {
    struct timespec t;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t);
    return (int64_t)t.tv_sec * 1000000000LL + t.tv_nsec;
}
static inline int64_t wall_ns() {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (int64_t)t.tv_sec * 1000000000LL + t.tv_nsec;
}

static inline uint64_t hash_T(const std::vector<double>& T,
                               const uint8_t* mask, int n) {
    uint64_t h = 14695981039346656037ULL;
    for (int i = 0; i < n; i++) {
        if (!mask[i]) continue;
        uint64_t bits; std::memcpy(&bits, &T[i], 8);
        h ^= bits;
        h *= 1099511628211ULL;
    }
    return h;
}

// ---------------------------------------------------------------- CPU_rule 측정
struct MeasureResult {
    int64_t cpu_ns;
    int64_t wall_ns_elapsed;
    uint64_t checksum;
    int64_t n_decisions;
    double R_sum;
    double H_sum;
};

static MeasureResult run_cpu_rule(const std::vector<Window>& windows) {
    uint64_t checksum = 0;
    int64_t n_dec = 0;
    double R_sum = 0.0, H_sum = 0.0;

    int64_t w0 = wall_ns();
    int64_t c0 = cpu_ns();

    for (const auto& W : windows) {
        Ledger led;
        led.build(W.n, W.ts.data());

        // grp_r 계산 (bin_ids + unique-remap) — 타이머 안
        auto grp_r = compute_grp_r(W.n, W.rule_n_rf, W.X_rule.data(), W.rule_edges, W.G_r);

        auto T_base = online_group_T(
            W.n, 1, nullptr,
            W.g.data(), W.h.data(), W.ts.data(),
            W.r.data(), W.dec_mask.data(),
            W.n_grid, W.grid.data(),
            W.b, W.cap, W.W_cal, W.U, led);

        auto T_rule = online_group_T(
            W.n, W.G_r, grp_r.data(),
            W.g.data(), W.h.data(), W.ts.data(),
            W.r.data(), W.dec_mask.data(),
            W.n_grid, W.grid.data(),
            W.b, W.cap, W.W_cal, W.U, led);

        auto fb_rb = run_fallback3(W.n,
            W.g.data(), W.h.data(), W.ts.data(), W.r.data(),
            T_rule, T_base, W.dec_mask.data(), W.b, W.W_fb);

        auto [R, H] = compute_costs(fb_rb.T,
            W.g.data(), W.h.data(), W.r.data(), W.te_mask.data(), W.n, W.b);

        checksum ^= hash_T(fb_rb.T, W.te_mask.data(), W.n);
        for (int i = 0; i < W.n; i++) if (W.te_mask[i]) n_dec++;
        R_sum += R; H_sum += H;
    }

    int64_t c1 = cpu_ns();
    int64_t w1 = wall_ns();

    volatile uint64_t sink = checksum; (void)sink;
    volatile double   dsink = R_sum + H_sum; (void)dsink;

    return {c1 - c0, w1 - w0, checksum, n_dec, R_sum, H_sum};
}

// ---------------------------------------------------------------- CPU_C3 측정
static MeasureResult run_cpu_c3(const std::vector<Window>& windows,
                                 const P1bModel& model) {
    uint64_t checksum = 0;
    int64_t n_dec = 0;
    double R_sum = 0.0, H_sum = 0.0;

    int64_t w0 = wall_ns();
    int64_t c0 = cpu_ns();

    for (const auto& W : windows) {
        Ledger led;
        led.build(W.n, W.ts.data());

        // grp_r 계산 — 타이머 안 (CPU_rule과 공통)
        auto grp_r = compute_grp_r(W.n, W.rule_n_rf, W.X_rule.data(), W.rule_edges, W.G_r);

        auto T_base = online_group_T(
            W.n, 1, nullptr,
            W.g.data(), W.h.data(), W.ts.data(),
            W.r.data(), W.dec_mask.data(),
            W.n_grid, W.grid.data(),
            W.b, W.cap, W.W_cal, W.U, led);

        auto T_rule = online_group_T(
            W.n, W.G_r, grp_r.data(),
            W.g.data(), W.h.data(), W.ts.data(),
            W.r.data(), W.dec_mask.data(),
            W.n_grid, W.grid.data(),
            W.b, W.cap, W.W_cal, W.U, led);

        auto fb_rb = run_fallback3(W.n,
            W.g.data(), W.h.data(), W.ts.data(), W.r.data(),
            T_rule, T_base, W.dec_mask.data(), W.b, W.W_fb);

        // C3 추가: P1b 특징 계산 — 타이머 안
        auto X_p1b = compute_p1b_features(W.n, W.pos.data(), W.D0.data(), W.mD);

        // encode + lookup + κ-scale
        auto Tk = compute_Tk(W.n, X_p1b.data(), model, W.cap);

        auto T_ai = online_kappa_T_inc<false>(
            W.n, Tk.data(),
            W.g.data(), W.h.data(), W.ts.data(),
            W.r.data(), W.dec_mask.data(),
            W.b, W.cap, W.W_cal, W.U, led);

        auto fb_aifb = run_fallback3(W.n,
            W.g.data(), W.h.data(), W.ts.data(), W.r.data(),
            T_ai, T_base, W.dec_mask.data(), W.b, W.W_fb);

        auto fb_c3 = run_fallback3(W.n,
            W.g.data(), W.h.data(), W.ts.data(), W.r.data(),
            fb_aifb.T, fb_rb.T, W.dec_mask.data(), W.b, W.W_fb);

        auto [R, H] = compute_costs(fb_c3.T,
            W.g.data(), W.h.data(), W.r.data(), W.te_mask.data(), W.n, W.b);

        checksum ^= hash_T(fb_c3.T, W.te_mask.data(), W.n);
        for (int i = 0; i < W.n; i++) if (W.te_mask[i]) n_dec++;
        R_sum += R; H_sum += H;
    }

    int64_t c1 = cpu_ns();
    int64_t w1 = wall_ns();

    volatile uint64_t sink = checksum; (void)sink;
    volatile double   dsink = R_sum + H_sum; (void)dsink;

    return {c1 - c0, w1 - w0, checksum, n_dec, R_sum, H_sum};
}

// ---------------------------------------------------------------- main
int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "사용:\n"
                "  %s parity  <window_dir> <model_path> <result_dir>\n"
                "  %s measure <window_dir> <model_path> [--dev-test] --out-json <path>\n"
                "  %s equiv   <window_dir> <model_path> [--out-json <path>]\n"
                "  %s kbench  <window_dir> <model_path> [--out-json <path>]\n",
                argv[0], argv[0], argv[0], argv[0]);
        return 1;
    }

    std::string mode_str  = argv[1];
    std::string win_dir   = argv[2];
    std::string model_path= argv[3];

    if (mode_str != "parity" && mode_str != "measure" &&
        mode_str != "equiv" && mode_str != "kbench") {
        fprintf(stderr, "알 수 없는 모드: %s\n", mode_str.c_str());
        return 1;
    }

    P1bModel model = read_model(model_path);
    printf("[모델] cuts=%d/%d/%d  knots=%d  sentinel=%d\n",
           model.n_cuts[0], model.n_cuts[1], model.n_cuts[2],
           (int)model.knots.size(), model.sentinel);

    // ---------------------------------------------------------------- parity 모드
    if (mode_str == "parity") {
        if (argc < 5) {
            fprintf(stderr, "parity 모드: <result_dir> 인자 필요\n"); return 1;
        }
        std::string res_dir = argv[4];
        fs::create_directories(res_dir);

        std::vector<fs::path> bins;
        for (const auto& e : fs::directory_iterator(win_dir)) {
            if (e.path().extension() == ".bin") bins.push_back(e.path());
        }
        std::sort(bins.begin(), bins.end());

        for (const auto& bp : bins) {
            auto W = read_window(bp.string());
            auto out = fs::path(res_dir) / bp.filename();
            process_window_parity(W, model, out.string());
            printf("  %s  n=%d  G_r=%d  n_rf=%d  n_grid=%d\n",
                   bp.filename().c_str(), W.n, W.G_r, W.rule_n_rf, W.n_grid);
        }
        printf("완료: %zu 창 처리\n", bins.size());
        return 0;
    }

    // ---------------------------------------------------------------- equiv / kbench 공통 로드
    if (mode_str == "equiv" || mode_str == "kbench") {
        std::string out_json;
        for (int i = 4; i < argc; i++)
            if (std::string(argv[i]) == "--out-json" && i+1 < argc) out_json = argv[i+1];
        std::vector<fs::path> bins;
        for (const auto& e : fs::directory_iterator(win_dir))
            if (e.path().extension() == ".bin") bins.push_back(e.path());
        std::sort(bins.begin(), bins.end());
        std::vector<Window> windows;
        for (const auto& bp : bins) windows.push_back(read_window(bp.string()));
        printf("[%s] 창 %zu개\n", mode_str.c_str(), windows.size());
        FILE* jf = out_json.empty() ? nullptr : fopen(out_json.c_str(), "w");

        if (mode_str == "equiv") {
            // 창마다: 단계별 k 수열, T_ai, T_ai_fb, T_c3 비트 비교 + R/H 비교 + 기여값 불변 직접 확인
            KincStats tot;
            int win_fail = 0;
            int64_t n_cmp_T = 0, n_bad_T = 0, n_bad_trace = 0;
            double R_o = 0, H_o = 0, R_i = 0, H_i = 0;
            uint64_t ck_o = 0, ck_i = 0;
            auto bit_eq = [](double a, double b) { return std::memcmp(&a, &b, 8) == 0; };
            for (size_t wi = 0; wi < windows.size(); wi++) {
                const Window& W = windows[wi];
                Ledger led; led.build(W.n, W.ts.data());
                auto grp_r = compute_grp_r(W.n, W.rule_n_rf, W.X_rule.data(), W.rule_edges, W.G_r);
                auto T_base = online_group_T(W.n, 1, nullptr, W.g.data(), W.h.data(), W.ts.data(),
                    W.r.data(), W.dec_mask.data(), W.n_grid, W.grid.data(), W.b, W.cap, W.W_cal, W.U, led);
                auto T_rule = online_group_T(W.n, W.G_r, grp_r.data(), W.g.data(), W.h.data(), W.ts.data(),
                    W.r.data(), W.dec_mask.data(), W.n_grid, W.grid.data(), W.b, W.cap, W.W_cal, W.U, led);
                auto fb_rb = run_fallback3(W.n, W.g.data(), W.h.data(), W.ts.data(), W.r.data(),
                    T_rule, T_base, W.dec_mask.data(), W.b, W.W_fb);
                auto X_p1b = compute_p1b_features(W.n, W.pos.data(), W.D0.data(), W.mD);
                auto Tk = compute_Tk(W.n, X_p1b.data(), model, W.cap);

                auto T_ai_o = online_kappa_T(W.n, Tk.data(), W.g.data(), W.h.data(), W.ts.data(),
                    W.r.data(), W.dec_mask.data(), W.b, W.cap, W.W_cal, W.U, led);
                auto tr_o = online_kappa_trace_orig(W.n, Tk.data(), W.g.data(), W.h.data(), W.ts.data(),
                    W.r.data(), W.dec_mask.data(), W.b, W.cap, W.W_cal, W.U, led);
                KincStats st; std::vector<int> tr_i;
                auto T_ai_i = online_kappa_T_inc<true>(W.n, Tk.data(), W.g.data(), W.h.data(), W.ts.data(),
                    W.r.data(), W.dec_mask.data(), W.b, W.cap, W.W_cal, W.U, led, &st, &tr_i);

                auto c3 = [&](const std::vector<double>& T_ai) {
                    auto fa = run_fallback3(W.n, W.g.data(), W.h.data(), W.ts.data(), W.r.data(),
                        T_ai, T_base, W.dec_mask.data(), W.b, W.W_fb);
                    auto fc = run_fallback3(W.n, W.g.data(), W.h.data(), W.ts.data(), W.r.data(),
                        fa.T, fb_rb.T, W.dec_mask.data(), W.b, W.W_fb);
                    return std::make_pair(fa.T, fc.T);
                };
                auto [fa_o, fc_o] = c3(T_ai_o);
                auto [fa_i, fc_i] = c3(T_ai_i);
                auto [Ro, Ho] = compute_costs(fc_o, W.g.data(), W.h.data(), W.r.data(), W.te_mask.data(), W.n, W.b);
                auto [Ri, Hi] = compute_costs(fc_i, W.g.data(), W.h.data(), W.r.data(), W.te_mask.data(), W.n, W.b);
                R_o += Ro; H_o += Ho; R_i += Ri; H_i += Hi;
                ck_o ^= hash_T(fc_o, W.te_mask.data(), W.n);
                ck_i ^= hash_T(fc_i, W.te_mask.data(), W.n);

                int64_t bad = 0;
                for (int j = 0; j < W.n; j++) {
                    n_cmp_T += 3;
                    bad += !bit_eq(T_ai_o[j], T_ai_i[j]);
                    bad += !bit_eq(fa_o[j], fa_i[j]);
                    bad += !bit_eq(fc_o[j], fc_i[j]);
                }
                bool trace_ok = (tr_o == tr_i);
                bool rh_ok = bit_eq(Ro, Ri) && bit_eq(Ho, Hi);
                if (!trace_ok) n_bad_trace++;
                n_bad_T += bad;
                bool ok = trace_ok && bad == 0 && rh_ok && st.contrib_mismatch == 0 && st.k_mismatch == 0;
                if (!ok) {
                    win_fail++;
                    printf("  [불일치] %s trace=%d badT=%lld RH=%d contrib=%lld kmis=%lld\n",
                        bins[wi].filename().c_str(), (int)trace_ok, (long long)bad, (int)rh_ok,
                        (long long)st.contrib_mismatch, (long long)st.k_mismatch);
                }
                tot.steps += st.steps; tot.exact_steps += st.exact_steps; tot.resets += st.resets;
                tot.adds += st.adds; tot.removes += st.removes;
                tot.contrib_mismatch += st.contrib_mismatch; tot.k_mismatch += st.k_mismatch;
                tot.max_rel_drift = std::max(tot.max_rel_drift, st.max_rel_drift);
            }
            printf("[equiv] 창 %zu개 중 불일치 %d개\n", windows.size(), win_fail);
            printf("[equiv] κ 단계 %lld, 정확 경로 %lld, 리셋 %lld, 기여값 불일치 %lld, k 불일치 %lld\n",
                (long long)tot.steps, (long long)tot.exact_steps, (long long)tot.resets,
                (long long)tot.contrib_mismatch, (long long)tot.k_mismatch);
            printf("[equiv] T 비교 %lld개 중 다름 %lld, k 수열 다른 창 %lld\n",
                (long long)n_cmp_T, (long long)n_bad_T, (long long)n_bad_trace);
            printf("[equiv] 증분 합 최대 상대 오차 %.3e (허용 %.0e)\n", tot.max_rel_drift, KINC_TIE_REL);
            printf("[equiv] R/H 원본 %.10g / %.10g MiB, 증분 %.10g / %.10g MiB\n",
                R_o / MiB, H_o / MiB, R_i / MiB, H_i / MiB);
            printf("[equiv] C3 체크섬 원본 %016llx 증분 %016llx\n",
                (unsigned long long)ck_o, (unsigned long long)ck_i);
            if (jf) {
                fprintf(jf, "{\"n_windows\": %zu, \"windows_mismatch\": %d, \"kappa_steps\": %lld,"
                    " \"exact_steps\": %lld, \"resets\": %lld, \"contrib_mismatch\": %lld,"
                    " \"k_mismatch\": %lld, \"T_compared\": %lld, \"T_diff\": %lld,"
                    " \"trace_diff_windows\": %lld, \"max_rel_drift\": %.6e, \"tie_rel\": %.1e,"
                    " \"R_orig_MiB\": %.12g, \"H_orig_MiB\": %.12g, \"R_inc_MiB\": %.12g, \"H_inc_MiB\": %.12g,"
                    " \"checksum_c3_orig\": \"%016llx\", \"checksum_c3_inc\": \"%016llx\"}\n",
                    windows.size(), win_fail, (long long)tot.steps, (long long)tot.exact_steps,
                    (long long)tot.resets, (long long)tot.contrib_mismatch, (long long)tot.k_mismatch,
                    (long long)n_cmp_T, (long long)n_bad_T, (long long)n_bad_trace, tot.max_rel_drift,
                    KINC_TIE_REL, R_o / MiB, H_o / MiB, R_i / MiB, H_i / MiB,
                    (unsigned long long)ck_o, (unsigned long long)ck_i);
                fclose(jf);
            }
            return win_fail == 0 ? 0 : 2;
        }

        // kbench: ③만 원본/증분 CPU 비교. Ledger·Tk는 타이머 밖에서 미리 만든다.
        std::vector<Ledger> leds(windows.size());
        std::vector<std::vector<double>> Tks(windows.size());
        for (size_t wi = 0; wi < windows.size(); wi++) {
            const Window& W = windows[wi];
            leds[wi].build(W.n, W.ts.data());
            auto X = compute_p1b_features(W.n, W.pos.data(), W.D0.data(), W.mD);
            Tks[wi] = compute_Tk(W.n, X.data(), model, W.cap);
        }
        auto run = [&](bool inc, uint64_t& ck) {
            ck = 0;
            int64_t c0 = cpu_ns();
            for (size_t wi = 0; wi < windows.size(); wi++) {
                const Window& W = windows[wi];
                auto T = inc
                    ? online_kappa_T_inc<false>(W.n, Tks[wi].data(), W.g.data(), W.h.data(), W.ts.data(),
                        W.r.data(), W.dec_mask.data(), W.b, W.cap, W.W_cal, W.U, leds[wi])
                    : online_kappa_T(W.n, Tks[wi].data(), W.g.data(), W.h.data(), W.ts.data(),
                        W.r.data(), W.dec_mask.data(), W.b, W.cap, W.W_cal, W.U, leds[wi]);
                ck ^= hash_T(T, W.dec_mask.data(), W.n);
            }
            return cpu_ns() - c0;
        };
        uint64_t cko, cki;
        run(false, cko); run(true, cki);   // 워밍업
        const char* ord[3] = {"orig_inc", "inc_orig", "orig_inc"};
        if (jf) fprintf(jf, "{\"n_windows\": %zu, \"iterations\": [\n", windows.size());
        for (int it = 0; it < 3; it++) {
            int64_t to, ti;
            if (it == 1) { ti = run(true, cki); to = run(false, cko); }
            else         { to = run(false, cko); ti = run(true, cki); }
            printf("[kbench] %s  orig_ns=%lld  inc_ns=%lld  ratio=%.1f  checksum %s\n", ord[it],
                (long long)to, (long long)ti, (double)to / (double)ti, cko == cki ? "같음" : "다름");
            if (jf) fprintf(jf, "  {\"order\": \"%s\", \"orig_ns\": %lld, \"inc_ns\": %lld,"
                " \"checksum_orig\": \"%016llx\", \"checksum_inc\": \"%016llx\"}%s\n", ord[it],
                (long long)to, (long long)ti, (unsigned long long)cko, (unsigned long long)cki,
                it < 2 ? "," : "");
        }
        if (jf) { fprintf(jf, "]}\n"); fclose(jf); }
        return cko == cki ? 0 : 2;
    }

    // ---------------------------------------------------------------- measure 모드
    bool dev_test = false;
    std::string out_json;
    for (int i = 4; i < argc; i++) {
        if (std::string(argv[i]) == "--dev-test") dev_test = true;
        if (std::string(argv[i]) == "--out-json" && i+1 < argc) out_json = argv[i+1];
    }
    if (out_json.empty()) {
        fprintf(stderr, "measure 모드: --out-json <path> 필요\n"); return 1;
    }

    std::vector<fs::path> bins;
    for (const auto& e : fs::directory_iterator(win_dir)) {
        if (e.path().extension() == ".bin") bins.push_back(e.path());
    }
    std::sort(bins.begin(), bins.end());
    if (dev_test && (int)bins.size() > 5) bins.resize(5);

    printf("[measure] 창 %zu 개 로드 중...\n", bins.size());

    std::vector<Window> windows;
    windows.reserve(bins.size());
    for (const auto& bp : bins) windows.push_back(read_window(bp.string()));

    printf("[measure] 적재 완료: %zu 창\n", windows.size());

    printf("[measure] 워밍업...\n");
    auto wu_rule = run_cpu_rule(windows);
    auto wu_c3   = run_cpu_c3(windows, model);
    printf("[measure] 워밍업 완료 (rule_ns=%lld  c3_ns=%lld)\n",
           (long long)wu_rule.cpu_ns, (long long)wu_c3.cpu_ns);

    struct IterResult {
        std::string order;
        MeasureResult rule, c3;
    };
    std::vector<IterResult> iters;

    auto do_iter = [&](const std::string& order) {
        IterResult ir; ir.order = order;
        if (order == "rule_c3") {
            ir.rule = run_cpu_rule(windows);
            ir.c3   = run_cpu_c3(windows, model);
        } else {
            ir.c3   = run_cpu_c3(windows, model);
            ir.rule = run_cpu_rule(windows);
        }
        printf("[측정] %s  rule_ns=%lld  c3_ns=%lld  dI_ns=%lld\n",
               order.c_str(),
               (long long)ir.rule.cpu_ns,
               (long long)ir.c3.cpu_ns,
               (long long)(ir.c3.cpu_ns - ir.rule.cpu_ns));
        iters.push_back(ir);
    };

    do_iter("rule_c3");
    do_iter("c3_rule");
    do_iter("rule_c3");

    fs::create_directories(fs::path(out_json).parent_path());
    FILE* jf = fopen(out_json.c_str(), "w");
    if (!jf) { perror(out_json.c_str()); return 1; }
    fprintf(jf, "{\n");
    fprintf(jf, "  \"n_windows\": %zu,\n", windows.size());
    fprintf(jf, "  \"n_decisions_rule\": %lld,\n", (long long)iters[0].rule.n_decisions);
    fprintf(jf, "  \"n_decisions_c3\": %lld,\n", (long long)iters[0].c3.n_decisions);
    fprintf(jf, "  \"R_sum_MiB\": %.10g,\n", iters[0].c3.R_sum / MiB);
    fprintf(jf, "  \"H_sum_MiB\": %.10g,\n", iters[0].c3.H_sum / MiB);
    fprintf(jf, "  \"warmup\": {\"cpu_rule_ns\": %lld, \"cpu_c3_ns\": %lld},\n",
            (long long)wu_rule.cpu_ns, (long long)wu_c3.cpu_ns);
    fprintf(jf, "  \"iterations\": [\n");
    for (int i = 0; i < (int)iters.size(); i++) {
        const auto& it = iters[i];
        fprintf(jf, "    {\"order\": \"%s\","
                " \"cpu_rule_ns\": %lld, \"cpu_c3_ns\": %lld,"
                " \"dI_ns\": %lld,"
                " \"wall_rule_ns\": %lld, \"wall_c3_ns\": %lld,"
                " \"checksum_rule\": \"%016llx\", \"checksum_c3\": \"%016llx\"}%s\n",
                it.order.c_str(),
                (long long)it.rule.cpu_ns, (long long)it.c3.cpu_ns,
                (long long)(it.c3.cpu_ns - it.rule.cpu_ns),
                (long long)it.rule.wall_ns_elapsed, (long long)it.c3.wall_ns_elapsed,
                (unsigned long long)it.rule.checksum, (unsigned long long)it.c3.checksum,
                (i+1 < (int)iters.size()) ? "," : "");
    }
    fprintf(jf, "  ]\n}\n");
    fclose(jf);
    printf("[measure] JSON 저장: %s\n", out_json.c_str());

    return 0;
}
