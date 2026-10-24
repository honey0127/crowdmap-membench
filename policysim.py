#!/usr/bin/env python3
"""policysim.py — 메모리 반납 정책 시뮬레이터: 같은 일정에서 '빈 메모리(낭비)'와 '새로 폴트한 메모리(CPU)'를 정책별로 계산한다.

모델(유체 근사, 단편화 없음)
  D(t) = 쓰고 있는 본문 바이트, H(t) = 잡고 있는 바이트(H ≥ D).
  D가 H를 넘으면 그만큼 새 페이지 폴트(+ 나중의 반납) = CPU 비용. 정책은 언제 H를 낮출지(반납)만 정한다.
  미리 잡아 두기(선확보)는 폴트 수를 줄이지 못한다(폴트 시점만 앞당김). 그래서 이 축에서 정책 = '언제 반납하나'.
  높이 y의 메모리 '조각'은 D가 y 아래로 내려간 때부터 다시 y에 닿을 때까지 논다(놀이 시간 g, 조각 두께 h).
    timeout(T) : 조각을 T초까지 들고 있다가 반납 → 낭비 h·min(g,T), g>T면 재폴트 h      (≈ jemalloc 감쇠; 스키 대여 결정론 해법)
    oracle(T*) : g를 미리 앎 → g≤T*면 끝까지 보유(낭비 h·g), 아니면 즉시 반납(낭비 0, 재폴트 h)
  처음 높이에 도달할 때의 폴트(cold)는 어떤 정책도 피할 수 없다.

두 가지 상한
  1) 오라클 상한: 조각마다 g를 정확히 안다고 할 때. '완벽한 정보의 가치'라서 무작위 도착에서도 크게 나온다(느슨함).
  2) 학습 가능 몫(--learn): 조각이 놀기 시작할 때 알 수 있는 특징으로 그룹(특징당 4칸, 최대 2개 조합)을 나눠
     그룹마다 타임아웃을 따로 고른다. 앞 절반에서 고르고 뒤 절반에서 평가한다(과적합 방지).
     특징: 높이(그때 사용량 위 높이), 사용량, 유입1s·유입10s(최근 유입률), 이전놀이(같은 높이 1MiB 칸에서
     직전에 끝난 놀이 시간), 위상(--period를 줄 때). 깊이 2 결정 트리 수준의 단순 학습 모델에 해당한다.
     비용 = 재폴트 MiB + 낭비(MiB·s) / T*.  T* = 손익분기 시간 = (폴트·반납 1MiB의 CPU) ÷ (1MiB를 1초 쥐는 값).

사용
  python3 policysim.py SCHEDULE.csv [--label 이름] [--meas out/results.csv] [--out out/policy]
                       [--learn] [--period 초] [--tstars 1,10] [--churn-ms-per-mib 0.6]
    SCHEDULE.csv : traces.py schedule 결과(t_alloc_s,t_free_s,size_bytes)
    --meas       : bench.sh sweep 결과. 같은 일정의 실측 점(jemalloc·glibc)을 표·그림에 겹친다.
    --learn      : 학습 가능 몫 표를 출력한다. --period를 주면 주기 위상 특징도 쓴다.
    --churn-ms-per-mib : 폴트+반납 1MiB당 CPU(ms). 기본 0.6 = 실측(je_0 − je_−1 CPU ÷ 폴트 차이)
  출력: 표(화면), <out>_curve.csv, <out>.png
필요: numpy, matplotlib(그림)
"""
import argparse
import bisect
import csv
import itertools
import math

import numpy as np

MiB = 1024.0 * 1024.0
T_GRID = np.concatenate([[0.0], np.logspace(-3, 4, 141), [math.inf]])


def load_schedule(path):
    ta, tf, sz = [], [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(line for line in f if not line.startswith("#")):
            ta.append(float(row["t_alloc_s"]))
            tf.append(float(row["t_free_s"]))
            sz.append(float(row["size_bytes"]))
    return np.array(ta), np.array(tf), np.array(sz)


def slices(ta, tf, sz):
    """메모리 조각별 (놀이 시간 g, 두께 h, 재사용 여부, 놀기 시작한 시각 ts, 그때의 사용량 D0, D0 위 높이 pos)"""
    t = np.concatenate([ta, tf])
    d = np.concatenate([sz, -sz])
    order = np.lexsort((d, t))  # 같은 시각이면 반납(음수) 먼저
    t, d = t[order], d[order]
    t0 = float(t[0])
    D = 0.0
    idle = []  # [lo, hi, ts, D0], idle[-1]이 가장 낮은(가장 최근에 놀기 시작한) 조각
    out = {k: [] for k in ("g", "h", "reused", "ts", "D0", "pos", "prevgap")}
    cold = 0.0
    area = 0.0
    prev = t0
    BUCKET = 1.0 * MiB  # 높이 1MiB 칸마다 '직전에 끝난 놀이 시간' 기록
    hist = {}  # 칸 → ([끝난 시각...], [놀이 시간...]) (시간 순)

    def lookup(b, ts):
        ends, gaps = hist.get(b, ((), ()))
        i = bisect.bisect_right(ends, ts) - 1
        return gaps[i] if i >= 0 else math.nan

    def emit(g, lo, hi, ts, D0, reused, t_end_gap):
        b = int(((lo + hi) / 2) // BUCKET)
        out["g"].append(g)
        out["h"].append(hi - lo)
        out["reused"].append(reused)
        out["ts"].append(ts)
        out["D0"].append(D0)
        out["pos"].append((lo + hi) / 2 - D0)
        out["prevgap"].append(lookup(b, ts))  # 놀기 시작한 시점(ts)까지 알 수 있던 정보만 쓴다
        if reused:
            e = hist.setdefault(b, ([], []))
            e[0].append(t_end_gap)
            e[1].append(g)

    for ti, di in zip(t.tolist(), d.tolist()):
        area += D * (ti - prev)
        prev = ti
        if di < 0:
            lo = D + di
            if idle and idle[-1][2] == ti and idle[-1][0] == D:  # 같은 시각의 연속 감소는 합친다
                idle[-1][0] = lo
                idle[-1][3] = lo
            else:
                idle.append([lo, D, ti, lo])
            D = lo
        else:
            target = D + di
            level = D
            while level < target and idle:
                lo, hi, ts, D0 = idle[-1]
                top = min(hi, target)
                emit(ti - ts, lo, top, ts, D0, True, ti)
                if hi <= target:
                    idle.pop()
                else:
                    idle[-1][0] = target
                level = top
            if level < target:
                cold += target - level
            D = target
    t_end = float(t[-1])
    for lo, hi, ts, D0 in idle:
        emit(t_end - ts, lo, hi, ts, D0, False, t_end)
    S = {k: np.array(v) for k, v in out.items()}
    S["reused"] = S["reused"].astype(bool)
    S.update(cold=cold, dur=t_end - t0, t0=t0, t_end=t_end, dmean=area / (t_end - t0))
    return S


def eval_timeout(T, S):
    g, h, r = S["g"], S["h"], S["reused"]
    waste = (h * np.minimum(g, T)).sum()
    refault = h[r & (g > T)].sum()
    return waste / S["dur"] / MiB, (S["cold"] + refault) / S["dur"] / MiB


def eval_oracle(T, S):
    g, h, r = S["g"], S["h"], S["reused"]
    keep = r & (g <= T)
    waste = (h[keep] * g[keep]).sum()
    refault = h[r & (g > T)].sum()
    return waste / S["dur"] / MiB, (S["cold"] + refault) / S["dur"] / MiB


def curves(S):
    to = np.array([eval_timeout(T, S) for T in T_GRID])  # (낭비 MiB, 폴트 MiB/s)
    orc = np.array([eval_oracle(T, S) for T in T_GRID])
    return to, orc


def lower_envelope_at(curve, w):
    """곡선에서 낭비 ≤ w일 때 가능한 최소 폴트(인접 점 선형 보간 = 두 정책을 섞어 쓰는 것과 같다)"""
    pts = sorted(set(map(tuple, curve.tolist())))
    xs = np.array([p[0] for p in pts])
    best = np.minimum.accumulate(np.array([p[1] for p in pts]))
    if w <= xs[0]:
        return float(best[0])
    if w >= xs[-1]:
        return float(best[-1])
    return float(np.interp(w, xs, best))


# ---------------------------------------------------------------- 학습 가능 몫
def slice_costs(g, h, reused, Tstar):
    """T_GRID의 각 타임아웃에 대한 조각별 비용 행렬 (len(T_GRID) × n). 비용 = 재폴트 + 낭비/T*"""
    Tg = T_GRID[:, None]
    waste = h[None, :] * np.minimum(g[None, :], Tg)
    refault = h[None, :] * (reused[None, :] & (g[None, :] > Tg))
    return refault + waste / Tstar


def features(S, ta, sz, period):
    ts = S["ts"]
    order = np.argsort(ta)
    cum = np.concatenate([[0.0], np.cumsum(sz[order])])
    tsorted = ta[order]

    def rate(W):
        hi = np.searchsorted(tsorted, ts, side="right")
        lo = np.searchsorted(tsorted, ts - W, side="right")
        return (cum[hi] - cum[lo]) / W / MiB

    pg = S["prevgap"]
    F = {"높이": S["pos"] / MiB, "유입1s": rate(1.0), "유입10s": rate(10.0), "사용량": S["D0"] / MiB,
         "이전놀이": np.where(np.isnan(pg), -1e9, np.log10(np.maximum(pg, 1e-6)))}
    if period:
        F["위상"] = (ts % period) / period
    return F


def bin_ids(x_train, x_all, nbins=4):
    qs = np.unique(np.quantile(x_train, np.linspace(0, 1, nbins + 1)[1:-1]))
    return np.searchsorted(qs, x_all, side="right")


def learnable(S, ta, sz, Tstars, period):
    F = features(S, ta, sz, period)
    mid = S["t0"] + S["dur"] / 2
    tr = S["ts"] < mid
    te = ~tr
    g, h, r = S["g"], S["h"], S["reused"]
    names = list(F)
    sets = [(n,) for n in names] + [p for p in itertools.combinations(names, 2)
                                    if "높이" in p or "위상" in p or "이전놀이" in p]
    rows = []
    for Tstar in Tstars:
        C = slice_costs(g, h, r, Tstar)  # (nT, n)
        i_single = int(np.argmin(C[:, tr].sum(axis=1)))
        single = C[i_single, te].sum()
        oracle = (h[te] * np.where(r[te], np.minimum(g[te] / Tstar, 1.0), 0.0)).sum()
        best = (single, "(없음)")
        per_set = {}
        for fs in sets:
            gid = np.zeros(len(g), dtype=np.int64)
            for n in fs:
                gid = gid * 8 + bin_ids(F[n][tr], F[n])
            cost = 0.0
            for k in np.unique(gid[te]):
                mtr, mte = tr & (gid == k), te & (gid == k)
                if h[mtr].sum() < 0.01 * h[tr].sum():  # 학습 구간에 표본이 거의 없으면 전체 타임아웃 사용
                    i = i_single
                else:
                    i = int(np.argmin(C[:, mtr].sum(axis=1)))
                cost += C[i, mte].sum()
            per_set["×".join(fs)] = cost
            if cost < best[0]:
                best = (cost, "×".join(fs))
        rows.append(dict(Tstar=Tstar, single=single, T_single=T_GRID[i_single], oracle=oracle, best=best,
                         per_set=per_set))
    return rows


def read_meas(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(line for line in f if not line.startswith("#")))
    by, out = {}, {}
    for r in rows:
        by.setdefault(r["label"], []).append(r)
    for lab, rs in by.items():
        dur = np.mean([float(r["dur_s"]) for r in rs])
        out[lab] = dict(waste=np.mean([float(r["waste_mean_MB"]) for r in rs]),
                        fault=np.mean([float(r["minflt"]) for r in rs]) * 4096 / MiB / dur,
                        cpu=np.mean([float(r["cpu_ms"]) for r in rs]) / dur / 10.0)  # % of one core
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("schedule")
    ap.add_argument("--label", default="")
    ap.add_argument("--meas", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--learn", action="store_true")
    ap.add_argument("--period", type=float, default=0.0)
    ap.add_argument("--tstars", default="1,10")
    ap.add_argument("--churn-ms-per-mib", type=float, default=0.6)
    a = ap.parse_args()
    label = a.label or a.schedule.rsplit("/", 1)[-1].removesuffix(".csv")
    ta, tf, sz = load_schedule(a.schedule)
    S = slices(ta, tf, sz)
    dur = S["dur"]
    to, orc = curves(S)
    c = a.churn_ms_per_mib
    inflow = sz.sum() / dur / MiB
    cpu = lambda f: f * c / 10.0  # MiB/s × ms/MiB → ms/s → % of one core
    print(f"[{label}] 요청 {len(sz)}건, 길이 {dur:.0f}s, 평균 유입 {inflow:.1f}MiB/s, 평균 사용 {S['dmean'] / MiB:.1f}MiB, "
          f"피할 수 없는 폴트 {S['cold'] / dur / MiB:.3f}MiB/s")
    w0, f0 = eval_timeout(0.0, S)
    w1, f1 = eval_timeout(math.inf, S)
    print(f"  즉시 반납 : 빈 메모리 {w0:6.2f}MiB, 새 폴트 {f0:6.2f}MiB/s (CPU {cpu(f0):.2f}% of 1 core)")
    print(f"  반납 안 함: 빈 메모리 {w1:6.2f}MiB, 새 폴트 {f1:6.3f}MiB/s (CPU {cpu(f1):.3f}%)")
    print(f"\n  {'timeout T':>10} | {'빈 메모리':>9} {'폴트 MiB/s':>10} | 오라클: 같은 빈 메모리의 폴트")
    rows = []
    for T, (w, fl) in zip(T_GRID, to):
        rows.append((T, w, fl, lower_envelope_at(orc, w)))
    for Tshow in (0.1, 1.0, 5.0, 30.0):
        T, w, fl, fo = min(rows, key=lambda x: abs(math.log10(max(x[0], 1e-9)) - math.log10(Tshow)))
        print(f"  {T:>9.3g}s | {w:9.2f} {fl:10.3f} | {fo:10.3f} MiB/s ({(fo / fl - 1) * 100:+6.1f}%)")
    r = S["reused"]
    order = np.argsort(S["g"][r])
    gg, hh = S["g"][r][order], S["h"][r][order]
    if len(gg):
        cw = np.cumsum(hh) / hh.sum()
        qs = [gg[min(np.searchsorted(cw, p), len(gg) - 1)] for p in (0.5, 0.9, 0.99)]
        print(f"  다시 쓰인 조각의 놀이 시간(바이트 가중) p50 {qs[0]:.3g}s / p90 {qs[1]:.3g}s / p99 {qs[2]:.3g}s")

    if a.learn:
        Tstars = [float(x) for x in a.tstars.split(",")]
        print(f"\n  학습 가능 몫 (뒤 절반 평가, 비용 = 재폴트 + 낭비/T*; 타임아웃은 앞 절반에서 고름, 단위 MiB)")
        print(f"  {'T*':>5} | {'타임아웃 1개':>17} | {'특징별 타임아웃(최선)':>27} | {'오라클':>15} | 오라클과의 차이 중 특징이 메운 비율")
        for R in learnable(S, ta, sz, Tstars, a.period):
            s, o, (b, bn) = R["single"], R["oracle"], R["best"]
            share = (s - b) / (s - o) if s > o else 0.0
            print(f"  {R['Tstar']:>4g}s | {s / MiB:8.1f} (T={R['T_single']:.3g}s) | {b / MiB:8.1f} ({(b / s - 1) * 100:+5.1f}%, {bn}) | "
                  f"{o / MiB:7.1f} ({(o / s - 1) * 100:+5.1f}%) | {share * 100:5.1f}%")
            detail = ", ".join(f"{k} {(v / s - 1) * 100:+.1f}%" for k, v in sorted(R["per_set"].items(), key=lambda kv: kv[1]))
            print(f"         특징별: {detail}")

    meas = read_meas(a.meas) if a.meas else {}
    if meas:
        print(f"\n  실측 점과 비교(같은 빈 메모리에서의 폴트 MiB/s; 실측은 페이지 폴트 수 × 4KiB)")
        print(f"  {'설정':>9} | {'빈 메모리':>8} | {'실측':>7} | {'이상적 timeout':>13} | {'오라클':>7} | {'실측 CPU%':>8}")
        for lab, m in sorted(meas.items(), key=lambda kv: kv[1]["waste"]):
            ft = lower_envelope_at(to, m["waste"])
            fo = lower_envelope_at(orc, m["waste"])
            print(f"  {lab:>9} | {m['waste']:8.2f} | {m['fault']:7.2f} | {ft:13.2f} | {fo:7.2f} | {m['cpu']:8.2f}")
    if a.out:
        with open(a.out + "_curve.csv", "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["T_s", "timeout_waste_MiB", "timeout_fault_MiBps", "oracle_waste_MiB", "oracle_fault_MiBps"])
            for T, (w, fl), (wo, fo) in zip(T_GRID, to, orc):
                wr.writerow([T, f"{w:.4f}", f"{fl:.5f}", f"{wo:.4f}", f"{fo:.5f}"])
        plot(a.out + ".png", label, to, orc, meas, c)
        print(f"  → {a.out}_curve.csv, {a.out}.png")


def eval_point(to, T):
    i = int(np.argmin(np.abs(np.log10(np.maximum(T_GRID, 1e-9)) - math.log10(T))))
    return to[i, 0], to[i, 1]


def plot(path, label, to, orc, meas, c):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#52514e"
    INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
    fig, ax = plt.subplots(figsize=(8, 4.8), facecolor=SURF)
    ax.plot(to[:, 0], to[:, 1], color=BLUE, lw=2, label="timeout T (≈ jemalloc decay): line to beat")
    ax.plot(orc[:, 0], orc[:, 1], color=ORANGE, lw=2, ls="--", label="oracle (knows each reuse time): loose upper bound")
    xs = to[:, 0]  # T가 커질수록 낭비는 늘고 폴트는 준다(단조)
    ax.fill_between(xs, [lower_envelope_at(orc, w) for w in xs], to[:, 1], color=ORANGE, alpha=0.12, lw=0,
                    label="gap between them")
    for T in (0.1, 1.0, 10.0):
        w, f = eval_point(to, T)
        ax.plot([w], [f], "o", color=BLUE, ms=5)
        ax.annotate(f"T={T:g}s", (w, f), xytext=(5, 4), textcoords="offset points", color=INK2, fontsize=8)
    for i, (lab, m) in enumerate(sorted(meas.items())):
        ax.plot([m["waste"]], [m["fault"]], "s", color=GRAY, ms=6, label="measured (replay)" if i == 0 else None)
        ax.annotate(lab, (m["waste"], m["fault"]), xytext=(5, -10), textcoords="offset points", color=INK2, fontsize=8)
    ax.set_facecolor(SURF)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(which="both", colors=MUTED, labelsize=9)
    for s in ax.spines.values():
        s.set_color(AXIS)
    ax.set_xscale("symlog", linthresh=0.1)
    ax.set_title(f"{label}: idle memory vs new page faults (lower-left is better)", color=INK, fontsize=10, loc="left")
    ax.set_xlabel("mean idle memory held (MiB)", color=INK2, fontsize=9)
    ax.set_ylabel(f"newly faulted memory (MiB/s)   [1 MiB/s ≈ {c / 10:.2f}% of a core]", color=INK2, fontsize=9)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=140)


if __name__ == "__main__":
    main()
