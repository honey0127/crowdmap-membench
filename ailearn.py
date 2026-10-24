#!/usr/bin/env python3
"""ailearn.py — AI(그래디언트 부스팅) 예측으로 메모리 조각의 반납 시점을 정하는 정책을 시뮬레이터에서 평가한다.

문제(스키 대여): 놀기 시작한 메모리 조각(두께 h)을 쥐고 있으면 초당 h/T*의 비용(낭비), 반납했다가 다시 쓰이면 h의 비용
(재폴트+반납 CPU)이 든다. 조각이 다시 쓰일 때까지의 시간 g는 모른다.
  - 고정 타임아웃: 모든 조각을 T초까지 쥔다(≈ jemalloc 감쇠).
  - AI + 학습 보강 스키 대여(Purohit·Svitkina·Kumar, NeurIPS 2018):
      모델이 'g > T*(오래 논다)'를 예측하면 λ·T*초 뒤 반납, 아니면 T*/λ초까지 쥔다.
      예측이 맞으면 최적에 가깝고(1+λ배 이내), 틀려도 최적의 (1+1/λ)배를 넘지 않는다(최악 보장).
  - 오라클: g를 미리 안다(느슨한 상한).
평가: 시간 앞 절반으로 학습, 뒤 절반에서 평가. 비용 = 재폴트 MiB + 낭비(MiB·s)/T*. 기준 = 앞 절반으로 고른 타임아웃 1개.
학습 표본은 앞 절반이 끝난 시점에 정답(g > T* 여부)을 알 수 있는 조각만 쓴다(미래 정보 누수 방지).

특징(조각이 놀기 시작한 시점에 서버가 아는 값)
  높이(그때 사용량 위 높이), 사용량, 수준(절대 높이), 유입 1·10·60s(MiB/s), 도착 수 1s,
  이전놀이(같은 높이 1MiB 칸에서 직전에 끝난 놀이 시간), 최근 최고 대비(지난 10·60s 최고 사용량 − 수준), 위상(--period)

사용
  python3 ailearn.py SCHEDULE.csv [--label 이름] [--tstars 0.1,0.3,1,3,10] [--lams 0.3] [--period 초]
                     [--fallback-window 300]
열: 타임아웃 1개(기준, 앞 절반으로 고른 값), 특징 표(2개 특징 조합 중 최선), AI(점수 10칸별 타임아웃),
    AI+대체(최근 W초 동안 AI가 기준보다 나쁘면 기준으로 바꿈), AI 보장 λ(타임아웃을 [λT*, T*/λ]로 제한 → 최적의
    (1+1/λ)배 이내), PSK(Purohit 등 2018 규칙), 오라클(g를 앎), AUC(분류 성능), 대체 사용(기준으로 바꾼 조각 비율)
필요: numpy, scikit-learn(sudo apt install python3-sklearn), policysim.py(같은 폴더)
"""
import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import policysim as ps  # noqa: E402

MiB = ps.MiB


def d_history(ta, tf, sz, bin_s=0.01):
    """10ms 칸별 최대 사용량 D와 칸 시작 시각"""
    t = np.concatenate([ta, tf])
    d = np.concatenate([sz, -sz])
    o = np.lexsort((d, t))
    t, d = t[o], d[o]
    Dv = np.cumsum(d)  # 각 사건 직후의 D
    t0, t1 = t[0], t[-1]
    nb = int((t1 - t0) / bin_s) + 2
    edges = t0 + np.arange(nb + 1) * bin_s
    # 칸 시작 시점의 D = 그 전 마지막 사건 직후 값
    idx = np.searchsorted(t, edges[:-1], side="right") - 1
    Dstart = np.where(idx >= 0, Dv[np.maximum(idx, 0)], 0.0)
    binmax = Dstart.copy()
    k = np.minimum(((t - t0) / bin_s).astype(np.int64), nb - 1)
    np.maximum.at(binmax, k, Dv)
    return t0, bin_s, binmax


def trailing_max(binmax, w):
    from scipy.ndimage import maximum_filter1d

    # 각 칸 i에서 [i-w+1, i] 구간 최대(과거만)
    return maximum_filter1d(binmax, size=w, origin=(w - 1) // 2, mode="nearest")


def build_features(S, ta, tf, sz, period):
    ts = S["ts"]
    order = np.argsort(ta)
    tsorted = ta[order]
    cum = np.concatenate([[0.0], np.cumsum(sz[order])])

    def rate(W):
        hi = np.searchsorted(tsorted, ts, side="right")
        lo = np.searchsorted(tsorted, ts - W, side="right")
        return (cum[hi] - cum[lo]) / W / MiB, (hi - lo)

    r1, c1 = rate(1.0)
    r10, _ = rate(10.0)
    r60, _ = rate(60.0)
    level = (S["D0"] + S["pos"]) / MiB
    t0, b, binmax = d_history(ta, tf, sz)
    kb = np.clip(((ts - t0) / b).astype(np.int64) - 1, 0, len(binmax) - 1)  # 놀기 시작 직전 칸까지
    m10 = trailing_max(binmax, int(10 / b))[kb] / MiB
    m60 = trailing_max(binmax, int(60 / b))[kb] / MiB
    pg = S["prevgap"]
    cols = {
        "높이": S["pos"] / MiB, "사용량": S["D0"] / MiB, "수준": level,
        "유입1s": r1, "유입10s": r10, "유입60s": r60, "도착수1s": c1.astype(float),
        "이전놀이": np.where(np.isnan(pg), np.nan, np.log10(np.maximum(pg, 1e-6))),
        "최근10s최고-수준": m10 - level, "최근60s최고-수준": m60 - level,
    }
    if period:
        cols["위상sin"] = np.sin(2 * np.pi * ts / period)
        cols["위상cos"] = np.cos(2 * np.pi * ts / period)
    names = list(cols)
    return names, np.column_stack([cols[n] for n in names])


def cost(g, h, reused, T, Tstar):
    """조각별 비용(타임아웃 T는 스칼라 또는 조각별 배열)"""
    return h * np.minimum(g, T) / Tstar + h * (reused & (g > T))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("schedule")
    ap.add_argument("--label", default="")
    ap.add_argument("--tstars", default="0.1,0.3,1,3,10")
    ap.add_argument("--lams", default="0.3")
    ap.add_argument("--fallback-window", type=float, default=300.0, help="대체 장치가 보는 최근 구간(s)")
    ap.add_argument("--period", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score

    label = a.label or os.path.basename(a.schedule).removesuffix(".csv")
    ta, tf, sz = ps.load_schedule(a.schedule)
    S = ps.slices(ta, tf, sz)
    names, X = build_features(S, ta, tf, sz, a.period)
    g, h, r, ts = S["g"], S["h"], S["reused"], S["ts"]
    mid = S["t0"] + S["dur"] / 2
    tr, te = ts < mid, ts >= mid
    lams = [float(x) for x in a.lams.split(",")]
    tstars = [float(x) for x in a.tstars.split(",")]
    table = {R["Tstar"]: R for R in ps.learnable(S, ta, sz, tstars, a.period)}  # 단순 특징 표(비교용)
    print(f"[{label}] 조각 {len(g)}개(학습 {tr.sum()}, 평가 {te.sum()}), 특징 {len(names)}개")
    hdr = ["타임아웃 1개", "특징 표(2개)", "AI", "AI+대체"] + [f"AI 보장 λ={l:g}" for l in lams] + ["PSK λ=0.3", "오라클"]
    print(f"  {'T*':>5} | " + " | ".join(f"{x:>12}" for x in hdr) + f" | {'AUC':>5} | 대체 사용")
    for Tstar in tstars:
        # 1) 분류기: '이 조각은 T*보다 오래 논다'를 예측. 정답을 앞 절반 끝(mid)에 알 수 있는 표본만 학습
        done = r & (ts + g <= mid)
        m = tr & (done | ((mid - ts) > Tstar))
        y_all = ((done & (g > Tstar)) | (~done & ((mid - ts) > Tstar))).astype(int)
        Ctr = np.array([cost(g[tr], h[tr], r[tr], T, Tstar).sum() for T in ps.T_GRID])
        i_best = int(np.argmin(Ctr))
        base = cost(g[te], h[te], r[te], ps.T_GRID[i_best], Tstar).sum()
        orc = (h[te] * np.where(r[te], np.minimum(g[te] / Tstar, 1.0), 0.0)).sum()
        tab = table[Tstar]["best"][0]
        if len(np.unique(y_all[m])) < 2:
            print(f"  {Tstar:>4g}s | 학습 표본의 정답이 한 종류뿐이라 건너뜀")
            continue
        mk = lambda: HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, max_leaf_nodes=31,
                                                    l2_regularization=1.0, random_state=a.seed)
        w = h / h[m].mean()
        clf = mk().fit(X[m], y_all[m], sample_weight=w[m])
        p = clf.predict_proba(X)[:, 1]
        # 학습 구간 점수는 교차 예측으로 다시 구한다(학습에 쓴 표본의 점수는 과신이라 칸별 타임아웃을 고를 때 쓰면 안 된다)
        idx_m = np.where(m)[0]
        for k in range(4):
            fold = idx_m[k::4]
            rest = np.setdiff1d(idx_m, fold)
            if len(np.unique(y_all[rest])) == 2:
                p[fold] = mk().fit(X[rest], y_all[rest], sample_weight=w[rest]).predict_proba(X[fold])[:, 1]
        yt = ((g > Tstar) | ~r)[te]
        auc = roc_auc_score(yt, p[te], sample_weight=h[te]) if len(np.unique(yt)) == 2 else math.nan
        # 2) 점수 10칸마다 타임아웃을 앞 절반에서 고른다(필요하면 [λT*, T*/λ]로 제한 → 최악 (1+1/λ)배 보장)
        edges = np.unique(np.quantile(p[tr], np.linspace(0, 1, 11)[1:-1]))
        bins = np.searchsorted(edges, p, side="right")

        def binned(lo=0.0, hi=math.inf):
            grid = np.clip(ps.T_GRID, lo, hi)
            T_i = np.empty(len(g))
            for b in np.unique(bins):
                sel_tr, sel = tr & (bins == b), bins == b
                if h[sel_tr].sum() == 0:
                    T_i[sel] = min(max(ps.T_GRID[i_best], lo), hi)
                    continue
                cb = [cost(g[sel_tr], h[sel_tr], r[sel_tr], T, Tstar).sum() for T in grid]
                T_i[sel] = grid[int(np.argmin(cb))]
            return T_i

        T_ai = binned()
        c_ai = cost(g, h, r, T_ai, Tstar)
        c_base = cost(g, h, r, ps.T_GRID[i_best], Tstar)
        # 대체 장치: 최근 W초 동안 결과가 확정된 조각들(다시 쓰인 시각 ≤ 지금)에서 AI 비용이 고정 타임아웃보다 크면
        # 이번 조각은 고정 타임아웃을 쓴다. 비교는 반사실 비용(두 정책 모두 조각마다 계산 가능)으로 한다.
        ti = np.where(te)[0]
        known_t = np.where(r[ti], ts[ti] + g[ti], np.inf)
        o = np.argsort(known_t)
        kt = known_t[o]
        cum = np.concatenate([[0.0], np.cumsum((c_ai[ti] - c_base[ti])[o])])
        W = a.fallback_window
        lo_i = np.searchsorted(kt, ts[ti] - W, side="left")
        hi_i = np.searchsorted(kt, ts[ti], side="right")
        use_base = (cum[hi_i] - cum[lo_i]) > 0
        fb = np.where(use_base, c_base[ti], c_ai[ti]).sum()
        cells = [c_ai[te].sum(), fb] + [cost(g[te], h[te], r[te], binned(l * Tstar, Tstar / l)[te], Tstar).sum()
                                        for l in lams]
        psk = cost(g[te], h[te], r[te], np.where(p[te] >= 0.5, 0.3 * Tstar, Tstar / 0.3), Tstar).sum()
        f = lambda v: f"{(v / base - 1) * 100:+6.1f}%"
        print(f"  {Tstar:>4g}s | {base / MiB:6.0f} T={ps.T_GRID[i_best]:.2g}s | {f(tab):>12} | " +
              " | ".join(f"{f(v):>12}" for v in cells) + f" | {f(psk):>12} | {f(orc):>12} | {auc:5.2f} | "
              f"{use_base.mean() * 100:4.0f}%")

if __name__ == "__main__":
    main()
