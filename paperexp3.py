#!/usr/bin/env python3
"""paperexp3.py — v2 후보 정책 탐색(개발 데이터 전용). paperexp2의 그림자 장부·온라인 보정·대체 장치를 그대로 쓴다.

v1 봉인 시험의 특징 통제 결과(같은 특징에서도 AI가 규칙에 졌다)는 차이가 특징이 아니라 '쓰는 방식'에서 난다고 가리켰다.
후보는 규칙의 칸별 보정을 살리면서 AI 정보를 더하는 방식이다.
  C1 AI세분규칙-온라인   규칙 칸 × AI 점수 3분위(생존 곡선의 P(g > T*))로 묶고 칸별 반납 시점을 장부로 고른다.
                         작은 칸은 부모 규칙 칸 값을, 부모도 작으면 전체 값을 쓴다.
  C2 AI생존κ칸별-온라인  v1 주 AI의 조각별 T(생존 곡선 기대 비용 최소) × κ, κ를 규칙 칸마다 장부로 고른다. 작은 칸은 전체 κ.
  C3 AI→규칙대체         v1 주 AI(+대체)와 규칙-온라인+대체 사이의 그림자 장부 전환(W·ε·H는 v1과 같다).
  C4 AI생존세밀κ-온라인  생존 곡선의 기대 비용을 고운 격자(GRID)에서 최소로 하는 T × κ(온라인). v1은 매듭 11점에서만 골랐다.
  C1·C2·C4에는 v1과 같은 대체 장치(기준 = 고정-온라인)를 붙인다(+대체).
비교: 고정-온라인, 규칙-온라인+대체, v1 주 AI(AI생존κ-온라인+대체). 생존 모델·규칙 선택·장부는 v1 within과 같다.

선택 규칙(결과를 보기 전에 정한다): v1 R 창(개발 데이터로 씀)에서 T* = 0.3·1·3초 각각 '후보 vs 규칙-온라인+대체'의
  부호 검정이 p < 0.05이고 승 > 패인 T* 수가 가장 많은 후보를 고른다. 같으면 세 T*의 d 중앙값 평균이 낮은 쪽.
  어느 T*에서도 고정-온라인 대비 후퇴 비율이 10%를 넘으면 뺀다.

  python3 paperexp3.py --out exp/v2dev/res_a.csv exp/sched/*.csv ...   # 이미 처리한 일정은 건너뛴다(이어 하기)
"""
import argparse
import csv
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paperexp as pe  # noqa: E402
import paperexp2 as p2  # noqa: E402

MiB = pe.MiB
KNOTS = np.concatenate([[0.0], pe.TAU])
SRC = {"exp/sched": "dev", "exp/fx/sched": "devfx", "exp/sealed/sched": "v1real", "exp/sealed/fx/sched": "v1fx"}


def s_at(Sk, b):
    """생존 곡선(매듭 사이 선형)에서 P(g > b)"""
    S = np.hstack([np.ones((len(Sk), 1)), Sk])
    j = int(np.clip(np.searchsorted(KNOTS, b, side="right") - 1, 0, len(KNOTS) - 2))
    w = (b - KNOTS[j]) / (KNOTS[j + 1] - KNOTS[j])
    return S[:, j] * (1 - w) + S[:, j + 1] * w


def survival_timeouts_fine(Sk, b, grid=pe.GRID):
    """조각마다 E[min(g,T)]/b + P(g>T)를 고운 격자 T에서 최소로 하는 T(생존 곡선은 매듭 사이 선형, survival_timeouts와 같은 가정)"""
    S = np.hstack([np.ones((len(Sk), 1)), Sk])
    taus = np.union1d(KNOTS, grid[grid <= KNOTS[-1]])
    St = np.empty((len(Sk), len(taus)))
    for i, t in enumerate(taus):
        j = int(np.clip(np.searchsorted(KNOTS, t, side="right") - 1, 0, len(KNOTS) - 2))
        w = (t - KNOTS[j]) / (KNOTS[j + 1] - KNOTS[j])
        St[:, i] = S[:, j] * (1 - w) + S[:, j + 1] * w
    area = np.concatenate([np.zeros((len(Sk), 1)), np.cumsum((St[:, 1:] + St[:, :-1]) / 2 * np.diff(taus), axis=1)], axis=1)
    pos = np.searchsorted(taus, grid[grid <= KNOTS[-1]])
    ec = area[:, pos] / b + St[:, pos]
    return grid[grid <= KNOTS[-1]][np.argmin(ec, axis=1)]


def online_group_T_hier(tr, led, b, child, parent_of, Gc, Gp, dec_mask, W_cal, U):
    """online_group_T와 같지만, 작은 자식 칸은 부모 칸 값(부모도 작으면 전체 값)을 쓴다"""
    cap, grid = p2.tcap(b), p2.grid_on(b)
    T = np.full(len(tr.g), np.nan)
    dec = np.where(dec_mask)[0]
    if len(dec) == 0:
        return T
    t0, t1 = tr.ts[dec].min(), tr.ts[dec].max()
    prev = np.zeros(Gc, dtype=np.int64)
    for u in np.arange(t0, t1 + U, U):
        D = led.window(u, u + U)
        D = D[dec_mask[D]]
        if len(D) == 0:
            continue
        J = led.window(u - cap - W_cal, u - cap)
        if len(J):
            g_obs = led.observed(J, u, cap)
            C = pe.cost_matrix(g_obs, tr.h[J], np.ones(len(J), dtype=bool), b, grid)
            ip = int(np.argmin(C.sum(1)))
            cj = child[J]
            pj = parent_of[cj]
            tp = np.argmin(p2.group_sums(C, pj, Gp), axis=0)
            wp, npn = np.bincount(pj, weights=tr.h[J], minlength=Gp), np.bincount(pj, minlength=Gp)
            tp[(wp < p2.MIN_FRAC * wp.sum()) | (npn < p2.MIN_N)] = ip
            tc = np.argmin(p2.group_sums(C, cj, Gc), axis=0)
            wc, nc = np.bincount(cj, weights=tr.h[J], minlength=Gc), np.bincount(cj, minlength=Gc)
            small = (wc < p2.MIN_FRAC * wc.sum()) | (nc < p2.MIN_N)
            tc[small] = tp[parent_of[np.arange(Gc)]][small]
            prev = tc
        T[D] = grid[prev[child[D]]]
    return T


def online_kappa_T_cells(tr, led, b, Tk, grp, G, dec_mask, W_cal, U):
    """online_kappa_T와 같지만 κ를 칸마다 고른다(작은 칸은 전체 κ)"""
    cap = p2.tcap(b)
    K = len(p2.KGRID)
    T = np.full(len(tr.g), np.nan)
    dec = np.where(dec_mask)[0]
    if len(dec) == 0:
        return T
    t0, t1 = tr.ts[dec].min(), tr.ts[dec].max()
    kc = np.full(G, int(np.argmin(np.abs(p2.KGRID - 1.0))))
    for u in np.arange(t0, t1 + U, U):
        D = led.window(u, u + U)
        D = D[dec_mask[D]]
        if len(D) == 0:
            continue
        J = led.window(u - cap - W_cal, u - cap)
        if len(J):
            g_obs = led.observed(J, u, cap)
            hJ, gj = tr.h[J], grp[J]
            Ck = np.stack([pe.cost(g_obs, hJ, np.ones(len(J), dtype=bool), Tk[k, J], b) for k in range(K)])
            kg = int(np.argmin(Ck.sum(1)))
            kc = np.argmin(p2.group_sums(Ck, gj, G), axis=0)
            w, n = np.bincount(gj, weights=hJ, minlength=G), np.bincount(gj, minlength=G)
            kc[(w < p2.MIN_FRAC * w.sum()) | (n < p2.MIN_N)] = kg
        T[D] = Tk[kc[grp[D]], D]
    return T


def evaluate3(tr, led, b, Sk, W_cal, U):
    VA, VA1, TE, h = tr.VA, tr.VA1, tr.TE, tr.h
    dec = VA | TE
    Cva = pe.cost_matrix(tr.gva[VA], h[VA], tr.rva[VA], b)
    pos_va = np.where(VA)[0]
    is1 = VA1[pos_va]
    i_fix = int(np.argmin(Cva.sum(1)))
    i_fix1 = int(np.argmin(Cva[:, is1].sum(1)))
    i_fix2 = int(np.argmin(Cva[:, ~is1].sum(1)))
    cap = p2.tcap(b)
    pols, info = {}, {}
    base = p2.online_group_T(tr, led, b, np.zeros(len(tr.g), dtype=np.int64), 1, dec, W_cal, U)
    pols["고정-온라인"] = base
    grp_r, G_r, rname, _ = p2.rule_groups(tr, b, Cva, pos_va, is1, i_fix, i_fix1, i_fix2)
    info["rule"] = rname
    T_rule = p2.online_group_T(tr, led, b, grp_r, G_r, dec, W_cal, U)
    rec = pe.survival_timeouts(Sk, b, 1.0)
    Tk = np.stack([np.minimum(rec * k, cap) for k in p2.KGRID])
    T_ai, _ = p2.online_kappa_T(tr, led, b, Tk, dec, W_cal, U)
    # C1: 규칙 칸 × AI 점수 3분위
    sb = s_at(Sk, b)
    e = pe.quant_edges(sb[VA], 3)
    child = grp_r * 3 + np.minimum(pe.bin_ids(e, sb), 2)
    parent_of = np.arange(G_r * 3) // 3
    T_c1 = online_group_T_hier(tr, led, b, child, parent_of, G_r * 3, G_r, dec, W_cal, U)
    # C2: 칸별 κ
    T_c2 = online_kappa_T_cells(tr, led, b, Tk, grp_r, G_r, dec, W_cal, U)
    # C4: 고운 격자 생존 최적 × κ
    rec_f = survival_timeouts_fine(Sk, b)
    Tkf = np.stack([np.minimum(rec_f * k, cap) for k in p2.KGRID])
    T_c4, _ = p2.online_kappa_T(tr, led, b, Tkf, dec, W_cal, U)
    raw = {"규칙-온라인": T_rule, "AI생존κ-온라인": T_ai, "C1 AI세분규칙-온라인": T_c1, "C2 AI생존κ칸별-온라인": T_c2,
           "C4 AI생존세밀κ-온라인": T_c4}
    for name, T in raw.items():
        T_fb, mode, idx_fb = p2.run_fallback2(tr, T, base, b, dec, **p2.FB)
        pols[name + "+대체"] = T_fb
        te_m = TE[idx_fb]
        info[name + "+대체"] = float(mode[te_m].mean()) if te_m.any() else math.nan
    # C3: v1 주 AI(+대체) ↔ 규칙-온라인+대체
    T_c3, mode, idx_fb = p2.run_fallback2(tr, pols["AI생존κ-온라인+대체"], pols["규칙-온라인+대체"], b, dec, **p2.FB)
    pols["C3 AI→규칙대체"] = T_c3
    te_m = TE[idx_fb]
    info["C3 AI→규칙대체"] = float(mode[te_m].mean()) if te_m.any() else math.nan
    gt, ht, rt = tr.g[TE], h[TE], tr.r[TE]
    res = {}
    for name, T in pols.items():
        Tt = T[TE]
        assert not np.isnan(Tt).any(), name
        ref, w = pe.parts(gt, ht, rt, Tt, b)
        res[name] = dict(cost=ref + w / b, refault=ref, waste=w)
    orc = np.where(rt & (gt < b), np.inf, 0.0)
    ref, w = pe.parts(gt, ht, rt, orc, b)
    res["오라클"] = dict(cost=ref + w / b, refault=ref, waste=w)
    return res, info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("schedules", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default="exp/frozen.json")
    ap.add_argument("--tstars", default="")
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    tstars = [float(x) for x in a.tstars.split(",")] if a.tstars else cfg["tstars"]
    seed = cfg["seed"]
    done = set()
    if os.path.exists(a.out):
        done = {r["trace"] for r in csv.DictReader(open(a.out))}
    first = not os.path.exists(a.out)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    t_start = time.time()
    for path in a.schedules:
        name = os.path.basename(path).removesuffix(".csv")
        if name in done:
            continue
        t1 = time.time()
        tr = p2.Trace2(path, cfg["feats"])
        led = p2.Ledger(tr)
        m = tr.TR
        Sk = pe.fit_survival(tr.X[m], tr.gtr[m], tr.h[m] / tr.h[m].mean(), tr.X, seed)  # v1 within과 같은 생존 모델
        te_dur = tr.t0 + tr.dur - tr.c2
        rows = []
        for b in tstars:
            if len(np.unique((tr.gtr[tr.TR] > b).astype(int))) < 2:  # v1과 같은 건너뛰기 규칙
                continue
            res, info = evaluate3(tr, led, b, Sk, cfg["wcal"], cfg["period"])
            b_on = res["고정-온라인"]["cost"]
            for pol, v in res.items():
                rows.append(dict(src=SRC.get(os.path.dirname(os.path.normpath(path)), "other"), trace=name, Tstar=b, policy=pol,
                                 cost_MiB=v["cost"] / MiB, rel_fixon=(v["cost"] / b_on - 1) * 100 if b_on > 0 else 0.0,
                                 refault_MiB=v["refault"] / MiB, held_mean_MiB=v["waste"] / MiB / te_dur,
                                 n_te=int(tr.TE.sum()), rule=info["rule"], base_frac=info.get(pol, "")))
        if rows:
            with open(a.out, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                if first:
                    w.writeheader()
                    first = False
                w.writerows(rows)
        print(f"[{name}] {time.time() - t1:.0f}s", flush=True)
    print(f"끝 ({time.time() - t_start:.0f}s) → {a.out}")


if __name__ == "__main__":
    main()
