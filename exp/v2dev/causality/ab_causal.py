#!/usr/bin/env python3
"""ab_causal.py — 같은 평가(paperexp3.evaluate3의 4개 정책)를 조각 분해 변형만 바꿔 돌린다(개발 데이터 전용).
  variant: orig(원래 분해) | cellmid(1 MiB 칸으로 나누고 특징은 원래 조각 값) | causal(칸 중점 특징, ts에 안다)
  (저장소 루트에서) python3 exp/v2dev/causality/ab_causal.py --variant causal --tstars 2.6 --out exp/v2dev/causality/ab_R_causal.csv $(일정 목록)
  목록: exp/sealed/windows.csv의 set에 R이 든 54창(exp/sealed/sched/<name>.csv)
"""
import argparse
import csv
import functools
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "../../.."))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402

import paperexp as pe  # noqa: E402
import paperexp2 as p2  # noqa: E402
import cellslices  # noqa: E402

MiB = pe.MiB


def evaluate_lean(tr, led, b, Sk, W_cal, U):
    """paperexp3.evaluate3에서 고정-온라인, 규칙-온라인+대체, AI생존κ-온라인+대체, C3 부분만(같은 코드 순서)"""
    VA, VA1, TE, h = tr.VA, tr.VA1, tr.TE, tr.h
    dec = VA | TE
    Cva = pe.cost_matrix(tr.gva[VA], h[VA], tr.rva[VA], b)
    pos_va = np.where(VA)[0]
    is1 = VA1[pos_va]
    i_fix = int(np.argmin(Cva.sum(1)))
    i_fix1 = int(np.argmin(Cva[:, is1].sum(1)))
    i_fix2 = int(np.argmin(Cva[:, ~is1].sum(1)))
    cap = p2.tcap(b)
    pols = {}
    base = p2.online_group_T(tr, led, b, np.zeros(len(tr.g), dtype=np.int64), 1, dec, W_cal, U)
    pols["고정-온라인"] = base
    grp_r, G_r, rname, _ = p2.rule_groups(tr, b, Cva, pos_va, is1, i_fix, i_fix1, i_fix2)
    T_rule = p2.online_group_T(tr, led, b, grp_r, G_r, dec, W_cal, U)
    rec = pe.survival_timeouts(Sk, b, 1.0)
    Tk = np.stack([np.minimum(rec * k, cap) for k in p2.KGRID])
    T_ai, _ = p2.online_kappa_T(tr, led, b, Tk, dec, W_cal, U)
    pols["규칙-온라인"] = T_rule
    pols["AI생존κ-온라인"] = T_ai
    for name, T in (("규칙-온라인", T_rule), ("AI생존κ-온라인", T_ai)):
        T_fb, _, _ = p2.run_fallback2(tr, T, base, b, dec, **p2.FB)
        pols[name + "+대체"] = T_fb
    T_c3, mode, idx_fb = p2.run_fallback2(tr, pols["AI생존κ-온라인+대체"], pols["규칙-온라인+대체"], b, dec, **p2.FB)
    pols["C3 AI→규칙대체"] = T_c3
    te_m = TE[idx_fb]
    c3_rule_frac = float(mode[te_m].mean()) if te_m.any() else math.nan
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
    return res, rname, c3_rule_frac


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("schedules", nargs="+")
    ap.add_argument("--variant", choices=["orig", "cellmid", "causal"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tstars", default="2.6")
    ap.add_argument("--config", default=f"{REPO}/exp/frozen.json")
    a = ap.parse_args()
    if a.variant != "orig":
        pe.ps.slices = functools.partial(cellslices.slices_cells, mode=a.variant)
    cfg = json.load(open(a.config))
    tstars = [float(x) for x in a.tstars.split(",")]
    done = set()
    if os.path.exists(a.out):
        done = {r["trace"] for r in csv.DictReader(open(a.out))}
    first = not os.path.exists(a.out)
    for path in a.schedules:
        name = os.path.basename(path).removesuffix(".csv")
        if name in done:
            continue
        t1 = time.time()
        tr = p2.Trace2(path, cfg["feats"])
        led = p2.Ledger(tr)
        m = tr.TR
        Sk = pe.fit_survival(tr.X[m], tr.gtr[m], tr.h[m] / tr.h[m].mean(), tr.X, cfg["seed"])
        rows = []
        for b in tstars:
            if len(np.unique((tr.gtr[tr.TR] > b).astype(int))) < 2:
                continue
            res, rname, frac = evaluate_lean(tr, led, b, Sk, cfg["wcal"], cfg["period"])
            b_on = res["고정-온라인"]["cost"]
            for pol, v in res.items():
                rows.append(dict(variant=a.variant, trace=name, Tstar=b, policy=pol, cost_MiB=v["cost"] / MiB,
                                 rel_fixon=(v["cost"] / b_on - 1) * 100 if b_on > 0 else 0.0,
                                 n_slices=len(tr.g), n_te=int(tr.TE.sum()), rule=rname, c3_rule_frac=frac))
        if rows:
            with open(a.out, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                if first:
                    w.writeheader()
                    first = False
                w.writerows(rows)
        print(f"[{name}] {time.time() - t1:.0f}s n={len(tr.g)}", flush=True)


if __name__ == "__main__":
    main()
