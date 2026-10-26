#!/usr/bin/env python3
"""A/B 요약: v1 R 창, T* = 2.6초. orig(v2dev res_t26) vs cellmid vs causal(정확한 크기 + 1 MiB 칸 특징)"""
import math
import sys

import numpy as np
import pandas as pd
from scipy import stats

import os
D = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(D, "../../.."))
RULE, BASE, AI, C3 = "규칙-온라인+대체", "고정-온라인", "AI생존κ-온라인+대체", "C3 AI→규칙대체"
TIE = 0.5
b = float(sys.argv[1]) if len(sys.argv) > 1 else 2.6

w = pd.read_csv(f"{REPO}/exp/sealed/windows.csv")
R = [n for n, s in zip(w.name, w.set) if "R" in s]
o = pd.concat([pd.read_csv(f"{REPO}/exp/v2dev/res_t26_a.csv"), pd.read_csv(f"{REPO}/exp/v2dev/res_t26_b.csv")])
o = o[(o.Tstar == b) & o.trace.isin(R)].assign(variant="orig")
parts = [o]
for v in ("cellmid", "causal"):
    try:
        x = pd.read_csv(f"{D}/ab_R_{v}.csv")
        parts.append(x[x.Tstar == b])
    except FileNotFoundError:
        pass
r = pd.concat(parts)
common = set.intersection(*[set(r[r.variant == v].trace) for v in r.variant.unique()])
r = r[r.trace.isin(common)]
print(f"T*={b:g}, 공통 창 {len(common)}개, 변형 {list(r.variant.unique())}")


def dser(c, A, B):
    base, diff = c[BASE], c[A] - c[B]
    d = 100.0 * diff / base.where(base > 0)
    z = base <= 0
    d[z] = np.where(diff[z] == 0, 0.0, np.sign(diff[z]) * math.inf)
    return d


def fmt(x):
    wn, l = int((x < -TIE).sum()), int((x > TIE).sum())
    t = len(x) - wn - l
    p = stats.binomtest(min(wn, l), wn + l, 0.5).pvalue if wn + l else float("nan")
    return f"{wn:2d}/{t:2d}/{l:2d} p={p:.2g} 중앙 {x.median():+6.2f} 최악 {x.max():+6.2f}"


piv = {v: r[r.variant == v].pivot(index="trace", columns="policy", values="cost_MiB").sort_index() for v in r.variant.unique()}
for v, c in piv.items():
    bg = ~c.index.str.startswith("az24")
    print(f"\n[{v}]")
    for nm, A, B in (("규칙 vs 고정", RULE, BASE), ("AI+대체 vs 고정", AI, BASE), ("C3 vs 고정", C3, BASE),
                     ("C3 vs 규칙", C3, RULE), ("AI+대체 vs 규칙", AI, RULE)):
        d = dser(c, A, B)
        print(f"  {nm:14s} 전체 {fmt(d)} | BurstGPT {fmt(d[bg])}")
if "causal" in piv and "cellmid" in piv:
    a, c = piv["cellmid"], piv["causal"]
    print("\n[미래 정보 몫] 같은 분해에서 특징만 바꾼 비용 변화: (causal − cellmid) ÷ 고정-온라인 × 100 (%p, 창별 중앙·사분위)")
    for p in (RULE, AI, C3, "규칙-온라인", "AI생존κ-온라인"):
        x = 100 * (c[p] - a[p]) / a[BASE]
        print(f"  {p:16s} 중앙 {x.median():+.2f}  사분위 {x.quantile(.25):+.2f}~{x.quantile(.75):+.2f}  최대 {x.max():+.2f}  양수 {int((x > 0).sum())}/{len(x)}")
    print("  고정-온라인 차이 최대(검산, 0이어야 한다):", float((c[BASE] - a[BASE]).abs().max()))
