#!/usr/bin/env python3
"""v1 R 창: v1 모델(봉인 판정 res_main within) vs 청크 모델(exp/v2dev/chunk) 비교표 (sealed-results-v1.md §8.1)

  python3 exp/v2dev/chunk/erratum_tab.py
"""
import sys

import pandas as pd

import os
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, REPO)
from analyze_v2dev import dser, fmt, wtl  # noqa: E402

chunk_files = sys.argv[1:] or [f"{REPO}/exp/v2dev/chunk/res_q1.csv", f"{REPO}/exp/v2dev/chunk/res_q2.csv"]
w = pd.read_csv(f"{REPO}/exp/sealed/windows.csv")
R = [n for n, s in zip(w.name, w.set) if "R" in s]
m = pd.read_csv(f"{REPO}/exp/sealed/res/res_main.csv")
m = m[m.proto == "within"]
c = pd.concat([pd.read_csv(f) for f in chunk_files])
AI, RU, FX, C3 = "AI생존κ-온라인+대체", "규칙-온라인+대체", "고정-온라인", "C3 AI→규칙대체"
for b in (1.0, 2.6):
    for nm, df in (("v1 모델(봉인 판정)", m), ("청크 모델", c)):
        x = df[df.Tstar == b]
        if x.empty:
            continue
        t = x.pivot_table(index="trace", columns="policy", values="cost_MiB")
        t = t[t.index.isin(R)]
        bg = ~t.index.str.startswith("az24")
        print(f"[{nm}] T*={b:g} n={len(t)}")
        for lab, A, B in (("H1 주 AI vs 고정", AI, FX), ("H2 주 AI vs 규칙", AI, RU), ("규칙 vs 고정", RU, FX)):
            d = dser(t, A, B)
            l = wtl(d)[2]
            print(f"  {lab}: 전체 {fmt(d)} 후퇴 {l / len(d) * 100:.1f}% | BurstGPT {fmt(d[bg])} | Azure {fmt(d[~bg])}")
