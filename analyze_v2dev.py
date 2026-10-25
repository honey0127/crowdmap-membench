#!/usr/bin/env python3
"""analyze_v2dev.py — v2 후보 개발 평가 요약과 선택 규칙 적용(paperexp3.py 출력)

  python3 analyze_v2dev.py exp/v2dev/res_a.csv exp/v2dev/res_b.csv --windows exp/sealed/windows.csv > exp/v2dev/summary.txt

묶음: dev(개발 트레이스 10), devfx(개발 요인 실험 16), v1R(v1 봉인 R 54), v1S(v1 봉인 S 80), v1fx(v1 합성 확인 40).
선택 규칙(paperexp3.py 머리말과 같다, 결과를 보기 전에 정했다):
  v1R에서 T* = 0.3·1·3초 각각 '후보 vs 규칙-온라인+대체' 부호 검정 p < 0.05이고 승 > 패인 T* 수가 가장 많은 후보.
  같으면 세 T*의 d 중앙값 평균이 낮은 쪽. 어느 T*(0.3·1·3·10)에서든 고정-온라인 대비 후퇴 비율이 10%를 넘으면 뺀다.
"""
import argparse
import math
import statistics as st

import pandas as pd
from scipy import stats

RULE, BASE, V1 = "규칙-온라인+대체", "고정-온라인", "AI생존κ-온라인+대체"
CANDS = ["C1 AI세분규칙-온라인+대체", "C2 AI생존κ칸별-온라인+대체", "C3 AI→규칙대체", "C4 AI생존세밀κ-온라인+대체"]
TIE = 0.5


def wtl(x):
    w, l = int((x < -TIE).sum()), int((x > TIE).sum())
    return w, len(x) - w - l, l


def sp(w, l):
    return stats.binomtest(min(w, l), w + l, 0.5).pvalue if w + l else float("nan")


def fmt(x):
    if len(x) == 0:
        return "자료 없음"
    w, t, l = wtl(x)
    return f"{w}/{t}/{l} p={sp(w, l):.3g} 중앙 {x.median():+.2f} 최악 {x.max():+.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("res", nargs="+")
    ap.add_argument("--windows", default="exp/sealed/windows.csv")
    a = ap.parse_args()
    r = pd.concat([pd.read_csv(p) for p in a.res], ignore_index=True)
    win = pd.read_csv(a.windows).set_index("name")
    groups = {
        "dev": r[r.src == "dev"].trace.unique(),
        "devfx": r[r.src == "devfx"].trace.unique(),
        "v1R": [t for t in r[r.src == "v1real"].trace.unique() if "R" in win.loc[t, "set"]],
        "v1S": [t for t in r[r.src == "v1real"].trace.unique() if "S" in win.loc[t, "set"]],
        "v1fx": r[r.src == "v1fx"].trace.unique(),
    }
    print("단위 수:", {k: len(v) for k, v in groups.items()}, "| 행", len(r), "| 중복", int(r.duplicated(["trace", "Tstar", "policy"]).sum()))
    piv = {b: r[r.Tstar == b].pivot(index="trace", columns="policy", values="rel_fixon") for b in sorted(r.Tstar.unique())}
    pols = [V1] + CANDS
    for gname, units in groups.items():
        print(f"\n■ {gname} (단위 {len(units)})")
        for b, t in piv.items():
            t = t.loc[[u for u in units if u in t.index]]
            if t.empty:
                continue
            print(f"  T*={b:g} (n={len(t)}) 규칙 vs 고정: {fmt(t[RULE])}")
            for p in pols:
                print(f"    {p:28s} vs 규칙 {fmt(t[p] - t[RULE]):42s} | vs 고정 {fmt(t[p])}, 후퇴 {wtl(t[p])[2] / len(t) * 100:.0f}%")
    print("\n■ 선택 규칙 적용 (v1R)")
    best = None
    for p in CANDS:
        wins, meds, worst_lose = 0, [], 0.0
        for b, t in piv.items():
            t = t.loc[[u for u in groups["v1R"] if u in t.index]]
            if t.empty:
                continue
            lose = wtl(t[p])[2] / len(t)
            worst_lose = max(worst_lose, lose)
            if b in (0.3, 1.0, 3.0):
                d = t[p] - t[RULE]
                w, _, l = wtl(d)
                wins += int(sp(w, l) < 0.05 and w > l)
                meds.append(d.median())
        if not meds:
            continue
        ok = worst_lose <= 0.10
        score = (wins, -st.mean(meds))
        print(f"  {p}: 규칙을 이긴 T* 수 {wins}/3, d 중앙 평균 {st.mean(meds):+.2f}%p, 고정 대비 최대 후퇴 {worst_lose * 100:.1f}% → "
              f"{'후보' if ok else '제외(후퇴 비율 > 10%)'}")
        if ok and (best is None or score > best[0]):
            best = (score, p)
    print(f"  선택: {best[1] if best else '없음'}")


if __name__ == "__main__":
    main()
