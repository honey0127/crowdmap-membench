#!/usr/bin/env python3
"""summarize.py — paperexp.py 결과(within.csv, fallback.csv, loto.csv)를 표로 요약한다.
  python3 summarize.py exp/res
"""
import csv
import math
import os
import statistics as st
import sys
from collections import defaultdict

POLS = ["2특징 규칙", "2특징+대체", "AI", "AI+대체", "AI 생존", "AI 생존+대체", "PSK 결정적", "PSK 무작위", "오라클"]


def rd(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def fmt(v):
    return "   —  " if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:+6.1f}"


def wtl(a, c, band=0.5):
    """승/무/패: a가 c보다 band(%p) 넘게 작으면 승"""
    w = sum(x < y - band for x, y in zip(a, c))
    l = sum(x > y + band for x, y in zip(a, c))
    return w, len(a) - w - l, l


def sign_p(w, l):
    """무승부를 뺀 양측 부호 검정"""
    n = w + l
    if n == 0:
        return float("nan")
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(w, l) + 1)) / 2 ** n)


def main(d):
    W = rd(os.path.join(d, "within.csv"))
    by = defaultdict(dict)
    meta = {}
    for r in W:
        k = (r["trace"], float(r["Tstar"]))
        by[k][r["policy"]] = float(r["rel_pct"])
        meta[k] = (float(r["auc"]), int(r["n_te"]))
    traces = list(dict.fromkeys(r["trace"] for r in W))
    tstars = sorted({float(r["Tstar"]) for r in W})
    print("■ 트레이스 안(시험 구간), 고정 타임아웃 대비 비용 변화(%)")
    for b in tstars:
        print(f"\nT* = {b:g}s")
        print(f"  {'트레이스':<13}" + "".join(f"{p:>11}" for p in POLS) + f"{'AUC':>7}{'시험 조각':>9}")
        for t in traces:
            R = by.get((t, b))
            if not R:
                print(f"  {t:<13}  (학습 정답이 한 종류라 건너뜀)")
                continue
            auc, n = meta[(t, b)]
            print(f"  {t:<13}" + "".join(f"{fmt(R.get(p)):>11}" for p in POLS) + f"{auc:7.2f}{n:9d}")
        vals = {p: [by[(t, b)][p] for t in traces if (t, b) in by] for p in POLS}
        print(f"  {'중앙값':<13}" + "".join(f"{fmt(st.median(v)) if v else '':>11}" for v in vals.values()))
        print(f"  {'최악(최대)':<12}" + "".join(f"{fmt(max(v)) if v else '':>11}" for v in vals.values()))
        zero = [0.0] * len(vals["AI+대체"])
        print("  승/무/패(±0.5%p 안은 무), 부호 검정 p, 차이 중앙값(%p):")
        for a, c in (("AI+대체", None), ("2특징+대체", None), ("AI+대체", "2특징+대체"), ("PSK 결정적", None), ("PSK 무작위", None),
                     ("AI 생존+대체", None), ("AI 생존+대체", "2특징+대체"), ("AI 생존+대체", "AI+대체")):
            A, C = vals[a], (vals[c] if c else zero)
            w, t, l = wtl(A, C)
            print(f"    {a} vs {c or '고정'}: {w}/{t}/{l} (p={sign_p(w, l):.3f}), {st.median([x - y for x, y in zip(A, C)]):+.1f}")
        fb = vals["AI+대체"]
        print(f"  오라클 차이 중 AI+대체가 메운 비율(중앙값) {st.median([(f / o) * 100 for f, o in zip(fb, vals['오라클']) if o < -1]):.0f}%")

    pp = os.path.join(d, "within_parts.csv")
    if os.path.exists(pp):
        P = defaultdict(dict)
        for r in rd(pp):
            P[(r["trace"], float(r["Tstar"]))][r["policy"]] = (float(r["refault_MiB"]), float(r["held_mean_MiB"]))
        print("\n■ 비용 분해: 고정 대비 재확보량 / 평균 유휴 보유 변화(%, 중앙값). 고정이 0.1MiB 이상 쥐는 트레이스만")
        for b in tstars:
            ks = [k for k in P if k[1] == b and P[k]["고정 타임아웃"][1] >= 0.1 and P[k]["고정 타임아웃"][0] >= 1]
            line = f"  T*={b:g} (n={len(ks)}):"
            for p in ("2특징+대체", "AI+대체", "AI 생존+대체", "오라클"):
                dr = [(P[k][p][0] / P[k]["고정 타임아웃"][0] - 1) * 100 for k in ks]
                dh = [(P[k][p][1] / P[k]["고정 타임아웃"][1] - 1) * 100 for k in ks]
                line += f"  {p} {st.median(dr):+.1f} / {st.median(dh):+.1f}" if ks else ""
            print(line)

    F = rd(os.path.join(d, "fallback.csv"))
    print("\n■ 대체 장치 민감도(시험 구간, 모든 트레이스·T*)")
    g = defaultdict(list)
    for r in F:
        g[(float(r["W"]), float(r["eps"]), float(r["H"]))].append(r)
    print(f"  {'W':>5} {'ε':>5} {'H':>4} | {'평균':>6} {'최악':>6} | {'전환 횟수(중앙)':>14} {'기준 사용 비율':>12} | "
          f"{'손해 구간':>8} {'지연 중앙(s)':>11} {'놓침':>4}")
    for (w, e, h), rs in sorted(g.items()):
        rel = [float(r["rel_pct"]) for r in rs]
        sw = [int(float(r["switches"])) for r in rs]
        bf = [float(r["base_frac"]) for r in rs]
        runs = sum(int(float(r["runs"])) for r in rs)
        lats = [float(r["lat_med"]) for r in rs if r["lat_med"] not in ("", "nan") and not math.isnan(float(r["lat_med"]))]
        miss = sum(int(float(r["miss"])) for r in rs)
        print(f"  {w:5.0f} {e:5.2f} {h:4.0f} | {st.mean(rel):+6.1f} {max(rel):+6.1f} | {st.median(sw):14.0f} "
              f"{st.mean(bf) * 100:11.0f}% | {runs:8d} {st.median(lats) if lats else float('nan'):11.0f} {miss:4d}")

    p = os.path.join(d, "loto.csv")
    if os.path.exists(p):
        L = rd(p)
        print("\n■ 트레이스 단위 일반화(뺀 트레이스의 시험 구간), 뺀 트레이스 검증으로 고른 고정 타임아웃 대비(%)")
        lb = defaultdict(dict)
        for r in L:
            lb[(r["trace"], float(r["Tstar"]))][r["policy"]] = float(r["rel_local_pct"])
        cols = ["고정 무보정", "AI 무보정", "AI 국소 보정+대체", "AI 생존 무보정", "AI 생존 국소+대체"]
        for b in tstars:
            print(f"\nT* = {b:g}s")
            print(f"  {'뺀 트레이스':<13}" + "".join(f"{c:>12}" for c in cols) + f"{'(안) AI+대체':>13}")
            for t in traces:
                R = lb.get((t, b))
                if not R:
                    continue
                within_fb = by.get((t, b), {}).get("AI+대체")
                print(f"  {t:<13}" + "".join(f"{fmt(R.get(c)):>12}" for c in cols) + f"{fmt(within_fb):>13}")
            vals = {c: [lb[(t, b)][c] for t in traces if (t, b) in lb] for c in cols}
            print(f"  {'중앙값':<13}" + "".join(f"{fmt(st.median(v)):>12}" for v in vals.values()))
            print(f"  {'최악(최대)':<12}" + "".join(f"{fmt(max(v)):>12}" for v in vals.values()))
            print(f"  {'승/무/패':<12}" + "".join(f"{'/'.join(map(str, wtl(v, [0.0] * len(v)))):>12}" for v in vals.values()))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "exp/res")
