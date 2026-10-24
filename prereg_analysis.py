#!/usr/bin/env python3
"""prereg_analysis.py — 사전 등록 v1.1의 판정을 그대로 계산한다(봉인 시험 전에 고정한 분석 코드).

  python3 prereg_analysis.py --res exp/sealed/res/res_main.csv --windows exp/sealed/windows.csv \
      [--factorial exp/sealed/res/res_fx.csv] [--sens exp/sealed/res/res_sens.csv] \
      [--irr exp/sealed/res/res_main_irregularity.csv] --ai "AI생존κ-온라인+대체"

계산 정의(사전 등록 §2.2와 같다)
  창 평가값  Δ_p = (C_p / C_고정-온라인 − 1) × 100. C는 그 창 시험 구간의 유체 모델 비용(재확보 MiB + 낭비 MiB·s ÷ T*).
  차이       d(A, B) = Δ_A − Δ_B (%p). B = 고정-온라인이면 d = Δ_A. 반올림하지 않은 배정밀도 값으로 비교한다.
  승·무·패   d < −0.5 → 승(A가 낫다), d > +0.5 → 패, 그 밖은 무.
  부호 검정  무승부를 뺀 단위로 양측 정확 이항 검정: p = min(1, 2·Σ_{i ≤ min(승, 패)} C(n, i) / 2^n).
  후퇴 비율  패 ÷ (승 + 무 + 패), 분모는 그 T*에서 결과가 있는 모든 단위.
  단위       R·S 집합의 창. 창마다 서로 다른 (파일, 날) 군집에서 뽑았으므로 창 하나가 독립 단위 하나다.
판정
  H1  R 집합, T*=1초: 주 AI+대체 vs 고정-온라인. p < 0.05, 승 > 패, 후퇴 비율 ≤ 0.10이면 충족.
  H2  R 집합, T*=1초: 주 AI+대체 vs 규칙-온라인+대체. p < 0.05, 승 > 패이면 충족.
  H3  S 집합 네 칸 각각, T*=1초: 주 AI+대체 vs 규칙-온라인+대체. Holm(4) 보정 p < 0.05면 그 칸에서 방향(승>패 또는 패>승) 확정.
  H4  합성 확인 요인 실험, T*=1초: d(주 AI+대체, 규칙-온라인+대체)를 요인(크기 S/s, 몰림 b/p, 수준 w/n)의 높은 수준 20개와
      낮은 수준 20개 사이에서 Mann–Whitney U 양측(정규 근사, 동점·연속성 보정). Holm(3) 보정 p < 0.05면 효과 확정.
"""
import argparse
import csv
import math
import statistics as st
from collections import defaultdict

TIE = 0.5
RULE = "규칙-온라인+대체"
BASE = "고정-온라인"


def sign_p(w, l):
    n = w + l
    if n == 0:
        return float("nan")
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(w, l) + 1)) / 2 ** n)


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, run = [0.0] * len(ps), 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - r) * ps[i]))
        adj[i] = run
    return adj


def mann_whitney_p(x, y):
    """양측 Mann–Whitney U, 정규 근사(동점 보정 + 연속성 보정)"""
    allv = sorted([(v, 0) for v in x] + [(v, 1) for v in y])
    ranks, i, ties = [0.0] * len(allv), 0, 0.0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1][0] == allv[i][0]:
            j += 1
        for k in range(i, j + 1):
            ranks[k] = (i + j) / 2 + 1
        t = j - i + 1
        ties += t ** 3 - t
        i = j + 1
    n1, n2 = len(x), len(y)
    R1 = sum(r for r, (_, g) in zip(ranks, allv) if g == 0)
    U1 = R1 - n1 * (n1 + 1) / 2
    N = n1 + n2
    sd = math.sqrt(n1 * n2 / 12 * ((N + 1) - ties / (N * (N - 1)))) if N > 1 else 0.0
    if sd == 0:
        return 1.0, U1
    z = max(abs(U1 - n1 * n2 / 2) - 0.5, 0.0) / sd
    return min(1.0, math.erfc(z / math.sqrt(2))), U1


def spearman(x, y):
    def rank(v):
        o = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(o):
            j = i
            while j + 1 < len(o) and v[o[j + 1]] == v[o[i]]:
                j += 1
            for k in range(i, j + 1):
                r[o[k]] = (i + j) / 2
            i = j + 1
        return r
    rx, ry = rank(x), rank(y)
    mx, my = st.mean(rx), st.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den > 0 else float("nan")


def load(path, proto="within"):
    by = defaultdict(dict)
    for r in csv.DictReader(open(path)):
        if r.get("proto", "within") != proto:
            continue
        by[(r["trace"], float(r["Tstar"]))][r["policy"]] = float(r["rel_fixon"])
    return by


def wtl(d):
    w = sum(x < -TIE for x in d)
    l = sum(x > TIE for x in d)
    return w, len(d) - w - l, l


def diffs(by, units, b, a, c):
    out = []
    for u in units:
        R = by.get((u, b))
        if R is None or a not in R or (c != BASE and c not in R):
            continue
        out.append(R[a] - (0.0 if c == BASE else R[c]))
    return out


def q(d, p):
    s = sorted(d)
    k = (len(s) - 1) * p
    lo, hi = math.floor(k), math.ceil(k)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def fmt(d):
    if not d:
        return "자료 없음"
    w, t, l = wtl(d)
    return (f"n={len(d)} 승/무/패 {w}/{t}/{l}, p={sign_p(w, l):.4f}, 차이 중앙 {st.median(d):+.2f}%p "
            f"(사분위 {q(d, .25):+.2f}~{q(d, .75):+.2f}), 최악 {max(d):+.2f}%p, 후퇴 비율 {l / len(d) * 100:.1f}%")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--res", required=True)
    ap.add_argument("--windows", required=True)
    ap.add_argument("--factorial", default="")
    ap.add_argument("--sens", default="", help="솎기 민감도 결과 CSV(창 이름 뒤에 _r16, _r32)")
    ap.add_argument("--irr", default="", help="불규칙성 지표 CSV(run_frozen 출력)")
    ap.add_argument("--ai", required=True, help="주 AI 정책 이름(+대체 포함)")
    a = ap.parse_args()
    ai = a.ai
    by = load(a.res)
    W = list(csv.DictReader(open(a.windows)))
    R = [w["name"] for w in W if "R" in w["set"]]
    S = {c: [w["name"] for w in W if "S" in w["set"] and w["cell"] == c] for c in ("sa", "sA", "Sa", "SA")}
    src = {w["name"]: w["source"] for w in W}
    for nm, units in (("R", R), ("S", sum(S.values(), []))):
        miss = [u for u in units if (u, 1.0) not in by]
        print(f"{nm} 집합 {len(units)}개 중 T*=1초 결과 없음 {len(miss)}개: {miss}")

    print("\n■ H1 (1차) R 집합, T*=1초, 주 AI+대체 vs 고정-온라인")
    d1 = diffs(by, R, 1.0, ai, BASE)
    w, t, l = wtl(d1)
    p1 = sign_p(w, l)
    ok1 = p1 < 0.05 and w > l and l / len(d1) <= 0.10 if d1 else False
    print(f"  {fmt(d1)} → 판정: {'충족' if ok1 else '미충족'}")
    print("\n■ H2 (2차) R 집합, T*=1초, 주 AI+대체 vs 규칙-온라인+대체")
    d2 = diffs(by, R, 1.0, ai, RULE)
    w2, _, l2 = wtl(d2)
    ok2 = sign_p(w2, l2) < 0.05 and w2 > l2 if d2 else False
    print(f"  {fmt(d2)} → 판정: {'충족' if ok2 else '미충족'}")
    print("\n■ H3 (2차) S 집합 칸별, T*=1초, 주 AI+대체 vs 규칙-온라인+대체 (Holm 4개)")
    cells = ["sa", "sA", "Sa", "SA"]
    ds = [diffs(by, S[c], 1.0, ai, RULE) for c in cells]
    ps = [sign_p(*[wtl(d)[0], wtl(d)[2]]) if d else 1.0 for d in ds]
    adj = holm([1.0 if math.isnan(p) else p for p in ps])
    for c, d, qv in zip(cells, ds, adj):
        ww, _, ll = wtl(d) if d else (0, 0, 0)
        verdict = "AI 우세" if (qv < 0.05 and ww > ll) else ("규칙 우세" if (qv < 0.05 and ll > ww) else "검출 못 함")
        print(f"  {c}: {fmt(d)}, Holm p={qv:.4f} → {verdict}")

    print("\n■ 탐색: R 집합 T*별")
    for b in (0.3, 1.0, 3.0, 10.0):
        print(f"  T*={b:g} vs 고정-온라인: {fmt(diffs(by, R, b, ai, BASE))}")
        print(f"         vs 규칙-온라인+대체: {fmt(diffs(by, R, b, ai, RULE))}")
    print("\n■ 탐색: R 집합 출처별, T*=1초")
    for s_ in ("Azure", "BurstGPT"):
        U = [u for u in R if (src[u].startswith("az") if s_ == "Azure" else src[u].startswith("bg"))]
        print(f"  {s_}: vs 고정-온라인 {fmt(diffs(by, U, 1.0, ai, BASE))}")
        print(f"  {s_}: vs 규칙 {fmt(diffs(by, U, 1.0, ai, RULE))}")
    print("\n■ 탐색: S 칸별 vs 고정-온라인, T*=1·3초")
    for c in cells:
        for b in (1.0, 3.0):
            print(f"  {c} T*={b:g}: {fmt(diffs(by, S[c], b, ai, BASE))}")
    print("\n■ 탐색: 다른 정책 비교 (R 집합, T*=1·3초)")
    pairs = [("AI분류-온라인+대체", RULE), ("AI생존그룹-온라인+대체", RULE), ("AI생존κ-2특징-온라인+대체", RULE),
             ("AI생존κ-f10-온라인+대체", RULE), (RULE, BASE), ("고정(정적)", BASE), ("AI분류(정적)", BASE),
             ("AI생존κ-온라인", BASE)]
    for p, c in pairs:
        for b in (1.0, 3.0):
            print(f"  {p} vs {c} T*={b:g}: {fmt(diffs(by, R, b, p, c))}")
    print("\n■ 탐색: 특징 통제 비교 (R 집합, T*=1·3초)")
    for b in (1.0, 3.0):
        print(f"  주 AI vs 2특징 AI(같은 모델, 특징 15개 vs 1~2개) T*={b:g}: {fmt(diffs(by, R, b, ai, 'AI생존κ-2특징-온라인+대체'))}")
        print(f"  2특징 AI vs 규칙(같은 특징, 모델링·보정 차이) T*={b:g}: {fmt(diffs(by, R, b, 'AI생존κ-2특징-온라인+대체', RULE))}")
        print(f"  주 AI vs f10 AI(입력 패턴 특징 5개의 몫) T*={b:g}: {fmt(diffs(by, R, b, ai, 'AI생존κ-f10-온라인+대체'))}")
    pre = load(a.res, "pre")
    if pre:
        print("\n■ 탐색: 사전 학습 판(pre), R 집합 T*=1·3초")
        for b in (1.0, 3.0):
            print(f"  pre 주 AI vs 고정-온라인 T*={b:g}: {fmt(diffs(pre, R, b, ai, BASE))}")
            print(f"  pre 주 AI vs 규칙 T*={b:g}: {fmt(diffs(pre, R, b, ai, RULE))}")
    if a.irr:
        I = {r["trace"]: r for r in csv.DictReader(open(a.irr))}
        U = [u for u in dict.fromkeys(R + sum(S.values(), [])) if (u, 1.0) in by and u in I]
        d = [by[(u, 1.0)][ai] - by[(u, 1.0)][RULE] for u in U]
        print(f"\n■ 탐색: d(주 AI, 규칙)와 불규칙성 지표의 스피어만 상관 (R∪S 창 {len(U)}개, T*=1초)")
        for k in ("size_cv", "burst", "level", "fano10", "drift"):
            x = [float(I[u][k]) if I[u][k] not in ("", "nan") else float("nan") for u in U]
            ok = [i for i, v in enumerate(x) if not math.isnan(v)]
            print(f"  {k}: ρ = {spearman([x[i] for i in ok], [d[i] for i in ok]):+.2f} (n={len(ok)})")
    if a.sens:
        sb = load(a.sens)
        print("\n■ 탐색: 솎기 민감도(Azure 창 6개), T*=1초, 남김 상한 8·16·32건/s")
        names = sorted({t.rsplit("_r", 1)[0] for t, _ in sb})
        for nm in names:
            row = []
            for tag, b_ in (("8", by), ("16", sb), ("32", sb)):
                key = nm if tag == "8" else f"{nm}_r{tag}"
                Rr = b_.get((key, 1.0))
                row.append("—" if Rr is None else f"{Rr[ai]:+.1f}/{Rr[ai] - Rr[RULE]:+.1f}")
            print(f"  {nm}: (vs 고정-온라인 / vs 규칙) 8건/s {row[0]}, 16건/s {row[1]}, 32건/s {row[2]}")
    if a.factorial:
        fx = load(a.factorial)
        names = sorted({t for t, _ in fx})
        print("\n■ H4 (2차) 합성 확인 요인 실험, T*=1초, d(주 AI+대체, 규칙-온라인+대체)의 요인 효과 (Holm 3개)")
        diff = {t: fx[(t, 1.0)][ai] - fx[(t, 1.0)][RULE] for t in names if (t, 1.0) in fx}
        res = []
        for pos, (fac, hi, lo) in enumerate((("크기 변동", "S", "s"), ("짧은 몰림", "b", "p"), ("수준 변화", "w", "n"))):
            H = [v for t, v in diff.items() if t.split("_")[1][pos] == hi]
            L = [v for t, v in diff.items() if t.split("_")[1][pos] == lo]
            p, _ = mann_whitney_p(H, L)
            res.append((fac, H, L, p))
        adj = holm([r_[3] for r_ in res])
        for (fac, H, L, p), qv in zip(res, adj):
            print(f"  {fac}: 높음 중앙 {st.median(H):+.2f}%p (n={len(H)}) vs 낮음 {st.median(L):+.2f}%p (n={len(L)}), "
                  f"p={p:.4f}, Holm p={qv:.4f} → {'효과 확정' if qv < 0.05 else '검출 못 함'}")
        print(f"  전체: 주 AI vs 규칙 {fmt(list(diff.values()))}")
        print(f"  전체: 주 AI vs 고정-온라인 {fmt([fx[(t, 1.0)][ai] for t in diff])}")
        print("  조건별 중앙값 (vs 규칙 %p / vs 고정-온라인 %):")
        for cfg in sorted({t.split("_")[1] for t in diff}):
            ts_ = [t for t in diff if t.split("_")[1] == cfg]
            print(f"    {cfg}: {st.median([diff[t] for t in ts_]):+.2f} / {st.median([fx[(t, 1.0)][ai] for t in ts_]):+.2f} (n={len(ts_)})")


if __name__ == "__main__":
    main()
