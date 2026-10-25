#!/usr/bin/env python3
"""prereg_v2_analysis.py — 사전 등록 v2(초안)의 판정 계산. 입력은 paperexp3.py 출력이다.

  python3 prereg_v2_analysis.py --res exp/sealed_v2/res/res_real.csv --fx exp/sealed_v2/res/res_fx.csv \
      --primary "C3 AI→규칙대체" --tstar <판정 T*>

계산 정의는 v1(prereg_analysis.py)과 같다.
  창 평가값 Δ_p = (C_p / C_고정-온라인 − 1) × 100, 기준 비용이 0이면 Δ = 0(실행 코드 규칙).
  d(A, B) = Δ_A − Δ_B(%p). 승 d < −0.5, 패 d > +0.5, 그 밖 무. 무승부를 뺀 양측 정확 부호 검정. 후퇴 비율 = 패 ÷ 단위 수.
판정(판정 T* 하나에서)
  H1v2 (1차) v2 봉인 실제 창: 주 후보 vs 규칙-온라인+대체. p < 0.05이고 승 > 패면 충족. p < 0.05이고 패 > 승이면 '반대 방향'.
  H2v2 (2차) 같은 창: 주 후보 vs 고정-온라인. Holm(2) p < 0.05, 승 > 패, 후퇴 비율 ≤ 0.10이면 충족.
  H3v2 (2차) 합성 확인 40개(시드 201~205): 주 후보 vs 규칙-온라인+대체. Holm(2) p < 0.05이고 승 > 패면 충족.
  2차 가설 둘은 Holm(2)로 보정한다.
탐색(판정 없음): 다른 T*, v1 주 AI vs 규칙(v1 H2의 새 데이터 재현), 다른 후보, 합성 요인 효과(시드 블록 짝 Wilcoxon).
"""
import argparse
import math
import statistics as st

import pandas as pd
from scipy import stats

RULE, BASE, V1 = "규칙-온라인+대체", "고정-온라인", "AI생존κ-온라인+대체"
TIE = 0.5


def wtl(x):
    w = sum(v < -TIE for v in x)
    l = sum(v > TIE for v in x)
    return w, len(x) - w - l, l


def sign_p(w, l):
    n = w + l
    return float("nan") if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(w, l) + 1)) / 2 ** n)


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, run = [0.0] * len(ps), 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - r) * ps[i]))
        adj[i] = run
    return adj


def fmt(x):
    if len(x) == 0:
        return "자료 없음"
    w, t, l = wtl(x)
    return (f"n={len(x)} 승/무/패 {w}/{t}/{l}, p={sign_p(w, l):.4f}, d 중앙 {st.median(x):+.2f}%p, "
            f"최악 {max(x):+.2f}%p, 후퇴 비율 {l / len(x) * 100:.1f}%")


def table(df, b):
    return df[df.Tstar == b].pivot(index="trace", columns="policy", values="rel_fixon")


def direction(p, w, l):
    if not (p < 0.05):
        return "미충족"
    return "충족" if w > l else "미충족(반대 방향: 비교 정책 쪽이 유의하게 낮음)"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--res", required=True)
    ap.add_argument("--fx", required=True)
    ap.add_argument("--primary", required=True)
    ap.add_argument("--tstar", type=float, required=True)
    a = ap.parse_args()
    real, fx = pd.read_csv(a.res), pd.read_csv(a.fx)
    P, b = a.primary, a.tstar
    n_win = real.trace.nunique()
    t = table(real, b)
    print(f"v2 봉인 실제 창 {n_win}개 중 T*={b:g}초 결과 없음 {n_win - len(t)}개(학습 정답이 한 종류뿐)")
    print(f"\n■ H1v2 (1차) T*={b:g}초, {P} vs {RULE}")
    d1 = (t[P] - t[RULE]).tolist()
    w, _, l = wtl(d1)
    print(f"  {fmt(d1)} → 판정: {direction(sign_p(w, l), w, l) if d1 else '자료 없음'}")
    d2 = t[P].tolist()
    tf = table(fx, b)
    d3 = (tf[P] - tf[RULE]).tolist()
    w2, _, l2 = wtl(d2)
    w3, _, l3 = wtl(d3)
    q2, q3 = holm([sign_p(w2, l2) if d2 else 1.0, sign_p(w3, l3) if d3 else 1.0])
    print(f"\n■ H2v2 (2차) T*={b:g}초, {P} vs {BASE} (Holm 2)")
    ok2 = bool(d2) and q2 < 0.05 and w2 > l2 and l2 / len(d2) <= 0.10
    print(f"  {fmt(d2)}, Holm p={q2:.4f} → 판정: {'충족' if ok2 else ('미충족(반대 방향)' if q2 < 0.05 and l2 > w2 else '미충족')}")
    print(f"\n■ H3v2 (2차) 합성 {len(tf)}개, T*={b:g}초, {P} vs {RULE} (Holm 2)")
    ok3 = bool(d3) and q3 < 0.05 and w3 > l3
    print(f"  {fmt(d3)}, Holm p={q3:.4f} → 판정: {'충족' if ok3 else ('미충족(반대 방향)' if q3 < 0.05 and l3 > w3 else '미충족')}")

    print("\n■ 탐색: T*별 (실제 창)")
    for bb in sorted(real.Tstar.unique()):
        tt = table(real, bb)
        print(f"  T*={bb:g}: 주 후보 vs 규칙 {fmt((tt[P] - tt[RULE]).tolist())}")
        print(f"         주 후보 vs 고정 {fmt(tt[P].tolist())}")
        print(f"         v1 주 AI vs 규칙 {fmt((tt[V1] - tt[RULE]).tolist())}")
        print(f"         규칙 vs 고정 {fmt(tt[RULE].tolist())}")
    print("\n■ 탐색: 다른 후보 (실제 창, 판정 T*)")
    for c in [c for c in t.columns if c.startswith("C") and c != P]:
        print(f"  {c} vs 규칙 {fmt((t[c] - t[RULE]).tolist())}")
    print("\n■ 탐색: 합성 요인 효과, d(주 후보, 규칙), 시드 블록 짝 Wilcoxon (판정 T*)")
    dd = (tf[P] - tf[RULE]).rename("d").to_frame()
    dd["cfg"] = [n.split("_")[1] for n in dd.index]
    dd["seed"] = [n.split("_")[2] for n in dd.index]
    for pos, (fac, hi, lo) in enumerate((("크기 변동", "S", "s"), ("짧은 몰림", "b", "p"), ("수준 변화", "w", "n"))):
        diffs = []
        for _, ds in dd.groupby("seed"):
            ds = ds.set_index("cfg").d
            diffs += [ds[c] - ds[c[:pos] + lo + c[pos + 1:]] for c in ds.index
                      if c[pos] == hi and (c[:pos] + lo + c[pos + 1:]) in ds.index]
        if len(diffs) >= 2 and any(v != 0 for v in diffs):
            print(f"  {fac}: 짝 {len(diffs)}개, 차이 중앙 {st.median(diffs):+.2f}%p, Wilcoxon p={stats.wilcoxon(diffs).pvalue:.4f}")


if __name__ == "__main__":
    main()
