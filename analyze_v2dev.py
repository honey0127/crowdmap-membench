#!/usr/bin/env python3
"""analyze_v2dev.py — v2 후보 개발 평가 요약과 선택 규칙 적용(paperexp3.py 출력)

  python3 analyze_v2dev.py exp/v2dev/res_a.csv exp/v2dev/res_b.csv --windows exp/sealed/windows.csv > exp/v2dev/summary.txt
  python3 analyze_v2dev.py exp/v2dev/chunk/res_q1.csv exp/v2dev/chunk/res_q2.csv > exp/v2dev/chunk/summary.txt

묶음: dev(개발 트레이스 10), devfx(개발 요인 실험 16), v1R(v1 봉인 R 54), v1S(v1 봉인 S 80, R과 겹친 3창 포함), v1fx(v1 합성 확인 40).
d(A, B)는 prereg_v2_analysis.py와 같다: 100 × (C_A − C_B) ÷ C_고정-온라인, 기준 비용이 0이면 극한 규칙(같으면 0, 아니면 ±∞).
선택 규칙(paperexp3.py 머리말과 같다, 첫 결과를 보기 전에 정했다):
  v1R에서 T* = 0.3·1·3초 각각 '후보 vs 규칙-온라인+대체' 부호 검정 p < 0.05이고 승 > 패인 T* 수가 가장 많은 후보.
  같으면 세 T*의 d 중앙값 평균이 낮은 쪽. 어느 T*(0.3·1·3·10)에서든 고정-온라인 대비 후퇴 비율이 10%를 넘으면 뺀다.
  (판정 T* 2.6초 결과가 같은 파일에 있어도 선택에는 쓰지 않는다.)
검정력(판정 T*가 있으면): 주 후보 vs 규칙의 승/무/패 비율을 41창에 옮긴 모의 실험(2만 회, 시드 0).
"""
import argparse
import math
import statistics as st

import numpy as np
import pandas as pd
from scipy import stats

RULE, BASE, V1 = "규칙-온라인+대체", "고정-온라인", "AI생존κ-온라인+대체"
CANDS = ["C1 AI세분규칙-온라인+대체", "C2 AI생존κ칸별-온라인+대체", "C3 AI→규칙대체", "C4 AI생존세밀κ-온라인+대체"]
TIE = 0.5
SEL_T, EXCL_T = (0.3, 1.0, 3.0), (0.3, 1.0, 3.0, 10.0)


def dser(c, A, B):
    base, diff = c[BASE], c[A] - c[B]
    d = 100.0 * diff / base.where(base > 0)
    zero = base <= 0
    d[zero] = np.where(diff[zero] == 0, 0.0, np.sign(diff[zero]) * math.inf)
    return d


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


def power(pw, pt, pl, n=41, reps=20000, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.multinomial(n, [pw, pt, pl], size=reps)
    ok = 0
    for w, _, l in x:
        ok += int(w > l and sp(int(w), int(l)) < 0.05)
    return ok / reps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("res", nargs="+")
    ap.add_argument("--windows", default="exp/sealed/windows.csv")
    ap.add_argument("--tstar-eval", type=float, default=2.6)
    a = ap.parse_args()
    r = pd.concat([pd.read_csv(p) for p in a.res], ignore_index=True)
    win = pd.read_csv(a.windows).set_index("name")
    groups = {
        "dev": list(r[r.src == "dev"].trace.unique()),
        "devfx": list(r[r.src == "devfx"].trace.unique()),
        "v1R": [t for t in r[r.src == "v1real"].trace.unique() if "R" in win.loc[t, "set"]],
        "v1S": [t for t in r[r.src == "v1real"].trace.unique() if "S" in win.loc[t, "set"]],
        "v1fx": list(r[r.src == "v1fx"].trace.unique()),
    }
    models = sorted(r.model.unique()) if "model" in r.columns else ["v1"]
    print("평가 모델:", models, "| 단위 수:", {k: len(v) for k, v in groups.items()}, "| 행", len(r),
          "| 중복", int(r.duplicated(["trace", "Tstar", "policy"]).sum()))
    piv = {b: r[r.Tstar == b].pivot(index="trace", columns="policy", values="cost_MiB") for b in sorted(r.Tstar.unique())}
    pols = [V1] + [c for c in CANDS if c in r.policy.unique()]
    for gname, units in groups.items():
        print(f"\n■ {gname} (단위 {len(units)})")
        for b, t in piv.items():
            t = t.loc[[u for u in units if u in t.index]]
            if t.empty:
                continue
            print(f"  T*={b:g} (n={len(t)}) 규칙 vs 고정: {fmt(dser(t, RULE, BASE))}")
            for p in pols:
                dv = dser(t, p, BASE)
                print(f"    {p:28s} vs 규칙 {fmt(dser(t, p, RULE)):42s} | vs 고정 {fmt(dv)}, 후퇴 {wtl(dv)[2] / len(t) * 100:.0f}%")
    print("\n■ 출처별: v1R, C3·v1 주 AI vs 규칙")
    for b, t in piv.items():
        t = t.loc[[u for u in groups["v1R"] if u in t.index]]
        for src, m in (("BurstGPT", ~t.index.str.startswith("az24")), ("Azure", t.index.str.startswith("az24"))):
            tt = t[m]
            if len(tt) and "C3 AI→규칙대체" in tt.columns:
                print(f"  T*={b:g} {src} (n={len(tt)}): C3 {fmt(dser(tt, 'C3 AI→규칙대체', RULE))} | v1 주 AI {fmt(dser(tt, V1, RULE))}")
    if "base_frac" in r.columns:
        c3 = r[r.policy == "C3 AI→규칙대체"]
        print("\n■ C3가 규칙을 쓴 비율(시험 구간, 모든 개발 일정 평균): " +
              ", ".join(f"T*={b:g} {pd.to_numeric(c3[c3.Tstar == b].base_frac, errors='coerce').mean():.2f}" for b in sorted(c3.Tstar.unique())))
    print("\n■ 선택 규칙 적용 (v1R; 이기기 T* = 0.3·1·3, 후퇴 T* = 0.3·1·3·10)")
    best = None
    for p in [c for c in CANDS if c in r.policy.unique()]:
        wins, meds, worst_lose = 0, [], 0.0
        for b, t in piv.items():
            if not any(abs(b - x) < 1e-9 for x in EXCL_T):
                continue
            t = t.loc[[u for u in groups["v1R"] if u in t.index]]
            if t.empty:
                continue
            lose = wtl(dser(t, p, BASE))[2] / len(t)
            worst_lose = max(worst_lose, lose)
            if any(abs(b - x) < 1e-9 for x in SEL_T):
                d = dser(t, p, RULE)
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
    b = a.tstar_eval
    if best and any(abs(x - b) < 1e-9 for x in piv):
        P = best[1]
        t = piv[[x for x in piv if abs(x - b) < 1e-9][0]]
        print(f"\n■ 검정력 추정 (T*={b:g}초, {P} vs 규칙, 41창, 모의 2만 회)")
        bgR = [u for u in groups["v1R"] if not u.startswith("az24") and u in t.index]
        bgRS = bgR + [u for u in groups["v1S"] if not u.startswith("az24") and u in t.index and u not in bgR]
        for nm, units in (("v1R BurstGPT", bgR), ("v1R∪S BurstGPT", bgRS)):
            w, ti, l = wtl(dser(t.loc[units], P, RULE))
            n = w + ti + l
            print(f"  {nm} 비율 {w}/{ti}/{l} (승 {w / n:.2f}, 무 {ti / n:.2f}, 패 {l / n:.2f}) → 충족 확률 {power(w / n, ti / n, l / n):.2f}")
        print(f"  보수적 가정(승 0.35, 무 0.50, 패 0.15) → 충족 확률 {power(0.35, 0.50, 0.15):.2f}")


if __name__ == "__main__":
    main()
