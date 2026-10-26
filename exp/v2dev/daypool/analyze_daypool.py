#!/usr/bin/env python3
"""analyze_daypool.py — 개발 데이터로 판정 규칙의 충족 확률을 잰다(v1 R의 BurstGPT 40일, 자격 창 420개, T* = 2.6초).

  python3 exp/v2dev/daypool/analyze_daypool.py > exp/v2dev/daypool/summary.txt

계산 규칙은 prereg_v2_analysis.py(사전 등록 v2 개정 5)와 같다.
  - 창은 빼지 않는다. AI를 쓸 수 없는 창은 paperexp3.py가 대체 정책으로 계산했다(res_q3.csv, ai = none).
    v1의 '학습 정답이 한 종류뿐인 T*는 건너뛴다' 규칙은 쓰지 않는다(그 9창도 res_q3.csv에 있다).
  - 날 비용 = 그 날 자격 창 비용의 합(개정 3에서 결과를 보기 전에 정했다, 2026-10-25 18:00 KST).
  - 주 블록 = 같은 파일에서 (날 번호 // 7)이 같은 날들의 비용 합. 날 번호 = 파일 첫 요청부터 센 24시간 단위 번호.
  - H1v2(개정 5) = 총비용 d_total(C3, 규칙) < −0.5%p(관측 판정). 주 블록 부호 검정은 참고.
  - H2v2 = d_total(C3, 고정) < −0.5%p이고 주 블록 후퇴 비율 ≤ 10%. H3v2 = 합성 부호 검정 p < 0.05(보정 없음).
충족 확률: 개발 주 블록 21개에서 v2 봉인 블록 수(22개)를 복원 추출하는 붓스트랩 2만 회(시드 0, 블록 독립 가정, 계획용).
  효과 축소 민감도: 블록마다 C3 비용에 δ × 고정-온라인 비용 ÷ 100을 더해(δ = 1, 1.5, 2%p, 모든 블록에서 C3가 그만큼 불리)
  같은 확률을 다시 잰다.
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)
from prereg_v2_analysis import BASE, RULE, V1, agg, dser, fmt, lag1, sign_p, total_d, week_of, zero  # noqa: E402

P = "C3 AI→규칙대체"
B = 2.6
REPS, SEED = 20000, 0
N_BLOCK = 22  # v2 봉인 41일의 주 블록 수(입력 목록 exp/sealed_v2/day/blocks.csv, 정책 결과와 무관)
H3P = (26 / 40, 9 / 40, 5 / 40)  # v1 합성 40, 청크 모델, T* = 2.6초 (v2-dev-results.md §7.4)


def main():
    res = pd.concat([pd.read_csv(os.path.join(HERE, f)) for f in ("res_q1.csv", "res_q2.csv", "res_q3.csv")], ignore_index=True)
    assert set(res.model) == {"chunk"} and not res.duplicated(["trace", "Tstar", "policy"]).any()
    res["ai"] = res["ai"].fillna("ok") if "ai" in res.columns else "ok"
    win = pd.read_csv(os.path.join(HERE, "windows.csv"))
    day_of = dict(zip(win.name, win.cluster))
    v1 = pd.read_csv(os.path.join(REPO, "exp/sealed/windows.csv"))
    v1pick = {r.cluster: r.name for r in v1.itertuples() if "R" in r.set and r.group == "BurstGPT"}
    cw = res[res.Tstar == B].pivot(index="trace", columns="policy", values="cost_MiB")
    print(f"자격 창 {len(win)}개, T*={B:g}초 결과가 있는 창 {len(cw)}개, AI를 쓸 수 없던 창 "
          f"{sorted(res[res.ai == 'none'].trace.unique())}, 날 {win.cluster.nunique()}개")
    cd = agg(cw, lambda n: day_of[n])
    cb = agg(cd, week_of)
    print(f"주 블록 {len(cb)}개(블록마다 날 {cd.index.map(week_of).value_counts().min()}~{cd.index.map(week_of).value_counts().max()}개); "
          f"기준 비용 0인 단위: 창 {len(zero(cw))}개, 날 {len(zero(cd))}개, 주 블록 {len(zero(cb))}개")

    print(f"\n■ 40일 총비용(MiB): C3 {cw[P].sum():.1f}, 규칙 {cw[RULE].sum():.1f}, 고정-온라인 {cw[BASE].sum():.1f}, "
          f"v1 주 AI {cw[V1].sum():.1f}, 오라클 {cw['오라클'].sum():.1f}")
    print(f"  d_total: C3 − 규칙 {total_d(cw, P, RULE):+.2f}%p, C3 − 고정 {total_d(cw, P, BASE):+.2f}%p, "
          f"v1 주 AI − 규칙 {total_d(cw, V1, RULE):+.2f}%p, 규칙 − 고정 {total_d(cw, RULE, BASE):+.2f}%p")
    pairs = ((P, RULE, "C3 vs 규칙"), (P, BASE, "C3 vs 고정"), (V1, RULE, "v1 주 AI vs 규칙"), (RULE, BASE, "규칙 vs 고정"))
    for lab, c in (("주 블록(방향 참고)", cb), ("날(방향 요약)", cd)):
        print(f"\n■ {lab}, 단위 {len(c)}개")
        for A, Bp, nm in pairs:
            print(f"  {nm}: {fmt(dser(c, A, Bp))}")
    picked = [v1pick[d] for d in cd.index if v1pick.get(d) in cw.index]
    print("\n■ 창 하나(개정 2 판정 단위): v1 R 표본 창, 같은 40일")
    for A, Bp, nm in pairs:
        print(f"  {nm}: {fmt(dser(cw.loc[picked], A, Bp))}")
    print("\n■ 의존 점검: 시간 순 이웃 단위끼리 d(C3, 규칙)의 Spearman 순위상관")
    print(f"  날 {lag1(dser(cd, P, RULE))} | 주 블록 {lag1(dser(cb, P, RULE))}")

    rng = np.random.default_rng(SEED)
    cache = {}

    def sp(w, l):
        if (w, l) not in cache:
            cache[(w, l)] = sign_p(w, l) if w + l else 1.0
        return cache[(w, l)]

    F, Rb, Cb = cb[BASE].to_numpy(), cb[RULE].to_numpy(), cb[P].to_numpy()
    print(f"\n■ 충족 확률 (개발 주 블록 {len(cb)}개에서 {N_BLOCK}개 복원 추출 {REPS:,}회, 시드 {SEED}; 블록 독립 가정, 계획용 추정)")
    for s in (0.0, 1.0, 1.5, 2.0):
        C = Cb + s / 100 * F
        ok = np.zeros(4)
        for _ in range(REPS):
            i = rng.integers(len(cb), size=N_BLOCK)
            f, r_, c_ = F[i], Rb[i], C[i]
            dt1 = 100 * (c_.sum() - r_.sum()) / f.sum()
            dt2 = 100 * (c_.sum() - f.sum()) / f.sum()
            db1 = 100 * (c_ - r_) / f
            db2 = 100 * (c_ - f) / f
            w1, l1 = int((db1 < -0.5).sum()), int((db1 > 0.5).sum())
            l2 = int((db2 > 0.5).sum())
            x3 = rng.multinomial(40, H3P)
            ok += [dt1 < -0.5, sp(w1, l1) < 0.05 and w1 > l1, dt2 < -0.5 and l2 / N_BLOCK <= 0.10,
                   sp(int(x3[0]), int(x3[2])) < 0.05 and x3[0] > x3[2]]
        h = ok / REPS
        lab = "개발 효과 그대로" if s == 0.0 else f"모든 블록에서 C3를 {s:g}%p 불리하게"
        print(f"  {lab}: H1v2(총비용) {h[0]:.3f}, 주 블록 방향 참고 p<0.05 {h[1]:.3f}, H2v2 {h[2]:.3f}, H3v2 {h[3]:.3f}")


if __name__ == "__main__":
    main()
