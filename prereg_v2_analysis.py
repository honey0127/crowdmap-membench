#!/usr/bin/env python3
"""prereg_v2_analysis.py — 사전 등록 v2(개정 6)의 판정 계산. 입력은 paperexp3.py --model chunk 출력이다.

  python3 prereg_v2_analysis.py --res exp/sealed_v2/res/res_real.csv --fx exp/sealed_v2/res/res_fx.csv \
      --windows exp/sealed_v2/day/windows_day.csv --blocks exp/sealed_v2/day/blocks.csv \
      --primary "<주 후보>" --tstar <판정 T*> --tlist <판정 T*,0.3,1,3,10>

완전성 확인(개정 6, 판정 전에 한다): 하나라도 어긋나면 판정을 내지 않고 '판정 중단'을 쓰고 종료 코드 2로 끝난다.
  - 실제: windows_day.csv의 창 집합 = 결과 파일의 창 집합. 창 × --tlist의 T* × 정책 8개 조합이 빠짐없이 한 번씩 있다.
    비용은 유한한 0 이상 값이다. ai 열이 있고 값은 ok 또는 none이다(none = AI를 쓸 수 없어 대체 정책으로 계산한 정상 처리).
  - 합성: 일정 집합 = 사전 등록한 40개(조건 8 × 반복 5, 시드 201~240, run_sealed_v2.sh의 조건 순서)이고, 같은 완전성을 지킨다.
  - 블록: windows_day.csv의 block 열 = 정의식 = blocks.csv.
  --test-mode(시험 실행용)는 합성 일정 집합 검사만 끄고 출력 머리에 '시험 실행, 판정 아님'을 적는다. 나머지 검사는 그대로다.

단위(docs/prereg-v2.md §2.2)
  창 w: cost_MiB 열(시험 구간 재확보 MiB + 낭비 MiB·초 ÷ T*). 창은 빼지 않는다. AI를 쓸 수 없는 창은 paperexp3.py가
        미리 정한 대체 정책으로 계산한다(ai 열 = none).
  날 k: 그 날 자격 창 비용의 합(자격 시간창 합으로 정의한 일별 비용).
  주 블록 j: windows_day.csv의 block 열(날 번호 = 파일 첫 요청부터 센 24시간 단위 번호, 블록 = 파일_w(날 번호 // 7)).
        블록별 날 목록은 blocks.csv로 실행 전에 고정했고, 여기서 두 파일이 맞는지 확인한다.
  d(A, B) = 100 × (C_A − C_B) ÷ C_고정-온라인 (%p)을 단위의 합 비용으로 계산한다. 창 하나에서는 v1의 Δ_A − Δ_B와 같다.
  기준 비용 C_고정-온라인 = 0인 단위는 d를 정의하지 않고 그 비교에서 뺀다. 그런 단위의 이름과 정책별 비용을 따로 적는다.
  승 d < −0.5, 패 d > +0.5, 그 밖 무. 무승부를 뺀 양측 정확 부호 검정. 후퇴 비율 = 패 ÷ (승 + 무 + 패).
판정(판정 T* 하나에서, 개정 5)
  H1v2 (1차, 관측 판정) 41일 총비용 d_total(주 후보, 규칙) = 100 × (ΣC_주후보 − ΣC_규칙) ÷ ΣC_고정 (모든 자격 창 합).
       d_total < −0.5면 충족, > +0.5면 반대 방향, 그 밖은 차이 없음. 통계 검정이 아니라 41일에서 관측한 값이다.
  H1v2-방향 (함께 보고, 참고) 주 블록 승/무/패와 부호 검정 p. p는 '주 블록이 서로 독립'이라는 근사 위의 참고값이고 판정에 쓰지 않는다.
  H2v2 (2차, 관측 판정) d_total(주 후보, 고정) < −0.5이고 주 블록 후퇴 비율 ≤ 0.10이면 충족. > +0.5면 반대 방향.
  H3v2 (2차, 통계 검정) 합성 확인 40개(조건 8 × 반복 5, 일정마다 다른 시드 201~240): 주 후보 vs 규칙-온라인+대체.
       부호 검정 p < 0.05이고 승 > 패면 충족(p를 쓰는 확증 검정이 이것 하나라 보정하지 않는다). H0은 '40개 일정 모두에서
       P(승) = P(패)'다. 일정은 서로 독립이라 H0에서 무승부를 뺀 승 수는 조건마다 승 확률이 달라도 이항(n, 1/2)을 따른다.
       결론은 8조건을 같은 비중으로 합친 경향이다.
미리 정한 보고(판정 아님): 날 단위 승/무/패와 부호 검정(날 독립을 가정한 요약), 날별 d 목록, 날·주 단위 d의 시간 순
  이웃 순위상관(의존 점검), 합성 조건별 승/무/패, AI를 쓸 수 없던 창, 쓸 문장 틀(§2.3)의 경우.
탐색(판정 없음): 날마다 무작위 창 하나(picked = 1)만 쓴 결과, 모든 창(검정 없음), 다른 T*, v1 주 AI, 다른 후보,
  주 후보가 규칙을 쓴 비율, 합성 요인 효과(Mann–Whitney, 수준별 20개).
"""
import argparse
import math
import statistics as st
import sys

import numpy as np
import pandas as pd
from scipy import stats

RULE, BASE, V1 = "규칙-온라인+대체", "고정-온라인", "AI생존κ-온라인+대체"
POLICIES = [BASE, RULE, V1, "C1 AI세분규칙-온라인+대체", "C2 AI생존κ칸별-온라인+대체", "C4 AI생존세밀κ-온라인+대체",
            "C3 AI→규칙대체", "오라클"]  # paperexp3.evaluate3가 창·T*마다 내는 정책 8개
TIE = 0.5
CONDS = ["spn", "spw", "sbn", "sbw", "Spn", "Spw", "Sbn", "Sbw"]  # run_sealed_v2.sh의 조건 순서(조건 번호 0~7)
FX_EXPECTED = {f"fx_{c}_{201 + 8 * r + k}" for r in range(5) for k, c in enumerate(CONDS)}  # 시드 = 201 + 8 × (반복 − 1) + 조건 번호


def stop(msg):
    print(f"판정 중단(판정 문장을 쓰지 않는다): {msg}")
    print(f"판정 중단: {msg}", file=sys.stderr)
    sys.exit(2)


def complete(df, names, tlist, label):
    """이름 × T* × 정책 8개 조합이 빠짐없이 한 번씩 있고 비용이 유한한 0 이상 값인지 확인한다. 어긋난 것의 설명 목록을 돌려준다"""
    probs = []
    names = set(names)
    got = set(df.trace)
    for what, xs in (("결과가 없는 일정", sorted(names - got)), ("목록에 없는 일정", sorted(got - names))):
        if xs:
            probs.append(f"{label} {what} {len(xs)}개: {', '.join(xs[:10])}{' …' if len(xs) > 10 else ''}")
    tk = {round(float(t), 9) for t in tlist}
    extra_t = sorted({round(float(t), 9) for t in df.Tstar} - tk)
    if extra_t:
        probs.append(f"{label} --tlist에 없는 T*: {extra_t}")
    bad_pol = sorted(set(df.policy) - set(POLICIES))
    if bad_pol:
        probs.append(f"{label} 알 수 없는 정책: {bad_pol}")
    have = set(zip(df.trace, (round(float(t), 9) for t in df.Tstar), df.policy))
    miss = [(n, t, p) for n in sorted(names & got) for t in sorted(tk) for p in POLICIES if (n, t, p) not in have]
    if miss:
        probs.append(f"{label} 빠진 (일정, T*, 정책) {len(miss)}개: " + "; ".join(f"{n} T*={t:g} {p}" for n, t, p in miss[:10])
                     + (" …" if len(miss) > 10 else ""))
    c = pd.to_numeric(df.cost_MiB, errors="coerce")
    bad = df[~np.isfinite(c) | (c < 0)]
    if len(bad):
        probs.append(f"{label} 비용이 숫자가 아니거나 음수인 행 {len(bad)}개: " +
                     "; ".join(f"{r.trace} T*={r.Tstar:g} {r.policy}" for r in bad.head(10).itertuples()))
    return probs


def sign_p(w, l):
    w, l = int(w), int(l)  # numpy 정수면 2 ** n이 n ≥ 63에서 넘친다
    n = w + l
    return float("nan") if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(w, l) + 1)) / 2 ** n)


def wtl(x):
    w = int(sum(v < -TIE for v in x))
    l = int(sum(v > TIE for v in x))
    return w, len(x) - w - l, l


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, run = [0.0] * len(ps), 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - r) * ps[i]))
        adj[i] = run
    return adj


def fmt(x, test=True):
    x = list(x)
    if len(x) == 0:
        return "자료 없음"
    w, t, l = wtl(x)
    return (f"n={len(x)} 승/무/패 {w}/{t}/{l}, {f'p={sign_p(w, l):.4g}, ' if test else ''}d 중앙 {st.median(x):+.2f}%p, "
            f"최악 {max(x):+.2f}%p, 후퇴 비율 {l / len(x) * 100:.1f}%")


def costs(df, b):
    return df[df.Tstar == b].pivot(index="trace", columns="policy", values="cost_MiB")


def week_of(day):
    src, d = day.split("_d")
    return f"{src}_w{int(d) // 7:02d}"


def agg(c, key_of):
    """창 × 정책 비용 → 단위 × 정책 합 비용. 빈 칸이 있으면 멈춘다(완전성 확인을 거친 입력에서는 없다)"""
    assert not c.isna().any().any(), "비용 표에 빈 칸이 있다"
    return c.groupby(c.index.map(key_of)).sum()


def dser(c, A, B):
    """단위별 d(A, B). 기준 비용이 0인 단위는 뺀다(zero()로 따로 적는다)"""
    pos = c[c[BASE] > 0]
    return 100.0 * (pos[A] - pos[B]) / pos[BASE]


def zero(c):
    return c[c[BASE] <= 0]


def lag1(d):
    """같은 파일 안에서 시간 순으로 이웃한 단위끼리 d의 Spearman 순위상관"""
    xs, ys = [], []
    for src in sorted({i.split("_")[0] for i in d.index}):
        s = d[[i for i in d.index if i.split("_")[0] == src]].sort_index()
        xs += list(s.values[:-1])
        ys += list(s.values[1:])
    if len(xs) < 3:
        return "자료 부족"
    r = stats.spearmanr(xs, ys)
    return f"r={r.statistic:+.3f}, p={r.pvalue:.4f}, 쌍 {len(xs)}개"


def total_d(c, A, B):
    assert not c.isna().any().any(), "비용 표에 빈 칸이 있다"
    t = c.sum()
    return float("nan") if t[BASE] <= 0 else 100.0 * (t[A] - t[B]) / t[BASE]


def obs(dt):
    if math.isnan(dt):
        return "정의되지 않음(기준 총비용 0)"
    return "충족(총비용이 낮았다)" if dt < -TIE else ("미충족(반대 방향: 총비용이 높았다)" if dt > TIE else "미충족(총비용 차이가 ±0.5%p 안)")


def direction(p, w, l):
    if not (p < 0.05):
        return "미충족(차이를 확인하지 못함)"
    return "충족" if w > l else "미충족(반대 방향: 비교 정책 쪽이 유의하게 낮음)"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--res", required=True)
    ap.add_argument("--fx", required=True)
    ap.add_argument("--windows", required=True, help="자격 창 목록(name, cluster, picked, block)")
    ap.add_argument("--blocks", required=True, help="주 블록별 날 목록(blocks.csv, 실행 전에 고정)")
    ap.add_argument("--primary", required=True)
    ap.add_argument("--tstar", type=float, required=True)
    ap.add_argument("--tlist", required=True, help="실행한 T* 목록(쉼표, 판정 T* 포함). 모든 창·일정에서 이 T*가 다 있어야 한다")
    ap.add_argument("--test-mode", action="store_true", help="시험 실행: 합성 일정 집합 검사만 끄고 '판정 아님'을 적는다")
    a = ap.parse_args()
    real, fx = pd.read_csv(a.res), pd.read_csv(a.fx)
    W = pd.read_csv(a.windows)
    P, b = a.primary, a.tstar
    tlist = [float(x) for x in a.tlist.split(",")]
    if a.test_mode:
        print("시험 실행(--test-mode): 사전 등록 입력이 아닐 수 있어 아래 판정 줄은 판정이 아니다")
    if round(b, 9) not in {round(t, 9) for t in tlist}:
        stop(f"판정 T* {b:g}가 --tlist {a.tlist}에 없다")
    if P not in POLICIES:
        stop(f"주 후보 {P}가 정책 목록에 없다")
    probs = []
    for df, nm in ((real, "실제"), (fx, "합성")):
        if df.duplicated(["trace", "Tstar", "policy"]).any():
            probs.append(f"{nm}: 중복 행 {int(df.duplicated(['trace', 'Tstar', 'policy']).sum())}개")
        if set(df.model) != {"chunk"}:
            probs.append(f"{nm}: 평가 모델이 chunk가 아니다 {sorted(set(df.model))}")
        if "ai" not in df.columns:
            probs.append(f"{nm}: ai 열이 없다(개정 4 이전 코드의 출력)")
        elif not set(df.ai) <= {"ok", "none"}:
            probs.append(f"{nm}: ai 열 값이 ok·none이 아니다 {sorted(set(df.ai) - {'ok', 'none'})}")
    day_of, picked = dict(zip(W.name, W.cluster)), set(W.name[W.picked == 1])
    block_of = dict(zip(W.cluster, W.block))
    if not all(block_of[d] == week_of(d) for d in block_of):
        probs.append("windows_day.csv의 block 열이 정의(파일_w(날 번호 // 7))와 다르다")
    Bk = pd.read_csv(a.blocks)
    if {d: r.block for r in Bk.itertuples() for d in r.days.split()} != block_of:
        probs.append("blocks.csv와 windows_day.csv의 블록 구성이 다르다")
    probs += complete(real, W.name, tlist, "실제")
    fx_names = set(fx.trace) if a.test_mode else FX_EXPECTED
    if not a.test_mode and set(fx.trace) != FX_EXPECTED:
        probs.append("합성: 일정 집합이 사전 등록(조건 8 × 반복 5, 시드 201~240)과 다르다")
    probs += complete(fx, fx_names, tlist, "합성")
    if probs:
        stop(" | ".join(probs) + " → 빠진 묶음을 같은 코드·설정으로 다시 실행한 뒤 판정한다(prereg-v2.md §7)")
    seeds = sorted(int(n.rsplit("_", 1)[1]) for n in fx.trace.unique())
    print(f"완전성 확인 통과: 실제 창 {len(W)}개 × T* {len(tlist)}개 × 정책 {len(POLICIES)}개, "
          f"합성 일정 {fx.trace.nunique()}개(시드 {seeds[0]}~{seeds[-1]}{'' if a.test_mode else ', 사전 등록과 같음'})")

    cw = costs(real, b)[POLICIES]
    cf = costs(fx, b)[POLICIES]
    print(f"자격 창 {len(W)}개(날 {W.cluster.nunique()}개) 중 T*={b:g}초 결과가 있는 창 {len(cw)}개")
    no_ai = sorted(real[(real.Tstar == b) & (real.ai == "none")].trace.unique())
    print(f"AI를 쓸 수 없어 AI 정책을 대체 정책으로 계산한 창 {len(no_ai)}개{': ' + ', '.join(no_ai) if no_ai else ''}")
    cd = agg(cw, lambda n: day_of[n])
    cb = agg(cd, lambda d: block_of[d])
    print(f"단위: 창 {len(cw)}개 → 날 {len(cd)}개 → 주 블록 {len(cb)}개(blocks.csv와 같음, 블록마다 날 {Bk.n_days.min()}~{Bk.n_days.max()}개)")
    for lab, c in (("주 블록", cb), ("날", cd), ("합성 일정", cf)):
        z = zero(c)
        if len(z):
            print(f"기준 비용 0인 {lab} {len(z)}개(비교에서 뺀다): " +
                  "; ".join(f"{i} 주 후보 {r[P]:.3f}, 규칙 {r[RULE]:.3f}, 고정 {r[BASE]:.3f} MiB" for i, r in z.iterrows()))
        else:
            print(f"기준 비용 0인 {lab}: 0개")
    t = real[real.Tstar == b].pivot(index="trace", columns="policy", values="rel_fixon")
    pw = cw[cw[BASE] > 0]
    gap = (dser(cw, P, RULE) - (t.loc[pw.index, P] - t.loc[pw.index, RULE])).abs()
    print(f"검산: 창에서 d와 Δ 차의 최대 차이 {gap.max() if len(gap) else 0:.2e}%p")
    rr = real[(real.Tstar == b) & real.trace.isin(cw.index)]
    alt = rr.assign(day=rr.trace.map(day_of)).groupby(["day", "policy"]).cost_MiB.sum().unstack()[cd.columns]
    print(f"검산: 날 합 비용의 두 계산 경로 최대 차이 {float((alt.loc[cd.index] - cd).abs().max().max()):.2e} MiB")

    tot = cw.sum()
    print(f"\n총비용(날 {len(cd)}개, 자격 창 {len(cw)}개 합, MiB): 주 후보 {tot[P]:.1f}, 규칙 {tot[RULE]:.1f}, 고정-온라인 {tot[BASE]:.1f}, "
          f"v1 주 AI {tot[V1]:.1f}, 오라클 {tot['오라클']:.1f}")
    dt1, dt2 = total_d(cw, P, RULE), total_d(cw, P, BASE)
    d1, d2, d3 = dser(cb, P, RULE), dser(cb, P, BASE), dser(cf, P, RULE)
    w1, _, l1 = wtl(d1)
    w2, _, l2 = wtl(d2)
    w3, _, l3 = wtl(d3)
    print(f"\n■ H1v2 (1차, 관측 판정) T*={b:g}초, 41일 총비용, {P} vs {RULE}")
    print(f"  d_total {dt1:+.2f}%p → 판정: {obs(dt1)}")
    print(f"  방향(참고, 주 블록 독립을 가정한 근사 p): {fmt(d1)}")
    ok2 = (not math.isnan(dt2)) and dt2 < -TIE and len(d2) > 0 and l2 / len(d2) <= 0.10
    print(f"\n■ H2v2 (2차, 관측 판정) T*={b:g}초, 41일 총비용, {P} vs {BASE}, 주 블록 후퇴 비율 ≤ 10%")
    print(f"  d_total {dt2:+.2f}%p, 주 블록 후퇴 비율 {l2 / len(d2) * 100 if len(d2) else float('nan'):.1f}% → 판정: "
          f"{'충족' if ok2 else ('미충족(반대 방향: 총비용이 높았다)' if dt2 > TIE else '미충족')}")
    print(f"  방향(참고, 주 블록 독립을 가정한 근사 p): {fmt(d2)}")
    p3 = sign_p(w3, l3) if len(d3) else float("nan")
    print(f"\n■ H3v2 (2차, 통계 검정) 합성 {len(cf)}개, T*={b:g}초, {P} vs {RULE} (H0: 모든 일정에서 P(승) = P(패))")
    print(f"  {fmt(d3)} → 판정: {direction(p3, w3, l3) if len(d3) else '자료 없음'}")
    dirn = "주 후보 쪽" if (w1 > l1 and sign_p(w1, l1) < 0.05) else ("규칙 쪽" if (l1 > w1 and sign_p(w1, l1) < 0.05) else "뚜렷하지 않음")
    print(f"\n■ 쓸 문장 틀(§2.3): H1v2 = {obs(dt1)}, 주 블록 방향(참고 p < 0.05 기준) = {dirn}")

    print(f"\n■ 미리 정한 보고 (판정 아님, T*={b:g}초)")
    dd = dser(cd, P, RULE)
    print(f"  날 단위(날 독립을 가정한 요약): 주 후보 vs 규칙 {fmt(dd)}")
    print(f"                              주 후보 vs 고정 {fmt(dser(cd, P, BASE))}")
    print(f"  의존 점검(이웃 순위상관, d(주 후보, 규칙)): 날 {lag1(dd)} | 주 블록 {lag1(d1)}")
    print("  날별 d(주 후보 − 규칙, %p), 파일·날 순: " + ", ".join(f"{i} {v:+.2f}" for i, v in dd.sort_index().items()))
    cfg = pd.Series([n.split("_")[1] for n in cf.index], index=cf.index)
    print("  합성 조건별 승/무/패(주 후보 vs 규칙): " +
          ", ".join(f"{k} {'/'.join(map(str, wtl(d3[cfg[d3.index] == k])))}" for k in sorted(cfg.unique())))

    print("\n■ 탐색: 날마다 무작위 창 하나(원래 v2 표본, picked = 1)만 쓴 창 단위 결과 (판정 T*)")
    cp = cw[cw.index.isin(picked)]
    print(f"  주 후보 vs 규칙 {fmt(dser(cp, P, RULE))}")
    print(f"  주 후보 vs 고정 {fmt(dser(cp, P, BASE))}")
    print("\n■ 탐색: 모든 자격 창을 각각 단위로 본 결과 (판정 T*, 같은 날의 창은 독립이 아니라 검정하지 않는다)")
    print(f"  주 후보 vs 규칙 {fmt(dser(cw, P, RULE), test=False)}")
    print("\n■ 탐색: T*별 (주 블록 단위)")
    for bb in sorted(real.Tstar.unique()):
        cc = agg(agg(costs(real, bb)[POLICIES], lambda n: day_of[n]), lambda d: block_of[d])
        print(f"  T*={bb:g}: 주 후보 vs 규칙 {fmt(dser(cc, P, RULE))}")
        print(f"         주 후보 vs 고정 {fmt(dser(cc, P, BASE))}")
        print(f"         v1 주 AI vs 규칙 {fmt(dser(cc, V1, RULE))}")
        print(f"         v1 주 AI vs 고정 {fmt(dser(cc, V1, BASE))}")
        print(f"         규칙 vs 고정 {fmt(dser(cc, RULE, BASE))}")
    if "base_frac" in real.columns:
        bf = pd.to_numeric(real[(real.Tstar == b) & (real.policy == P)].base_frac, errors="coerce")
        print(f"\n■ 탐색: 주 후보가 규칙을 쓴 비율(시험 구간, 창 평균, 판정 T*): {bf.mean():.3f} (창 {bf.notna().sum()}개)")
    print("\n■ 탐색: 다른 후보 (주 블록 단위, 판정 T*)")
    for cand in [x for x in cb.columns if x.startswith("C") and x != P]:
        print(f"  {cand} vs 규칙 {fmt(dser(cb, cand, RULE))}")
    print("\n■ 탐색: 합성 요인 효과, d(주 후보, 규칙), 수준별 Mann–Whitney (판정 T*)")
    for pos_, (fac, hi, lo) in enumerate((("크기 변동", "S", "s"), ("짧은 몰림", "b", "p"), ("수준 변화", "w", "n"))):
        x = d3[cfg[d3.index].str[pos_] == hi].tolist()
        y = d3[cfg[d3.index].str[pos_] == lo].tolist()
        if len(x) >= 2 and len(y) >= 2:
            print(f"  {fac}: {hi} {len(x)}개 중앙 {st.median(x):+.2f}%p, {lo} {len(y)}개 중앙 {st.median(y):+.2f}%p, "
                  f"Mann–Whitney p={stats.mannwhitneyu(x, y).pvalue:.4f}")


if __name__ == "__main__":
    main()
