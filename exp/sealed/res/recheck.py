#!/usr/bin/env python3
"""recheck.py — 봉인 시험 판정의 독립 재계산. 실행 뒤에 쓴 검산용이며 사전 등록 코드가 아니다.

  python3 recheck.py exp/sealed/res/res_main.csv exp/sealed/windows.csv exp/sealed/res/res_fx.csv exp/sealed/res/verdict.txt

prereg_analysis.py와 다른 경로로 계산한다.
  - 창 평가값 Δ를 rel_fixon 열 대신 cost_MiB 열에서 다시 만든다(같은 값이어야 한다).
  - 부호 검정은 scipy.stats.binomtest, Mann–Whitney는 scipy.stats.mannwhitneyu(정규 근사, 연속성 보정)로 한다.
  - verdict.txt의 승/무/패·p·Holm p와 맞춰 본다.
"""
import math
import re
import sys

import pandas as pd
from scipy import stats

AI, RULE, BASE, TIE = "AI생존κ-온라인+대체", "규칙-온라인+대체", "고정-온라인", 0.5
res_p, win_p, fx_p, ver_p = sys.argv[1:5]
res, win, fx = pd.read_csv(res_p), pd.read_csv(win_p), pd.read_csv(fx_p)
verdict = open(ver_p).read()
bad = []


def with_delta(df):
    base = df[df.policy == BASE].set_index(["proto", "trace", "Tstar"])["cost_MiB"].rename("base")
    d = df.join(base, on=["proto", "trace", "Tstar"])
    d["delta"] = (d["cost_MiB"] / d["base"] - 1) * 100
    zero = d["base"] == 0                      # 시험 구간 비용이 0인 단위: run_frozen 규칙대로 Δ = 0(무승부)
    d.loc[zero, "delta"] = 0.0
    gap = (d["delta"] - d["rel_fixon"]).abs()
    print(f"Δ(cost_MiB) vs rel_fixon 최대 차이: {gap.max():.3g}, NaN {int(gap.isna().sum())}개, "
          f"기준 비용 0인 단위 {d.loc[zero & (d.policy == BASE), ['proto', 'trace', 'Tstar']].values.tolist()}")
    if not (gap.max() < 1e-9 and gap.notna().all()):
        bad.append("Δ 불일치")
    return d


def table(d, proto, b):
    t = d[(d.proto == proto) & (d.Tstar == b)].pivot(index="trace", columns="policy", values="delta")
    return t


def wtl(x):
    w, l = int((x < -TIE).sum()), int((x > TIE).sum())
    return w, len(x) - w - l, l


def sign_p(w, l):
    return float("nan") if w + l == 0 else stats.binomtest(min(w, l), w + l, 0.5).pvalue


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, run = [0.0] * len(ps), 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - r) * ps[i]))
        adj[i] = run
    return adj


d = with_delta(res)
t1 = table(d, "within", 1.0)
R = win.loc[win["set"].str.contains("R"), "name"].tolist()
cells = ["sa", "sA", "Sa", "SA"]
S = {c: win.loc[win["set"].str.contains("S") & (win["cell"] == c), "name"].tolist() for c in cells}
print(f"창: R {len(R)}, S {sum(map(len, S.values()))}, 서로 다른 창 {win['name'].nunique()}, 결과의 창 {d.trace.nunique()}")
for nm, U in (("R", R), ("S", sum(S.values(), []))):
    miss = [u for u in U if u not in t1.index]
    print(f"{nm}: T*=1 within 결과 없음 {len(miss)}개 {miss}")
print(f"행 수: within {int((d.proto == 'within').sum())}, pre {int((d.proto == 'pre').sum())}, "
      f"T*별 within 창 수 {d[d.proto == 'within'].groupby('Tstar').trace.nunique().to_dict()}")


def grab(pattern):
    m = re.search(pattern, verdict, flags=re.S)
    return m.groups() if m else None


def check(name, x, pat):
    w, t, l = wtl(x)
    p = sign_p(w, l)
    g = grab(pat)
    ok = g is not None and (int(g[0]), int(g[1]), int(g[2])) == (w, t, l) and abs(float(g[3]) - p) < 6e-5
    print(f"{name}: n={len(x)} 승/무/패 {w}/{t}/{l}, p={p:.4f}, 중앙 {x.median():+.2f}%p, 후퇴 {l / len(x) * 100:.1f}%"
          f"  | verdict {g} → {'일치' if ok else '불일치'}")
    if not ok:
        bad.append(name)
    return w, t, l, p


tR = t1.loc[[u for u in R if u in t1.index]]
w1, t1_, l1, p1 = check("H1", tR[AI], r"■ H1.*?n=\d+ 승/무/패 (\d+)/(\d+)/(\d+), p=([0-9.]+)")
print(f"  H1 판정(재계산): {'충족' if (p1 < 0.05 and w1 > l1 and l1 / len(tR) <= 0.10) else '미충족'}")
w2, _, l2, p2 = check("H2", tR[AI] - tR[RULE], r"■ H2.*?n=\d+ 승/무/패 (\d+)/(\d+)/(\d+), p=([0-9.]+)")
print(f"  H2 판정(재계산): {'충족' if (p2 < 0.05 and w2 > l2) else '미충족'}")
ps, out = [], []
for c in cells:
    tc = t1.loc[[u for u in S[c] if u in t1.index]]
    r_ = check(f"H3 {c}", tc[AI] - tc[RULE], rf"■ H3.*?\n  {c}: n=\d+ 승/무/패 (\d+)/(\d+)/(\d+), p=([0-9.]+)")
    out.append(r_)
    ps.append(1.0 if math.isnan(r_[3]) else r_[3])
for c, (w, _, l, _), q in zip(cells, out, holm(ps)):
    g = grab(rf"■ H3.*?\n  {c}: .*?Holm p=([0-9.]+) → (\S+ \S+|\S+)")
    v = "AI 우세" if (q < 0.05 and w > l) else ("규칙 우세" if (q < 0.05 and l > w) else "검출 못 함")
    ok = g is not None and abs(float(g[0]) - q) < 6e-5 and g[1].startswith(v.split()[0])
    print(f"  H3 {c}: Holm p={q:.4f} → {v} | verdict {g} → {'일치' if ok else '불일치'}")
    if not ok:
        bad.append(f"H3 {c} Holm")

f = with_delta(fx)
tf = table(f, "within", 1.0)
dfx = tf[AI] - tf[RULE]
print(f"합성: 창 {len(dfx)}개")
facs = (("크기 변동", "S", "s"), ("짧은 몰림", "b", "p"), ("수준 변화", "w", "n"))
pm = []
for pos, (fac, hi, lo) in enumerate(facs):
    H = dfx[[n.split("_")[1][pos] == hi for n in dfx.index]]
    L = dfx[[n.split("_")[1][pos] == lo for n in dfx.index]]
    p = stats.mannwhitneyu(H, L, alternative="two-sided", method="asymptotic", use_continuity=True).pvalue
    pm.append((fac, H, L, p))
for (fac, H, L, p), q in zip(pm, holm([x[3] for x in pm])):
    g = grab(rf"■ H4.*?\n  {fac}: .*?p=([0-9.]+), Holm p=([0-9.]+)")
    ok = g is not None and abs(float(g[0]) - p) < 6e-5 and abs(float(g[1]) - q) < 6e-5
    print(f"H4 {fac}: 높음 {H.median():+.2f} (n={len(H)}) vs 낮음 {L.median():+.2f} (n={len(L)}), p={p:.4f}, Holm {q:.4f}"
          f" | verdict {g} → {'일치' if ok else '불일치'}")
    if not ok:
        bad.append(f"H4 {fac}")
print("검산: 모두 일치" if not bad else f"검산: 불일치 {bad}")
sys.exit(1 if bad else 0)
