#!/usr/bin/env python3
"""posthoc.py — 봉인 시험 v1 사후 분석(판정 뒤에 정했다. 판정을 바꾸지 않는다).

  python3 exp/sealed/res/posthoc/posthoc.py > exp/sealed/res/posthoc/posthoc_out.txt   # 저장소 루트에서

1. 단위 구조: R 집합 창의 출처·파일·날짜. Azure 두 파일의 첫 요청 시각은 원본 TIMESTAMP 첫 행이다
   (code 2024-05-10 00:00:00.009930+00:00, conv 2024-05-12 00:00:00.001163+00:00). BurstGPT_2·3은 서로 다른 기간이다(BurstGPT README:
   _1·_2는 이어진 121일, _3은 다른 110일).
2. H1·H2를 Azure 같은 달력 날짜의 창(code·conv)을 한 단위로 합쳐(d 평균) 다시 계산한다.
3. H4를 시드 블록으로 다시 계산한다. 같은 시드 안에서 다른 두 요인 수준이 같은 두 트레이스를 짝지어(요인마다 20쌍)
   짝 차이의 Wilcoxon 부호 순위 검정, 시드별 짝 차이 평균 5개의 t-검정을 한다. 각각 Holm(3)으로 보정한다.
4. 결과 파일 점검: (proto, trace, T*, policy) 중복 행 수.
"""
import datetime as dt
import os

import numpy as np
import pandas as pd
from scipy import stats

AI, RULE, TIE = "AI생존κ-온라인+대체", "규칙-온라인+대체", 0.5
RES = "exp/sealed/res"
AZ_START = {"az24_code": dt.date(2024, 5, 10), "az24_conv": dt.date(2024, 5, 12)}


def wtl(x):
    w, l = int((x < -TIE).sum()), int((x > TIE).sum())
    return w, len(x) - w - l, l


def sign(x):
    w, t, l = wtl(x)
    p = stats.binomtest(min(w, l), w + l, 0.5).pvalue if w + l else float("nan")
    return f"n={len(x)} 승/무/패 {w}/{t}/{l}, p={p:.2g}, d 중앙 {np.median(x):+.2f}%p, 후퇴 비율 {l / len(x) * 100:.1f}%"


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, run = [0.0] * len(ps), 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - r) * ps[i]))
        adj[i] = run
    return adj


def main():
    win = pd.read_csv("exp/sealed/windows.csv").set_index("name")
    res = pd.read_csv(f"{RES}/res_main.csv")
    t1 = res[(res.proto == "within") & (res.Tstar == 1.0)].pivot(index="trace", columns="policy", values="rel_fixon")
    R = [u for u in win.index[win["set"].str.contains("R")] if u in t1.index]
    az = [u for u in R if win.loc[u, "source"].startswith("az")]
    bg = [u for u in R if u not in az]
    date = {u: AZ_START[win.loc[u, "source"]] + dt.timedelta(days=int(win.loc[u, "cluster"].split("_d")[-1])) for u in az}

    print("■ 1. 단위 구조 (R 집합, T* = 1초 결과가 있는 창)")
    print(f"  Azure {len(az)}창: 파일별 날짜 code {sorted(str(date[u]) for u in az if 'code' in u)}")
    print(f"                      conv {sorted(str(date[u]) for u in az if 'conv' in u)}")
    cnt = pd.Series(date).value_counts()
    print(f"  Azure 달력 날짜 {len(cnt)}일, 그중 창 2개(code·conv)인 날 {int((cnt == 2).sum())}일")
    print(f"  BurstGPT {len(bg)}창: 서로 다른 (파일, 날) 군집, 파일별 {pd.Series([win.loc[u, 'source'] for u in bg]).value_counts().to_dict()}")

    print("\n■ 2. H1·H2, Azure 같은 달력 날짜를 한 단위로 합침(d 평균), T* = 1초")
    for name, col in (("H1 주 AI vs 고정-온라인", None), ("H2 주 AI vs 규칙-온라인+대체", RULE)):
        d = t1[AI] - (0.0 if col is None else t1[col])
        m = pd.Series(d[az].values, index=[date[u] for u in az]).groupby(level=0).mean()
        print(f"  {name}: 등록 분석 {sign(d[R])}")
        print(f"  {' ' * len(name)}  합친 분석 {sign(pd.concat([m, d[bg]]))}")

    print("\n■ 3. H4 시드 블록 분석, T* = 1초, d(주 AI, 규칙)의 요인 효과(높음 − 낮음, 짝 차이)")
    fx = pd.read_csv(f"{RES}/res_fx.csv")
    tf = fx[(fx.proto == "within") & (fx.Tstar == 1.0)].pivot(index="trace", columns="policy", values="rel_fixon")
    d = (tf[AI] - tf[RULE]).rename("d").to_frame()
    d["cfg"] = [n.split("_")[1] for n in d.index]
    d["seed"] = [n.split("_")[2] for n in d.index]
    print(f"  시드별 d 평균: {d.groupby('seed').d.mean().round(2).to_dict()}")
    out = []
    for pos, (fac, hi, lo) in enumerate((("크기 변동", "S", "s"), ("짧은 몰림", "b", "p"), ("수준 변화", "w", "n"))):
        diffs, per_seed = [], []
        for seed, ds in d.groupby("seed"):
            ds = ds.set_index("cfg").d
            v = [ds[c] - ds[c[:pos] + lo + c[pos + 1:]] for c in ds.index if c[pos] == hi]
            diffs += v
            per_seed.append(float(np.mean(v)))
        diffs = np.array(diffs)
        out.append((fac, diffs, per_seed, stats.wilcoxon(diffs).pvalue, stats.ttest_1samp(per_seed, 0.0).pvalue))
    hw, ht = holm([o[3] for o in out]), holm([o[4] for o in out])
    for (fac, diffs, per_seed, pw, pt), qw, qt in zip(out, hw, ht):
        print(f"  {fac}: 짝 {len(diffs)}개 차이 중앙 {np.median(diffs):+.2f}%p(양수 {int((diffs > 0).sum())}개), "
              f"Wilcoxon p={pw:.4f}(Holm {qw:.4f}) | 시드별 평균 {np.round(per_seed, 2).tolist()}, t p={pt:.4f}(Holm {qt:.4f})")

    print("\n■ 4. 결과 파일 점검")
    for f in ("res_main", "res_fx", "res_sens"):
        r = pd.read_csv(f"{RES}/{f}.csv")
        print(f"  {f}.csv: {len(r)}행, (proto, trace, T*, policy) 중복 {int(r.duplicated(['proto', 'trace', 'Tstar', 'policy']).sum())}행")
    log = open(f"{RES}/parts/driver.log").read()
    print(f"  driver.log: 묶음 끝 {log.count('] 끝 ')}개, 실패 {log.count('실패')}개")


if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".."))
    main()
