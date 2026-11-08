#!/usr/bin/env python3
"""peak_summary.py — peak_windows.csv 요약: 정책별 피크 통계, 날짜 클러스터 부트스트랩, 예약 가치(사후)"""
import csv, sys
import numpy as np
rs = list(csv.DictReader(open("exp/v2dev/peak/peak_windows.csv")))
clus = {r["name"]: r["cluster"] for r in csv.DictReader(open("exp/v2dev/daypool/windows.csv"))}
P = ["반납안함", "고정-온라인", "규칙-온라인+대체", "C3-P1b"]
v = lambda p, k: np.array([float(r[f"{p}|{k}"]) for r in rs])
tau = np.array([float(r["tau_s"]) for r in rs]); E = tau.sum()
print(f"창 {len(rs)}개, 날 {len(set(clus[r['trace']] for r in rs))}개, E = Στ = {E:.0f} s")
print("정책 | 평균피크 | 최대 | Q95 | Q99 | 창별피크 σ | 시간평균 B | B(t) σ(창 평균) | 놀이피크 평균")
for p in P:
    pk = v(p, "peak_MiB")
    print(p, f"{pk.mean():.2f} {pk.max():.0f} {np.quantile(pk,.95):.2f} {np.quantile(pk,.99):.2f} {pk.std():.2f} "
          f"{(v(p,'mean_MiB')*tau).sum()/E:.2f} {(v(p,'sd_MiB')*tau).sum()/E:.2f} {v(p,'idle_peak_MiB').mean():.2f}")
d = v("규칙-온라인+대체", "peak_MiB") - v("C3-P1b", "peak_MiB")
print(f"창별 피크 차(규칙 − C3): 양 {int((d>0).sum())}, 0 {int((d==0).sum())}, 음 {int((d<0).sum())}; 합 {d.sum():.0f} MiB")
d2 = v("반납안함", "peak_MiB") - v("C3-P1b", "peak_MiB")
print(f"창별 피크 차(반납안함 − C3): 양 {int((d2>0).sum())}, 0 {int((d2==0).sum())}")
# 날짜 클러스터 부트스트랩
days = sorted(set(clus[r["trace"]] for r in rs)); di = {d: [i for i, r in enumerate(rs) if clus[r["trace"]] == d] for d in days}
rng = np.random.default_rng(20261108); B = 2000
a, c = v("규칙-온라인+대체", "peak_MiB"), v("C3-P1b", "peak_MiB")
out = {q: [] for q in ("max", "q95", "q99", "mean", "sd")}
for _ in range(B):
    idx = np.concatenate([di[days[j]] for j in rng.integers(0, len(days), len(days))])
    for q, f in (("max", np.max), ("q95", lambda x: np.quantile(x, .95)), ("q99", lambda x: np.quantile(x, .99)), ("mean", np.mean), ("sd", np.std)):
        out[q].append(f(a[idx]) - f(c[idx]))
for q in out:
    x = np.array(out[q]); print(f"Δ{q}(규칙 − C3) 95% 구간 [{np.quantile(x,.025):.3f}, {np.quantile(x,.975):.3f}]")
# 예약 가치: c/T* × ΔQ × E (계산)
c_s, Ts = 281.8e-6, 2.6
for q, f in (("max", np.max), ("q95", lambda x: np.quantile(x, .95)), ("q99", lambda x: np.quantile(x, .99))):
    dq = f(a) - f(c); print(f"V_reserve[{q}] = {c_s/Ts*dq*E:.3f} s (ΔQ {dq:.3f} MiB)")
print(f"참고: 1 MiB 예약 감소 = {c_s/Ts*E:.2f} s")
pw = (tau * d).sum(); print(f"창별 예약(민감도) Σ(Δpeak·τ) = {pw:.0f} MiB·s → {c_s/Ts*pw:.3f} s")
