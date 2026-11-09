#!/usr/bin/env python3
"""overhang_summary.py — 과잉 보유 O(t) = B − D 요약, 규칙 대비 절감률, 날짜 클러스터 부트스트랩(사후)"""
import csv
import numpy as np
rs = list(csv.DictReader(open("exp/v2dev/peak/overhang_windows.csv")))
clus = {r["name"]: r["cluster"] for r in csv.DictReader(open("exp/v2dev/daypool/windows.csv"))}
P = ["반납안함", "고정-온라인", "규칙-온라인+대체", "C3-P1b"]
v = lambda p, k: np.array([float(r[f"{p}|{k}"]) for r in rs])
tau = np.array([float(r["tau_s"]) for r in rs]); E = tau.sum()
tw = lambda x: float((x * tau).sum() / E)  # 시험 시간 가중 창 평균
print(f"창 {len(rs)}, E = {E:.0f} s")
print("정책 | O 시간평균(MiB) | O 시간σ(창 가중평균) | O 창별 최대(평균) | O 창별 최대(최대)")
for p in P:
    print(p, f"{tw(v(p,'O_mean_MiB')):.3f} {tw(v(p,'O_sd_MiB')):.3f} {v(p,'idle_peak_MiB').mean():.2f} {v(p,'idle_peak_MiB').max():.0f}")
days = sorted(set(clus[r["trace"]] for r in rs)); di = {d: [i for i, r in enumerate(rs) if clus[r["trace"]] == d] for d in days}
rng = np.random.default_rng(20261109)
def rate(a, b, idx):  # 규칙 대비 절감률
    A, B = tw2(a, idx), tw2(b, idx); return (A - B) / A
def tw2(x, idx): return float((x[idx] * tau[idx]).sum() / tau[idx].sum())
for k, lab in (("O_sd_MiB", "σ_O"), ("O_mean_MiB", "μ_O"), ("idle_peak_MiB", "O_peak")):
    a, b = v("규칙-온라인+대체", k), v("C3-P1b", k)
    if k == "idle_peak_MiB":
        f = lambda idx: (a[idx].mean() - b[idx].mean()) / a[idx].mean()
    else:
        f = lambda idx: rate(a, b, idx)
    allidx = np.arange(len(rs)); r0 = f(allidx)
    bs = [f(np.concatenate([di[days[j]] for j in rng.integers(0, len(days), len(days))])) for _ in range(2000)]
    print(f"{lab} 절감률(규칙→C3) {r0*100:.2f}%  95% [{np.quantile(bs,.025)*100:.2f}, {np.quantile(bs,.975)*100:.2f}]  창별 C3<규칙 {int((b<a).sum())}, = {int((b==a).sum())}, > {int((b>a).sum())}")
a, b = v("규칙-온라인+대체", "O_sd_MiB"), v("C3-P1b", "O_sd_MiB")
dsig = tw(a) - tw(b)
c, Ts = 281.8e-6, 2.6
for z in (1.0, 2.33):
    print(f"V_σ(z={z}) = c·z·Δσ_O·E/T* = {c*z*dsig*E/Ts*1e3:.1f} ms  (Δσ_O = {dsig:.4f} MiB)")
