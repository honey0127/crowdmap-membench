#!/usr/bin/env python3
"""peak_sim.py — 사후 분석: 정책별 청크 풀 보유량 B(t)의 피크 (개발 420창, T* = 2.6초, C3-P1b).

B(t) = D(t) + 놀고 있지만 아직 반납하지 않은 청크.
  D(t): 청크 단위로 올린 사용 중 본문 바이트(일정의 도착·완료).
  놀이 청크: 조각 i는 ts_i부터 ts_i + min(g_i, T_i)까지 들고 있다(유체 모델, pe.parts와 같은 가정).
정책 T 배열은 g1_simulate.py와 같은 순서로 만든다(고정-온라인, 규칙-온라인+대체, C3-P1b). 반납 안 함은 T = ∞.
검산: 규칙·C3-P1b의 R·H가 exp/v2dev/g1/simulation/windows-P1b.csv와 같아야 한다.
구간: 시험 구간 [c2, t0+dur). 결정 대상(VA|TE) 조각만 놀이 보유에 넣는다(TR 조각은 T가 없다).
출력: 창마다 한 줄(정책별 피크·시간 평균·시간 가중 표준편차·놀이 피크).
사용: python3 exp/v2dev/peak/peak_sim.py --out OUT.csv SCHED...
"""
import argparse
import csv
import math
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, os.path.join(REPO, "exp/v2dev/g1"))
os.chdir(REPO)
import g1_common as G  # noqa: E402  (numpy보다 먼저 부른다)
import numpy as np  # noqa: E402

MiB = 1024.0 * 1024.0
POLS = ["반납안함", "고정-온라인", "규칙-온라인+대체", "C3-P1b"]


def series_stats(ta, tf, sz, ts, hold_end, h, lo_t, hi_t):
    """[lo_t, hi_t)에서 B(t)의 최대·시간 평균·시간 가중 표준편차, 놀이 보유 최대"""
    r = lambda x: np.round(np.asarray(x, float), 6)  # 같은 시각 사건을 한 번에 처리한다(부동소수 잡음 제거)
    tD = np.concatenate([r(ta), r(tf)])
    dD = np.concatenate([sz, -sz])
    tI = np.concatenate([r(ts), r(hold_end)])
    dI = np.concatenate([h, -h])
    t = np.concatenate([tD, tI])
    dB = np.concatenate([dD, dI])
    dIdle = np.concatenate([np.zeros_like(dD), dI])
    o = np.argsort(t, kind="stable")
    t, dB, dIdle = t[o], dB[o], dIdle[o]
    ut, first = np.unique(t, return_index=True)
    B = np.cumsum(dB)[np.r_[first[1:] - 1, len(t) - 1]]      # 각 고유 시각의 사건을 모두 반영한 값
    I = np.cumsum(dIdle)[np.r_[first[1:] - 1, len(t) - 1]]
    # [lo_t, hi_t)로 자른 계단 함수
    k0 = np.searchsorted(ut, lo_t, side="right") - 1
    Bv = np.r_[B[k0] if k0 >= 0 else 0.0, B[k0 + 1:][ut[k0 + 1:] < hi_t]]
    Iv = np.r_[I[k0] if k0 >= 0 else 0.0, I[k0 + 1:][ut[k0 + 1:] < hi_t]]
    tv = np.r_[lo_t, ut[k0 + 1:][ut[k0 + 1:] < hi_t], hi_t]
    w = np.diff(tv)
    mean = float((Bv * w).sum() / w.sum())
    sd = float(math.sqrt(max(((Bv - mean) ** 2 * w).sum() / w.sum(), 0.0)))
    return float(Bv.max()), mean, sd, float(Iv.max()), float((Iv * w).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("schedules", nargs="+")
    a = ap.parse_args()
    pe, p2, p3, _ = G.research_modules()
    cfg = G.read_json(G.REPO / "exp/frozen.json")
    grid = G.grid_from_json(G.read_json(G.REPO / "exp/v2dev/g1/grids/grid-P1b.json"))
    table = np.load(G.REPO / "exp/v2dev/g1/tables/lookup-P1b.npy", allow_pickle=False)
    ref = {r["trace"]: r for r in csv.DictReader(open(G.REPO / "exp/v2dev/g1/simulation/windows-P1b.csv"))}
    K = G.knots(pe)
    b = G.TSTAR
    rows = []
    for path in a.schedules:
        tr = p2.Trace2(path, G.FEATS)
        led = p2.Ledger(tr)
        cap = p2.tcap(b)
        VA, VA1, TE, h = tr.VA, tr.VA1, tr.TE, tr.h
        dec = VA | TE
        base = p2.online_group_T(tr, led, b, np.zeros(len(tr.g), dtype=np.int64), 1, dec, cfg["wcal"], cfg["period"])
        Cva = pe.cost_matrix(tr.gva[VA], h[VA], tr.rva[VA], b)
        pos_va = np.where(VA)[0]
        is1 = VA1[pos_va]
        i_fix = int(np.argmin(Cva.sum(1)))
        i_fix1 = int(np.argmin(Cva[:, is1].sum(1)))
        i_fix2 = int(np.argmin(Cva[:, ~is1].sum(1)))
        grp_r, G_r, _, _ = p2.rule_groups(tr, b, Cva, pos_va, is1, i_fix, i_fix1, i_fix2)
        T_rule = p2.online_group_T(tr, led, b, grp_r, G_r, dec, cfg["wcal"], cfg["period"])
        T_rb, _, _ = p3.FALLBACK(tr, T_rule, base, b, dec, **p2.FB)
        rec = K[table[tuple(G.encode_matrix(tr.X, grid).T)]]
        Tk = np.stack([np.minimum(rec * k, cap) for k in p2.KGRID])
        T_ai, _ = p2.online_kappa_T(tr, led, b, Tk, dec, cfg["wcal"], cfg["period"])
        T_ai_fb, _, _ = p3.FALLBACK(tr, T_ai, base, b, dec, **p2.FB)
        T_c3, _, _ = p3.FALLBACK(tr, T_ai_fb, T_rb, b, dec, **p2.FB)

        # 검산: g1 결과와 R·H가 같은가
        gt, ht, rt = tr.g[TE], h[TE], tr.r[TE]
        R0 = ref[tr.name]
        par = {}
        for tag, T, kR, kH in (("rule", T_rb, "rule_refault_MiB", "rule_waste_MiB"), ("c3", T_c3, "c3_refault_MiB", "c3_waste_MiB")):
            R, W = pe.parts(gt, ht, rt, T[TE], b)
            par[f"parity_{tag}_dR_MiB"] = R / MiB - float(R0[kR])
            par[f"parity_{tag}_dH_MiBs"] = W / MiB - float(R0[kH])

        lo_t, hi_t = float(tr.c2), float(tr.t0 + tr.dur)
        row = dict(trace=tr.name, tau_s=hi_t - lo_t, n_te=int(TE.sum()), **par)
        for name, T in zip(POLS, (np.full(len(tr.g), np.inf), base, T_rb, T_c3)):
            m = dec
            hold_end = tr.ts[m] + np.minimum(tr.g[m], T[m])
            pk, mu, sd, ipk, iarea = series_stats(tr.ta, tr.tf, tr.sz, tr.ts[m], hold_end, h[m], lo_t, hi_t)
            row[f"{name}|peak_MiB"] = pk / MiB
            row[f"{name}|mean_MiB"] = mu / MiB
            row[f"{name}|sd_MiB"] = sd / MiB
            row[f"{name}|idle_peak_MiB"] = ipk / MiB
            row[f"{name}|idle_area_MiBs"] = iarea / MiB
        rows.append(row)
        print(f"[{tr.name}] ok", flush=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
