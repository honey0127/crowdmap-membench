#!/usr/bin/env python3
"""causality_check.py — 미래 교란 시험: 결정이 결정 시각 뒤의 정보를 쓰지 않는지 확인한다(개발 데이터 전용).

방법: 자르는 시각 t_p 뒤에 도착하는 요청의 크기만 무작위로 섞은 일정을 만든다(도착·끝 시각은 그대로라 학습·검증·시험 분할은 같다).
원래 일정과 섞은 일정에서 같은 정책을 돌리고, t_p 전에 놀기 시작한 결정 대상 조각(검증·시험 구간)을 비교한다.
  - 같은 조각을 찾지 못한 무게: t_p 전에 놀기 시작했는데 조각의 모양(두께·위치 특징)이 t_p 뒤 사건으로 바뀐 조각
  - T가 바뀐 무게: 같은 조각인데 정책이 준 반납 시점이 달라진 조각
두 값이 모두 0이면 t_p 전의 결정은 t_p 뒤 정보와 무관하다. t_p = 시험 구간 시작(q = 0)과 중간(q = 0.5)에서 본다.
  python3 causality_check.py --model chunk --tstar 2.6 exp/sched/*.csv > exp/v2dev/chunk/causality_chunk.txt
"""
import argparse
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402

import paperexp as pe  # noqa: E402
import paperexp2 as p2  # noqa: E402
import paperexp3 as p3  # noqa: E402

POLS = ["고정-온라인", "규칙-온라인", "AI생존κ-온라인", "규칙-온라인+대체", "AI생존κ-온라인+대체", "C3 AI→규칙대체"]


def decisions(tr, b, cfg):
    """paperexp3.evaluate3와 같은 순서로 정책별 조각 T를 만든다(시험 비용 계산 전 단계)"""
    led = p2.Ledger(tr)
    m = tr.TR
    Sk = p3.fit_survival3(tr.X[m], tr.gtr[m], tr.h[m] / tr.h[m].mean(), tr.X, cfg["seed"])  # paperexp3.main과 같은 생존 모델
    W_cal, U = cfg["wcal"], cfg["period"]
    VA, VA1, TE, h = tr.VA, tr.VA1, tr.TE, tr.h
    dec = VA | TE
    Cva = pe.cost_matrix(tr.gva[VA], h[VA], tr.rva[VA], b)
    pos_va = np.where(VA)[0]
    is1 = VA1[pos_va]
    i_fix = int(np.argmin(Cva.sum(1)))
    i_fix1 = int(np.argmin(Cva[:, is1].sum(1)))
    i_fix2 = int(np.argmin(Cva[:, ~is1].sum(1)))
    cap = p2.tcap(b)
    P = {}
    base = p2.online_group_T(tr, led, b, np.zeros(len(tr.g), dtype=np.int64), 1, dec, W_cal, U)
    P["고정-온라인"] = base
    grp_r, G_r, rname, _ = p2.rule_groups(tr, b, Cva, pos_va, is1, i_fix, i_fix1, i_fix2)
    P["규칙-온라인"] = p2.online_group_T(tr, led, b, grp_r, G_r, dec, W_cal, U)
    rec = pe.survival_timeouts(Sk, b, 1.0)
    Tk = np.stack([np.minimum(rec * k, cap) for k in p2.KGRID])
    P["AI생존κ-온라인"], _ = p2.online_kappa_T(tr, led, b, Tk, dec, W_cal, U)
    for name in ("규칙-온라인", "AI생존κ-온라인"):
        P[name + "+대체"], _, _ = p3.FALLBACK(tr, P[name], base, b, dec, **p2.FB)
    P["C3 AI→규칙대체"], _, _ = p3.FALLBACK(tr, P["AI생존κ-온라인+대체"], P["규칙-온라인+대체"], b, dec, **p2.FB)
    return P, rname


def keyed(tr, P, t_p):
    m = (tr.VA | tr.TE) & (tr.ts < t_p)
    idx = np.where(m)[0]
    out = {}
    for i in idx:
        k = (round(float(tr.ts[i]), 9), round(float(tr.D0[i]), 3), round(float(tr.pos[i]), 3), round(float(tr.h[i]), 3))
        out[k] = (float(tr.h[i]), tuple(float(P[p][i]) for p in POLS))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("schedules", nargs="+")
    ap.add_argument("--model", choices=["v1", "chunk"], required=True)
    ap.add_argument("--tstar", type=float, default=2.6)
    ap.add_argument("--cuts", default="0,0.5")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--config", default="exp/frozen.json")
    a = ap.parse_args()
    if a.model == "chunk":
        p3.use_chunk_model()
    cfg = json.load(open(a.config))
    # 조각의 D0·pos를 Trace에 남긴다(비교용 열쇠)
    init0, sl0, last = pe.Trace.__init__, pe.ps.slices, {}

    def sl(ta, tf, sz):
        last["S"] = sl0(ta, tf, sz)
        return last["S"]

    def init(self, path):
        init0(self, path)
        self.D0, self.pos = last["S"]["D0"], last["S"]["pos"]
    pe.ps.slices = sl
    pe.Trace.__init__ = init
    tmpd = tempfile.mkdtemp()
    tot = {"un": 0.0, "all": 0.0, **{p: 0.0 for p in POLS}}
    print(f"모델 {a.model}, T* {a.tstar:g}초, 자르는 위치 q = {a.cuts}(시험 구간 안 비율), 섞기 시드 {a.seed}")
    for path in a.schedules:
        name = os.path.basename(path).removesuffix(".csv")
        lines = [ln for ln in open(path) if not ln.startswith("#")]
        hdr, arr = lines[0], np.array([[float(x) for x in ln.strip().split(",")] for ln in lines[1:]])
        tr0 = p2.Trace2(path, cfg["feats"])
        P0, r0 = decisions(tr0, a.tstar, cfg)
        for q in [float(x) for x in a.cuts.split(",")]:
            t_p = tr0.c2 + q * (tr0.t0 + tr0.dur - tr0.c2)
            rng = np.random.default_rng(a.seed)
            fut = arr[:, 0] > t_p
            arr2 = arr.copy()
            arr2[fut, 2] = rng.permutation(arr[fut, 2])
            p_alt = os.path.join(tmpd, f"{name}_q{q:g}.csv")
            with open(p_alt, "w") as f:
                f.write(hdr)
                for r in arr2:
                    f.write(f"{float(r[0])!r},{float(r[1])!r},{float(r[2])!r}\n")
            tr1 = p2.Trace2(p_alt, cfg["feats"])
            assert abs(tr1.c2 - tr0.c2) < 1e-9 and abs(tr1.dur - tr0.dur) < 1e-9
            P1, r1 = decisions(tr1, a.tstar, cfg)
            K0, K1 = keyed(tr0, P0, t_p), keyed(tr1, P1, t_p)
            w_all = sum(v[0] for v in K0.values())
            common = K0.keys() & K1.keys()
            w_un = sum(K0[k][0] for k in K0.keys() - common)
            chg = {p: sum(K0[k][0] for k in common if K0[k][1][j] != K1[k][1][j]) for j, p in enumerate(POLS)}
            tot["un"] += w_un
            tot["all"] += w_all
            for p in POLS:
                tot[p] += chg[p]
            print(f"{name} q={q:g}: 조각 {len(K0)}개, 섞인 요청 {int(fut.sum())}개, 같은 조각 없음 {w_un / w_all * 100:.3f}%, "
                  f"규칙 조합 {r0}{'' if r0 == r1 else ' → ' + r1} | T 바뀜 " +
                  ", ".join(f"{p} {chg[p] / w_all * 100:.3f}%" for p in POLS), flush=True)
    print(f"\n합계(무게 기준): 같은 조각 없음 {tot['un'] / tot['all'] * 100:.3f}% | T 바뀜 " +
          ", ".join(f"{p} {tot[p] / tot['all'] * 100:.3f}%" for p in POLS))


if __name__ == "__main__":
    main()
