#!/usr/bin/env python3
"""export_decisions.py — 3단계 재생용 정책 결정 내보내기(유체 모델 → 청크 풀 사건 목록)

  python3 export_decisions.py exp/sched/az_code.csv --chunk 1M --tstar 1 --outdir exp/stage3/dec

- 요청 크기를 청크 배수로 올림한다(청크 풀은 본문마다 청크를 통째로 준다). 그러면 사용량 D(t)와 유체 모델의 조각이
  모두 청크 경계에 맞아, 조각 하나 = 청크 몇 개의 놀이 구간이 된다.
- 올림한 일정으로 v1과 같은 방법(paperexp2: 생존 모델·규칙 선택·온라인 보정·대체 장치)을 돌려 정책별 조각 반납 시점 T를 만든다.
  정책: fixed = 고정-온라인, rule = 규칙-온라인+대체, ai = AI생존κ-온라인+대체(v1 주 AI), never = 반납 안 함, now = 즉시 반납.
- 사건: C(처음 쓰기, 정책과 무관), R(ts + T에 반납, T < g일 때), F(ts + g에 다시 쓰기, 반납했고 다시 쓰인 조각).
  평가 구간(시험 구간, 시각 ≥ c2)에서 놀기 시작한 조각은 tag 1, 나머지는 tag 0(준비 구간: 검증 구간까지는 정책 T, 학습 구간은
  검증으로 고른 고정 타임아웃)이다.
- 출력: <outdir>/<이름>_c<청크>_t<T*>_<정책>.csv(사건 목록, chunkpool replay 입력), <outdir>/<이름>_c<청크>_t<T*>_fluid.json
  (tag 1 조각의 유체 모델 예측: 재확보 MiB, 낭비 MiB·초, 반납 MiB·횟수)
"""
import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paperexp as pe  # noqa: E402
import paperexp2 as p2  # noqa: E402
import policysim as ps  # noqa: E402

MiB = pe.MiB


def parse_size(s):
    s = s.strip().upper()
    mult = {"K": 1024, "M": MiB, "G": 1024 * MiB}
    return int(float(s[:-1]) * mult[s[-1]]) if s[-1] in mult else int(s)


def cold_events(ta, tf, sz):
    """사용량이 지금까지의 최고치를 넘을 때 새로 쓰는 범위(policysim.slices와 같은 사건 순서)"""
    t = np.concatenate([ta, tf])
    d = np.concatenate([sz, -sz])
    order = np.lexsort((d, t))
    D, M, out = 0.0, 0.0, []
    for ti, di in zip(t[order].tolist(), d[order].tolist()):
        D += di
        if D > M:
            out.append((ti, M, D))
            M = D
    return out, M


def policy_T(tr, led, b, Sk, cfg):
    VA, VA1, TE, h = tr.VA, tr.VA1, tr.TE, tr.h
    dec = VA | TE
    Cva = pe.cost_matrix(tr.gva[VA], h[VA], tr.rva[VA], b)
    pos_va = np.where(VA)[0]
    is1 = VA1[pos_va]
    i_fix = int(np.argmin(Cva.sum(1)))
    i_fix1 = int(np.argmin(Cva[:, is1].sum(1)))
    i_fix2 = int(np.argmin(Cva[:, ~is1].sum(1)))
    W, U, cap = cfg["wcal"], cfg["period"], p2.tcap(b)
    base = p2.online_group_T(tr, led, b, np.zeros(len(tr.g), dtype=np.int64), 1, dec, W, U)
    grp_r, G_r, rname, _ = p2.rule_groups(tr, b, Cva, pos_va, is1, i_fix, i_fix1, i_fix2)
    T_rule = p2.online_group_T(tr, led, b, grp_r, G_r, dec, W, U)
    rec = pe.survival_timeouts(Sk, b, 1.0)
    Tk = np.stack([np.minimum(rec * k, cap) for k in p2.KGRID])
    T_ai, _ = p2.online_kappa_T(tr, led, b, Tk, dec, W, U)
    out = {"fixed": base,
           "rule": p2.run_fallback2(tr, T_rule, base, b, dec, **p2.FB)[0],
           "ai": p2.run_fallback2(tr, T_ai, base, b, dec, **p2.FB)[0],
           "never": np.full(len(tr.g), np.inf),
           "now": np.zeros(len(tr.g))}
    warm = pe.GRID[i_fix]  # 결정 구간 밖(학습 구간) 조각은 검증으로 고른 고정 타임아웃
    for k in ("fixed", "rule", "ai"):
        out[k] = np.where(np.isnan(out[k]), warm, out[k])
    return out, rname


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("schedule")
    ap.add_argument("--chunk", default="1M")
    ap.add_argument("--tstar", type=float, default=1.0)
    ap.add_argument("--outdir", default="exp/stage3/dec")
    ap.add_argument("--config", default="exp/frozen.json")
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    C = parse_size(a.chunk)
    os.makedirs(a.outdir, exist_ok=True)
    name = os.path.basename(a.schedule).removesuffix(".csv")
    tag = f"{name}_c{a.chunk}_t{a.tstar:g}"
    ta, tf, sz = ps.load_schedule(a.schedule)
    szq = np.ceil(sz / C) * C
    qpath = os.path.join(a.outdir, f"{name}_c{a.chunk}.sched.csv")
    with open(qpath, "w") as f:
        f.write("t_alloc_s,t_free_s,size_bytes\n")
        for x, y, s in zip(ta, tf, szq):
            f.write(f"{x:.6f},{y:.6f},{int(s)}\n")
    tr = p2.Trace2(qpath, cfg["feats"])
    if len(np.unique((tr.gtr[tr.TR] > a.tstar).astype(int))) < 2:
        sys.exit(f"{name}: 학습 정답이 한 종류뿐이다(T*={a.tstar:g})")
    led = p2.Ledger(tr)
    m = tr.TR
    Sk = pe.fit_survival(tr.X[m], tr.gtr[m], tr.h[m] / tr.h[m].mean(), tr.X, cfg["seed"])
    pols, rname = policy_T(tr, led, a.tstar, Sk, cfg)
    S = ps.slices(*ps.load_schedule(qpath))  # 조각의 높이 범위(Trace와 같은 순서)
    assert np.array_equal(S["g"], tr.g) and np.array_equal(S["h"], tr.h)
    lo = S["D0"] + S["pos"] - S["h"] / 2
    lo_c = np.rint(lo / C).astype(np.int64)
    n_c = np.rint(S["h"] / C).astype(np.int64)
    assert np.allclose(lo_c * C, lo, atol=1.0) and np.allclose(n_c * C, S["h"], atol=1.0), "조각이 청크 경계에 맞지 않는다"
    colds, Mmax = cold_events(*ps.load_schedule(qpath))
    n_chunks = int(round(Mmax / C))
    te = (tr.ts >= tr.c2).astype(int)
    fluid = {"schedule": a.schedule, "chunk_bytes": C, "tstar": a.tstar, "rule": rname, "n_chunks": n_chunks,
             "te_duration_s": float(tr.t0 + tr.dur - tr.c2), "policies": {}}
    for pol, T in pols.items():
        g, h, r = tr.g, tr.h, tr.r
        rel = T < g
        refm = rel & r
        ev = [("C", t, int(round(x / C)), int(round((y - x) / C)), 0) for t, x, y in colds]
        ev += [("R", float(tr.ts[i] + T[i]), int(lo_c[i]), int(n_c[i]), int(te[i])) for i in np.where(rel & (n_c > 0))[0]]
        ev += [("F", float(tr.ts[i] + g[i]), int(lo_c[i]), int(n_c[i]), int(te[i])) for i in np.where(refm & (n_c > 0))[0]]
        prio = {"C": 0, "F": 1, "R": 2}
        ev.sort(key=lambda e: (e[1], prio[e[0]]))
        path = os.path.join(a.outdir, f"{tag}_{pol}.csv")
        with open(path, "w") as f:
            f.write(f"# chunk_bytes={C} n_chunks={n_chunks} policy={pol} tstar={a.tstar:g} schedule={name}\n")
            f.write("type,t,lo,n,tag\n")
            for e in ev:
                f.write(f"{e[0]},{e[1]:.6f},{e[2]},{e[3]},{e[4]}\n")
        k = te.astype(bool)
        fluid["policies"][pol] = dict(
            refault_MiB=float((h * refm)[k].sum() / MiB), waste_MiB_s=float((h * np.minimum(g, T))[k].sum() / MiB),
            released_MiB=float((h * rel)[k].sum() / MiB), n_release=int((rel & k).sum()),
            cost_fluid_MiB=float(((h * refm) + h * np.minimum(g, T) / a.tstar)[k].sum() / MiB), events=len(ev))
    with open(os.path.join(a.outdir, f"{tag}_fluid.json"), "w") as f:
        json.dump(fluid, f, ensure_ascii=False, indent=1)
    print(json.dumps(fluid["policies"], ensure_ascii=False))


if __name__ == "__main__":
    main()
