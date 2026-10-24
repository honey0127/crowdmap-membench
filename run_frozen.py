#!/usr/bin/env python3
"""run_frozen.py — 고정한 설정(frozen.json)으로 일정 파일을 하나씩 처리한다. 봉인 시험과 합성 확인 실험용.

  python3 run_frozen.py --config exp/frozen.json --out exp/sealed/res/res_main.csv exp/sealed/sched/*.csv
  python3 run_frozen.py --config exp/frozen.json --out res.csv --pretrain exp/dev_sched.txt sched...   # 사전 학습 판도 잰다

- within: 창마다 학습 50 / 검증 25 / 시험 25. paperexp2.fit_models·evaluate를 그대로 쓴다.
- pre(탐색): 모델만 --pretrain 목록(개발 일정)의 학습 구간으로 한 번 학습해 쓰고, 나머지는 within과 같다.
- 결과 열은 paperexp2.py의 res2.csv와 같다. 불규칙성 지표는 <out>의 _irregularity.csv에 쓴다.
"""
import argparse
import csv
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paperexp as pe  # noqa: E402
import paperexp2 as p2  # noqa: E402

MiB = pe.MiB


def apply_config(cfg):
    p2.CAP_MULT, p2.CAP_MAX = cfg["cap_mult"], cfg["cap_max"]
    p2.MIN_FRAC, p2.MIN_N = cfg["min_frac"], cfg["min_n"]
    p2.KGRID = np.array(cfg["kgrid"])
    p2.FB = dict(W=cfg["fb_W"], eps=cfg["fb_eps"], H=cfg["fb_H"])


class Pretrained:
    """개발 일정으로 한 번 학습한 모델(생존 매듭별 분류기 + T*별 분류기)"""

    def __init__(self, paths, feats, tstars, seed):
        from sklearn.ensemble import HistGradientBoostingClassifier
        Xs, gs, ws = [], [], []
        for p in paths:
            t = p2.Trace2(p, feats)
            m = t.TR
            if m.sum() == 0:
                continue
            Xs.append(t.X[m])
            gs.append(t.gtr[m])
            ws.append(t.h[m] / t.h[m].sum() * 1000)
        X, g, w = np.vstack(Xs), np.concatenate(gs), np.concatenate(ws)
        self.surv = []
        for tau in pe.TAU:
            y = (g > tau).astype(int)
            if len(np.unique(y)) < 2:
                self.surv.append(float(y.mean()))
                continue
            self.surv.append(HistGradientBoostingClassifier(max_iter=120, learning_rate=0.1, max_leaf_nodes=31,
                                                            l2_regularization=1.0, random_state=seed).fit(X, y, sample_weight=w))
        self.clf = {}
        for b in tstars:
            y = (g > b).astype(int)
            self.clf[b] = pe.make_clf(seed).fit(X, y, sample_weight=w) if len(np.unique(y)) == 2 else None
        self.n_train = len(g)

    def predict(self, X, tstars):
        Sk = np.column_stack([m.predict_proba(X)[:, 1] if not isinstance(m, float) else np.full(len(X), m)
                              for m in self.surv])
        out = {"surv": np.minimum.accumulate(Sk, axis=1)}
        for b in tstars:
            out[b] = self.clf[b].predict_proba(X)[:, 1] if self.clf[b] is not None else None
        return out


def rows_for(proto, tr, b, res, info, cfg):
    b_on, b_st = res["고정-온라인"]["cost"], res["고정(정적)"]["cost"]
    te_dur = tr.t0 + tr.dur - tr.c2
    out = []
    for name, v in res.items():
        out.append(dict(proto=proto, trace=tr.name, Tstar=b, policy=name, cost_MiB=v["cost"] / MiB,
                        rel_fixon=(v["cost"] / b_on - 1) * 100 if b_on > 0 else 0.0,
                        rel_fixst=(v["cost"] / b_st - 1) * 100 if b_st > 0 else 0.0,
                        refault_MiB=v["refault"] / MiB, held_mean_MiB=v["waste"] / MiB / te_dur,
                        auc=info["auc"], n_te=int(tr.TE.sum()), feats=cfg["feats"], wcal=cfg["wcal"], U=cfg["period"],
                        rule=info["규칙 조합"], kappa_med=info["κ 중앙값"],
                        base_frac=info.get(name.replace("+대체", "") + " 기준 사용", "") if name.endswith("+대체") else ""))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("schedules", nargs="+")
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pretrain", default="", help="사전 학습에 쓸 개발 일정 경로 목록 파일(한 줄에 하나)")
    ap.add_argument("--seed", type=int, default=None, help="설정 파일의 seed 대신 쓸 시드")
    ap.add_argument("--no-ablate", action="store_true", help="설정의 ablate를 끈다(솎기 민감도처럼 주 비교만 볼 때)")
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    apply_config(cfg)
    seed = cfg["seed"] if a.seed is None else a.seed
    tstars = cfg["tstars"]
    t0 = time.time()
    pre = None
    if a.pretrain:
        paths = [l.strip() for l in open(a.pretrain) if l.strip()]
        pre = Pretrained(paths, cfg["feats"], tstars, seed)
        print(f"사전 학습: 일정 {len(paths)}개, 조각 {pre.n_train}개 ({time.time() - t0:.0f}s)", flush=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    irr_path = a.out.replace(".csv", "_irregularity.csv")
    first = not os.path.exists(a.out)
    first_irr = not os.path.exists(irr_path)
    for path in a.schedules:
        tr = p2.Trace2(path, cfg["feats"])
        end = tr.t0 + tr.dur
        m_all = p2.irregularity(tr.ta, tr.sz, tr.c1, end)
        m_va = p2.irregularity(tr.ta, tr.sz, tr.c1, tr.c2)
        m_te = p2.irregularity(tr.ta, tr.sz, tr.c2, end)
        drift = math.log2(m_te["rate"] / m_va["rate"]) if m_va["rate"] > 0 and m_te["rate"] > 0 else math.nan
        irr = dict(trace=tr.name, **m_all, drift=drift, size_cv_te=m_te["size_cv"], fano10_te=m_te["fano10"])
        with open(irr_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(irr))
            if first_irr:
                w.writeheader()
                first_irr = False
            w.writerow(irr)
        led = p2.Ledger(tr)
        ablate = bool(cfg.get("ablate", False)) and not a.no_ablate
        jobs = [("within", p2.fit_models([tr], tr.X, tstars, seed, f10=ablate))]
        if pre is not None:
            jobs.append(("pre", pre.predict(tr.X, tstars)))
        rows = []
        for proto, M in jobs:
            for b in tstars:
                single = len(np.unique((tr.gtr[tr.TR] > b).astype(int))) < 2
                if M[b] is None or (proto == "within" and single):
                    print(f"[{proto} {tr.name}] T*={b:g}: 학습 정답이 한 종류뿐이라 건너뜀", flush=True)
                    continue
                mod = {"clf_p": M[b], "surv": M["surv"]}
                if proto == "within" and ablate:
                    mod["surv_f10"] = M["surv_f10"]
                res, info = p2.evaluate(tr, led, b, mod, cfg["wcal"], cfg["period"], seed,
                                        ablate=(proto == "within" and ablate))
                rows += rows_for(proto, tr, b, res, info, cfg)
                b_on = res["고정-온라인"]["cost"]
                print(f"[{proto} {tr.name}] T*={b:g}: " + ", ".join(
                    f"{n} {(res[n]['cost'] / b_on - 1) * 100:+.1f}%" for n in
                    ("규칙-온라인+대체", cfg["primary"], "AI분류-온라인+대체", "AI생존그룹-온라인+대체", "AI생존κ-온라인+대체")
                    if n in res), flush=True)
        if rows:
            with open(a.out, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                if first:
                    w.writeheader()
                    first = False
                w.writerows(rows)
        del tr, led, jobs
    print(f"끝 ({time.time() - t0:.0f}s) → {a.out}")


if __name__ == "__main__":
    main()
