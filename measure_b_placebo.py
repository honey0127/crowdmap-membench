#!/usr/bin/env python3
"""measure_b_placebo.py — 추가 분석 계획 v3 §5의 작업 B(위약 대조). 사후 분석이다(계획 §4 '둘 다 탈락' 분기, a-cpu-results.md).

대상과 데이터
  within C3(v2 봉인 정책). 개발 40일(v1 R의 BurstGPT 날)의 자격 창 420개(exp/v2dev/daypool/all.txt), T* = 2.6초,
  청크 모델(paperexp3 --model chunk), exp/frozen.json 설정(paperexp2 기본값과 같다). 봉인 데이터는 쓰지 않는다.

정책 계산(창마다, paperexp3.evaluate3의 해당 줄과 같은 함수·순서)
  base   = 고정-온라인 = p2.online_group_T(묶음 1개)
  rule   = 규칙-온라인+대체 = FALLBACK(p2.online_group_T(규칙 칸), base)
  AI(Sk) = AI생존κ-온라인+대체 = FALLBACK(p2.online_kappa_T(Tk(Sk)), base), Tk = min(survival_timeouts(Sk) × κ, T_cap)
  C3(Sk) = FALLBACK(AI(Sk), rule)          전환 규칙(run_fallback3, W 300초, ε 0, H 0)은 모든 변형에서 같다
  비용 = 시험 구간의 재확보 MiB + 낭비 MiB·초 ÷ T*.

변형(시드 1~5로 고정한다. 결과를 본 뒤 늘리지 않는다). '예측값'은 조각별 생존 곡선 Sk의 행(매듭 10개)이다.
  real       실제 생존 모델(paperexp3.fit_survival3, 학습 구간, GBDT random_state = 설정 seed)
  P_shuffle  같은 보정 구간 안의 결정 대상 조각끼리 Sk 행을 무작위로 섞는다. 같은 구간 뒤쪽 조각의 값도 쓰므로 비인과적이다.
  P_draw     결정 대상 조각마다, 가장 최근의 비어 있지 않은 이전 구간에서 학습 구간 밖 조각 하나를 무작위로(복원 추출) 골라
             그 Sk 행을 쓴다. 이전 구간에 후보가 없으면 학습 구간 조각들의 평균 Sk를 쓰고 그 조각 수를 기록한다. 미래 정보를 쓰지 않는다.
  P_label    학습 구간의 정답(관측 놀이 시간 gtr)을 무작위로 섞어 같은 함수·설정으로 다시 학습한 Sk
  보정 구간  online_kappa_T와 같은 격자다. t0 = 결정 대상 조각의 첫 시작 시각, u in np.arange(t0, t1 + U, U), U = 30초이고,
             시작 시각 ts가 [u, u + U)인 조각이 그 구간에 든다(led.window와 같다). t0 앞의 조각은 floor((ts − t0)/U)로 정한다.
  난수       np.random.default_rng([시드, 변형 번호, crc32(창 이름)]). 변형 번호는 P_label 1, P_draw 2, P_shuffle 3이다.
  AI 불가    실제 AI를 학습할 수 없는 창(bg2_h0629)은 모든 변형에서 AI를 쓰지 않는다(AI = base, C3 = rule, evaluate3와 같다).
             P_label 학습이 실패한 창도 같게 계산하고 기록한다.

보고(--summarize). 비용이 낮을수록 좋다. d = 100 × (총비용 차) ÷ 고정-온라인 총비용(%p)이고, 시드마다 계산해 5개의 범위를 낸다.
  주 비교   C3 − P_shuffle(조각별 대응 정보)
  진단      P_shuffle − P_draw(시점), P_draw − P_label(예측값 분포·흩어짐), P_label − rule(전환 구조 + 위약 모델)
  상호작용  (C3 − P_shuffle,C3) − (AI − P_shuffle,AI)
  분해 검산 C3 − rule = 위 네 항의 합(시드마다 같아야 한다)
  판정      시드 5개의 범위가 −0.5 또는 +0.5%p 경계에 걸치면 '판정 불가', 모두 −0.5 미만이면 '앞쪽 비용이 낮다',
            모두 +0.5 초과면 '앞쪽 비용이 높다', 모두 ±0.5 안이면 '차이 없음'.
  'AI 정보가 비용을 줄였다'는 쓰지 않는다(작업 A의 AI 경로 오버헤드 332.5%p를 넘을 수 없다). 결과는 시뮬레이션 총이득의 분해다.
검산: real의 base·rule·AI·C3 비용을 개발 결과(res_q1~q3)와 창마다 비교해 기록한다. 앞 창 3개가 모두 0.1% 넘게 다르면 멈춘다.
진단: 변형마다 C3가 규칙을 쓴 비율과 AI가 고정-온라인으로 대체한 비율(시험 구간), P_draw 대체 조각 수, 학습 CPU.

실행(끝난 창은 건너뛴다)
  python3 measure_b_placebo.py --targets exp/v2dev/daypool/list_q1.txt --out OUT/part_q1.csv
  python3 measure_b_placebo.py --summarize --targets exp/v2dev/daypool/all.txt --parts OUT/part_q1.csv,OUT/part_q2.csv --outdir OUT
"""
import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
import zlib

import numpy as np

REPO = os.environ.get("MEMBENCH_REPO", "/home/claude/honey0127/crowdmap-membench")
sys.path.insert(0, REPO)
import paperexp as pe  # noqa: E402
import paperexp2 as p2  # noqa: E402
import paperexp3 as p3  # noqa: E402

MiB = pe.MiB
TSTAR = 2.6
SEEDS = (1, 2, 3, 4, 5)
VAR_ID = {"P_label": 1, "P_draw": 2, "P_shuffle": 3}
VARIANTS = ("P_shuffle", "P_draw", "P_label")
RES = ("exp/v2dev/daypool/res_q1.csv", "exp/v2dev/daypool/res_q2.csv", "exp/v2dev/daypool/res_q3.csv")
POL = {"base": "고정-온라인", "rule": "규칙-온라인+대체", "ai": "AI생존κ-온라인+대체", "c3": "C3 AI→규칙대체"}
ROWS_PER_WINDOW = 1 + len(VARIANTS) * len(SEEDS)
FIELDS = ["trace", "variant", "seed", "ai_ok", "base_MiB", "rule_MiB", "ai_MiB", "c3_MiB", "ai_base_frac", "c3_rule_frac",
          "n_te", "n_dec", "draw_fallback", "fit_cpu_s", "check"]


def rel(a, b):
    return abs(a - b) / max(abs(b), 1e-12)


def classify(rd):
    return "exact" if rd <= 1e-9 else ("float" if rd <= 1e-3 else "MISMATCH")


def test_cost(tr, T, b):
    TE = tr.TE
    Tt = T[TE]
    assert not np.isnan(Tt).any()
    ref, w = pe.parts(tr.g[TE], tr.h[TE], tr.r[TE], Tt, b)
    return (ref + w / b) / MiB


def mode_frac(tr, mode, idx):
    te_m = tr.TE[idx]
    return float(mode[te_m].mean()) if te_m.any() else math.nan


def base_rule(tr, led, b, cfg):
    """paperexp3.evaluate3와 같은 고정-온라인, 규칙-온라인+대체"""
    VA, VA1, h = tr.VA, tr.VA1, tr.h
    dec = VA | tr.TE
    Cva = pe.cost_matrix(tr.gva[VA], h[VA], tr.rva[VA], b)
    pos_va = np.where(VA)[0]
    is1 = VA1[pos_va]
    i_fix = int(np.argmin(Cva.sum(1)))
    i_fix1 = int(np.argmin(Cva[:, is1].sum(1)))
    i_fix2 = int(np.argmin(Cva[:, ~is1].sum(1)))
    base = p2.online_group_T(tr, led, b, np.zeros(len(tr.g), dtype=np.int64), 1, dec, cfg["wcal"], cfg["period"])
    grp_r, G_r, _, _ = p2.rule_groups(tr, b, Cva, pos_va, is1, i_fix, i_fix1, i_fix2)
    T_rule = p2.online_group_T(tr, led, b, grp_r, G_r, dec, cfg["wcal"], cfg["period"])
    T_rb, _, _ = p3.FALLBACK(tr, T_rule, base, b, dec, **p2.FB)
    return base, T_rb


def ai_c3(tr, led, b, Sk, base, T_rb, cfg):
    """paperexp3.evaluate3와 같은 AI생존κ-온라인+대체와 C3. Sk가 None이면 AI를 쓰지 않는다."""
    dec = tr.VA | tr.TE
    if Sk is None:  # evaluate3와 같다: AI = base(대체 비율 1), C3 = rule(규칙 비율 1)
        return base, T_rb, 1.0, 1.0
    cap = p2.tcap(b)
    rec = pe.survival_timeouts(Sk, b, 1.0)
    Tk = np.stack([np.minimum(rec * k, cap) for k in p2.KGRID])
    T_ai, _ = p2.online_kappa_T(tr, led, b, Tk, dec, cfg["wcal"], cfg["period"])
    T_afb, mode_a, idx_a = p3.FALLBACK(tr, T_ai, base, b, dec, **p2.FB)
    T_c3, mode_c, idx_c = p3.FALLBACK(tr, T_afb, T_rb, b, dec, **p2.FB)
    return T_afb, T_c3, mode_frac(tr, mode_a, idx_a), mode_frac(tr, mode_c, idx_c)


def bucket_ids(tr, led, U):
    """보정 구간 번호(online_kappa_T의 격자). t0 이후 조각은 led.window(u, u + U)로, t0 앞 조각은 floor로 정한다."""
    dec = tr.VA | tr.TE
    t0, t1 = tr.ts[dec].min(), tr.ts[dec].max()
    bid = np.full(len(tr.ts), np.iinfo(np.int64).min, dtype=np.int64)
    for k, u in enumerate(np.arange(t0, t1 + U, U)):
        bid[led.window(u, u + U)] = k
    before = tr.ts < t0
    bid[before] = np.floor((tr.ts[before] - t0) / U).astype(np.int64)
    return bid


def placebo_shuffle(Sk, tr, bid, rng):
    dec = tr.VA | tr.TE
    out = Sk.copy()
    for k in np.unique(bid[dec]):
        idx = np.where(dec & (bid == k))[0]
        out[idx] = Sk[idx[rng.permutation(len(idx))]]
    return out


def placebo_draw(Sk, tr, bid, rng):
    dec = tr.VA | tr.TE
    out = Sk.copy()
    donor_ok = ~tr.TR  # 학습 구간 밖(표본 밖 예측)
    prior = Sk[tr.TR].mean(axis=0) if tr.TR.any() else Sk.mean(axis=0)
    ks = np.unique(bid[(dec | donor_ok) & (bid > np.iinfo(np.int64).min)])
    last_donors, fallback = None, 0
    for k in ks:  # 오름차순
        idx = np.where(dec & (bid == k))[0]
        if len(idx):
            if last_donors is None:
                out[idx] = prior
                fallback += len(idx)
            else:
                out[idx] = Sk[last_donors[rng.integers(0, len(last_donors), size=len(idx))]]
        donors = np.where(donor_ok & (bid == k))[0]
        if len(donors):
            last_donors = donors
    return out, fallback


def window_rows(path, cfg, ref):
    name = os.path.basename(path).removesuffix(".csv")
    b = TSTAR
    tr = p2.Trace2(path, cfg["feats"])
    if int(tr.TE.sum()) != ref["n_te"]:
        raise SystemExit(f"[{name}] 시험 결정 수가 개발 결과와 다르다({int(tr.TE.sum())} vs {ref['n_te']})")
    led = p2.Ledger(tr)
    base, T_rb = base_rule(tr, led, b, cfg)
    c_base, c_rule = test_cost(tr, base, b), test_cost(tr, T_rb, b)
    m = tr.TR
    Xtr, gtr, wtr = tr.X[m], tr.gtr[m], tr.h[m] / tr.h[m].mean()  # paperexp3.main과 같은 입력
    t0 = time.process_time()
    try:
        Sk = p3.fit_survival3(Xtr, gtr, wtr, tr.X, cfg["seed"])
    except ValueError:
        Sk = None
    fit_real = time.process_time() - t0
    n_te, n_dec = int(tr.TE.sum()), int((tr.VA | tr.TE).sum())
    rows = []

    def add(variant, seed, Skv, fit_s, fallback=0, check=""):
        T_afb, T_c3, fa, fc = ai_c3(tr, led, b, Skv, base, T_rb, cfg)
        rows.append(dict(trace=name, variant=variant, seed=seed, ai_ok=int(Skv is not None), base_MiB=c_base, rule_MiB=c_rule,
                         ai_MiB=test_cost(tr, T_afb, b), c3_MiB=test_cost(tr, T_c3, b), ai_base_frac=fa, c3_rule_frac=fc,
                         n_te=n_te, n_dec=n_dec, draw_fallback=fallback, fit_cpu_s=fit_s, check=check))

    add("real", 0, Sk, fit_real)
    r0 = rows[0]
    st = [classify(rel(r0[k + "_MiB"], ref["cost"][POL[k]])) for k in ("base", "rule", "ai", "c3")]
    r0["check"] = max(st, key=["exact", "float", "MISMATCH"].index)
    if (Sk is None) != (ref["ai"] == "none"):
        raise SystemExit(f"[{name}] AI 학습 가능 여부가 개발 결과와 다르다")
    bid = bucket_ids(tr, led, cfg["period"]) if Sk is not None else None
    crc = zlib.crc32(name.encode())
    for seed in SEEDS:
        for v in VARIANTS:
            rng = np.random.default_rng([seed, VAR_ID[v], crc])
            if Sk is None:  # 실제 AI를 쓸 수 없는 창: 모든 변형에서 AI를 쓰지 않는다
                add(v, seed, None, 0.0)
            elif v == "P_shuffle":
                add(v, seed, placebo_shuffle(Sk, tr, bid, rng), 0.0)
            elif v == "P_draw":
                Skd, fb = placebo_draw(Sk, tr, bid, rng)
                add(v, seed, Skd, 0.0, fallback=fb)
            else:
                perm = rng.permutation(len(gtr))
                t0 = time.process_time()
                try:
                    Skl = p3.fit_survival3(Xtr, gtr[perm], wtr, tr.X, cfg["seed"])
                except ValueError:
                    Skl = None
                add(v, seed, Skl, time.process_time() - t0, check="" if Skl is not None else "label_fit_failed")
    return rows


# ---------------------------------------------------------------- 실행
def load_res():
    res = {}
    for f in RES:
        with open(os.path.join(REPO, f), encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if r.get("model") != "chunk" or abs(float(r["Tstar"]) - TSTAR) > 1e-9:
                    continue
                d = res.setdefault(r["trace"], dict(cost={}, n_te=int(r["n_te"]), ai="ok"))
                d["cost"][r["policy"]] = float(r["cost_MiB"])
                if r.get("ai"):
                    d["ai"] = r["ai"]
    return res


def read_rows(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def targets_of(path):
    p = path if os.path.isabs(path) else os.path.join(REPO, path)
    out = [x.strip() for x in open(p, encoding="utf-8") if x.strip()]
    return [x if os.path.isabs(x) else os.path.join(REPO, x) for x in out]


def run(a):
    p3.use_chunk_model()
    cfg = json.load(open(os.path.join(REPO, "exp/frozen.json"), encoding="utf-8"))
    want = dict(cap_mult=p2.CAP_MULT, cap_max=p2.CAP_MAX, min_frac=p2.MIN_FRAC, min_n=p2.MIN_N,
                fb_W=p2.FB["W"], fb_eps=p2.FB["eps"], fb_H=p2.FB["H"])
    if any(float(cfg[k]) != float(v) for k, v in want.items()) or cfg["feats"] != "f15":
        raise SystemExit("frozen.json과 paperexp2 기본값이 다르다")
    paths = targets_of(a.targets)
    if a.limit:
        paths = paths[:a.limit]
    res = load_res()
    rows = read_rows(a.out)
    cnt = {}
    for r in rows:
        cnt[r["trace"]] = cnt.get(r["trace"], 0) + 1
    done = {t for t, c in cnt.items() if c == ROWS_PER_WINDOW}
    if len(done) != len(cnt):  # 끝나지 않은 창의 줄을 뺀다
        os.replace(a.out, a.out + f".partial_{int(time.time())}")
        with open(a.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows([r for r in rows if r["trace"] in done])
    meta = a.out + ".env.json"
    if not os.path.exists(meta):
        import sklearn
        sha = hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()
        json.dump({"started": time.strftime("%F %T %z"), "script_sha256": sha, "python": sys.version.split()[0],
                   "numpy": np.__version__, "sklearn": sklearn.__version__, "repo": REPO, "targets": a.targets,
                   "seeds": SEEDS, "Tstar": TSTAR, "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS")},
                  open(meta, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    t_start, guard = time.time(), []
    for i, path in enumerate(paths, 1):
        name = os.path.basename(path).removesuffix(".csv")
        if name in done:
            continue
        if name not in res:
            raise SystemExit(f"[{name}] 개발 결과에 없다")
        t1 = time.time()
        out = window_rows(path, cfg, res[name])
        new = not os.path.exists(a.out) or os.path.getsize(a.out) == 0
        with open(a.out, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if new:
                w.writeheader()
            w.writerows(out)
        r0 = out[0]
        if i <= 3:
            guard.append(r0["check"])
            if len(guard) == min(3, len(paths)) and all(s == "MISMATCH" for s in guard):
                raise SystemExit("앞 창 3개 모두 real 비용이 개발 결과와 0.1% 넘게 다르다. 설정이나 일정을 확인한다")
        print(f"[{i}/{len(paths)}] {name} n_te={r0['n_te']} check={r0['check']} C3={r0['c3_MiB']:.1f} "
              f"rule={r0['rule_MiB']:.1f} ({time.time() - t1:.0f}s, 누적 {(time.time() - t_start) / 60:.1f}분)", flush=True)
    print(f"끝 → {a.out}", flush=True)


# ---------------------------------------------------------------- 요약
def verdict(lo, hi):
    if lo < -0.5 <= hi or lo <= 0.5 < hi:
        return "판정 불가(범위가 ±0.5%p 경계에 걸친다)"
    if hi < -0.5:
        return "앞쪽 비용이 낮다"
    if lo > 0.5:
        return "앞쪽 비용이 높다"
    return "차이 없음(±0.5%p 안)"


def summarize(a):
    names = [os.path.basename(p).removesuffix(".csv") for p in targets_of(a.targets)]
    rows = []
    for p in a.parts.split(","):
        rows += read_rows(p)
    by = {}
    for r in rows:
        by.setdefault(r["trace"], []).append(r)
    miss = [n for n in names if len(by.get(n, [])) != ROWS_PER_WINDOW]
    if miss:
        raise SystemExit(f"결과가 빠지거나 겹친 창 {len(miss)}개(예: {miss[:5]})")
    rows = [r for n in names for r in by[n]]
    real = {r["trace"]: r for r in rows if r["variant"] == "real"}
    base_tot = sum(float(real[n]["base_MiB"]) for n in names)
    rule_tot = sum(float(real[n]["rule_MiB"]) for n in names)
    tot = {}
    for r in rows:
        k = (r["variant"], int(r["seed"]))
        t = tot.setdefault(k, dict(ai=0.0, c3=0.0, c3_rule=[], ai_base=[], fb=0, fit=0.0, fail=0))
        t["ai"] += float(r["ai_MiB"])
        t["c3"] += float(r["c3_MiB"])
        for key, col in (("c3_rule", "c3_rule_frac"), ("ai_base", "ai_base_frac")):
            if r[col] not in ("", "nan") and math.isfinite(float(r[col])):
                t[key].append(float(r[col]))
        t["fb"] += int(r["draw_fallback"])
        t["fit"] += float(r["fit_cpu_s"])
        t["fail"] += int(r["check"] == "label_fit_failed")

    def d(x):
        return 100.0 * x / base_tot

    R = tot[("real", 0)]
    comp = {}
    for s in SEEDS:
        sh, dr, lb = tot[("P_shuffle", s)], tot[("P_draw", s)], tot[("P_label", s)]
        comp.setdefault("C3 − P_shuffle", []).append(d(R["c3"] - sh["c3"]))
        comp.setdefault("P_shuffle − P_draw", []).append(d(sh["c3"] - dr["c3"]))
        comp.setdefault("P_draw − P_label", []).append(d(dr["c3"] - lb["c3"]))
        comp.setdefault("P_label − rule", []).append(d(lb["c3"] - rule_tot))
        comp.setdefault("상호작용 (C3 − P_shuffle,C3) − (AI − P_shuffle,AI)", []).append(
            d((R["c3"] - sh["c3"]) - (R["ai"] - sh["ai"])))
        comp.setdefault("참고: C3 − P_draw", []).append(d(R["c3"] - dr["c3"]))
        comp.setdefault("참고: C3 − P_label", []).append(d(R["c3"] - lb["c3"]))
        comp.setdefault("참고: AI − P_shuffle,AI", []).append(d(R["ai"] - sh["ai"]))
    c3_rule = d(R["c3"] - rule_tot)
    decomp_err = max(abs(sum(comp[k][i] for k in ("C3 − P_shuffle", "P_shuffle − P_draw", "P_draw − P_label", "P_label − rule"))
                         - c3_rule) for i in range(len(SEEDS)))
    checks = {}
    for r in real.values():
        checks[r["check"]] = checks.get(r["check"], 0) + 1
    out = {"n_windows": len(names), "seeds": SEEDS, "Tstar_s": TSTAR, "fixed_online_total_MiB": base_tot,
           "rule_total_MiB": rule_tot, "real_C3_total_MiB": R["c3"], "real_AI_total_MiB": R["ai"],
           "C3_minus_rule_pp": c3_rule, "decomposition_max_error_pp": decomp_err, "real_check_vs_dev_results": checks,
           "comparisons_pp": {k: {"by_seed": v, "range": [min(v), max(v)], "verdict": verdict(min(v), max(v))}
                              for k, v in comp.items()},
           "diagnostics": {f"{v}{'' if s == 0 else f' seed {s}'}": {
               "C3_total_MiB": t["c3"], "AI_total_MiB": t["ai"],
               "C3_rule_frac_mean": float(np.mean(t["c3_rule"])) if t["c3_rule"] else None,
               "AI_base_frac_mean": float(np.mean(t["ai_base"])) if t["ai_base"] else None,
               "draw_fallback_chunks": t["fb"], "fit_cpu_s": t["fit"], "label_fit_failed": t["fail"]}
               for (v, s), t in sorted(tot.items())}}
    os.makedirs(a.outdir, exist_ok=True)
    with open(os.path.join(a.outdir, "b_summary.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: out[k] for k in ("n_windows", "C3_minus_rule_pp", "decomposition_max_error_pp",
                                          "real_check_vs_dev_results")}, ensure_ascii=False))
    for k, v in out["comparisons_pp"].items():
        print(f"{k}: {' '.join(f'{x:+.3f}' for x in v['by_seed'])}  범위 [{v['range'][0]:+.3f}, {v['range'][1]:+.3f}] → {v['verdict']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--parts", default="")
    ap.add_argument("--outdir", default="")
    a = ap.parse_args()
    if a.summarize:
        summarize(a)
    else:
        if not a.out:
            raise SystemExit("--out이 필요하다")
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        run(a)


if __name__ == "__main__":
    main()
