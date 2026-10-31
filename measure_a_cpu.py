#!/usr/bin/env python3
"""measure_a_cpu.py — 추가 분석 계획 v3(docs/additional-analysis-plan-v3.md) §2–§4의 작업 A. 개발 데이터 전용이다.

무엇을 재는가 (모두 프로세스 CPU 시간 time.process_time_ns, 창마다 --reps회 반복)
  within C3     창마다 생존 모델(매듭 10개)을 paperexp3.fit_survival3와 같은 방식으로 학습하는 CPU = F_within,
                시험 구간 결정 N_w개를 매듭 모델 10개로 한 번에 예측하는 CPU = I_opt, 매듭별 n_iter_.
                시뮬레이션 이득 S_w는 기존 개발 결과(--res, T* = 2.6초)의 '규칙-온라인+대체 − C3' 비용이다.
  사전 학습 C3  v1 pre(run_frozen.Pretrained)와 같은 학습원(--pretrain)으로 청크 모델 생존 모델을 학습하는 CPU = F_pre(반복),
                창마다 그 모델로 paperexp3.evaluate3를 한 번 돌려 S_pre를 내고, 일괄 예측 CPU = I_pre를 반복해 잰다.
  참고값        1행 예측 CPU(호출 고정비). I_opt에서 이것을 뺀 값도 함께 적는다(판정은 계획대로 I_opt로 한다).
                사전 학습 C3의 evaluate3 안에서 AI 경로의 κ 온라인 보정·대체 장부·C3 전환 장부 CPU(시뮬레이터 오프라인 값).
판정(계획 §2.1·§3)은 summary.json에 계산한다. 이 계측으로는 탈락만 확정하고, 생존은 C++ 온라인 측정을 거쳐야 쓴다.

검산
  어긋나면 멈춘다(같은 기계에서 정해지는 것)
  - 대상 목록의 창이 모두 있고 이름이 겹치지 않으며, 사전 학습원과 겹치지 않는다.
  - 모든 창이 --res에 고정-온라인·규칙·C3 행으로 정확히 한 번씩 있고, n_te가 다시 만든 트레이스의 시험 결정 수와 같다.
  - within: AI 학습 가능 여부(ai 열)가 --res와 같고, 앞 --check-sim개 창에서 계측용 학습이 paperexp3.fit_survival3와
            같은 생존 곡선을 낸다(같은 기계·같은 라이브러리에서 돌리므로 배열이 정확히 같아야 한다).
  - 앞 창 3개 모두에서 고정-온라인 비용이 --res와 0.1% 넘게 다르면 T*·단위·청크 설정이나 일정이 다른 것으로 본다.
  기록만 한다(다른 기계에서는 조금 다를 수 있다)
  - 다시 돌린 고정-온라인·규칙·C3 비용과 --res의 차이. CPU·BLAS·라이브러리 판이 다르면 합 순서 차이로 argmin이 갈려
    일부 창의 비용이 조금 달라질 수 있다(sealed-results-v2 §6 재현 확인). 창별 상태(exact/float/MISMATCH)와 합계 차이를
    summary.json의 repro_check에 적는다. 판정에는 within S = --res 값, pre S = 이번 실행 값을 쓴다.
  - S와 T_eval은 창마다 한 번만 더한다(반복은 CPU 시간에만 쓴다).

실행(대상 목록을 반드시 준다. 다른 목록으로 넘어가는 대체 경로는 없다)
  taskset -c 1 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \\
    python3 measure_a_cpu.py --targets exp/v2dev/daypool/all.txt --mode both --outdir exp/v2dev/cpu_a
  끊기면 같은 명령에 --resume을 붙여 이어 간다(끝난 창은 건너뛴다).
"""
import argparse
import csv
import gc
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import paperexp as pe  # noqa: E402
import paperexp2 as p2  # noqa: E402
import paperexp3 as p3  # noqa: E402

MiB = pe.MiB
TSTAR = 2.6  # 판정 T*(prereg-v2 §3.4, 3단계 실측)
C_NS = 281_800  # 비용 1단위(재확보 1 MiB)의 CPU 등가: stage3 판정 조건 반납 25.6 + 재확보 256.2 µs/MiB
POL_BASE, POL_RULE, POL_C3 = "고정-온라인", "규칙-온라인+대체", "C3 AI→규칙대체"
RES_DEFAULT = "exp/v2dev/daypool/res_q1.csv,exp/v2dev/daypool/res_q2.csv,exp/v2dev/daypool/res_q3.csv"
CODE_FILES = ["paperexp.py", "paperexp2.py", "paperexp3.py", "chunkslices.py", "policysim.py", "ailearn.py", "measure_a_cpu.py"]
TOL_EXACT, TOL_GROSS = 1e-9, 1e-3  # 상대 차이: 이하면 같음, 넘으면 설정·단위 오류로 보고 멈춘다(사이는 부동소수점 차이로 기록)
cpu_ns = time.process_time_ns

F_WITHIN_RUNS = ["trace", "rep", "ai", "n_te", "tau_s", "n_train", "S_units", "Sc_ns", "F_cpu_ns", "Iopt_cpu_ns",
                 "I_1row_cpu_ns", "n_iter_json"]
F_KNOTS = ["trace", "rep", "knot", "tau", "kind", "n_iter", "fit_cpu_ns", "n_rows", "n_pos"]
F_CHECK = ["trace", "ai_res", "ai_now", "Sk_equal_fit_survival3", "policy", "cost_MiB_now", "cost_MiB_res", "rel_diff", "status"]
F_PRE_FIT = ["rep", "F_fit_cpu_ns", "prep_cpu_ns", "n_train", "n_pretrain_files", "n_iter_json"]
F_PRE_SIM = ["trace", "n_te", "tau_s", "base_cost_MiB", "rule_cost_MiB", "c3_cost_MiB", "S_units", "Sc_ns",
             "rule_rel_diff_vs_res", "base_rel_diff_vs_res", "c3_within_res_MiB", "c3_rule_mode_frac", "status",
             "n_dec", "sim_kappa_ai_cpu_ns", "sim_ledger_ai_fb_cpu_ns", "sim_ledger_c3_switch_cpu_ns"]
F_PRE_RUNS = ["trace", "rep", "n_te", "Iopt_cpu_ns", "I_1row_cpu_ns"]


class AIUnavailable(Exception):
    """생존 모델을 학습할 수 없는 창(paperexp3.main에서 Sk = None이 되는 경우)"""

    def __init__(self, spent_ns, knot, err):
        super().__init__(f"매듭 {knot}: {type(err).__name__}: {err}")
        self.spent_ns = spent_ns


# ---------------------------------------------------------------- 계측 대상 계산(paperexp3와 같은 방식)
def fit_knots(Xtr, gtr, wtr, seed):
    """paperexp3.fit_survival3의 학습 부분과 같다(같은 학습기·설정·오류 처리). 예측은 따로 한다(predict_knots).
    반환: 매듭 모델 목록(학습기 또는 상수), 매듭별 기록, 학습 CPU 합(ns)."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    models, rows, spent = [], [], 0
    for k, tau in enumerate(pe.TAU):
        t0 = cpu_ns()
        y = (gtr > tau).astype(int)
        kind, n_iter = "gbdt", 0
        if len(np.unique(y)) < 2:
            m, kind = (float(y.mean()) if len(y) else 0.0), "const_one_class"
        else:
            try:
                m = HistGradientBoostingClassifier(max_iter=120, learning_rate=0.1, max_leaf_nodes=31, l2_regularization=1.0,
                                                   random_state=seed).fit(Xtr, y, sample_weight=wtr)
                n_iter = int(m.n_iter_)
            except ValueError as e:
                if min(int(y.sum()), len(y) - int(y.sum())) >= 2:
                    raise AIUnavailable(spent + cpu_ns() - t0, k, e) from e
                m, kind = float(y.mean()), "const_fit_error"
        dt = cpu_ns() - t0
        spent += dt
        models.append(m)
        rows.append(dict(knot=k, tau=float(tau), kind=kind, n_iter=n_iter, fit_cpu_ns=dt, n_rows=int(len(y)), n_pos=int(y.sum())))
    return models, rows, spent


def predict_knots(models, X):
    """생존 곡선 n × 매듭(단조 감소로 맞춤). paperexp3.fit_survival3·run_frozen.Pretrained.predict와 같은 계산이다."""
    Sk = np.empty((len(X), len(models)))
    if len(X) == 0:
        return Sk
    for k, m in enumerate(models):
        Sk[:, k] = m if isinstance(m, float) else m.predict_proba(X)[:, 1]
    return np.minimum.accumulate(Sk, axis=1)


def timed_fit(Xtr, gtr, wtr, seed):
    gc.collect()
    gc.disable()
    try:
        models, rows, spent = fit_knots(Xtr, gtr, wtr, seed)
        return models, rows, spent, "ok"
    except AIUnavailable as e:
        return None, [], e.spent_ns, "none"
    finally:
        gc.enable()


def timed_predict(models, X):
    """일괄 예측 CPU(ns)와 1행 예측 CPU(ns, 호출 고정비 참고값)"""
    if models is None or len(X) == 0:
        return 0, 0
    gc.collect()
    gc.disable()
    try:
        t0 = cpu_ns()
        predict_knots(models, X)
        t_all = cpu_ns() - t0
        t0 = cpu_ns()
        predict_knots(models, X[:1])
        t_one = cpu_ns() - t0
    finally:
        gc.enable()
    return t_all, t_one


# ---------------------------------------------------------------- 입력·검산
def resolve(p):
    return p if os.path.isabs(p) else os.path.join(ROOT, p)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def name_of(path):
    return os.path.basename(path).removesuffix(".csv")


def read_list(path, what):
    p = resolve(path)
    if not os.path.isfile(p):
        raise SystemExit(f"{what} 목록이 없다: {path}")
    paths = [resolve(l.strip()) for l in open(p, encoding="utf-8") if l.strip()]
    missing = [x for x in paths if not os.path.isfile(x)]
    if missing:
        raise SystemExit(f"{what} 목록의 파일 {len(missing)}개가 없다(예: {missing[:3]}). 개발 일정을 먼저 만든다.")
    names = [name_of(x) for x in paths]
    if len(set(names)) != len(names):
        raise SystemExit(f"{what} 목록에 같은 이름이 두 번 이상 있다")
    return paths


def load_res(files):
    """--res의 T* = 2.6초·청크 모델 행: 창 → {cost: 정책 → 비용(MiB), n_te, ai}"""
    res, seen = {}, set()
    for f in files:
        with open(resolve(f), encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if r.get("model") != "chunk" or abs(float(r["Tstar"]) - TSTAR) > 1e-9:
                    continue
                key = (r["trace"], r["policy"])
                if key in seen:
                    raise SystemExit(f"--res에 같은 창·정책 행이 두 번 있다: {key}")
                seen.add(key)
                d = res.setdefault(r["trace"], dict(cost={}, n_te=int(r["n_te"]), ai="ok"))
                if int(r["n_te"]) != d["n_te"]:
                    raise SystemExit(f"--res의 n_te가 정책마다 다르다: {r['trace']}")
                d["cost"][r["policy"]] = float(r["cost_MiB"])
                if r.get("ai"):
                    d["ai"] = r["ai"]
    return res


def rel_diff(a, b):
    return abs(a - b) / max(abs(b), 1e-12)


def classify(rd):
    return "exact" if rd <= TOL_EXACT else ("float" if rd <= TOL_GROSS else "MISMATCH")


def append_rows(path, fields, rows):
    if not rows:
        return
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerows(rows)


def read_rows(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def check_config(cfg):
    """paperexp3.main은 frozen.json의 장부·보정 상수를 paperexp2 기본값으로 쓴다. 두 값이 같은지 확인한다."""
    want = dict(cap_mult=p2.CAP_MULT, cap_max=p2.CAP_MAX, min_frac=p2.MIN_FRAC, min_n=p2.MIN_N,
                fb_W=p2.FB["W"], fb_eps=p2.FB["eps"], fb_H=p2.FB["H"])
    bad = {k: (cfg[k], v) for k, v in want.items() if float(cfg[k]) != float(v)}
    if bad or not np.array_equal(np.array(cfg["kgrid"], dtype=float), p2.KGRID) or cfg["feats"] != "f15":
        raise SystemExit(f"frozen.json과 paperexp2 기본값이 다르다: {bad}")


def environment(args, targets, pretrain, res_files):
    def git(*a):
        try:
            return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, timeout=30).stdout.strip()
        except Exception as e:  # noqa: BLE001
            return f"(git 오류: {e})"

    import sklearn
    from threadpoolctl import threadpool_info

    cpu = ""
    try:
        cpu = next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), "")
    except OSError:
        pass
    try:
        aff = sorted(os.sched_getaffinity(0))
    except AttributeError:
        aff = None
    return {
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "argv": sys.argv, "mode": args.mode, "reps": args.reps, "limit": args.limit, "check_sim": args.check_sim,
        "Tstar_s": TSTAR, "cpu_cost_unit_ns": C_NS, "timer": "time.process_time_ns",
        "git_head": git("rev-parse", "HEAD"), "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty_files": git("status", "--porcelain").splitlines(),
        "targets": {"list": args.targets, "sha256": sha256(resolve(args.targets)), "count_used": len(targets),
                    "first": name_of(targets[0]) if targets else None, "last": name_of(targets[-1]) if targets else None},
        "pretrain": {"list": args.pretrain, "sha256": sha256(resolve(args.pretrain)), "count": len(pretrain)} if pretrain else None,
        "res_files": {f: sha256(resolve(f)) for f in res_files},
        "code_sha256": {f: sha256(os.path.join(ROOT, f)) for f in CODE_FILES if os.path.exists(os.path.join(ROOT, f))},
        "python": sys.version.split()[0], "numpy": np.__version__, "sklearn": sklearn.__version__,
        "platform": platform.platform(), "cpu_model": cpu, "cpu_affinity": aff,
        "env_threads": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
        "threadpools": [{k: d.get(k) for k in ("user_api", "internal_api", "num_threads")} for d in threadpool_info()],
    }


# ---------------------------------------------------------------- 사전 학습 모델
def prepare_pre(paths, cfg, seed, reps, outdir, record):
    """run_frozen.Pretrained와 같은 학습원·가중치로 청크 모델 생존 모델을 학습한다(T*별 분류기는 C3에 쓰지 않아 만들지 않는다)."""
    gc.collect()
    t0 = cpu_ns()
    Xs, gs, ws = [], [], []
    for p in paths:
        t = p2.Trace2(p, cfg["feats"])
        m = t.TR
        if m.sum() == 0:
            continue
        Xs.append(t.X[m])
        gs.append(t.gtr[m])
        ws.append(t.h[m] / t.h[m].sum() * 1000)
    X, g, w = np.vstack(Xs), np.concatenate(gs), np.concatenate(ws)
    prep = cpu_ns() - t0
    models, iters = None, []
    for rep in range(1, reps + 1):
        mdl, krows, spent, ai = timed_fit(X, g, w, seed)
        if ai != "ok":
            raise SystemExit("사전 학습 모델을 학습할 수 없다(학습원 확인 필요)")
        it = [r["n_iter"] for r in krows]
        iters.append(it)
        if models is None:
            models = mdl
        if record:
            append_rows(os.path.join(outdir, "pre_fit_runs.csv"), F_PRE_FIT,
                        [dict(rep=rep, F_fit_cpu_ns=spent, prep_cpu_ns=prep, n_train=len(g), n_pretrain_files=len(Xs),
                              n_iter_json=json.dumps(it))])
            append_rows(os.path.join(outdir, "pre_knots.csv"), F_KNOTS, [dict(trace="(pretrain)", rep=rep, **r) for r in krows])
        print(f"[pre fit {rep}/{reps}] 조각 {len(g)}개, fit {spent / 1e6:.1f} ms (입력 준비 {prep / 1e6:.1f} ms)", flush=True)
        if not record:
            break  # 이어 가기: 시간은 이미 기록했다. 같은 시드라 모델은 같다
    if any(i != iters[0] for i in iters):
        raise SystemExit(f"반복마다 n_iter_가 다르다(결정적이지 않다): {iters}")
    return models


# ---------------------------------------------------------------- 창 하나
def run_within(tr, name, ref, args, cfg, outdir, check):
    m = tr.TR
    Xtr, gtr, wtr = tr.X[m], tr.gtr[m], tr.h[m] / tr.h[m].mean()  # paperexp3.main과 같은 입력
    Xte = tr.X[tr.TE]
    n_te, tau = int(tr.TE.sum()), float(tr.t0 + tr.dur - tr.c2)
    S = ref["cost"][POL_RULE] - ref["cost"][POL_C3]
    rows, krows, first_models, ai_seen = [], [], None, None
    for rep in range(1, args.reps + 1):
        models, kr, F, ai = timed_fit(Xtr, gtr, wtr, cfg["seed"])
        if ai_seen is None:
            ai_seen, first_models = ai, models
        elif ai != ai_seen:
            raise SystemExit(f"[{name}] 반복마다 AI 학습 가능 여부가 다르다")
        I, I1 = timed_predict(models, Xte)
        rows.append(dict(trace=name, rep=rep, ai=ai, n_te=n_te, tau_s=tau, n_train=int(m.sum()), S_units=S, Sc_ns=S * C_NS,
                         F_cpu_ns=F, Iopt_cpu_ns=I, I_1row_cpu_ns=I1, n_iter_json=json.dumps([r["n_iter"] for r in kr])))
        krows += [dict(trace=name, rep=rep, **r) for r in kr]
    if ai_seen != ref["ai"]:
        raise SystemExit(f"[{name}] AI 학습 가능 여부가 --res와 다르다(지금 {ai_seen}, --res {ref['ai']}): 입력이나 코드가 다르다")
    if ai_seen == "none" and abs(S) > 0:
        raise SystemExit(f"[{name}] AI를 쓸 수 없는 창인데 C3 비용이 규칙과 다르다(S = {S})")
    if len({r["n_iter_json"] for r in rows}) != 1:
        raise SystemExit(f"[{name}] 반복마다 n_iter_가 다르다(결정적이지 않다)")
    base_status = None
    if check:
        crow = []
        if ai_seen == "ok":
            Sk_now = predict_knots(first_models, tr.X)
            Sk_ref = p3.fit_survival3(Xtr, gtr, wtr, tr.X, cfg["seed"])
            same = bool(np.array_equal(Sk_now, Sk_ref))
        else:
            Sk_now, same = None, True
            try:
                p3.fit_survival3(Xtr, gtr, wtr, tr.X, cfg["seed"])
                same = False  # 계측용 학습은 실패했는데 원래 함수는 성공: 둘이 다르다
            except ValueError:
                pass
        r_now, _ = p3.evaluate3(tr, p2.Ledger(tr), TSTAR, Sk_now, cfg["wcal"], cfg["period"])
        for pol in (POL_BASE, POL_RULE, POL_C3):
            a, b = r_now[pol]["cost"] / MiB, ref["cost"][pol]
            rd = rel_diff(a, b)
            crow.append(dict(trace=name, ai_res=ref["ai"], ai_now=ai_seen, Sk_equal_fit_survival3=same, policy=pol,
                             cost_MiB_now=a, cost_MiB_res=b, rel_diff=rd, status=classify(rd)))
        append_rows(os.path.join(outdir, "within_sim_check.csv"), F_CHECK, crow)
        if not same:
            raise SystemExit(f"[{name}] 계측용 학습이 paperexp3.fit_survival3와 다른 생존 곡선을 낸다")
        base_status = crow[0]["status"]  # 비용 차이는 기록만 한다(머리말 '검산')
    append_rows(os.path.join(outdir, "within_knots.csv"), F_KNOTS, krows)
    append_rows(os.path.join(outdir, "within_runs.csv"), F_WITHIN_RUNS, rows)  # 창의 마지막 기록(이어 가기 기준)
    return rows, base_status


class CallTimer:
    """evaluate3 안의 함수 호출별 CPU 시간(참고값). evaluate3는 실행 시점에 p2.online_kappa_T와 p3.FALLBACK을 찾으므로
    두 이름을 잠시 감싼다. 호출 순서는 evaluate3 코드로 정해진다(코드 지문은 environment.json):
      online_kappa_T  1번째 = AI생존κ(C3의 AI 경로), 2번째 = C4
      FALLBACK        2번째 = AI생존κ+대체(AI vs 고정-온라인 장부), 6번째 = C3 전환(AI+대체 vs 규칙+대체 장부)"""

    def __init__(self):
        self.calls, self.orig = {}, []

    def __enter__(self):
        for mod, name in ((p2, "online_kappa_T"), (p3, "FALLBACK")):
            fn = getattr(mod, name)
            self.orig.append((mod, name, fn))

            def wrapped(*a, _fn=fn, _name=name, **k):
                t0 = cpu_ns()
                out = _fn(*a, **k)
                self.calls.setdefault(_name, []).append(cpu_ns() - t0)
                return out

            setattr(mod, name, wrapped)
        return self

    def __exit__(self, *exc):
        for mod, name, fn in self.orig:
            setattr(mod, name, fn)
        return False

    def ai_path(self):
        k, f = self.calls.get("online_kappa_T", []), self.calls.get("FALLBACK", [])
        if len(k) != 2 or len(f) != 6:
            return math.nan, math.nan, math.nan
        return k[0], f[1], f[5]


def run_pre(tr, name, ref, args, cfg, outdir, models):
    Sk_all = predict_knots(models, tr.X)  # 시뮬레이션용(시간을 재지 않는다)
    with CallTimer() as ct:
        r, info = p3.evaluate3(tr, p2.Ledger(tr), TSTAR, Sk_all, cfg["wcal"], cfg["period"])
    kap, led_ai, led_sw = ct.ai_path()
    base, rule, c3 = (r[p]["cost"] / MiB for p in (POL_BASE, POL_RULE, POL_C3))
    rd_rule, rd_base = rel_diff(rule, ref["cost"][POL_RULE]), rel_diff(base, ref["cost"][POL_BASE])
    status = max((classify(rd_rule), classify(rd_base)), key=["exact", "float", "MISMATCH"].index)  # 기록만 한다
    n_te, tau = int(tr.TE.sum()), float(tr.t0 + tr.dur - tr.c2)
    S = rule - c3
    Xte = tr.X[tr.TE]
    runs = []
    for rep in range(1, args.reps + 1):
        I, I1 = timed_predict(models, Xte)
        runs.append(dict(trace=name, rep=rep, n_te=n_te, Iopt_cpu_ns=I, I_1row_cpu_ns=I1))
    append_rows(os.path.join(outdir, "pre_window_runs.csv"), F_PRE_RUNS, runs)
    append_rows(os.path.join(outdir, "pre_sim_costs.csv"), F_PRE_SIM,  # 창의 마지막 기록(이어 가기 기준)
                [dict(trace=name, n_te=n_te, tau_s=tau, base_cost_MiB=base, rule_cost_MiB=rule, c3_cost_MiB=c3, S_units=S,
                      Sc_ns=S * C_NS, rule_rel_diff_vs_res=rd_rule, base_rel_diff_vs_res=rd_base,
                      c3_within_res_MiB=ref["cost"][POL_C3], c3_rule_mode_frac=info.get(POL_C3, math.nan), status=status,
                      n_dec=int((tr.VA | tr.TE).sum()), sim_kappa_ai_cpu_ns=kap, sim_ledger_ai_fb_cpu_ns=led_ai,
                      sim_ledger_c3_switch_cpu_ns=led_sw)])
    return S, runs, classify(rd_base), status


# ---------------------------------------------------------------- 요약·판정
def band(lo, hi, thr):
    """반복 범위 [lo, hi]와 경계 thr: 'below'(모두 아래), 'above'(모두 같거나 위), 'straddle'(걸침)"""
    if hi < thr:
        return "below"
    if lo >= thr:
        return "above"
    return "straddle"


def summarize(outdir, names, mode, reps, partial, res):
    out = {"partial_run": partial, "n_windows": len(names), "Tstar_s": TSTAR, "cpu_cost_unit_ns": C_NS, "reps": reps}
    if partial:
        out["note"] = "부분 실행(--limit)이라 판정하지 않는다. 숫자는 동작 확인용이다."
    out["repro_note"] = ("repro_check는 다시 돌린 비용과 --res의 차이다. 다른 CPU·라이브러리에서는 argmin이 갈려 일부 창이 조금 다를 수 있다. "
                         "판정에는 within S = --res 값, pre S = 이번 실행 값을 쓴다.")
    base_tot = sum(res[n]["cost"][POL_BASE] for n in names)
    out["fixed_online_total_units"] = base_tot
    if mode in ("within", "both"):
        rows = [r for r in read_rows(os.path.join(outdir, "within_runs.csv")) if r["trace"] in names]
        got = {(r["trace"], int(r["rep"])) for r in rows}
        if len(rows) != len(got) or got != {(n, k) for n in names for k in range(1, reps + 1)}:
            raise SystemExit("within_runs.csv의 (창, 반복) 기록이 빠지거나 겹친다")
        per = {n: next(r for r in rows if r["trace"] == n and int(r["rep"]) == 1) for n in names}
        N = len(names)
        S_tot = sum(float(per[n]["S_units"]) for n in names)
        Sc_tot = S_tot * C_NS
        F_tot = [sum(int(r["F_cpu_ns"]) for r in rows if int(r["rep"]) == k) for k in range(1, reps + 1)]
        I_tot = [sum(int(r["Iopt_cpu_ns"]) for r in rows if int(r["rep"]) == k) for k in range(1, reps + 1)]
        I_net = [sum(max(int(r["Iopt_cpu_ns"]) - int(r["I_1row_cpu_ns"]), 0) for r in rows if int(r["rep"]) == k)
                 for k in range(1, reps + 1)]
        taus = np.array([float(per[n]["tau_s"]) for n in names])
        Sc_m, F_lo, F_hi = Sc_tot / N, min(F_tot) / N, max(F_tot) / N
        w = {"S_total_units": S_tot, "S_total_pp_of_fixed_online": 100 * S_tot / base_tot if base_tot else math.nan,
             "Sc_total_s": Sc_tot / 1e9, "Sc_mean_ms": Sc_m / 1e6,
             "F_total_s_by_rep": [x / 1e9 for x in F_tot], "F_mean_ms_by_rep": [x / N / 1e6 for x in F_tot],
             "Iopt_total_s_by_rep": [x / 1e9 for x in I_tot], "Iopt_minus_1row_total_s_by_rep": [x / 1e9 for x in I_net],
             "n_te_total": int(sum(int(per[n]["n_te"]) for n in names)),
             "ai_unavailable": [n for n in names if per[n]["ai"] == "none"],
             "tau_s": {"min": float(taus.min()), "median": float(np.median(taus)), "max": float(taus.max())}}
        # 1단계: I = 0, 창당 평균 F와 창당 평균 Sc. R*/τ = F/Sc. 2τ에서 순이득이 되려면 F < 2·Sc
        if Sc_m <= 0:
            s1 = ("탈락", "시뮬레이션 이득이 0 이하다")
        else:
            b2 = band(F_lo, F_hi, 2 * Sc_m)
            s1 = {"above": ("탈락", "F 범위 전체가 2·Sc 이상(R* 하한 ≥ 2τ)"), "straddle": ("판정 불가", "F 범위가 2·Sc 경계에 걸친다"),
                  "below": ("통과", "F 범위 전체가 2·Sc 미만")}[b2]
        w["step1"] = {"rule": "I = 0, 창당 평균: F ≥ 2·Sc 이면 탈락", "F_mean_ms_range": [F_lo / 1e6, F_hi / 1e6],
                      "threshold_2Sc_ms": 2 * Sc_m / 1e6, "Rstar_over_tau_I0_range": [F_lo / Sc_m, F_hi / Sc_m] if Sc_m > 0 else None,
                      "verdict": s1[0], "reason": s1[1]}
        # 2단계: 시험 구간 일괄 예측 CPU 합 vs Sc 합
        b = band(min(I_tot), max(I_tot), Sc_tot)
        s2 = {"above": ("탈락", "I_opt 범위 전체가 Sc 이상"), "straddle": ("판정 불가", "I_opt 범위가 Sc 경계에 걸친다"),
              "below": ("통과", "I_opt 범위 전체가 Sc 미만")}[b]
        b_net = band(min(I_net), max(I_net), Sc_tot)
        w["step2"] = {"rule": "Sc_total ≤ I_opt_total 이면 탈락", "Iopt_total_s_range": [min(I_tot) / 1e9, max(I_tot) / 1e9],
                      "Sc_total_s": Sc_tot / 1e9, "verdict": s2[0], "reason": s2[1],
                      "reference_minus_1row_band": b_net}
        if s1[0] == "통과" and s2[0] == "통과":
            lo = F_lo / (Sc_m - min(I_tot) / N)
            hi = F_hi / (Sc_m - max(I_tot) / N)
            w["Rstar_over_tau_range_with_Iopt"] = [lo, hi]
            final = "C++ 온라인 측정 대상(생존 판정 보류)"
        elif "탈락" in (s1[0], s2[0]):
            final = "탈락(" + ("1단계" if s1[0] == "탈락" else "2단계") + ")"
        else:
            final = "판정 불가"
        w["verdict"] = "판정 없음(부분 실행)" if partial else final
        ov = (min(I_tot) + min(F_tot) / 2) / C_NS  # 계획 §5: within은 I + R = 2τ에서의 F·τ/R
        w["ai_path_overhead_pp_for_B"] = 100 * ov / base_tot if base_tot else math.nan
        chk = [r for r in read_rows(os.path.join(outdir, "within_sim_check.csv")) if r["trace"] in names]
        w["repro_check"] = {pol: {"windows": {s: sum(1 for r in chk if r["policy"] == pol and r["status"] == s)
                                              for s in ("exact", "float", "MISMATCH")},
                                  "max_rel_diff": max([float(r["rel_diff"]) for r in chk if r["policy"] == pol], default=0.0)}
                            for pol in (POL_BASE, POL_RULE, POL_C3)}
        w["repro_check"]["Sk_equal_fit_survival3"] = all(r["Sk_equal_fit_survival3"] == "True" for r in chk)
        out["within"] = w
    if mode in ("pre", "both"):
        sim = [r for r in read_rows(os.path.join(outdir, "pre_sim_costs.csv")) if r["trace"] in names]
        runs = [r for r in read_rows(os.path.join(outdir, "pre_window_runs.csv")) if r["trace"] in names]
        fits = read_rows(os.path.join(outdir, "pre_fit_runs.csv"))
        if sorted(r["trace"] for r in sim) != sorted(names):
            raise SystemExit("pre_sim_costs.csv의 창 기록이 빠지거나 겹친다(S·T_eval 중복 합산 방지)")
        got = {(r["trace"], int(r["rep"])) for r in runs}
        if len(runs) != len(got) or got != {(n, k) for n in names for k in range(1, reps + 1)}:
            raise SystemExit("pre_window_runs.csv의 (창, 반복) 기록이 빠지거나 겹친다")
        if len(fits) != reps:
            raise SystemExit("pre_fit_runs.csv의 반복 수가 다르다")
        S_tot = sum(float(r["S_units"]) for r in sim)
        Sc_tot = S_tot * C_NS
        T_eval = sum(float(r["tau_s"]) for r in sim)
        I_tot = [sum(int(r["Iopt_cpu_ns"]) for r in runs if int(r["rep"]) == k) for k in range(1, reps + 1)]
        I_net = [sum(max(int(r["Iopt_cpu_ns"]) - int(r["I_1row_cpu_ns"]), 0) for r in runs if int(r["rep"]) == k)
                 for k in range(1, reps + 1)]
        F_fit = [int(r["F_fit_cpu_ns"]) for r in fits]
        prep = int(fits[0]["prep_cpu_ns"])
        p = {"S_pre_total_units": S_tot, "S_pre_total_pp_of_fixed_online": 100 * S_tot / base_tot if base_tot else math.nan,
             "S_pre_total_times_c_s": Sc_tot / 1e9, "T_eval_s": T_eval,
             "Iopt_total_s_by_rep": [x / 1e9 for x in I_tot], "Iopt_minus_1row_total_s_by_rep": [x / 1e9 for x in I_net],
             "F_pre_fit_s_by_rep": [x / 1e9 for x in F_fit], "F_pre_input_prep_s": prep / 1e9,
             "rule_cost_check": sorted({r["status"] for r in sim}),
             "c3_rule_mode_frac_mean": float(np.nanmean([float(r["c3_rule_mode_frac"]) for r in sim]))}
        b = band(min(I_tot), max(I_tot), Sc_tot)
        op = {"above": ("탈락", "I_pre 범위 전체가 S_pre·c 이상"), "straddle": ("판정 불가", "I_pre 범위가 경계에 걸친다"),
              "below": ("통과", "I_pre 범위 전체가 S_pre·c 미만")}[b]
        p["operating"] = {"rule": "S_pre·c ≤ I_pre(일괄 예측 합)이면 탈락", "net_s_range": [(Sc_tot - max(I_tot)) / 1e9, (Sc_tot - min(I_tot)) / 1e9],
                          "verdict": op[0], "reason": op[1], "reference_minus_1row_band": band(min(I_net), max(I_net), Sc_tot)}

        def hstar(F, I):
            d = Sc_tot - I
            return F * T_eval / d if d > 0 else None

        h_opt, h_con = hstar(min(F_fit), min(I_tot)), hstar(max(F_fit), max(I_tot))
        h_opt2, h_con2 = hstar(min(F_fit) + prep, min(I_tot)), hstar(max(F_fit) + prep, max(I_tot))
        p["lifecycle"] = {"Hstar_s_fit_only": [h_opt, h_con], "Hstar_s_with_input_prep": [h_opt2, h_con2],
                          "Hstar_over_T_eval_fit_only": [h / T_eval if h is not None else None for h in (h_opt, h_con)],
                          "note": "분모(S_pre·c − I_pre)가 0 이하면 유한한 H*가 없다(None)"}
        final = {"탈락": "탈락(운영 기준)", "판정 불가": "판정 불가"}.get(op[0], "C++ 온라인 측정 대상(운영 기준 생존 판정 보류)")
        p["verdict"] = "판정 없음(부분 실행)" if partial else final
        p["ai_path_overhead_pp_for_B"] = 100 * (min(I_tot) / C_NS) / base_tot if base_tot else math.nan

        def repro(col_now, col_rd, pol):
            rds = [float(r[col_rd]) for r in sim]
            now, ref_ = sum(float(r[col_now]) for r in sim), sum(res[r["trace"]]["cost"][pol] for r in sim)
            return {"windows": {s: sum(1 for x in rds if classify(x) == s) for s in ("exact", "float", "MISMATCH")},
                    "max_rel_diff": max(rds, default=0.0), "total_rel_diff": rel_diff(now, ref_) if ref_ else math.nan}

        p["repro_check"] = {"fixed_online": repro("base_cost_MiB", "base_rel_diff_vs_res", POL_BASE),
                            "rule": repro("rule_cost_MiB", "rule_rel_diff_vs_res", POL_RULE)}

        def fsum(col):
            v = [float(r[col]) for r in sim]
            return sum(v) / 1e9 if all(math.isfinite(x) for x in v) else math.nan

        n_dec, n_te = sum(int(r["n_dec"]) for r in sim), sum(int(r["n_te"]) for r in sim)
        p["reference_sim_ai_path_s"] = {
            "kappa_ai": fsum("sim_kappa_ai_cpu_ns"), "ledger_ai_fb": fsum("sim_ledger_ai_fb_cpu_ns"),
            "ledger_c3_switch": fsum("sim_ledger_c3_switch_cpu_ns"), "n_dec_total": n_dec, "n_te_total": n_te,
            "note": "시뮬레이터의 오프라인 일괄 값(검증+시험 구간 결정 전체). 판정에 쓰지 않는다. 시험 구간 몫은 약 × n_te/n_dec"}
        out["pre"] = p
    return out


# ---------------------------------------------------------------- 실행
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--targets", required=True, help="대상 일정 경로 목록(예: exp/v2dev/daypool/all.txt)")
    ap.add_argument("--outdir", required=True, help="새 출력 폴더(비어 있어야 한다. --resume이면 이어 쓴다)")
    ap.add_argument("--mode", choices=["within", "pre", "both"], default="both")
    ap.add_argument("--res", default=RES_DEFAULT, help="기존 개발 결과 CSV(쉼표로 구분)")
    ap.add_argument("--pretrain", default="exp/dev_sched.txt", help="사전 학습원 목록(v1 pre와 같다)")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--check-sim", type=int, default=3, help="within에서 evaluate3를 다시 돌려 --res와 맞춰 볼 앞 창 수")
    ap.add_argument("--limit", type=int, default=0, help="앞 N개 창만(스모크 시험). 이때는 판정하지 않는다")
    ap.add_argument("--resume", action="store_true", help="같은 --outdir에서 끝난 창을 건너뛰고 이어 간다")
    a = ap.parse_args()

    outdir = resolve(a.outdir)
    if os.path.isdir(outdir) and os.listdir(outdir) and not a.resume:
        raise SystemExit(f"출력 폴더가 비어 있지 않다(덮어쓰지 않는다. 이어 가려면 --resume): {outdir}")
    if a.resume and not os.path.isdir(outdir):
        raise SystemExit(f"--resume인데 출력 폴더가 없다: {outdir}")
    os.makedirs(outdir, exist_ok=True)
    if os.environ.get("OMP_NUM_THREADS") != "1":
        print("경고: OMP_NUM_THREADS가 1이 아니다. 계측은 threadpool_limits(1)로 단일 스레드에서 하지만, 계획대로 1로 둔다.", flush=True)

    p3.use_chunk_model()  # v2 평가 모델(청크 모델, run_fallback3)
    cfg = json.load(open(os.path.join(ROOT, "exp/frozen.json"), encoding="utf-8"))
    check_config(cfg)
    targets = read_list(a.targets, "대상")
    if a.limit:
        targets = targets[:a.limit]
    names = [name_of(p) for p in targets]
    pretrain = read_list(a.pretrain, "사전 학습원") if a.mode in ("pre", "both") else []
    if set(names) & {name_of(p) for p in pretrain}:
        raise SystemExit("대상 창과 사전 학습원이 겹친다")
    res_files = [f for f in a.res.split(",") if f]
    res = load_res(res_files)
    lack = [n for n in names if n not in res or any(p not in res[n]["cost"] for p in (POL_BASE, POL_RULE, POL_C3))]
    if lack:
        raise SystemExit(f"--res에 결과가 없는 대상 창 {len(lack)}개(예: {lack[:5]}). res_q3.csv 포함 여부를 확인한다")

    env = environment(a, targets, pretrain, res_files)
    env_name = "environment.json" if not os.path.exists(os.path.join(outdir, "environment.json")) else f"environment_resume_{int(time.time())}.json"
    with open(os.path.join(outdir, env_name), "w", encoding="utf-8") as f:
        json.dump(env, f, ensure_ascii=False, indent=2)
    print(f"대상 {len(names)}창, 모드 {a.mode}, 반복 {a.reps}, T* {TSTAR}, 커밋 {env['git_head'][:10]}, CPU {env['cpu_affinity']}", flush=True)

    # 이어 가기: 창마다 마지막에 쓰는 파일(within_runs.csv, pre_sim_costs.csv)에 있는 창만 끝난 것으로 본다.
    # 끝나지 않은 창이 다른 파일에 남긴 부분 기록은 지운다(같은 창이 두 번 합산되지 않게).
    done_w = {r["trace"] for r in read_rows(os.path.join(outdir, "within_runs.csv"))}
    done_p = {r["trace"] for r in read_rows(os.path.join(outdir, "pre_sim_costs.csv"))}
    for fname, fields, keep in (("within_knots.csv", F_KNOTS, done_w), ("within_sim_check.csv", F_CHECK, done_w),
                                ("pre_window_runs.csv", F_PRE_RUNS, done_p)):
        path = os.path.join(outdir, fname)
        rows = read_rows(path)
        kept = [r for r in rows if r["trace"] in keep]
        if len(kept) != len(rows):
            os.replace(path, path + f".partial_{int(time.time())}")
            append_rows(path, fields, kept)
            print(f"이어 가기: {fname}에서 끝나지 않은 창의 기록 {len(rows) - len(kept)}줄을 뺐다(원본은 .partial_*로 남긴다)", flush=True)
    for fname in ("within_runs.csv", "pre_window_runs.csv"):
        cnt = {}
        for r in read_rows(os.path.join(outdir, fname)):
            cnt[r["trace"]] = cnt.get(r["trace"], 0) + 1
        if any(c != a.reps for c in cnt.values()):
            raise SystemExit(f"{fname}에 반복 수가 맞지 않는 창이 있다(이어 가기 불가, 새 폴더로 다시 실행한다)")

    from threadpoolctl import threadpool_limits

    t_start = time.time()
    with threadpool_limits(limits=1):
        models_pre = None
        if a.mode in ("pre", "both"):
            n_fit = len(read_rows(os.path.join(outdir, "pre_fit_runs.csv")))
            if n_fit not in (0, a.reps):
                raise SystemExit("pre_fit_runs.csv의 반복 수가 맞지 않는다(새 폴더로 다시 실행한다)")
            models_pre = prepare_pre(pretrain, cfg, cfg["seed"], a.reps, outdir, record=(n_fit == 0))
        guard, n_guard = {"within": [], "pre": []}, min(3, len(names))
        for i, (path, name) in enumerate(zip(targets, names), 1):
            need_w = a.mode in ("within", "both") and name not in done_w
            need_p = a.mode in ("pre", "both") and name not in done_p
            if not (need_w or need_p):
                continue
            t1 = time.time()
            tr = p2.Trace2(path, cfg["feats"])
            ref = res[name]
            if int(tr.TE.sum()) != ref["n_te"]:
                raise SystemExit(f"[{name}] 시험 결정 수가 --res와 다르다({int(tr.TE.sum())} vs {ref['n_te']}): 일정이 다르다")
            msg = []
            if need_w:
                rows, bst = run_within(tr, name, ref, a, cfg, outdir, check=(i <= a.check_sim))
                if bst is not None and i <= n_guard:
                    guard["within"].append(bst)
                msg.append(f"within ai={rows[0]['ai']} F={min(r['F_cpu_ns'] for r in rows) / 1e6:.1f}ms "
                           f"I={min(r['Iopt_cpu_ns'] for r in rows) / 1e6:.2f}ms S={rows[0]['S_units']:+.2f}"
                           + (f" 재현={bst}" if bst is not None else ""))
            if need_p:
                S, runs, bst, st = run_pre(tr, name, ref, a, cfg, outdir, models_pre)
                if i <= n_guard:
                    guard["pre"].append(bst)
                msg.append(f"pre I={min(r['Iopt_cpu_ns'] for r in runs) / 1e6:.2f}ms S={S:+.2f}" + ("" if st == "exact" else f" 재현={st}"))
            print(f"[{i}/{len(names)}] {name} n_te={ref['n_te']} " + " | ".join(msg) +
                  f" ({time.time() - t1:.0f}s, 누적 {(time.time() - t_start) / 60:.1f}분)", flush=True)
            del tr
            if i == n_guard:
                for k, v in guard.items():
                    if len(v) == n_guard and all(s == "MISMATCH" for s in v):
                        raise SystemExit(f"앞 창 {n_guard}개 모두 고정-온라인 비용이 --res와 0.1% 넘게 다르다({k}): "
                                         "T*·단위·청크 설정이나 일정이 다른 것으로 보고 멈춘다(고친 뒤 새 폴더로 다시 실행한다)")

    summary = summarize(outdir, names, a.mode, a.reps, bool(a.limit), res)
    summary["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    with open(os.path.join(outdir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"A 계측 완료 → {outdir}/summary.json", flush=True)


if __name__ == "__main__":
    main()
