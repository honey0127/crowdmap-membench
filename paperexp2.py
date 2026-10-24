#!/usr/bin/env python3
"""paperexp2.py — 방법 확정용 실험(유체 모델): 온라인 보정, 같은 결정 단계를 쓰는 공정한 비교, 입력 패턴 특징, 불규칙성 지표.

결정 단계(모든 '-온라인' 정책 공통)
  - 조각을 묶음(그룹)으로 나누고, 그룹별 반납 시점을 U초마다 그림자 장부로 다시 고른다.
  - 장부는 놀기 시작한 지 T_cap초가 지난 조각(성숙 조각)만 쓴다. 성숙 조각은 후보 반납 시점 T ≤ T_cap 모두에 대해 비용이
    확정된다(아직 다시 쓰이지 않았으면 나중에 다시 쓰인다고 둔다). T_cap = min(300, 20·T*).
  - 창은 [t − T_cap − W_cal, t − T_cap)이다. 그룹 무게가 창 무게의 2% 미만이거나 20조각 미만이면 전체 최적값을 쓴다.
정책(묶는 방법만 다르다)
  고정-온라인        묶음 1개
  규칙-온라인        특징 1~2개의 검증 4분위 칸 조합(조합은 검증 교차 적합으로 고른다)
  AI분류-온라인      'g > T*' 분류기 점수의 검증 10분위
  AI생존그룹-온라인  생존 곡선이 권하는 반납 시점(매듭)별 묶음
  AI생존κ-온라인     묶지 않고 조각별 기대 비용 최소 T(생존 곡선) × κ, κ를 같은 장부로 온라인 선택
  +대체              고정-온라인을 기준으로 그림자 장부 전환(W, ε, H는 고정값)
참고: 고정(정적) = 검증으로 한 번 고른 고정 타임아웃(v1 기준), AI분류(정적) = 검증으로 한 번 맞춘 v1 AI, 오라클.
프로토콜
  within: 트레이스마다 학습 50 / 검증 25 / 시험 25(paperexp.py와 같다). 모델은 학습 구간, 묶음 경계·규칙 조합은 검증 구간.
  pre(사전 학습): 모델만 다른 트레이스들의 학습 구간으로 학습하고 나머지는 within과 같다(--pretrained-from로 학습 일정 목록).
사용
  python3 paperexp2.py exp/sched/*.csv --out exp/res2 [--feats f15] [--wcal 900] [--period 30] [--loto]
"""
import argparse
import csv
import itertools
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paperexp as pe  # noqa: E402

MiB = pe.MiB
CAP_MULT, CAP_MAX = 20.0, 300.0
MIN_FRAC, MIN_N = 0.02, 20
KGRID = np.array([0.25, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0, 2.8, 4.0])
FEATS_IN = ["크기평균10s", "크기평균60s", "크기CV60s", "직전크기", "직전도착경과"]
FB = dict(W=300.0, eps=0.0, H=0.0)  # 대체 장치 고정값(v1 민감도에서 평균·최악 모두 가장 좋았다)


# ---------------------------------------------------------------- 데이터·특징·불규칙성
class Trace2(pe.Trace):
    def __init__(self, path, feats="f15"):
        self.featset = feats
        super().__init__(path)
        self.ta, self.tf, self.sz = pe.ps.load_schedule(path)
        self.feat_names = pe.FEATS + (FEATS_IN if feats == "f15" else [])

    def _features(self, S, ta, tf, sz):
        X = super()._features(S, ta, tf, sz)
        if self.featset != "f15":
            return X
        ts = S["ts"]
        o = np.argsort(ta, kind="stable")
        t_s, s_s = ta[o], sz[o].astype(float)
        msk = (ta >= self.t0) & (ta < self.t0 + 0.5 * self.dur)
        med = float(np.median(sz[msk])) if msk.any() else float(np.median(sz))
        c1 = np.concatenate([[0.0], np.cumsum(s_s / med)])
        c2 = np.concatenate([[0.0], np.cumsum((s_s / med) ** 2)])
        hi = np.searchsorted(t_s, ts, side="right")

        def win(W):
            lo = np.searchsorted(t_s, ts - W, side="right")
            n = hi - lo
            with np.errstate(invalid="ignore", divide="ignore"):
                mean = (c1[hi] - c1[lo]) / n
                var = (c2[hi] - c2[lo]) / n - mean ** 2
                cv = np.sqrt(np.maximum(var, 0.0)) / mean
            return np.where(n > 0, mean, np.nan), np.where(n > 1, cv, np.nan)

        m10, _ = win(10.0)
        m60, cv60 = win(60.0)
        last = np.where(hi > 0, s_s[np.maximum(hi - 1, 0)] / med, np.nan)
        since = np.where(hi > 0, np.log10(np.maximum(ts - t_s[np.maximum(hi - 1, 0)], 1e-4)), np.nan)
        return np.column_stack([X, m10, m60, cv60, last, since])


def irregularity(ta, sz, lo, hi):
    """입력만으로 계산하는 불규칙성 지표(구간 [lo, hi))"""
    m = (ta >= lo) & (ta < hi)
    t, s = ta[m] - lo, sz[m].astype(float)
    dur = hi - lo
    out = dict(n=int(m.sum()), rate=m.sum() / dur if dur > 0 else math.nan)
    out["size_cv"] = float(s.std() / s.mean()) if len(s) > 1 else math.nan
    nb10 = int(dur // 10)
    c10 = np.bincount((t // 10).astype(int), minlength=nb10)[:nb10] if nb10 else np.array([])
    out["fano10"] = float(c10.var() / c10.mean()) if len(c10) and c10.mean() > 0 else math.nan
    nb1 = int(dur // 60) * 60
    if nb1:
        c1 = np.bincount(t.astype(int), minlength=nb1)[:nb1].reshape(-1, 60)
        mu, var = c1.mean(1), c1.var(1)
        ok = mu > 0
        out["burst"] = float(np.median(var[ok] / mu[ok])) if ok.any() else math.nan
    else:
        out["burst"] = math.nan
    nb300 = int(dur // 300)
    c300 = np.bincount((t // 300).astype(int), minlength=nb300)[:nb300] if nb300 else np.array([])
    out["level"] = float(c300.std() / c300.mean()) if len(c300) and c300.mean() > 0 else math.nan
    return out


# ---------------------------------------------------------------- 그림자 장부와 온라인 보정
def tcap(b):
    return min(CAP_MAX, CAP_MULT * b)


def grid_on(b):
    return pe.GRID[pe.GRID <= tcap(b) * (1 + 1e-9)]


class Ledger:
    def __init__(self, tr):
        self.tr = tr
        self.o = np.argsort(tr.ts, kind="stable")
        self.tss = tr.ts[self.o]

    def window(self, lo_t, hi_t):
        lo = np.searchsorted(self.tss, lo_t, "left")
        hi = np.searchsorted(self.tss, hi_t, "left")
        return self.o[lo:hi]

    def observed(self, J, t, cap):
        """시각 t에 아는 놀이 시간: 이미 다시 쓰였으면 g, 아니면 cap보다 큰 값(후보 T ≤ cap 모두에서 반납 후 재확보)"""
        tr = self.tr
        age = t - tr.ts[J]
        done = tr.r[J] & (tr.g[J] <= age)
        return np.where(done, tr.g[J], np.maximum(age, cap) * (1 + 1e-9) + 1e-9)


def group_sums(C, gid, G):
    onehot = np.zeros((len(gid), G))
    onehot[np.arange(len(gid)), gid] = 1.0
    return C @ onehot  # nT × G


def online_group_T(tr, led, b, grp, G, dec_mask, W_cal, U):
    """그룹별 반납 시점을 U초마다 장부로 다시 고른다. 반환: 조각별 T(dec_mask 밖은 NaN), 갱신 기록"""
    cap, grid = tcap(b), grid_on(b)
    T = np.full(len(tr.g), np.nan)
    dec = np.where(dec_mask)[0]
    if len(dec) == 0:
        return T
    t0, t1 = tr.ts[dec].min(), tr.ts[dec].max()
    prev = np.zeros(G, dtype=np.int64)
    for u in np.arange(t0, t1 + U, U):
        D = led.window(u, u + U)
        D = D[dec_mask[D]]
        if len(D) == 0:
            continue
        J = led.window(u - cap - W_cal, u - cap)
        if len(J):
            g_obs = led.observed(J, u, cap)
            C = pe.cost_matrix(g_obs, tr.h[J], np.ones(len(J), dtype=bool), b, grid)
            gj = grp[J]
            S = group_sums(C, gj, G)
            ip = int(np.argmin(C.sum(1)))
            table = np.argmin(S, axis=0)
            w = np.bincount(gj, weights=tr.h[J], minlength=G)
            n = np.bincount(gj, minlength=G)
            table[(w < MIN_FRAC * w.sum()) | (n < MIN_N)] = ip
            prev = table
        T[D] = grid[prev[grp[D]]]
    return T


def online_kappa_T(tr, led, b, Tk, dec_mask, W_cal, U):
    """조각별 후보 T(κ) 배열 Tk(len(KGRID) × n, 이미 cap으로 자름)에서 κ를 U초마다 장부로 고른다"""
    cap = tcap(b)
    T = np.full(len(tr.g), np.nan)
    dec = np.where(dec_mask)[0]
    if len(dec) == 0:
        return T, []
    t0, t1 = tr.ts[dec].min(), tr.ts[dec].max()
    k_prev, chosen = int(np.argmin(np.abs(KGRID - 1.0))), []
    for u in np.arange(t0, t1 + U, U):
        D = led.window(u, u + U)
        D = D[dec_mask[D]]
        if len(D) == 0:
            continue
        J = led.window(u - cap - W_cal, u - cap)
        if len(J):
            g_obs = led.observed(J, u, cap)
            hJ = tr.h[J]
            costs = [pe.cost(g_obs, hJ, np.ones(len(J), dtype=bool), Tk[k, J], b).sum() for k in range(len(KGRID))]
            k_prev = int(np.argmin(costs))
        chosen.append(KGRID[k_prev])
        T[D] = Tk[k_prev, D]
    return T, chosen


def run_fallback2(tr, T_ai, T_b, b, dec_mask, W, eps, H):
    """조각별 T를 쓰는 두 정책 사이의 그림자 장부 전환(paperexp.run_fallback의 배열 기준판)"""
    ts, g, h, r = tr.ts, tr.g, tr.h, tr.r
    idx = np.where(dec_mask)[0]
    idx = idx[np.argsort(ts[idx], kind="stable")]
    ea = pe.ledger_events(ts[idx], g[idx], h[idx], r[idx], T_ai[idx], b)
    eb = pe.ledger_events(ts[idx], g[idx], h[idx], r[idx], T_b[idx], b)
    A = pe.window_sums(ea[0], ea[1], ts[idx], W)
    B = pe.window_sums(eb[0], eb[1], ts[idx], W)
    want = A > (1.0 + eps) * B
    if H <= 0:
        mode = want
    else:
        mode = np.zeros(len(idx), dtype=bool)
        cur, last = False, -math.inf
        for i, t in enumerate(ts[idx]):
            if want[i] != cur and t - last >= H:
                cur, last = want[i], t
            mode[i] = cur
    T = T_ai.copy()
    T[idx] = np.where(mode, T_b[idx], T_ai[idx])
    return T, mode, idx


# ---------------------------------------------------------------- 묶음 만들기
def rule_groups(tr, b, Cva, pos_va, is1, i_fix, i_fix1, i_fix2):
    """paperexp.within의 2특징 규칙 선택(검증 교차 적합)을 특징 수에 맞게 일반화. 반환: 조각별 묶음 번호, 묶음 수, 조합 이름"""
    X, VA = tr.X, tr.VA
    nf = X.shape[1]
    Xv = X[VA]
    best = None
    for fs in [(i,) for i in range(nf)] + list(itertools.combinations(range(nf), 2)):
        gid = np.zeros(VA.sum(), dtype=np.int64)
        edges = []
        for f in fs:
            x = np.nan_to_num(Xv[:, f], nan=-1e9)
            e = pe.quant_edges(x, 4)
            edges.append(e)
            gid = gid * 8 + pe.bin_ids(e, x)
        uniq, gc = np.unique(gid, return_inverse=True)
        cv = 0.0
        for fit_m, ev_m, fb in ((is1, ~is1, i_fix1), (~is1, is1, i_fix2)):
            idx = pe.fit_groups(Cva[:, fit_m], gc[fit_m], len(uniq), fb)
            cv += Cva[idx[gc[ev_m]], np.where(ev_m)[0]].sum()
        if best is None or cv < best[0]:
            best = (cv, fs, edges)
    _, fs, edges = best
    gid_all = np.zeros(len(tr.g), dtype=np.int64)
    for f, e in zip(fs, edges):
        gid_all = gid_all * 8 + pe.bin_ids(e, np.nan_to_num(X[:, f], nan=-1e9))
    uniq, grp = np.unique(gid_all, return_inverse=True)
    return grp, len(uniq), "×".join(tr.feat_names[f] for f in fs), fs


def score_groups(p, VA, nbins=10):
    e = pe.quant_edges(p[VA], nbins)
    return pe.bin_ids(e, p), len(e) + 1


# ---------------------------------------------------------------- 한 트레이스·한 T*
def surv_kappa(tr, led, b, Sk, dec, W_cal, U):
    """생존 곡선 → 조각별 기대 비용 최소 T(κ=1) → κ를 장부로 온라인 선택한 T"""
    rec = pe.survival_timeouts(Sk, b, 1.0)
    Tk = np.stack([np.minimum(rec * k, tcap(b)) for k in KGRID])
    return online_kappa_T(tr, led, b, Tk, dec, W_cal, U)


def evaluate(tr, led, b, models, W_cal, U, seed=0, ablate=False):
    """models: dict(clf_p=조각별 P(g>b), surv=생존 곡선 n×len(TAU), [surv_f10]). 반환: 정책별 시험 비용·분해, 부가 정보
    ablate=True면 특징을 통제한 AI 변형 두 개를 더한다(탐색):
      AI생존κ-2특징: 규칙이 그 T*에서 고른 특징 1~2개만으로 학습한 생존 모델(같은 정보, 다른 모델링)
      AI생존κ-f10: 입력 패턴 특징 5개를 뺀 생존 모델(models['surv_f10'])"""
    VA, VA1, TE = tr.VA, tr.VA1, tr.TE
    h = tr.h
    dec = VA | TE
    gv, rv = tr.gva, tr.rva
    Cva = pe.cost_matrix(gv[VA], h[VA], rv[VA], b)
    pos_va = np.where(VA)[0]
    is1 = VA1[pos_va]
    i_fix = int(np.argmin(Cva.sum(1)))
    i_fix1 = int(np.argmin(Cva[:, is1].sum(1)))
    i_fix2 = int(np.argmin(Cva[:, ~is1].sum(1)))
    cap = tcap(b)
    pols, info = {}, {}
    # 정적 참고
    pols["고정(정적)"] = np.full(len(tr.g), pe.GRID[i_fix])
    p = models["clf_p"]
    gs, Gs = score_groups(p, VA)
    idx = pe.fit_groups(Cva, gs[VA], Gs, i_fix)
    pols["AI분류(정적)"] = pe.GRID[idx[gs]]
    # 온라인: 묶는 방법만 다르다
    pols["고정-온라인"] = online_group_T(tr, led, b, np.zeros(len(tr.g), dtype=np.int64), 1, dec, W_cal, U)
    grp_r, G_r, rname, rfs = rule_groups(tr, b, Cva, pos_va, is1, i_fix, i_fix1, i_fix2)
    info["규칙 조합"] = rname
    pols["규칙-온라인"] = online_group_T(tr, led, b, grp_r, G_r, dec, W_cal, U)
    pols["AI분류-온라인"] = online_group_T(tr, led, b, gs, Gs, dec, W_cal, U)
    Sk = models["surv"]
    rec = pe.survival_timeouts(Sk, b, 1.0)
    knots = np.concatenate([[0.0], pe.TAU])
    grp_s = np.searchsorted(knots, rec * (1 - 1e-9))
    pols["AI생존그룹-온라인"] = online_group_T(tr, led, b, grp_s, len(knots), dec, W_cal, U)
    Tk = np.stack([np.minimum(rec * k, cap) for k in KGRID])
    pols["AI생존κ-온라인"], kch = online_kappa_T(tr, led, b, Tk, dec, W_cal, U)
    info["κ 중앙값"] = float(np.median(kch)) if kch else math.nan
    names_fb = ["규칙-온라인", "AI분류-온라인", "AI생존그룹-온라인", "AI생존κ-온라인"]
    if ablate:
        m = tr.TR
        Sk2 = pe.fit_survival(tr.X[m][:, list(rfs)], tr.gtr[m], h[m] / h[m].mean(), tr.X[:, list(rfs)], seed)
        pols["AI생존κ-2특징-온라인"], _ = surv_kappa(tr, led, b, Sk2, dec, W_cal, U)
        pols["AI생존κ-f10-온라인"], _ = surv_kappa(tr, led, b, models["surv_f10"], dec, W_cal, U)
        names_fb += ["AI생존κ-2특징-온라인", "AI생존κ-f10-온라인"]
    # 대체 장치(기준 = 고정-온라인)
    base = pols["고정-온라인"]
    for name in names_fb:
        T_fb, mode, idx_fb = run_fallback2(tr, pols[name], base, b, dec, **FB)
        pols[name + "+대체"] = T_fb
        te_m = TE[idx_fb]
        info[name + " 기준 사용"] = float(mode[te_m].mean()) if te_m.any() else math.nan
    # 시험 구간 평가
    gt, ht, rt = tr.g[TE], h[TE], tr.r[TE]
    res = {}
    for name, T in pols.items():
        Tt = T[TE]
        assert not np.isnan(Tt).any(), name
        ref, w = pe.parts(gt, ht, rt, Tt, b)
        res[name] = dict(cost=ref + w / b, refault=ref, waste=w)
    orc = np.where(rt & (gt < b), np.inf, 0.0)
    ref, w = pe.parts(gt, ht, rt, orc, b)
    res["오라클"] = dict(cost=ref + w / b, refault=ref, waste=w)
    yt = ((tr.g > b) | ~tr.r)[TE]
    from sklearn.metrics import roc_auc_score
    info["auc"] = roc_auc_score(yt, p[TE], sample_weight=ht) if len(np.unique(yt)) == 2 else math.nan
    return res, info


def fit_models(tr_list_train, X_target, b_list, seed=0, weights="own", f10=False):
    """학습 구간 조각으로 분류기(b마다)와 생존 모델을 학습해 대상 조각 점수를 낸다.
    tr_list_train: 학습에 쓸 Trace2 목록(within이면 자기 자신 하나)"""
    Xs, gs, ws = [], [], []
    for t in tr_list_train:
        m = t.TR
        if m.sum() == 0:
            continue
        Xs.append(t.X[m])
        gs.append(t.gtr[m])
        hm = t.h[m]
        ws.append(hm / hm.mean() if len(tr_list_train) == 1 else hm / hm.sum() * 1000)
    X, g, w = np.vstack(Xs), np.concatenate(gs), np.concatenate(ws)
    out = {"surv": pe.fit_survival(X, g, w, X_target, seed)}
    if f10:  # 입력 패턴 특징(뒤 5개)을 뺀 생존 모델(특징 통제 비교용)
        out["surv_f10"] = pe.fit_survival(X[:, :10], g, w, X_target[:, :10], seed)
    for b in b_list:
        y = (g > b).astype(int)
        if len(np.unique(y)) < 2:
            out[b] = None
            continue
        clf = pe.make_clf(seed).fit(X, y, sample_weight=w)
        out[b] = clf.predict_proba(X_target)[:, 1]
    return out


# ---------------------------------------------------------------- 실행
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("schedules", nargs="+")
    ap.add_argument("--out", default="exp/res2")
    ap.add_argument("--tstars", default="0.3,1,3,10")
    ap.add_argument("--feats", default="f15", choices=["f10", "f15"])
    ap.add_argument("--wcal", type=float, default=900.0, help="온라인 보정 창(초)")
    ap.add_argument("--period", type=float, default=30.0, help="온라인 보정 주기 U(초)")
    ap.add_argument("--loto", action="store_true", help="사전 학습(다른 트레이스 학습 구간) 판도 잰다")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tstars = [float(x) for x in a.tstars.split(",")]
    t_start = time.time()
    traces = [Trace2(p, a.feats) for p in a.schedules]
    rows, irr = [], []
    for tr in traces:
        lo, end = tr.c1, tr.t0 + tr.dur
        m_all = irregularity(tr.ta, tr.sz, lo, end)
        m_va = irregularity(tr.ta, tr.sz, tr.c1, tr.c2)
        m_te = irregularity(tr.ta, tr.sz, tr.c2, end)
        drift = math.log2(m_te["rate"] / m_va["rate"]) if m_va["rate"] > 0 and m_te["rate"] > 0 else math.nan
        irr.append(dict(trace=tr.name, **{k: v for k, v in m_all.items()}, drift=drift,
                        size_cv_te=m_te["size_cv"], fano10_te=m_te["fano10"]))
    protos = [("within", None)] + ([("pre", "loto")] if a.loto else [])
    for proto, _ in protos:
        for k, tr in enumerate(traces):
            led = Ledger(tr)
            train_set = [tr] if proto == "within" else [t for i, t in enumerate(traces) if i != k]
            M = fit_models(train_set, tr.X, tstars, a.seed)
            for b in tstars:
                if M[b] is None:
                    print(f"[{proto} {tr.name}] T*={b:g}: 학습 정답이 한 종류뿐이라 건너뜀", flush=True)
                    continue
                if proto == "within" and len(np.unique((tr.gtr[tr.TR] > b).astype(int))) < 2:
                    continue
                res, info = evaluate(tr, led, b, {"clf_p": M[b], "surv": M["surv"]}, a.wcal, a.period, a.seed)
                b_on, b_st = res["고정-온라인"]["cost"], res["고정(정적)"]["cost"]
                te_dur = tr.t0 + tr.dur - tr.c2
                for name, v in res.items():
                    rows.append(dict(proto=proto, trace=tr.name, Tstar=b, policy=name, cost_MiB=v["cost"] / MiB,
                                     rel_fixon=(v["cost"] / b_on - 1) * 100 if b_on > 0 else 0.0,
                                     rel_fixst=(v["cost"] / b_st - 1) * 100 if b_st > 0 else 0.0,
                                     refault_MiB=v["refault"] / MiB, held_mean_MiB=v["waste"] / MiB / te_dur,
                                     auc=info["auc"], n_te=int(tr.TE.sum()), feats=a.feats, wcal=a.wcal, U=a.period,
                                     rule=info["규칙 조합"], kappa_med=info["κ 중앙값"],
                                     base_frac=info.get(name.replace("+대체", "") + " 기준 사용", "")
                                     if name.endswith("+대체") else ""))
                show = ["고정(정적)", "규칙-온라인", "규칙-온라인+대체", "AI분류-온라인+대체", "AI생존그룹-온라인+대체",
                        "AI생존κ-온라인+대체", "AI분류(정적)", "오라클"]
                print(f"[{proto} {tr.name}] T*={b:g} (고정-온라인 대비): " +
                      ", ".join(f"{n} {(res[n]['cost'] / b_on - 1) * 100:+.1f}%" for n in show), flush=True)
    name = f"_{a.tag}" if a.tag else ""
    with open(os.path.join(a.out, f"res2{name}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(a.out, f"irregularity{name}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(irr[0]))
        w.writeheader()
        w.writerows(irr)
    print(f"끝 ({time.time() - t_start:.0f}s) → {a.out}/res2{name}.csv, irregularity{name}.csv")


if __name__ == "__main__":
    main()
