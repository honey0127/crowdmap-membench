#!/usr/bin/env python3
"""paperexp.py — 논문용 시뮬레이션 재실험(유체 모델): 학습/검증/시험 분리, 트레이스 단위 일반화, PSK, 대체 장치 민감도.

프로토콜
  - 트레이스마다 시간 순으로 학습 50% / 검증 25% / 시험 25%로 나눈다. 학습·검증 구간 끝 300초(엠바고) 안에 놀기 시작한
    조각은 그 구간에서 뺀다.
  - 학습·검증 구간의 결과(놀이 시간)는 그 구간 끝까지 관찰된 것만 쓴다. 끝까지 다시 쓰이지 않은 조각은 '그때까지 놀았고
    나중에 다시 쓰인다'로 둔다. 그래서 학습·검증에서 고르는 타임아웃은 0~300초로 제한한다(이 범위에서는 이 가정이 선택을
    치우치게 하지 않는다: 모든 후보가 같은 재확보 비용을 낸다).
  - 분류기(그래디언트 부스팅)만 학습 구간에서 학습한다. 결정 매개변수(고정 타임아웃, 2특징 규칙의 조합·칸별 타임아웃,
    AI 점수 칸별 타임아웃, PSK의 λ, 대체 장치의 W·ε·H)는 모두 검증 구간에서 정한다. 검증 구간 안의 선택은 검증 구간을 시간
    순 두 쪽으로 나눠 서로 맞춘 값(교차 적합)으로 평가해 과적합을 막는다. 시험 구간은 마지막에 한 번만 평가한다.
  - 특징은 그 트레이스의 학습 구간 평균(유입률, 사용량)으로 나눠 규모를 없앤다.
  - 트레이스 단위 일반화(LOTO): 한 트레이스를 빼고 나머지 트레이스의 학습 구간으로 분류기를, 검증 구간으로 칸별 타임아웃을
    정해 뺀 트레이스의 시험 구간에서 평가한다(무보정). 뺀 트레이스의 검증 구간으로 칸별 타임아웃만 다시 고른 경우(국소 보정)도 잰다.
비용(유체 모델): 조각마다 재확보(0/1) × 두께 + 낭비(두께 × 초) / T*. 기준 = 검증 구간으로 고른 고정 타임아웃.
정책: 고정 타임아웃 / 2특징 규칙(+대체) / AI = 'g > T*' 분류기 + 점수 10칸별 타임아웃(+대체) /
      AI 생존 = 매듭 τ마다 'g > τ' 분류기로 생존 곡선을 예측하고 조각마다 기대 비용 최소 T, 배율 κ는 검증에서(+대체)
      [첫 결과를 본 뒤 추가한 탐색적 변형] / PSK 결정적·무작위(λ는 검증에서) / 오라클.

사용
  python3 paperexp.py exp/sched/*.csv --out exp/res [--tstars 0.3,1,3,10] [--no-loto]
출력: <out>/within.csv, <out>/within_parts.csv(재확보량·낭비 분해, 즉시 반납·반납 안 함 참고값),
      <out>/fallback.csv, <out>/loto.csv, 화면 요약
필요: numpy, scipy, scikit-learn, policysim.py·ailearn.py(같은 폴더)
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
import ailearn as al  # noqa: E402
import policysim as ps  # noqa: E402

MiB = ps.MiB
EMB = 300.0  # 엠바고(초), 학습·검증에서 고르는 타임아웃의 상한
SURV_CACHE = {}
GRID = np.concatenate([[0.0], np.logspace(-3, math.log10(EMB), 101)])
LAMS = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
FB_W, FB_EPS, FB_H = [60.0, 300.0, 900.0], [0.0, 0.05], [0.0, 60.0]
FEATS = ["높이", "사용량", "수준", "유입1s", "유입10s", "유입60s", "도착수1s", "이전놀이", "최근10s최고-수준", "최근60s최고-수준"]


# ---------------------------------------------------------------- 데이터
class Trace:
    def __init__(self, path):
        self.name = os.path.basename(path).removesuffix(".csv")
        ta, tf, sz = ps.load_schedule(path)
        S = ps.slices(ta, tf, sz)
        self.g, self.h, self.r, self.ts = S["g"], S["h"], S["reused"], S["ts"]
        self.t0, self.dur = S["t0"], S["dur"]
        self.c1 = self.t0 + 0.50 * self.dur
        self.c2 = self.t0 + 0.75 * self.dur
        self.TR = self.ts < self.c1 - EMB
        self.VA = (self.ts >= self.c1) & (self.ts < self.c2 - EMB)
        self.TE = self.ts >= self.c2
        mid_va = (self.c1 + self.c2 - EMB) / 2
        self.VA1 = self.VA & (self.ts < mid_va)
        self.VA2 = self.VA & (self.ts >= mid_va)
        self.X = self._features(S, ta, tf, sz)
        # 구간 끝까지 관찰된 결과
        self.gtr, self.rtr = self._observed(self.c1)
        self.gva, self.rva = self._observed(self.c2)

    def _observed(self, cut):
        done = self.r & (self.ts + self.g <= cut)
        g_obs = np.where(done, self.g, np.maximum(cut - self.ts, 0.0))
        r_obs = np.where(done, True, True)  # 다시 안 쓰인 조각은 나중에 다시 쓰인다고 둔다(타임아웃 ≤ 엠바고에서는 선택에 영향 없음)
        return g_obs, r_obs

    def _features(self, S, ta, tf, sz):
        ts = S["ts"]
        o = np.argsort(ta)
        tsorted, cum = ta[o], np.concatenate([[0.0], np.cumsum(sz[o])])

        def rate(W):
            hi = np.searchsorted(tsorted, ts, side="right")
            lo = np.searchsorted(tsorted, ts - W, side="right")
            return (cum[hi] - cum[lo]) / W / MiB, (hi - lo) / W

        r1, c1 = rate(1.0)
        r10, _ = rate(10.0)
        r60, _ = rate(60.0)
        level = (S["D0"] + S["pos"]) / MiB
        t0, b, binmax = al.d_history(ta, tf, sz)
        kb = np.clip(((ts - t0) / b).astype(np.int64) - 1, 0, len(binmax) - 1)
        m10 = al.trailing_max(binmax, int(10 / b))[kb] / MiB
        m60 = al.trailing_max(binmax, int(60 / b))[kb] / MiB
        # 학습 구간 평균(정규화 상수): 유입률, 도착 수, 사용량
        c1t = self.t0 + 0.5 * self.dur
        msk = (ta >= self.t0) & (ta < c1t)
        mrate = max(sz[msk].sum() / MiB / (c1t - self.t0), 1e-3)
        mcnt = max(msk.sum() / (c1t - self.t0), 1e-3)
        nb = int((c1t - t0) / b)
        mD = max(binmax[:max(nb, 1)].mean() / MiB, 0.25)
        pg = S["prevgap"]
        cols = [S["pos"] / MiB / mD, S["D0"] / MiB / mD, level / mD, r1 / mrate, r10 / mrate, r60 / mrate, c1 / mcnt,
                np.where(np.isnan(pg), np.nan, np.log10(np.maximum(pg, 1e-6))), (m10 - level) / mD, (m60 - level) / mD]
        return np.column_stack(cols)


# ---------------------------------------------------------------- 비용
def cost(g, h, r, T, b):
    return h * np.minimum(g, T) / b + h * (r & (g > T))


def cost_matrix(g, h, r, b, grid=GRID):
    Tg = grid[:, None]
    return h[None, :] * np.minimum(g[None, :], Tg) / b + h[None, :] * (r[None, :] & (g[None, :] > Tg))


def parts(g, h, r, T, b, rand=False):
    """비용을 재확보량(바이트)과 낭비(바이트·초)로 나눈다. rand=True면 T를 PSK 무작위의 K로 보고 기댓값을 쓴다."""
    if rand:
        D = np.expm1(T / b)
        x = np.minimum(g, T)
        ex = np.exp(x / b)
        F = (ex - 1) / D
        Etau = (ex * (x - b) + b) / D
        return (h * r * F).sum(), (h * (Etau + np.where(g <= T, (1 - F) * g, 0.0))).sum()
    return (h * (r & (g > T))).sum(), (h * np.minimum(g, T)).sum()


def fit_groups(C, gid, G, fallback_idx):
    """그룹별로 비용 합이 최소인 타임아웃 인덱스(C: nT × n, gid: 0..G-1)"""
    onehot = np.zeros((len(gid), G))
    onehot[np.arange(len(gid)), gid] = 1.0
    S = C @ onehot  # nT × G
    idx = np.argmin(S, axis=0)
    empty = onehot.sum(axis=0) == 0
    idx[empty] = fallback_idx
    return idx


def psk_rand_cost(g, h, r, K, b):
    """연속 시간 무작위 PSK: 반납 시각 τ ~ e^{τ/b}/(b(e^{K/b}-1)), τ∈[0,K]의 기대 비용"""
    D = np.expm1(K / b)
    x = np.minimum(g, K)
    ex = np.exp(x / b)
    F = (ex - 1) / D
    Etau = (ex * (x - b) + b) / D
    return h * (Etau / b + r * F + np.where(g <= K, (1 - F) * g / b, 0.0))


def bin_ids(edges, x):
    return np.searchsorted(edges, x, side="right")


def quant_edges(x, n):
    return np.unique(np.quantile(x, np.linspace(0, 1, n + 1)[1:-1])) if len(x) else np.array([])


# ---------------------------------------------------------------- 대체 장치
def ledger_events(ts, g, h, r, T, b):
    """정책 T(조각별)를 따를 때의 비용 발생 시각과 크기(그림자 장부)"""
    hold = np.minimum(g, T)
    t_w, a_w = ts + hold, h * hold / b
    ref = r & (g > T)
    return np.concatenate([t_w, (ts + g)[ref]]), np.concatenate([a_w, h[ref]])


def window_sums(ev_t, ev_a, t_dec, W):
    o = np.argsort(ev_t)
    et, cum = ev_t[o], np.concatenate([[0.0], np.cumsum(ev_a[o])])
    hi = np.searchsorted(et, t_dec, side="right")
    lo = np.searchsorted(et, t_dec - W, side="right")
    return cum[hi] - cum[lo]


def run_fallback(ts, g, h, r, T_ai, T_b, b, W, eps, H, dec_mask, cutoff=math.inf):
    """dec_mask 조각들(시간 순)에 대해 AI/기준 중 하나를 고른다. 장부는 cutoff 이전 사건만 쓴다(검증용 절단)."""
    idx = np.where(dec_mask)[0]
    idx = idx[np.argsort(ts[idx])]
    # 장부는 결정 대상 조각만으로 만든다(두 정책이 같은 결정을 내리는 구간의 조각은 비교에 섞지 않는다)
    ev_ai_t, ev_ai_a = ledger_events(ts[idx], g[idx], h[idx], r[idx], T_ai[idx], b)
    ev_b_t, ev_b_a = ledger_events(ts[idx], g[idx], h[idx], r[idx], np.full(len(idx), T_b), b)
    keep_ai, keep_b = ev_ai_t <= cutoff, ev_b_t <= cutoff
    A = window_sums(ev_ai_t[keep_ai], ev_ai_a[keep_ai], ts[idx], W)
    B = window_sums(ev_b_t[keep_b], ev_b_a[keep_b], ts[idx], W)
    want_base = A > (1.0 + eps) * B
    mode = np.zeros(len(idx), dtype=bool)
    cur, last = False, -math.inf
    tdec = ts[idx]
    if H <= 0:
        mode = want_base
        switches = int(np.count_nonzero(np.diff(mode.astype(np.int8)))) + int(mode[0]) if len(mode) else 0
    else:
        switches = 0
        for i in range(len(idx)):
            if want_base[i] != cur and tdec[i] - last >= H:
                cur, last = want_base[i], tdec[i]
                switches += 1
            mode[i] = cur
    T = np.where(mode, T_b, T_ai[idx])
    return idx, T, mode, switches


def switch_latency(ts, g, h, r, T_ai, T_b, b, idx, mode, t_start, t_end, bucket=60.0):
    """사후에 AI가 기준보다 5% 넘게 나빴던 1분 칸이 2개 이상 이어진 구간마다, 대체 장치가 켜질 때까지 걸린 시간"""
    nb = int(math.ceil((t_end - t_start) / bucket))
    k = np.clip(((ts[idx] - t_start) / bucket).astype(int), 0, nb - 1)
    ca = np.bincount(k, cost(g[idx], h[idx], r[idx], T_ai[idx], b), minlength=nb)
    cb = np.bincount(k, cost(g[idx], h[idx], r[idx], T_b, b), minlength=nb)
    losing = (ca > 1.05 * cb) & (ca - cb > 0.002 * max(cb.sum(), 1e-9))
    lat, miss = [], 0
    i = 0
    while i < nb:
        if losing[i]:
            j = i
            while j + 1 < nb and losing[j + 1]:
                j += 1
            if j > i:  # 2칸 이상 지속
                s, e = t_start + i * bucket, t_start + (j + 1) * bucket
                on = (ts[idx] >= s) & (ts[idx] < e) & mode
                if on.any():
                    lat.append(float(ts[idx][on].min() - s))
                else:
                    lat.append(float(e - s))
                    miss += 1
            i = j + 1
        else:
            i += 1
    return lat, miss


def fallback_select(tr, T_pol, T_fix, b):
    """정책 T_pol(조각별)에 대체 장치를 붙인다. 검증 구간 비용으로 (W, ε, H)를 고르고 시험 구간 결과를 돌려준다."""
    VA, TE, h = tr.VA, tr.TE, tr.h
    best, sens = None, []
    for W, eps, H in itertools.product(FB_W, FB_EPS, FB_H):
        iv, Tv, _, _ = run_fallback(tr.ts, tr.gva, h, tr.rva, T_pol, T_fix, b, W, eps, H, VA, cutoff=tr.c2)
        cv = cost(tr.gva[iv], h[iv], tr.rva[iv], Tv, b).sum()
        it, Tt, mt, sw = run_fallback(tr.ts, tr.g, h, tr.r, T_pol, T_fix, b, W, eps, H, VA | TE)
        te_m = TE[it]
        ct = cost(tr.g[it][te_m], h[it][te_m], tr.r[it][te_m], Tt[te_m], b).sum()
        lat, miss = switch_latency(tr.ts, tr.g, h, tr.r, T_pol, T_fix, b, it[te_m], mt[te_m], tr.c2, tr.t0 + tr.dur)
        T_full = np.full(len(tr.g), np.nan)
        T_full[it[te_m]] = Tt[te_m]
        sens.append(dict(W=W, eps=eps, H=H, cost=ct, val=cv, _T=T_full[TE],
                         switches=int(np.count_nonzero(np.diff(mt[te_m].astype(np.int8)))),
                         base_frac=float(mt[te_m].mean()) if te_m.any() else 0.0, runs=len(lat),
                         lat_med=float(np.median(lat)) if lat else math.nan, miss=miss))
        if best is None or cv < best[0]:
            best = (cv, len(sens) - 1)
    return sens[best[1]], sens


# ---------------------------------------------------------------- 생존(다시 쓰일 때까지 시간 분포) 모델
TAU = np.array([0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0])


def fit_survival(Xtr, gtr, wtr, Xall, seed=0):
    """매듭 τ마다 'g > τ' 확률을 그래디언트 부스팅으로 예측하고 단조 감소로 맞춘다 → (n × len(TAU))"""
    from sklearn.ensemble import HistGradientBoostingClassifier

    Sk = np.empty((len(Xall), len(TAU)))
    for k, tau in enumerate(TAU):
        y = (gtr > tau).astype(int)
        if len(np.unique(y)) < 2:
            Sk[:, k] = float(y.mean()) if len(y) else 0.0
            continue
        clf = HistGradientBoostingClassifier(max_iter=120, learning_rate=0.1, max_leaf_nodes=31, l2_regularization=1.0,
                                             random_state=seed).fit(Xtr, y, sample_weight=wtr)
        Sk[:, k] = clf.predict_proba(Xall)[:, 1]
    return np.minimum.accumulate(Sk, axis=1)


def survival_timeouts(Sk, b, kappa=1.0):
    """조각마다 기대 비용 E[min(g,T)]/b + P(g>T)를 최소로 하는 T(후보: 0과 매듭)"""
    knots = np.concatenate([[0.0], TAU])
    S = np.hstack([np.ones((len(Sk), 1)), Sk])  # S(0)=1
    area = np.concatenate([np.zeros((len(Sk), 1)), np.cumsum((S[:, 1:] + S[:, :-1]) / 2 * np.diff(knots), axis=1)], axis=1)
    ec = area / b + S  # T = 0이면 1(곧 다시 쓰이면 재확보)
    return knots[np.argmin(ec, axis=1)] * kappa


# ---------------------------------------------------------------- 분류기
def make_clf(seed=0):
    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, max_leaf_nodes=31, l2_regularization=1.0,
                                          random_state=seed)


def train_labels(tr, b):
    m = tr.TR
    y = (tr.gtr[m] > b).astype(int)
    return m, y


# ---------------------------------------------------------------- 트레이스 안 실험
def within(tr, b, seed=0):
    from sklearn.metrics import roc_auc_score

    out = {}
    VA, VA1, VA2, TE = tr.VA, tr.VA1, tr.VA2, tr.TE
    gv, rv, h = tr.gva, tr.rva, tr.h
    Cva = cost_matrix(gv[VA], h[VA], rv[VA], b)
    pos_va = np.where(VA)[0]
    is1 = VA1[pos_va]
    # 1) 고정 타임아웃(검증)
    i_fix = int(np.argmin(Cva.sum(1)))
    T_fix = GRID[i_fix]
    base = cost(tr.g[TE], h[TE], tr.r[TE], T_fix, b).sum()
    out["고정 타임아웃"] = dict(cost=base, param=f"T={T_fix:.3g}s", T=np.full(TE.sum(), T_fix))
    i_fix1 = int(np.argmin(Cva[:, is1].sum(1)))
    i_fix2 = int(np.argmin(Cva[:, ~is1].sum(1)))
    # 2) 2특징 규칙: 검증 두 쪽 교차 적합으로 조합 선택, 전체 검증으로 칸별 타임아웃 확정
    best = None
    sets = [(i,) for i in range(len(FEATS))] + list(itertools.combinations(range(len(FEATS)), 2))
    Xv = tr.X[VA]
    for fs in sets:
        gid = np.zeros(VA.sum(), dtype=np.int64)
        edges_list = []
        for f in fs:
            x = np.nan_to_num(Xv[:, f], nan=-1e9)
            e = quant_edges(x, 4)
            edges_list.append(e)
            gid = gid * 8 + bin_ids(e, x)
        uniq, gid_c = np.unique(gid, return_inverse=True)
        G = len(uniq)
        cv = 0.0
        for fit_m, ev_m, fb in ((is1, ~is1, i_fix1), (~is1, is1, i_fix2)):
            idx = fit_groups(Cva[:, fit_m], gid_c[fit_m], G, fb)
            cv += Cva[idx[gid_c[ev_m]], np.where(ev_m)[0]].sum()
        if best is None or cv < best[0]:
            best = (cv, fs, edges_list)
    _, fs, edges_list = best
    gid_v = np.zeros(VA.sum(), dtype=np.int64)
    gid_t = np.zeros(TE.sum(), dtype=np.int64)
    for f, e in zip(fs, edges_list):
        gid_v = gid_v * 8 + bin_ids(e, np.nan_to_num(Xv[:, f], nan=-1e9))
        gid_t = gid_t * 8 + bin_ids(e, np.nan_to_num(tr.X[TE][:, f], nan=-1e9))
    uniq = np.unique(gid_v)
    G = len(uniq)
    gv_c = np.searchsorted(uniq, gid_v)
    idx = fit_groups(Cva, gv_c, G, i_fix)
    gt_c = np.searchsorted(uniq, gid_t)
    known = (gt_c < G) & (uniq[np.minimum(gt_c, G - 1)] == gid_t)
    T_tab = np.where(known, GRID[idx[np.minimum(gt_c, G - 1)]], T_fix)
    out["2특징 규칙"] = dict(cost=cost(tr.g[TE], h[TE], tr.r[TE], T_tab, b).sum(), param="×".join(FEATS[f] for f in fs), T=T_tab)
    T_tab_all = np.full(len(tr.g), T_fix)
    T_tab_all[np.where(TE)[0]] = T_tab
    for fit_m, ev_m, fb in ((is1, ~is1, i_fix1), (~is1, is1, i_fix2)):  # 검증 구간: 두 쪽 교차 적합
        idx_h = fit_groups(Cva[:, fit_m], gv_c[fit_m], G, fb)
        T_tab_all[pos_va[ev_m]] = GRID[idx_h[gv_c[ev_m]]]
    # 3) 분류기(학습 구간)
    m, y = train_labels(tr, b)
    if len(np.unique(y)) < 2:
        return None
    clf = make_clf(seed).fit(tr.X[m], y, sample_weight=h[m] / h[m].mean())
    p = clf.predict_proba(tr.X)[:, 1]
    yt = ((tr.g > b) | ~tr.r)[TE]
    out["_auc"] = roc_auc_score(yt, p[TE], sample_weight=h[TE]) if len(np.unique(yt)) == 2 else math.nan
    # 4) AI 점수 10칸: 교차 적합(검증 두 쪽)으로 검증 결정, 전체 검증으로 시험용 확정
    def ai_fit(fit_m_va):
        e = quant_edges(p[pos_va][fit_m_va], 10)
        bv = bin_ids(e, p[pos_va])
        idx = fit_groups(Cva[:, fit_m_va], bv[fit_m_va], len(e) + 1, i_fix)
        return e, idx

    T_ai = np.full(len(tr.g), T_fix)
    for fit_m, ev_m in ((is1, ~is1), (~is1, is1)):  # 검증 구간 조각의 표본 밖 AI 결정(대체 장치 선택용)
        e, idx = ai_fit(fit_m)
        sel = pos_va[ev_m]
        T_ai[sel] = GRID[idx[bin_ids(e, p[sel])]]
    e, idx = ai_fit(np.ones(len(pos_va), dtype=bool))
    pos_te = np.where(TE)[0]
    T_ai[pos_te] = GRID[idx[bin_ids(e, p[pos_te])]]
    out["AI"] = dict(cost=cost(tr.g[TE], h[TE], tr.r[TE], T_ai[TE], b).sum(), param="", T=T_ai[TE])
    # 5) PSK 결정적·무작위: λ를 검증에서 고른다
    for kind in ("PSK 결정적", "PSK 무작위"):
        best = None
        for lam in LAMS:
            K = np.where(p >= 0.5, lam * b, b / lam)
            if kind == "PSK 결정적":
                cv = cost(gv[VA], h[VA], rv[VA], K[VA], b).sum()
            else:
                cv = psk_rand_cost(gv[VA], h[VA], rv[VA], K[VA], b).sum()
            if best is None or cv < best[0]:
                best = (cv, lam)
        lam = best[1]
        K = np.where(p >= 0.5, lam * b, b / lam)
        c = cost(tr.g[TE], h[TE], tr.r[TE], K[TE], b).sum() if kind == "PSK 결정적" else \
            psk_rand_cost(tr.g[TE], h[TE], tr.r[TE], K[TE], b).sum()
        out[kind] = dict(cost=c, param=f"λ={lam:g}", T=K[TE], rand=(kind == "PSK 무작위"))
    # 6) 대체 장치: 검증에서 (W, ε, H)를 고르고 시험에서 평가. 모든 조합의 시험 결과는 민감도로 따로 남긴다
    s, sens = fallback_select(tr, T_ai, T_fix, b)
    out["AI+대체"] = dict(cost=s["cost"], param=f"W={s['W']:g},ε={s['eps']:g},H={s['H']:g}", T=s["_T"])
    out["_fb_sens"] = sens
    s2, _ = fallback_select(tr, T_tab_all, T_fix, b)
    out["2특징+대체"] = dict(cost=s2["cost"], param=f"W={s2['W']:g},ε={s2['eps']:g},H={s2['H']:g}", T=s2["_T"])
    # 6b) AI 생존 모델(탐색적): 다시 쓰일 때까지 시간의 분포를 예측하고 조각마다 기대 비용 최소 T, κ는 검증에서 선택
    if not hasattr(tr, "_surv"):
        mtr = tr.TR
        tr._surv = fit_survival(tr.X[mtr], tr.gtr[mtr], h[mtr] / h[mtr].mean(), tr.X, seed)
    best = None
    for kappa in (0.5, 0.7, 1.0, 1.4, 2.0):
        Ts = survival_timeouts(tr._surv, b, kappa)
        cv = cost(gv[VA], h[VA], rv[VA], Ts[VA], b).sum()
        if best is None or cv < best[0]:
            best = (cv, kappa, Ts)
    _, kappa, T_sv = best
    out["AI 생존"] = dict(cost=cost(tr.g[TE], h[TE], tr.r[TE], T_sv[TE], b).sum(), param=f"κ={kappa:g}", T=T_sv[TE])
    s3, _ = fallback_select(tr, T_sv, T_fix, b)
    out["AI 생존+대체"] = dict(cost=s3["cost"], param=f"κ={kappa:g},W={s3['W']:g},ε={s3['eps']:g},H={s3['H']:g}",
                             T=s3["_T"])
    # 7) 오라클
    gt, ht, rt = tr.g[TE], h[TE], tr.r[TE]
    out["오라클"] = dict(cost=(ht * np.where(rt, np.minimum(gt / b, 1.0), 0.0)).sum(), param="",
                      T=np.where(rt & (gt < b), np.inf, 0.0))  # 다시 쓰이고 b보다 빨리 쓰이면 쥔다, 아니면 바로 반납
    out["즉시 반납(T=0)"] = dict(cost=cost(gt, ht, rt, 0.0, b).sum(), param="", T=np.zeros(TE.sum()))
    out["반납 안 함(T=∞)"] = dict(cost=cost(gt, ht, rt, np.inf, b).sum(), param="", T=np.full(TE.sum(), np.inf))
    out["_n"] = (int(tr.TR.sum()), int(VA.sum()), int(TE.sum()))
    out["_Tfix"] = T_fix
    return out


# ---------------------------------------------------------------- 트레이스 단위 일반화
def loto(traces, k, b, seed=0):
    test = traces[k]
    others = [t for i, t in enumerate(traces) if i != k]
    Xs, ys, ws = [], [], []
    for t in others:
        m, y = train_labels(t, b)
        if len(y) == 0:
            continue
        Xs.append(t.X[m])
        ys.append(y)
        ws.append(t.h[m] / t.h[m].sum())  # 트레이스마다 같은 무게
    y = np.concatenate(ys)
    if len(np.unique(y)) < 2:
        return None
    clf = make_clf(seed).fit(np.vstack(Xs), y, sample_weight=np.concatenate(ws) * len(ws) * 1000)
    # 무보정: 다른 트레이스 검증 구간으로 점수 칸·칸별 타임아웃, 고정 타임아웃
    ps_all, C_all = [], []
    for t in others:
        if t.VA.sum() == 0:
            continue
        pv = clf.predict_proba(t.X[t.VA])[:, 1]
        C = cost_matrix(t.gva[t.VA], t.h[t.VA], t.rva[t.VA], b)
        C_all.append(C / max(t.h[t.VA].sum(), 1.0))
        ps_all.append(pv)
    P, C = np.concatenate(ps_all), np.hstack(C_all)
    i_fix0 = int(np.argmin(C.sum(1)))
    e = quant_edges(P, 10)
    idx = fit_groups(C, bin_ids(e, P), len(e) + 1, i_fix0)
    pt = clf.predict_proba(test.X)[:, 1]
    TE = test.TE
    g, h, r = test.g[TE], test.h[TE], test.r[TE]
    res = {"AI 무보정": cost(g, h, r, GRID[idx[bin_ids(e, pt[TE])]], b).sum(),
           "고정 무보정": cost(g, h, r, GRID[i_fix0], b).sum()}
    # 국소 보정: 뺀 트레이스의 검증 구간으로 칸별 타임아웃·고정 타임아웃만 다시 고른다
    if test.VA.sum() == 0:
        return None
    Cv = cost_matrix(test.gva[test.VA], test.h[test.VA], test.rva[test.VA], b)
    i_fix = int(np.argmin(Cv.sum(1)))
    pv = pt[test.VA]
    e2 = quant_edges(pv, 10)
    idx2 = fit_groups(Cv, bin_ids(e2, pv), len(e2) + 1, i_fix)
    res["AI 국소 보정"] = cost(g, h, r, GRID[idx2[bin_ids(e2, pt[TE])]], b).sum()
    res["고정 국소"] = cost(g, h, r, GRID[i_fix], b).sum()
    # 국소 보정 + 대체 장치(뺀 트레이스의 검증 구간으로 W·ε·H 선택)
    pos_va = np.where(test.VA)[0]
    is1 = test.VA1[pos_va]
    T_loc = np.full(len(test.g), GRID[i_fix])
    for fit_m, ev_m in ((is1, ~is1), (~is1, is1)):
        e3 = quant_edges(pv[fit_m], 10)
        i3 = fit_groups(Cv[:, fit_m], bin_ids(e3, pv[fit_m]), len(e3) + 1, i_fix)
        T_loc[pos_va[ev_m]] = GRID[i3[bin_ids(e3, pv[ev_m])]]
    T_loc[np.where(TE)[0]] = GRID[idx2[bin_ids(e2, pt[TE])]]
    s_fb, _ = fallback_select(test, T_loc, GRID[i_fix], b)
    res["AI 국소 보정+대체"] = s_fb["cost"]
    # 생존 모델(다른 트레이스로 학습) + 국소 κ + 대체
    key = ("_surv_loto", k)
    if key not in SURV_CACHE:
        Xs2, gs2, ws2 = [], [], []
        for t in others:
            m2 = t.TR
            if m2.sum() == 0:
                continue
            Xs2.append(t.X[m2])
            gs2.append(t.gtr[m2])
            ws2.append(t.h[m2] / t.h[m2].sum() * 1000)
        SURV_CACHE[key] = fit_survival(np.vstack(Xs2), np.concatenate(gs2), np.concatenate(ws2), test.X, seed)
    Sk = SURV_CACHE[key]
    best = None
    for kappa in (0.5, 0.7, 1.0, 1.4, 2.0):
        Ts = survival_timeouts(Sk, b, kappa)
        cv = cost(test.gva[test.VA], test.h[test.VA], test.rva[test.VA], Ts[test.VA], b).sum()
        if best is None or cv < best[0]:
            best = (cv, Ts)
    T_sv = best[1]
    res["AI 생존 무보정"] = cost(g, h, r, survival_timeouts(Sk, b, 1.0)[TE], b).sum()
    s_sv, _ = fallback_select(test, T_sv, GRID[i_fix], b)
    res["AI 생존 국소+대체"] = s_sv["cost"]
    return res


# ---------------------------------------------------------------- 실행
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("schedules", nargs="+")
    ap.add_argument("--out", default="exp/res")
    ap.add_argument("--tstars", default="0.3,1,3,10")
    ap.add_argument("--no-loto", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tstars = [float(x) for x in a.tstars.split(",")]
    t0 = time.time()
    traces = [Trace(p) for p in a.schedules]
    print(f"트레이스 {len(traces)}개 준비 {time.time() - t0:.0f}s")
    pols = ["고정 타임아웃", "2특징 규칙", "2특징+대체", "AI", "AI+대체", "AI 생존", "AI 생존+대체", "PSK 결정적", "PSK 무작위", "오라클"]
    refs = ["즉시 반납(T=0)", "반납 안 함(T=∞)"]
    wrows, frows, prows = [], [], []
    for tr in traces:
        for b in tstars:
            R = within(tr, b, a.seed)
            if R is None:
                print(f"[{tr.name}] T*={b:g}: 학습 정답이 한 종류뿐이라 건너뜀")
                continue
            base = R["고정 타임아웃"]["cost"]
            line = []
            for pol in pols:
                c = R[pol]["cost"]
                rel = (c / base - 1) * 100 if base > 0 else 0.0
                wrows.append(dict(trace=tr.name, Tstar=b, policy=pol, cost_MiB=c / MiB, rel_pct=rel, param=R[pol]["param"],
                                  auc=R["_auc"], n_tr=R["_n"][0], n_va=R["_n"][1], n_te=R["_n"][2]))
                line.append(f"{pol} {rel:+.1f}%")
            TE = tr.TE
            te_dur = tr.t0 + tr.dur - tr.c2
            for pol in pols + refs:
                P = R[pol]
                ref_b, w_bs = parts(tr.g[TE], tr.h[TE], tr.r[TE], P["T"], b, P.get("rand", False))
                c_chk = ref_b + w_bs / b
                assert abs(c_chk - P["cost"]) <= 1e-6 * max(P["cost"], 1.0), (tr.name, b, pol, c_chk, P["cost"])
                prows.append(dict(trace=tr.name, Tstar=b, policy=pol, refault_MiB=ref_b / MiB, waste_MiBs=w_bs / MiB,
                                  held_mean_MiB=w_bs / MiB / te_dur, cost_MiB=P["cost"] / MiB,
                                  rel_pct=(P["cost"] / base - 1) * 100 if base > 0 else 0.0, test_s=te_dur))
            for s in R["_fb_sens"]:
                frows.append(dict(trace=tr.name, Tstar=b, rel_pct=(s["cost"] / base - 1) * 100,
                                  **{k: v for k, v in s.items() if k not in ("cost", "val", "_T")}))
            print(f"[{tr.name}] T*={b:g} (기준 {R['_Tfix']:.3g}s, AUC {R['_auc']:.2f}, n={R['_n']}): " + ", ".join(line[1:]),
                  flush=True)
    with open(os.path.join(a.out, "within.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(wrows[0]))
        w.writeheader()
        w.writerows(wrows)
    with open(os.path.join(a.out, "within_parts.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(prows[0]))
        w.writeheader()
        w.writerows(prows)
    with open(os.path.join(a.out, "fallback.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(frows[0]))
        w.writeheader()
        w.writerows(frows)
    if not a.no_loto:
        lrows = []
        for k, tr in enumerate(traces):
            for b in tstars:
                R = loto(traces, k, b, a.seed)
                if R is None:
                    continue
                base = R["고정 국소"]
                for pol, c in R.items():
                    lrows.append(dict(trace=tr.name, Tstar=b, policy=pol, cost_MiB=c / MiB,
                                      rel_local_pct=(c / base - 1) * 100 if base > 0 else 0.0,
                                      rel_zeroshot_pct=(c / R["고정 무보정"] - 1) * 100 if R["고정 무보정"] > 0 else 0.0))
                print(f"[LOTO {tr.name}] T*={b:g}: " + ", ".join(f"{p} {(c / base - 1) * 100:+.1f}%" for p, c in R.items()),
                      flush=True)
        with open(os.path.join(a.out, "loto.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(lrows[0]))
            w.writeheader()
            w.writerows(lrows)
    print(f"끝 ({time.time() - t0:.0f}s) → {a.out}/within.csv, fallback.csv" + ("" if a.no_loto else ", loto.csv"))


if __name__ == "__main__":
    main()
