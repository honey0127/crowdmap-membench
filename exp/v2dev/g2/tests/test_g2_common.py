#!/usr/bin/env python3
"""test_g2_common.py — G2W3 이진 형식 단위 시험 + 합성 데이터로 C++ 대칭 검증.

Python 참조 구현(이 파일 안에 인라인)으로 합성 창과 합성 모델을 만들고,
g2_common으로 쓴 뒤 g2_harness(parity)로 처리하고, 결과를 비교한다.

실행:
  python3 exp/v2dev/g2/tests/test_g2_common.py
"""
import math
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
G2_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(G2_DIR))
import g2_common as gc

HARNESS = G2_DIR / "g2_harness"


# ---------------------------------------------------------------- Python 참조 구현
def _observed(J, t, cap, ts, g, r):
    age = t - ts[J]
    done = r[J] & (g[J] <= age)
    return np.where(done, g[J], np.maximum(age, cap) * (1.0 + 1e-9) + 1e-9)


def _ledger_window(tss, idx, lo_t, hi_t):
    lo = np.searchsorted(tss, lo_t, side="left")
    hi = np.searchsorted(tss, hi_t, side="left")
    return idx[lo:hi]


def _ledger_events(ts, g, h, r, T_arr, b):
    hold = np.minimum(g, T_arr)
    t_w, a_w = ts + hold, h * hold / b
    ref = r & (g > T_arr)
    return (np.concatenate([t_w, (ts + g)[ref]]),
            np.concatenate([a_w, h[ref]]))


def _run_fallback3_py(n, g, h, ts, r, T_ai, T_b, dec_mask, b, W):
    idx = np.where(dec_mask)[0]
    idx = idx[np.argsort(ts[idx], kind="stable")]
    Ta, Tb = T_ai[idx], T_b[idx]
    m = Ta != Tb
    if m.any():
        t_a, a_a = _ledger_events(ts[idx][m], g[idx][m], h[idx][m], r[idx][m], Ta[m], b)
        t_b_, a_b = _ledger_events(ts[idx][m], g[idx][m], h[idx][m], r[idx][m], Tb[m], b)
        ev_t = np.concatenate([t_a, t_b_])
        ev_a = np.concatenate([a_a, -a_b])
    else:
        ev_t = ev_a = np.array([])
    if len(ev_t):
        o = np.argsort(ev_t, kind="stable")
        et = ev_t[o]
        cum = np.concatenate([[0.0], np.cumsum(ev_a[o])])
    else:
        et = np.array([])
        cum = np.array([0.0])
    t_dec = ts[idx]
    D = (cum[np.searchsorted(et, t_dec, side="right")]
         - cum[np.searchsorted(et, t_dec - W, side="right")])
    mode = D > 0
    T = T_ai.copy()
    T[idx] = np.where(mode, Tb, Ta)
    return T, mode, idx


def _online_kappa_T_py(n, g, h, ts, r, Tk, dec_mask, cap, W_cal, U, b):
    T = np.full(n, np.nan)
    dec = np.where(dec_mask)[0]
    if len(dec) == 0:
        return T
    t0, t1 = ts[dec].min(), ts[dec].max()
    k_prev = gc.INITIAL_K
    sort_idx = np.argsort(ts, kind="stable")
    tss = ts[sort_idx]
    n_steps = max(0, math.ceil((t1 + U - t0) / U))
    for ki in range(n_steps):
        u = t0 + ki * U
        D_all = _ledger_window(tss, sort_idx, u, u + U)
        D = D_all[dec_mask[D_all]]
        if len(D) == 0:
            continue
        J = _ledger_window(tss, sort_idx, u - cap - W_cal, u - cap)
        if len(J):
            g_obs = _observed(J, u, cap, ts, g, r)
            costs = np.array([
                (h[J] * np.minimum(g_obs, Tk[k, J]) / b
                 + h[J] * (g_obs > Tk[k, J])).sum()
                for k in range(len(gc.KGRID))
            ])
            k_prev = int(np.argmin(costs))
        T[D] = Tk[k_prev, D]
    return T


def _online_group_T_py(n, g, h, ts, r, grp, G, dec_mask, grid, cap, W_cal, U, b):
    T = np.full(n, np.nan)
    dec = np.where(dec_mask)[0]
    if len(dec) == 0:
        return T
    t0, t1 = ts[dec].min(), ts[dec].max()
    prev = np.zeros(G, dtype=np.int64)
    sort_idx = np.argsort(ts, kind="stable")
    tss = ts[sort_idx]
    n_grid = len(grid)
    n_steps = max(0, math.ceil((t1 + U - t0) / U))
    MIN_FRAC, MIN_N = 0.02, 20

    for ki in range(n_steps):
        u = t0 + ki * U
        D_all = _ledger_window(tss, sort_idx, u, u + U)
        D = D_all[dec_mask[D_all]]
        if len(D) == 0:
            continue
        J = _ledger_window(tss, sort_idx, u - cap - W_cal, u - cap)
        if len(J):
            g_obs = _observed(J, u, cap, ts, g, r)
            Tg = grid[:, None]
            C = (h[J][None, :] * np.minimum(g_obs[None, :], Tg) / b
                 + h[J][None, :] * (g_obs[None, :] > Tg))
            gj = grp[J]
            ip = int(np.argmin(C.sum(1)))
            S = np.zeros((n_grid, G))
            for ig in range(n_grid):
                np.add.at(S[ig], gj, C[ig])
            table_g = np.argmin(S, axis=0)
            w = np.bincount(gj, weights=h[J], minlength=G)
            cnt = np.bincount(gj, minlength=G)
            table_g[(w < MIN_FRAC * w.sum()) | (cnt < MIN_N)] = ip
            prev = table_g
        T[D] = grid[prev[grp[D]]]
    return T


def _encode_one_py(x, cuts, sentinel):
    if np.isnan(x):
        return sentinel
    return int(np.searchsorted(cuts, x, side="right"))


def _compute_Tk_py(X_p1b, cuts_list, table_np, knots, cap, sentinel=64):
    n = len(X_p1b)
    K = len(gc.KGRID)
    kgrid = np.array(gc.KGRID)
    Tk = np.zeros((K, n))
    for j in range(n):
        c0 = _encode_one_py(X_p1b[j, 0], cuts_list[0], sentinel)
        c1 = _encode_one_py(X_p1b[j, 1], cuts_list[1], sentinel)
        c2 = _encode_one_py(X_p1b[j, 2], cuts_list[2], sentinel)
        idx_lut = int(table_np[c0, c1, c2])
        rec = knots[idx_lut]
        Tk[:, j] = np.minimum(rec * kgrid, cap)
    return Tk


# ---------------------------------------------------------------- G2W3 Python 참조 구현
def _compute_grp_r_py(X_rule, rule_edges_list):
    """C++ compute_grp_r의 Python 복제 (bin_ids=upper_bound, unique-remap)."""
    n = len(X_rule)
    n_rf = len(rule_edges_list)
    gid = np.zeros(n, dtype=np.int64)
    for rf in range(n_rf):
        edges = np.asarray(rule_edges_list[rf])
        bids = np.searchsorted(edges, X_rule[:, rf], side='right').astype(np.int64)
        gid = gid * 8 + bids
    uniq = np.sort(np.unique(gid))
    grp_r = np.searchsorted(uniq, gid).astype(np.int32)
    return grp_r, len(uniq)


def _compute_p1b_features_py(pos, D0, mD):
    """C++ compute_p1b_features의 Python 복제."""
    n = len(pos)
    MiB = gc.MiB
    X = np.zeros((n, 3))
    X[:, 0] = pos / MiB / mD
    X[:, 1] = D0 / MiB / mD
    X[:, 2] = (pos + D0) / MiB / mD
    return X


# ---------------------------------------------------------------- 합성 모델 생성
def make_synthetic_model():
    cuts_list = [
        np.array([0.3, 0.7]),
        np.array([0.4, 0.8]),
        np.array([0.2, 0.6]),
    ]
    knots = np.array(gc.KNOTS, dtype=np.float64)
    table = np.zeros((gc.TABLE_DIM, gc.TABLE_DIM, gc.TABLE_DIM), dtype=np.uint8)
    for c0 in range(gc.TABLE_DIM):
        for c1 in range(gc.TABLE_DIM):
            for c2 in range(gc.TABLE_DIM):
                table[c0, c1, c2] = (c0 + c1 + c2) % len(knots)
    return dict(cuts_list=cuts_list, knots=knots, table=table, sentinel=64)


# ---------------------------------------------------------------- 합성 창 생성 (G2W3 형식)
def make_synthetic_window(model=None):
    """소규모 합성 창(n=20): G2W3 형식, pos/D0/mD/X_rule/rule_edges."""
    if model is None:
        model = make_synthetic_model()

    rng = np.random.default_rng(42)
    n = 20
    b = gc.TSTAR; cap = gc.CAP; W_cal = gc.W_CAL; U = gc.U_PERIOD; W_fb = gc.W_FB
    MiB = gc.MiB

    ts = np.sort(rng.uniform(0.0, 500.0, n))
    g  = rng.exponential(10.0, n)
    h  = rng.uniform(0.5, 2.0, n) * MiB
    r  = rng.integers(0, 2, n).astype(bool)

    dec_mask = np.zeros(n, dtype=bool)
    mid = n // 4
    dec_mask[mid:mid+15] = True
    te_mask = np.zeros(n, dtype=bool)
    te_mask[mid+10:mid+15] = True

    # G2W3: 규칙 데이터 (X_rule, rule_edges)
    n_rf = 2
    rule_edges_list = [np.array([0.3, 0.7]), np.array([0.4, 0.8])]
    rule_ne = np.array([len(e) for e in rule_edges_list], dtype=np.uint32)
    rule_edges_flat = np.concatenate(rule_edges_list)
    X_rule = rng.uniform(0.0, 1.0, (n, n_rf))

    # grp_r: Python 버전 — C++와 동일 알고리즘 (상단 _compute_grp_r_py)
    grp_r, G_r = _compute_grp_r_py(X_rule, rule_edges_list)

    # G2W3: pos/D0/mD (P1b 특징 원시 입력)
    pos = rng.uniform(0.0, 100.0, n) * MiB   # bytes
    D0  = rng.uniform(0.0,  50.0, n) * MiB   # bytes
    mD  = 10.0

    # P1b 특징: Python 버전 — C++와 동일 (상단 _compute_p1b_features_py)
    X_p1b = _compute_p1b_features_py(pos, D0, mD)

    GRID_FULL = np.concatenate([[0.0], np.logspace(-3, math.log10(300.0), 101)])
    grid = GRID_FULL[GRID_FULL <= cap * (1.0 + 1e-9)]

    grp_base = np.zeros(n, dtype=np.int32)
    T_base = _online_group_T_py(n, g, h, ts, r, grp_base, 1, dec_mask, grid, cap, W_cal, U, b)
    T_rule = _online_group_T_py(n, g, h, ts, r, grp_r, G_r, dec_mask, grid, cap, W_cal, U, b)
    T_rb, _, _ = _run_fallback3_py(n, g, h, ts, r, T_rule, T_base, dec_mask, b, W_fb)

    cuts_list = model["cuts_list"]
    table_np  = model["table"]
    knots     = model["knots"]
    Tk = _compute_Tk_py(X_p1b, cuts_list, table_np, knots, cap, model["sentinel"])

    T_ai = _online_kappa_T_py(n, g, h, ts, r, Tk, dec_mask, cap, W_cal, U, b)
    T_ai_fb, _, _ = _run_fallback3_py(n, g, h, ts, r, T_ai, T_base, dec_mask, b, W_fb)
    T_c3, _, _ = _run_fallback3_py(n, g, h, ts, r, T_ai_fb, T_rb, dec_mask, b, W_fb)

    te_idx = np.where(te_mask)[0]
    r_bool = r.astype(bool)
    R_MiB = float(h[te_idx][(r_bool[te_idx]) & (g[te_idx] > T_c3[te_idx])].sum()) / MiB
    H_MiB = float((h[te_idx] * np.minimum(g[te_idx], T_c3[te_idx])).sum()) / MiB

    return dict(
        n=n, G_r=G_r, n_grid=len(grid), rule_n_rf=n_rf,
        b=b, cap=cap, W_cal=W_cal, U=U, W_fb=W_fb, mD=mD,
        g=g, h=h, r=r.astype(np.uint8), ts=ts,
        dec_mask=dec_mask.astype(np.uint8), te_mask=te_mask.astype(np.uint8),
        pos=pos, D0=D0,
        X_rule=X_rule, rule_ne=rule_ne, rule_edges_flat=rule_edges_flat,
        rule_edges_list=rule_edges_list,
        grid=grid,
        T_base=T_base, T_rule=T_rule, T_rb=T_rb,
        T_ai=T_ai, T_ai_fb=T_ai_fb, T_c3=T_c3,
        R_MiB=R_MiB, H_MiB=H_MiB,
    )


# ================================================================ 단위 시험
class TestBinaryFormat(unittest.TestCase):
    """g2_common G2W3 이진 형식 왕복 (쓰기 → 읽기)."""

    def setUp(self):
        self.d = make_synthetic_window()
        self.tmp = tempfile.NamedTemporaryFile(suffix=".bin", delete=False)
        self.tmp.close()

    def tearDown(self):
        Path(self.tmp.name).unlink(missing_ok=True)

    def _write_read(self):
        d = self.d
        gc.write_window(
            self.tmp.name,
            d["n"], d["G_r"], d["n_grid"], d["rule_n_rf"],
            d["b"], d["cap"], d["W_cal"], d["U"], d["W_fb"], d["mD"],
            d["g"], d["h"], d["r"], d["ts"],
            d["dec_mask"], d["te_mask"],
            d["pos"], d["D0"],
            d["X_rule"], d["rule_ne"], d["rule_edges_flat"],
            d["grid"],
            d["T_base"], d["T_rule"], d["T_rb"],
            d["T_ai"], d["T_ai_fb"], d["T_c3"],
            d["R_MiB"], d["H_MiB"],
        )
        return gc.read_window(self.tmp.name)

    def test_scalars_roundtrip(self):
        d = self.d; w = self._write_read()
        self.assertEqual(w["n"],         d["n"])
        self.assertEqual(w["G_r"],       d["G_r"])
        self.assertEqual(w["n_grid"],    d["n_grid"])
        self.assertEqual(w["rule_n_rf"], d["rule_n_rf"])
        self.assertAlmostEqual(w["b"],     d["b"])
        self.assertAlmostEqual(w["cap"],   d["cap"])
        self.assertAlmostEqual(w["W_cal"], d["W_cal"])
        self.assertAlmostEqual(w["U"],     d["U"])
        self.assertAlmostEqual(w["W_fb"],  d["W_fb"])
        self.assertAlmostEqual(w["mD"],    d["mD"])

    def test_arrays_roundtrip(self):
        d = self.d; w = self._write_read()
        for k in ("g", "h", "ts", "pos", "D0"):
            np.testing.assert_array_equal(
                w[k], np.asarray(d[k], np.float64), err_msg=k)
        for k in ("r", "dec_mask", "te_mask"):
            np.testing.assert_array_equal(
                w[k], np.asarray(d[k], bool), err_msg=k)
        np.testing.assert_array_equal(
            w["X_rule"], np.asarray(d["X_rule"], np.float64))
        np.testing.assert_array_equal(
            w["rule_ne"], np.asarray(d["rule_ne"], np.uint32))
        for i, (we, de) in enumerate(zip(w["rule_edges"], d["rule_edges_list"])):
            np.testing.assert_array_equal(
                we, np.asarray(de, np.float64), err_msg=f"rule_edges[{i}]")
        np.testing.assert_array_equal(
            w["grid"], np.asarray(d["grid"], np.float64))

    def test_baseline_roundtrip(self):
        d = self.d; w = self._write_read()
        for k_in, k_out in [
            ("T_base",   "T_base_py"),
            ("T_rule",   "T_rule_py"),
            ("T_rb",     "T_rb_py"),
            ("T_ai",     "T_ai_py"),
            ("T_ai_fb",  "T_ai_fb_py"),
            ("T_c3",     "T_c3_py"),
        ]:
            np.testing.assert_array_equal(
                w[k_out], np.asarray(d[k_in], np.float64), err_msg=k_in)
        self.assertAlmostEqual(w["R_py"], d["R_MiB"])
        self.assertAlmostEqual(w["H_py"], d["H_MiB"])

    def test_magic_mismatch_raises(self):
        self._write_read()
        with open(self.tmp.name, "r+b") as f:
            f.seek(0); f.write(b"XXXX")
        with self.assertRaises(ValueError):
            gc.read_window(self.tmp.name)

    def test_model_roundtrip(self):
        m = make_synthetic_model()
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tf:
            tp = tf.name
        try:
            gc.write_model(tp, m["cuts_list"], m["knots"], m["table"])
            m2 = gc.read_model(tp)
            for i in range(3):
                np.testing.assert_array_equal(m2["cuts"][i], m["cuts_list"][i])
            np.testing.assert_array_equal(m2["knots"], m["knots"])
            np.testing.assert_array_equal(m2["table"], m["table"])
            self.assertEqual(m2["sentinel"], m["sentinel"])
        finally:
            Path(tp).unlink(missing_ok=True)


class TestCppParity(unittest.TestCase):
    """합성 창으로 C++ 하니스 출력과 Python 기준선 대칭 검증 (G2W3)."""

    @unittest.skipUnless(HARNESS.exists(), "g2_harness 없음 (g2_build.sh 먼저 실행)")
    def test_parity_synthetic(self):
        m_dict = make_synthetic_model()
        d = make_synthetic_window(model=m_dict)

        with tempfile.TemporaryDirectory() as tmpdir:
            win_dir = Path(tmpdir) / "windows"
            res_dir = Path(tmpdir) / "results"
            mod_path = Path(tmpdir) / "model.bin"
            win_dir.mkdir()

            gc.write_model(str(mod_path),
                           m_dict["cuts_list"], m_dict["knots"], m_dict["table"])

            win_path = win_dir / "test.bin"
            gc.write_window(
                str(win_path),
                d["n"], d["G_r"], d["n_grid"], d["rule_n_rf"],
                d["b"], d["cap"], d["W_cal"], d["U"], d["W_fb"], d["mD"],
                d["g"], d["h"], d["r"], d["ts"],
                d["dec_mask"], d["te_mask"],
                d["pos"], d["D0"],
                d["X_rule"], d["rule_ne"], d["rule_edges_flat"],
                d["grid"],
                d["T_base"], d["T_rule"], d["T_rb"],
                d["T_ai"], d["T_ai_fb"], d["T_c3"],
                d["R_MiB"], d["H_MiB"],
            )

            proc = subprocess.run(
                [str(HARNESS), "parity",
                 str(win_dir), str(mod_path), str(res_dir)],
                capture_output=True, text=True,
            )
            if proc.returncode != 0:
                self.fail(f"g2_harness 실패: {proc.stderr}\n{proc.stdout}")

            res_path = res_dir / "test.bin"
            self.assertTrue(res_path.exists(), "결과 파일 없음")

            w = gc.read_window(str(win_path))
            r = gc.read_results(str(res_path), d["n"])

            def check_T(label, cpp, py):
                nan_eq = np.array_equal(np.isnan(cpp), np.isnan(py))
                self.assertTrue(nan_eq, f"{label}: NaN 위치 불일치")
                m = ~np.isnan(py)
                if m.any():
                    max_diff = float(np.abs(cpp[m] - py[m]).max())
                    self.assertEqual(max_diff, 0.0,
                                     f"{label}: 최대 오차 {max_diff:.3e}")

            check_T("T_base",   r["T_base"],   w["T_base_py"])
            check_T("T_rule",   r["T_rule"],   w["T_rule_py"])
            check_T("T_rb",     r["T_rb"],     w["T_rb_py"])
            check_T("T_ai",     r["T_ai"],     w["T_ai_py"])
            check_T("T_ai_fb",  r["T_ai_fb"],  w["T_ai_fb_py"])
            check_T("T_c3",     r["T_c3"],     w["T_c3_py"])

            self.assertAlmostEqual(r["R"], w["R_py"], delta=1e-9)
            self.assertAlmostEqual(r["H"], w["H_py"], delta=1e-9)

    @unittest.skipUnless(HARNESS.exists(), "g2_harness 없음 (g2_build.sh 먼저 실행)")
    def test_parity_all_equal_T(self):
        """T_rb == T_base인 경우: T_c3 = T_ai_fb (rule 모드 없음)."""
        m_dict = make_synthetic_model()
        d = make_synthetic_window(model=m_dict)

        d["T_rb"] = d["T_base"].copy()
        g_, h_, ts_ = d["g"], d["h"], d["ts"]
        r_ = d["r"].astype(bool)
        dec_mask = d["dec_mask"].astype(bool)
        b_, W_fb_ = d["b"], d["W_fb"]

        T_ai_fb, _, _ = _run_fallback3_py(
            d["n"], g_, h_, ts_, r_, d["T_ai"], d["T_base"], dec_mask, b_, W_fb_)
        T_c3, _, _ = _run_fallback3_py(
            d["n"], g_, h_, ts_, r_, T_ai_fb, d["T_rb"], dec_mask, b_, W_fb_)
        d["T_ai_fb"] = T_ai_fb
        d["T_c3"] = T_c3
        te_idx = np.where(d["te_mask"].astype(bool))[0]
        r_bool = r_.astype(bool)
        MiB = gc.MiB
        d["R_MiB"] = float(h_[te_idx][(r_bool[te_idx]) & (g_[te_idx] > T_c3[te_idx])].sum()) / MiB
        d["H_MiB"] = float((h_[te_idx] * np.minimum(g_[te_idx], T_c3[te_idx])).sum()) / MiB

        with tempfile.TemporaryDirectory() as tmpdir:
            win_dir = Path(tmpdir) / "windows"
            res_dir = Path(tmpdir) / "results"
            mod_path = Path(tmpdir) / "model.bin"
            win_dir.mkdir()
            gc.write_model(str(mod_path),
                           m_dict["cuts_list"], m_dict["knots"], m_dict["table"])
            win_path = win_dir / "eq.bin"
            gc.write_window(
                str(win_path),
                d["n"], d["G_r"], d["n_grid"], d["rule_n_rf"],
                d["b"], d["cap"], d["W_cal"], d["U"], d["W_fb"], d["mD"],
                d["g"], d["h"], d["r"], d["ts"],
                d["dec_mask"], d["te_mask"],
                d["pos"], d["D0"],
                d["X_rule"], d["rule_ne"], d["rule_edges_flat"],
                d["grid"],
                d["T_base"], d["T_rule"], d["T_rb"],
                d["T_ai"], d["T_ai_fb"], d["T_c3"],
                d["R_MiB"], d["H_MiB"],
            )
            proc = subprocess.run(
                [str(HARNESS), "parity",
                 str(win_dir), str(mod_path), str(res_dir)],
                capture_output=True, text=True,
            )
            self.assertEqual(proc.returncode, 0,
                             proc.stderr + proc.stdout)
            r2 = gc.read_results(str(res_dir / "eq.bin"), d["n"])
            m = ~np.isnan(d["T_ai_fb"])
            np.testing.assert_array_equal(r2["T_ai_fb"][m], T_ai_fb[m])


if __name__ == "__main__":
    unittest.main(verbosity=2)
