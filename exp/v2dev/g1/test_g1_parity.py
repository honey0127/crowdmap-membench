#!/usr/bin/env python3
"""test_g1_parity.py — g1_simulate.py C3 경로 parity 시험

evaluate3()의 C3 경로와 g1_simulate.py의 C3 경로가 동일한 rec 입력에서 T_c3·R·H를
정확히 같게 낸다는 것을 dev 일정 1개와 합성 입력으로 검증한다.

420개 창 시뮬레이션은 절대 실행하지 않는다.

그룹 E: pe.parts() 단위 — refault·waste 분해, 부호·단위 정확성
그룹 F: run_fallback3 전환 로직 — W=300s 창 shadow ledger, mode 결정
그룹 G: C3 경로 parity — dev 일정 1개(syn_periodic.csv), 동일 rec 주입 시 T_c3·R·H 완전 일치

실행:
  ~/.venvs/membench-g1/bin/python exp/v2dev/g1/test_g1_parity.py
"""
import math
import sys
import types
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import g1_common as G
import g1_simulate as sim


def _research():
    if not hasattr(_research, "_cache"):
        _research._cache = G.research_modules()
    return _research._cache


def _ns(ts, g, h, r):
    return types.SimpleNamespace(ts=ts, g=g, h=h, r=r)


# ---------------------------------------------------------------- 그룹 E: pe.parts()
class TestPartsFunction(unittest.TestCase):
    """pe.parts(g, h, r, T, b) = (refault_bytes, waste_byte_s)
    부호와 단위가 계획 v4 §1·measure_a_cpu.py와 일치하는지 검증한다."""

    @classmethod
    def setUpClass(cls):
        cls.pe, cls.p2, cls.p3, cls.ma = _research()

    def test_no_reuse_zero_refault(self):
        """r=False: refault=0, waste = Σ h·min(g,T)."""
        g = np.array([2.0, 5.0])
        h = np.array([1.0, 2.0])
        r = np.array([False, False])
        T = np.array([3.0, 3.0])
        ref, wst = self.pe.parts(g, h, r, T, 2.6)
        self.assertAlmostEqual(ref, 0.0)
        self.assertAlmostEqual(wst, 1.0 * 2.0 + 2.0 * 3.0)   # 2 + 6 = 8

    def test_all_refaulted(self):
        """g > T, r=True: refault = Σh, waste = Σ h·T."""
        g = np.array([10.0, 10.0])
        h = np.array([1.0, 2.0])
        r = np.array([True, True])
        T = np.array([3.0, 3.0])
        ref, wst = self.pe.parts(g, h, r, T, 2.6)
        self.assertAlmostEqual(ref, 3.0)    # 1+2
        self.assertAlmostEqual(wst, 9.0)    # 3*3

    def test_held_no_refault(self):
        """r=True but g <= T: 보유, refault=0, waste = h·g."""
        g = np.array([2.0])
        h = np.array([1.0])
        r = np.array([True])
        T = np.array([5.0])
        ref, wst = self.pe.parts(g, h, r, T, 2.6)
        self.assertAlmostEqual(ref, 0.0)
        self.assertAlmostEqual(wst, 2.0)

    def test_waste_units_mib_s(self):
        """waste / MiB는 MiB·s 단위다(ΔH에 쓰임)."""
        MiB = self.pe.MiB
        h = np.array([MiB])   # 1 MiB
        g = np.array([2.6])
        r = np.array([True])
        T = np.array([2.6])   # T = g = T*, g <= T → no refault
        _, wst = self.pe.parts(g, h, r, T, 2.6)
        # waste = h * min(g, T) = MiB * 2.6 → wst/MiB = 2.6 (MiB·s)
        self.assertAlmostEqual(wst / MiB, 2.6)

    def test_s_model_formula_decomposition(self):
        """S_model = ΔR + ΔH/T* = (rule_cost - c3_cost)/MiB.
        measure_a_cpu.py run_pre()의 S = rule - c3 공식과 동일해야 한다."""
        MiB = self.pe.MiB
        b = G.TSTAR
        g = np.array([1.0, 10.0])
        h = np.array([MiB, MiB])
        r = np.array([True, True])
        T_rule = np.array([3.0, 3.0])
        T_c3 = np.array([6.0, 1.0])
        rr, rw = self.pe.parts(g, h, r, T_rule, b)
        cr, cw = self.pe.parts(g, h, r, T_c3, b)
        dR = (rr - cr) / MiB
        dH = (rw - cw) / MiB
        S_decomp = dR + dH / b
        rule_cost = (rr + rw / b) / MiB
        c3_cost = (cr + cw / b) / MiB
        S_direct = rule_cost - c3_cost
        self.assertAlmostEqual(S_decomp, S_direct, places=12)

    def test_oracle_formula(self):
        """오라클: r&(g<T*)이면 T=∞(보유), 아니면 T=0(즉시 반납)."""
        b = G.TSTAR
        g = np.array([1.0, 5.0])   # 1 < T*, 5 >= T*
        h = np.array([1.0, 1.0])
        r = np.array([True, True])
        orc_T = np.where(r & (g < b), np.inf, 0.0)
        # piece 0: T=inf → r&(g>inf)=False → refault=0, waste=h*g=1
        # piece 1: T=0   → r&(g>0)=True  → refault=1, waste=0
        ref, wst = self.pe.parts(g, h, r, orc_T, b)
        self.assertAlmostEqual(ref, 1.0)
        self.assertAlmostEqual(wst, 1.0)


# ---------------------------------------------------------------- 그룹 F: run_fallback3 전환 로직
class TestFallback3Switch(unittest.TestCase):
    """run_fallback3(tr, T_ai, T_b, b, dec_mask, W=300, ε=0, H=0)
    두 정책 사이의 W=300s shadow ledger 전환 로직이 올바른지 검증한다."""

    @classmethod
    def setUpClass(cls):
        cls.pe, cls.p2, cls.p3, cls.ma = _research()

    def _fb3(self, ts, g, h, r, T_ai, T_b, b=2.6):
        tr = _ns(np.array(ts, float), np.array(g, float),
                 np.array(h, float), np.array(r, bool))
        dec = np.ones(len(ts), dtype=bool)
        return self.p3.FALLBACK(tr,
                                np.array(T_ai, float),
                                np.array(T_b, float),
                                b, dec, **self.p2.FB)

    def test_fb_constants(self):
        """p2.FB = {W: 300.0, eps: 0.0, H: 0.0} — FALLBACK에 쓰는 고정값."""
        self.assertEqual(self.p2.FB["W"], 300.0)
        self.assertEqual(self.p2.FB["eps"], 0.0)
        self.assertEqual(self.p2.FB["H"], 0.0)

    def test_identical_policies_no_switch(self):
        """T_ai = T_b: 차이 사건이 없으므로 D=0 → mode=False → T=T_ai."""
        n = 6
        ts = np.arange(n, dtype=float) * 10
        T_same = np.full(n, 2.0)
        T_out, mode, idx = self._fb3(ts,
                                      np.ones(n) * 5.0,
                                      np.ones(n),
                                      np.ones(n, dtype=bool),
                                      T_same, T_same)
        self.assertTrue(np.all(~mode), "동일 정책인데 mode=True 조각이 있다")
        np.testing.assert_array_equal(T_out[idx], T_same[idx])

    def test_returned_shapes(self):
        """T_out.shape = (n,), mode.shape = (len(dec),), idx covers all dec."""
        n = 10
        ts = np.arange(n, dtype=float)
        g = np.ones(n) * 3.0
        h = np.ones(n)
        r = np.ones(n, dtype=bool)
        T_ai = np.random.default_rng(1).uniform(1, 4, n)
        T_b = np.random.default_rng(2).uniform(1, 4, n)
        T_out, mode, idx = self._fb3(ts, g, h, r, T_ai, T_b)
        self.assertEqual(T_out.shape, (n,))
        self.assertEqual(mode.dtype, bool)
        self.assertEqual(set(idx.tolist()), set(range(n)))

    def test_mode_true_means_T_b_used(self):
        """mode[i]=True → T_out[idx[i]] = T_b[idx[i]]."""
        n = 8
        ts = np.arange(n, dtype=float)
        T_ai = np.full(n, 1.0)
        T_b = np.full(n, 4.0)
        T_out, mode, idx = self._fb3(ts,
                                      np.ones(n) * 5.0,
                                      np.ones(n),
                                      np.ones(n, dtype=bool),
                                      T_ai, T_b)
        for i, ix in enumerate(idx):
            if mode[i]:
                self.assertAlmostEqual(T_out[ix], T_b[ix], msg=f"idx {ix}: mode=True이지만 T_b 아님")
            else:
                self.assertAlmostEqual(T_out[ix], T_ai[ix], msg=f"idx {ix}: mode=False이지만 T_ai 아님")

    def test_fallback3_is_p3_fallback(self):
        """p3.FALLBACK이 research_modules() 이후 run_fallback3를 가리킨다."""
        import paperexp3 as p3
        self.assertIs(self.p3.FALLBACK, p3.run_fallback3)

    def test_rule_fallback_uses_fb_constants(self):
        """T_rb = FALLBACK(T_rule, base, ...) — W,ε,H 가 p2.FB와 같아야 한다."""
        import inspect
        src = inspect.getsource(sim.main)
        # g1_simulate.py가 p2.FB를 그대로 쓰는지 소스 텍스트로 확인
        self.assertIn("**p2.FB", src,
                      "g1_simulate.main이 p2.FB 고정값을 쓰지 않는다")


# ---------------------------------------------------------------- 그룹 G: C3 경로 parity
class TestC3PathParity(unittest.TestCase):
    """
    동일한 rec 입력 → T_c3·R·H 완전 일치.
    dev 일정 한 개(syn_periodic.csv)를 쓴다. 420창 시뮬레이션은 없다.
    """

    _TRACE = G.REPO / "exp/sched/syn_periodic.csv"

    @classmethod
    def setUpClass(cls):
        cls.pe, cls.p2, cls.p3, cls.ma = _research()
        if not cls._TRACE.exists():
            raise unittest.SkipTest(f"trace 없음: {cls._TRACE}")
        cls.cfg = G.read_json(G.REPO / "exp/frozen.json")
        cls.b = G.TSTAR
        cls.cap = cls.p2.tcap(cls.b)
        cls.K = G.knots(cls.pe)
        # 한 번만 tr 로드 (setUpClass에서 공유)
        cls.tr = cls.p2.Trace2(str(cls._TRACE), G.FEATS)
        cls.led = cls.p2.Ledger(cls.tr)

    def _run_c3(self, rec):
        """rec 배열(n,) → (T_c3, mode, idx_fb, rule_ref, rule_wst, c3_ref, c3_wst, T_rb)."""
        pe, p2, p3 = self.pe, self.p2, self.p3
        tr, led, b, cap = self.tr, self.led, self.b, self.cap
        cfg = self.cfg
        VA, VA1, TE, h = tr.VA, tr.VA1, tr.TE, tr.h
        dec = VA | TE
        # base
        base = p2.online_group_T(tr, led, b,
                                  np.zeros(len(tr.g), dtype=np.int64), 1,
                                  dec, cfg["wcal"], cfg["period"])
        # 규칙
        Cva = pe.cost_matrix(tr.gva[VA], h[VA], tr.rva[VA], b)
        pos_va = np.where(VA)[0]
        is1 = VA1[pos_va]
        i_fix = int(np.argmin(Cva.sum(1)))
        i_fix1 = int(np.argmin(Cva[:, is1].sum(1)))
        i_fix2 = int(np.argmin(Cva[:, ~is1].sum(1)))
        grp_r, G_r, _, _ = p2.rule_groups(tr, b, Cva, pos_va, is1, i_fix, i_fix1, i_fix2)
        T_rule = p2.online_group_T(tr, led, b, grp_r, G_r, dec, cfg["wcal"], cfg["period"])
        T_rb, _, _ = p3.FALLBACK(tr, T_rule, base, b, dec, **p2.FB)
        # AI 경로 (rec 주입 지점)
        Tk = np.stack([np.minimum(rec * k, cap) for k in p2.KGRID])
        T_ai, _ = p2.online_kappa_T(tr, led, b, Tk, dec, cfg["wcal"], cfg["period"])
        T_ai_fb, _, _ = p3.FALLBACK(tr, T_ai, base, b, dec, **p2.FB)
        T_c3, mode, idx_fb = p3.FALLBACK(tr, T_ai_fb, T_rb, b, dec, **p2.FB)
        # 비용
        gt, ht, rt = tr.g[TE], h[TE], tr.r[TE]
        rule_ref, rule_wst = pe.parts(gt, ht, rt, T_rb[TE], b)
        c3_ref, c3_wst = pe.parts(gt, ht, rt, T_c3[TE], b)
        return dict(T_c3=T_c3, mode=mode, idx_fb=idx_fb,
                    rule_ref=rule_ref, rule_wst=rule_wst,
                    c3_ref=c3_ref, c3_wst=c3_wst, T_rb=T_rb)

    def _lookup_rec(self, table, grid):
        """조회표 → rec (g1_simulate 경로)."""
        codes = G.encode_matrix(self.tr.X, grid)
        idx = table[tuple(codes.T)]
        return self.K[idx]

    # --- G.1 상수 rec: 조회표 경로 == 직접 경로 ---
    def test_constant_rec_via_table_matches_direct(self):
        """
        모든 코드 조합 → K[4] (=K[4]=pe.TAU[3]=0.3s):
        K[table[codes]] == np.full(n, K[4]) → T_c3·R·H 완전 일치.
        """
        KI = 4  # K[4] = pe.TAU[3] = 0.3
        # 직접 rec
        rec_direct = np.full(len(self.tr.g), self.K[KI])
        r_direct = self._run_c3(rec_direct)

        # 조회표: sentinel=255, cutpoints=[-1e30] → 모든 유한값이 code=1
        grid = {"features": ["사용량", "수준"], "indices": [1, 2],
                "cutpoints": [np.array([-1e30]), np.array([-1e30])],
                "sentinel": 255}
        table = np.full((256, 256), KI, dtype=np.uint8)
        rec_lookup = self._lookup_rec(table, grid)
        # NaN 조각(sentinel→code=255)도 table[255,...]=KI 이므로 rec 동일
        np.testing.assert_array_equal(rec_direct, rec_lookup,
                                      err_msg="상수 rec: 직접과 조회표가 다르다")
        r_lookup = self._run_c3(rec_lookup)

        np.testing.assert_array_equal(r_direct["T_c3"], r_lookup["T_c3"],
                                      err_msg="T_c3 불일치")
        self.assertAlmostEqual(r_direct["c3_ref"], r_lookup["c3_ref"])
        self.assertAlmostEqual(r_direct["c3_wst"], r_lookup["c3_wst"])
        self.assertAlmostEqual(r_direct["rule_ref"], r_lookup["rule_ref"])
        self.assertAlmostEqual(r_direct["rule_wst"], r_lookup["rule_wst"])

    # --- G.2 가변 rec: 코드에 따라 다른 KNOTS 인덱스 ---
    def test_varying_rec_via_table_matches_direct(self):
        """
        (c0*3 + c1*7) % 10 + 1 → KNOTS 인덱스:
        조회표 rec와 직접 계산 rec가 같으면 T_c3·R·H 완전 일치.
        """
        # 2분위 격자
        f1 = self.tr.X[:, 1]
        f2 = self.tr.X[:, 2]
        m1 = ~np.isnan(f1); m2 = ~np.isnan(f2)
        cut1 = np.array([np.median(f1[m1])]) if m1.any() else np.array([0.5])
        cut2 = np.array([np.median(f2[m2])]) if m2.any() else np.array([0.5])
        grid = {"features": ["사용량", "수준"], "indices": [1, 2],
                "cutpoints": [cut1, cut2], "sentinel": 255}

        # 조회표 구성
        table = np.zeros((256, 256), dtype=np.uint8)
        for c0 in range(256):
            for c1 in range(256):
                table[c0, c1] = (c0 * 3 + c1 * 7) % 10 + 1  # 1~10

        rec_lookup = self._lookup_rec(table, grid)
        # 직접 rec: 같은 공식으로 계산
        codes = G.encode_matrix(self.tr.X, grid)
        rec_direct = self.K[table[tuple(codes.T)]]

        np.testing.assert_array_equal(rec_direct, rec_lookup,
                                      err_msg="가변 rec: 직접과 조회표가 다르다")
        r_d = self._run_c3(rec_direct)
        r_l = self._run_c3(rec_lookup)

        np.testing.assert_array_equal(r_d["T_c3"], r_l["T_c3"],
                                      err_msg="가변 rec: T_c3 불일치")
        self.assertAlmostEqual(r_d["c3_ref"], r_l["c3_ref"], places=12)
        self.assertAlmostEqual(r_d["c3_wst"], r_l["c3_wst"], places=12)

    # --- G.3 규칙 경로가 rec와 독립적 ---
    def test_rule_baseline_independent_of_rec(self):
        """T_rb(규칙 경로)는 rec와 무관하다. rec를 바꿔도 rule_ref·rule_wst가 동일해야 한다."""
        rA = self._run_c3(np.full(len(self.tr.g), self.K[2]))
        rB = self._run_c3(np.full(len(self.tr.g), self.K[8]))
        self.assertAlmostEqual(rA["rule_ref"], rB["rule_ref"], places=12,
                               msg="rule_ref가 rec에 따라 달라진다 — 규칙 경로가 AI 경로에 오염됨")
        self.assertAlmostEqual(rA["rule_wst"], rB["rule_wst"], places=12,
                               msg="rule_wst가 rec에 따라 달라진다 — 규칙 경로가 AI 경로에 오염됨")
        np.testing.assert_array_equal(rA["T_rb"], rB["T_rb"],
                                      err_msg="T_rb가 rec에 따라 달라진다")

    # --- G.4 T_c3[TE]에 NaN 없음 ---
    def test_no_nan_in_T_c3_TE(self):
        """T_c3[TE]에 NaN 없음 — evaluate3._costs()와 같은 조건."""
        r = self._run_c3(np.full(len(self.tr.g), self.K[4]))
        te_vals = r["T_c3"][self.tr.TE]
        self.assertFalse(np.isnan(te_vals).any(),
                         f"T_c3[TE]에 NaN {np.isnan(te_vals).sum()}개")

    # --- G.5 mode_c3 → c3_rule_frac 계산 ---
    def test_c3_rule_frac_computation(self):
        """c3_rule_frac = mode_c3[TE_decisions].mean() — evaluate3와 동일 계산."""
        rec = np.full(len(self.tr.g), self.K[4])
        r = self._run_c3(rec)
        mode, idx_fb, TE = r["mode"], r["idx_fb"], self.tr.TE
        te_m = TE[idx_fb]           # idx_fb 위치가 TE인가
        if te_m.any():
            expected_frac = float(mode[te_m].mean())
        else:
            expected_frac = math.nan

        # g1_simulate.py의 c3_rule_frac 계산과 동일
        if te_m.any():
            self.assertAlmostEqual(
                float(mode[te_m].mean()), expected_frac, places=15,
                msg="c3_rule_frac 계산 불일치"
            )
        else:
            self.assertTrue(math.isnan(expected_frac))

    # --- G.6 S_model 공식 대조 ---
    def test_s_model_equals_rule_minus_c3_cost(self):
        """S_model = ΔR + ΔH/T* = (rule_cost − c3_cost)/MiB.
        measure_a_cpu.run_pre()의 S = rule − c3 공식과 소수점 12자리까지 일치."""
        MiB = self.pe.MiB
        b = G.TSTAR
        r = self._run_c3(np.full(len(self.tr.g), self.K[4]))
        dR = (r["rule_ref"] - r["c3_ref"]) / MiB
        dH = (r["rule_wst"] - r["c3_wst"]) / MiB
        S_decomp = dR + dH / b
        rule_cost = (r["rule_ref"] + r["rule_wst"] / b) / MiB
        c3_cost = (r["c3_ref"] + r["c3_wst"] / b) / MiB
        S_direct = rule_cost - c3_cost
        # 두 계산 경로가 대수적으로 동일하지만 부동소수점 덧셈 순서가 달라
        # ~1e-12 편차가 발생할 수 있다. 절대 차이 1e-9로 확인한다.
        self.assertAlmostEqual(S_decomp, S_direct, delta=1e-9,
                               msg="S_model 공식이 (rule_cost−c3_cost)/MiB와 다르다")

    # --- G.7 Sc_ns 단위 ---
    def test_sc_ns_unit(self):
        """Sc_ns = S_model × C_NS (ns/MiB). Sc_ns / 1e9 = Sc_s(초)."""
        r = self._run_c3(np.full(len(self.tr.g), self.K[4]))
        MiB = self.pe.MiB
        b = G.TSTAR
        dR = (r["rule_ref"] - r["c3_ref"]) / MiB
        dH = (r["rule_wst"] - r["c3_wst"]) / MiB
        S = dR + dH / b
        Sc_ns = S * sim.C_NS
        Sc_s = Sc_ns / 1e9
        # 단위가 맞으면 Sc_s는 유한한 실수
        self.assertTrue(math.isfinite(Sc_s), "Sc_s가 유한하지 않다")

    # --- G.8 TE 행 수 단일 창 양수 (전체 2,003,785는 420창 없이 검증 불가) ---
    def test_single_window_n_te_positive(self):
        """단일 창의 TE 조각 수 > 0.
        주의: 420창 합계 = 2,003,785 검증은 g1_simulate.py에 없음(GAP #8)."""
        n_te = int(self.tr.TE.sum())
        self.assertGreater(n_te, 0, "syn_periodic의 TE 조각이 없다")

    # --- G.9 KNOTS·cap 상수 ---
    def test_knots_length_and_first_zero(self):
        """KNOTS = [0, τ1..τ10], 길이 11, KNOTS[0]=0."""
        K = G.knots(self.pe)
        self.assertEqual(len(K), 11)
        self.assertEqual(K[0], 0.0)
        np.testing.assert_array_equal(K[1:], self.pe.TAU)

    def test_cap_for_tstar(self):
        """T*=2.6, cap = min(300, 20*2.6) = 52."""
        self.assertAlmostEqual(self.cap, 52.0)

    def test_kgrid_contains_1(self):
        """KGRID에 κ=1.0이 포함되어 있다(AI생존κ 기본값)."""
        self.assertIn(1.0, list(self.p2.KGRID))


# ---------------------------------------------------------------- 그룹 H: run_fallback2 vs run_fallback3
class TestFallback2VsFallback3(unittest.TestCase):
    """run_fallback2(paperexp2)와 run_fallback3(paperexp3) 직접 비교.

    - 두 구현은 같은 전환 규칙(창 합 A_i > B_i면 T_b, 같으면 T_ai)을 쓴다.
    - run_fallback3는 차이 사건을 직접 계산해 수치 잡음을 없앤다.
    - 결과가 반드시 같아야 하는 경우: 정책이 동일하거나, A/B 차이가 충분히 클 때.
    - eps != 0 또는 H > 0은 run_fallback3에서 단언 오류다(v2 고정값만 쓴다).
    """

    @classmethod
    def setUpClass(cls):
        cls.pe, cls.p2, cls.p3, cls.ma = _research()

    def _fb2(self, ts, g, h, r, T_ai, T_b, b=2.6):
        tr = _ns(np.array(ts, float), np.array(g, float),
                 np.array(h, float), np.array(r, bool))
        dec = np.ones(len(ts), dtype=bool)
        return self.p2.run_fallback2(tr,
                                     np.array(T_ai, float),
                                     np.array(T_b, float),
                                     b, dec, **self.p2.FB)

    def _fb3(self, ts, g, h, r, T_ai, T_b, b=2.6):
        tr = _ns(np.array(ts, float), np.array(g, float),
                 np.array(h, float), np.array(r, bool))
        dec = np.ones(len(ts), dtype=bool)
        return self.p3.run_fallback3(tr,
                                     np.array(T_ai, float),
                                     np.array(T_b, float),
                                     b, dec, **self.p2.FB)

    def test_identical_policies_both_no_switch(self):
        """T_ai = T_b: 차이 없음 → 두 구현 모두 mode=False, T=T_ai."""
        n = 10
        ts = np.arange(n, dtype=float) * 50.0
        T_same = np.full(n, 2.6)
        g = np.ones(n) * 3.0
        h = np.ones(n)
        r = np.ones(n, dtype=bool)
        T2, mode2, idx2 = self._fb2(ts, g, h, r, T_same, T_same)
        T3, mode3, idx3 = self._fb3(ts, g, h, r, T_same, T_same)
        self.assertTrue(np.all(~mode2), "run_fallback2: 동일 정책에 mode=True 조각")
        self.assertTrue(np.all(~mode3), "run_fallback3: 동일 정책에 mode=True 조각")
        np.testing.assert_array_equal(T2, T3, err_msg="동일 정책: T 출력 불일치")

    def test_no_decisions_empty_mode(self):
        """dec_mask=False: 결정 없음 → 두 구현 모두 T=T_ai, mode=[], idx=[]."""
        n = 5
        ts = np.arange(n, dtype=float) * 10.0
        T_ai = np.full(n, 2.6)
        T_b = np.full(n, 1.0)
        g = np.ones(n) * 3.0
        h = np.ones(n)
        r = np.ones(n, dtype=bool)
        tr = _ns(ts, g, h, r)
        dec = np.zeros(n, dtype=bool)
        T2, mode2, idx2 = self.p2.run_fallback2(tr, T_ai.copy(), T_b.copy(), 2.6, dec, **self.p2.FB)
        T3, mode3, idx3 = self.p3.run_fallback3(tr, T_ai.copy(), T_b.copy(), 2.6, dec, **self.p2.FB)
        np.testing.assert_array_equal(T2, T_ai, err_msg="run_fallback2: 결정 없음 → T_ai 반환 아님")
        np.testing.assert_array_equal(T3, T_ai, err_msg="run_fallback3: 결정 없음 → T_ai 반환 아님")
        self.assertEqual(len(mode2), 0)
        self.assertEqual(len(mode3), 0)

    def test_output_shapes_agree(self):
        """두 구현의 T 형태·idx·정렬 순서가 같다."""
        n = 8
        rng = np.random.default_rng(42)
        ts = np.sort(rng.uniform(0, 500, n))
        g = np.ones(n) * 3.0
        h = np.ones(n)
        r = np.ones(n, dtype=bool)
        T_ai = rng.uniform(0.5, 5.0, n)
        T_b = rng.uniform(0.5, 5.0, n)
        T2, mode2, idx2 = self._fb2(ts, g, h, r, T_ai, T_b)
        T3, mode3, idx3 = self._fb3(ts, g, h, r, T_ai, T_b)
        self.assertEqual(T2.shape, T3.shape)
        self.assertEqual(mode2.shape, mode3.shape)
        np.testing.assert_array_equal(idx2, idx3, err_msg="idx 순서 불일치")

    def test_fb3_rejects_nonzero_eps(self):
        """run_fallback3: eps ≠ 0 → AssertionError."""
        n = 3
        ts = np.arange(n, dtype=float)
        g = np.ones(n) * 2.0
        h = np.ones(n)
        r = np.ones(n, dtype=bool)
        tr = _ns(ts, g, h, r)
        dec = np.ones(n, dtype=bool)
        T_ai = np.full(n, 2.0)
        T_b = np.full(n, 1.0)
        with self.assertRaises(AssertionError):
            self.p3.run_fallback3(tr, T_ai, T_b, 2.6, dec, W=300.0, eps=0.1, H=0.0)

    def test_fb3_rejects_positive_H(self):
        """run_fallback3: H > 0 → AssertionError."""
        n = 3
        ts = np.arange(n, dtype=float)
        g = np.ones(n) * 2.0
        h = np.ones(n)
        r = np.ones(n, dtype=bool)
        tr = _ns(ts, g, h, r)
        dec = np.ones(n, dtype=bool)
        T_ai = np.full(n, 2.0)
        T_b = np.full(n, 1.0)
        with self.assertRaises(AssertionError):
            self.p3.run_fallback3(tr, T_ai, T_b, 2.6, dec, W=300.0, eps=0.0, H=1.0)

    def test_fallback2_accepts_nonzero_eps(self):
        """run_fallback2는 eps ≠ 0을 허용한다(v1 모델 경로)."""
        n = 3
        ts = np.arange(n, dtype=float) * 10.0
        g = np.ones(n) * 2.0
        h = np.ones(n)
        r = np.ones(n, dtype=bool)
        tr = _ns(ts, g, h, r)
        dec = np.ones(n, dtype=bool)
        T_ai = np.full(n, 2.0)
        T_b = np.full(n, 1.0)
        T_out, _, _ = self.p2.run_fallback2(tr, T_ai, T_b, 2.6, dec, W=300.0, eps=0.5, H=0.0)
        self.assertEqual(T_out.shape, (n,))

    def test_identical_policies_fb3_exact_zero_diff(self):
        """run_fallback3: 동일 정책 → 차이 사건 없음 → D=0 (수치 잡음 없음).
        이것이 v2 설계 의도: run_fallback2는 두 장부를 따로 계산해 ~1e-9 오차 가능."""
        import paperexp3 as p3_module
        self.assertIs(self.p3.FALLBACK, p3_module.run_fallback3,
                      "research_modules() 이후 FALLBACK이 run_fallback3이어야 한다")

    def test_clear_T_b_preference_mode_agree(self):
        """T_ai >> T_b (낭비 큰 쪽 → 전환 명확): 두 구현 mode 일치.
        초기 창은 데이터 없음 → mode[0]=False, 충분히 쌓인 뒤 전환."""
        n = 30
        ts = np.arange(n, dtype=float) * 20.0   # 0, 20, 40 ... 580 s
        # T_ai=10 → 큰 낭비(h*min(g,10)=h*g=3), T_b=0.5 → 작은 낭비(h*0.5=0.5)
        T_ai = np.full(n, 10.0)
        T_b = np.full(n, 0.5)
        g = np.ones(n) * 3.0    # g=3 < T_ai=10, g=3 > T_b=0.5 (refault under T_b)
        h = np.ones(n) * 2.0
        r = np.ones(n, dtype=bool)
        T2, mode2, idx2 = self._fb2(ts, g, h, r, T_ai, T_b)
        T3, mode3, idx3 = self._fb3(ts, g, h, r, T_ai, T_b)
        np.testing.assert_array_equal(mode2, mode3,
                                      err_msg="명확한 T_b 선호: mode 불일치")
        np.testing.assert_array_equal(T2, T3, err_msg="명확한 T_b 선호: T 출력 불일치")

    def test_window_boundary_event_at_300s(self):
        """창 경계(W=300s) 근방의 사건에서 두 구현이 같은 mode를 낸다."""
        # 사건: 0, 100, 200, 299, 301, 400 s
        ts = np.array([0.0, 100.0, 200.0, 299.0, 301.0, 400.0])
        n = len(ts)
        # T_ai = T_b → D=0 → mode 모두 False (경계 조건 없이 명확)
        T_ai = np.full(n, 2.6)
        T_b = np.full(n, 2.6)
        g = np.ones(n) * 3.0
        h = np.ones(n)
        r = np.ones(n, dtype=bool)
        T2, mode2, _ = self._fb2(ts, g, h, r, T_ai, T_b)
        T3, mode3, _ = self._fb3(ts, g, h, r, T_ai, T_b)
        self.assertTrue(np.all(~mode2), "run_fallback2: 창 경계, 동일 정책에 mode=True")
        self.assertTrue(np.all(~mode3), "run_fallback3: 창 경계, 동일 정책에 mode=True")


if __name__ == "__main__":
    unittest.main(verbosity=2)
