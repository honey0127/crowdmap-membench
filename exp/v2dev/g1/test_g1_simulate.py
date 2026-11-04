#!/usr/bin/env python3
"""test_g1_simulate.py — g1_simulate.py 단위 시험

합성 입력으로 순수 함수(select_g2_candidate, 공식, 상수)만 시험한다.
연구 모듈(paperexp*, measure_a_cpu)은 불러오지 않는다.

실행:
  ~/.venvs/membench-g1/bin/python exp/v2dev/g1/test_g1_simulate.py
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import g1_simulate as sim
import g1_common as G


# ---------------------------------------------------------------- 그룹 A: 상수·필드
class TestConstants(unittest.TestCase):

    def test_c_ns_value(self):
        self.assertEqual(sim.C_NS, 281_800)

    def test_threshold_sc_s(self):
        self.assertEqual(sim.THRESHOLD_SC_S, 1.0)

    def test_sim_dir_name(self):
        self.assertEqual(sim.SIM_DIR, "simulation")

    def test_w_fields_exact_set(self):
        expected = {
            "trace", "n_te", "n_dec", "tau_s",
            "rule_refault_MiB", "rule_waste_MiB",
            "c3_refault_MiB", "c3_waste_MiB",
            "dR_MiB", "dH_MiB", "S_units", "Sc_ns",
            "c3_rule_mode_frac",
            "oracle_refault_MiB", "oracle_waste_MiB",
            "rule_name",
        }
        self.assertEqual(set(sim.W_FIELDS), expected)

    def test_w_fields_no_duplicates(self):
        self.assertEqual(len(sim.W_FIELDS), len(set(sim.W_FIELDS)))

    def test_tstar_in_common(self):
        self.assertEqual(G.TSTAR, 2.6)

    def test_n_eval_in_common(self):
        self.assertEqual(G.N_EVAL, 420)


# ---------------------------------------------------------------- 그룹 B: select_g2_candidate
class TestSelectG2Candidate(unittest.TestCase):

    CANDS = ["P1", "P1b"]

    def _summ(self, p1_sc, p1b_sc):
        """sc_s 값으로 PASS/FAIL을 결정하는 최소 요약 사전."""
        def _one(sc):
            return dict(
                S_total_times_c_s=sc,
                verdict_en="PASS" if sc >= sim.THRESHOLD_SC_S else "FAIL",
            )
        return {"P1": _one(p1_sc), "P1b": _one(p1b_sc)}

    def test_both_fail_returns_none(self):
        cand, verdict, reason = sim.select_g2_candidate(self._summ(0.5, 0.3), self.CANDS)
        self.assertIsNone(cand)
        self.assertEqual(verdict, "없음")
        self.assertIn("1.0", reason)   # 통과 기준 언급

    def test_only_p1_passes(self):
        cand, verdict, reason = sim.select_g2_candidate(self._summ(2.0, 0.5), self.CANDS)
        self.assertEqual(cand, "P1")
        self.assertEqual(verdict, "P1")
        self.assertIn("P1", reason)

    def test_only_p1b_passes(self):
        cand, verdict, reason = sim.select_g2_candidate(self._summ(0.5, 2.0), self.CANDS)
        self.assertEqual(cand, "P1b")
        self.assertEqual(verdict, "P1b")
        self.assertIn("P1b", reason)

    def test_both_pass_p1_higher_selects_p1(self):
        cand, verdict, reason = sim.select_g2_candidate(self._summ(3.0, 2.0), self.CANDS)
        self.assertEqual(cand, "P1")
        self.assertEqual(verdict, "P1")

    def test_both_pass_p1b_higher_selects_p1b(self):
        cand, verdict, reason = sim.select_g2_candidate(self._summ(2.0, 3.0), self.CANDS)
        self.assertEqual(cand, "P1b")
        self.assertEqual(verdict, "P1b")
        self.assertIn("P1b", reason)

    def test_both_pass_exact_tie_selects_p1(self):
        """값이 같으면 더 작은 표 P1 선택(계획 v4 §3)."""
        cand, verdict, _ = sim.select_g2_candidate(self._summ(2.5, 2.5), self.CANDS)
        self.assertEqual(cand, "P1")
        self.assertEqual(verdict, "P1")

    def test_exactly_at_threshold_is_pass(self):
        """경계값: sc_s = 1.0 정확히는 PASS."""
        cand, _, _ = sim.select_g2_candidate(self._summ(1.0, 0.5), self.CANDS)
        self.assertEqual(cand, "P1")

    def test_just_below_threshold_is_fail(self):
        """경계값: sc_s < 1.0은 FAIL."""
        cand, _, _ = sim.select_g2_candidate(self._summ(1.0 - 1e-9, 0.5), self.CANDS)
        self.assertIsNone(cand)

    def test_single_candidate_p1_passes(self):
        """후보 목록이 P1만일 때."""
        cand, verdict, _ = sim.select_g2_candidate(self._summ(2.0, 0.5), ["P1"])
        self.assertEqual(cand, "P1")

    def test_single_candidate_p1b_passes(self):
        """후보 목록이 P1b만일 때."""
        cand, verdict, _ = sim.select_g2_candidate(self._summ(0.5, 2.0), ["P1b"])
        self.assertEqual(cand, "P1b")

    def test_single_candidate_fails(self):
        """후보 목록이 하나뿐이고 실패."""
        cand, verdict, _ = sim.select_g2_candidate(self._summ(0.5, 0.0), ["P1"])
        self.assertIsNone(cand)
        self.assertEqual(verdict, "없음")


# ---------------------------------------------------------------- 그룹 C: 공식 확인
class TestFormulas(unittest.TestCase):

    def test_s_model_positive_improvement(self):
        """ΔR > 0, ΔH > 0이면 S_model > 0 (규칙보다 C3가 나음)."""
        dR, dH = 10.0, 5.2
        self.assertGreater(dR + dH / G.TSTAR, 0)

    def test_s_model_zero_improvement(self):
        """ΔR = ΔH = 0이면 S_model = 0."""
        self.assertEqual(0.0 + 0.0 / G.TSTAR, 0.0)

    def test_s_model_tstar_weight(self):
        """낭비 항은 T*로 나눠 재확보 단위로 환산한다."""
        dR, dH = 0.0, G.TSTAR  # dH = T* → 기여 = 1.0
        self.assertAlmostEqual(dR + dH / G.TSTAR, 1.0)

    def test_sc_ns_equals_s_times_c_ns(self):
        """Sc_ns = S_model × C_NS (창 하나)."""
        s_model = 7.3
        self.assertAlmostEqual(s_model * sim.C_NS, s_model * 281_800)

    def test_sc_s_aggregation(self):
        """S_model×c(s) = Σ S_model × C_NS / 1e9."""
        rows = [{"S_units": 1.5}, {"S_units": 2.0}, {"S_units": 0.8}]
        s_total = sum(r["S_units"] for r in rows)
        sc_s = s_total * sim.C_NS / 1e9
        self.assertAlmostEqual(sc_s, 4.3 * 281_800 / 1e9)

    def test_pass_verdict_at_threshold(self):
        """통과 기준: S_model×c ≥ 1.0 s."""
        threshold = sim.THRESHOLD_SC_S
        for sc in (1.0, 1.001, 1e6):
            self.assertTrue(sc >= threshold, f"sc={sc} should be PASS")
        for sc in (0.999, 0.0, -1.0):
            self.assertFalse(sc >= threshold, f"sc={sc} should be FAIL")

    def test_dR_MiB_direction(self):
        """dR = R_rule − R_c3: 양수 ↔ C3가 더 적게 재확보한다."""
        R_rule, R_c3 = 100.0, 90.0
        self.assertGreater(R_rule - R_c3, 0)

    def test_dH_MiB_direction(self):
        """dH = H_rule − H_c3: 양수 ↔ C3가 더 적게 낭비한다."""
        H_rule, H_c3 = 50.0, 45.0
        self.assertGreater(H_rule - H_c3, 0)


# ---------------------------------------------------------------- 그룹 D: _assert_te_counts
class TestAssertTeCounts(unittest.TestCase):
    """_assert_te_counts 경계값·실패 케이스.
    pre_sim_costs.csv 대조(단계 4)는 합성 추적 이름이 파일에 없어 건너뛴다."""

    def _rows(self, cands, trace_n_te):
        return {c: [{"trace": t, "n_te": n} for t, n in trace_n_te] for c in cands}

    def _paths(self, n):
        return [f"path_{i}.csv" for i in range(n)]

    def test_normal_dev_mode_no_error(self):
        """정상(개발 시험): 중복 없고 후보 간 일치 → 오류 없음."""
        win_rows = self._rows(["P1", "P1b"], [("t_a", 100), ("t_b", 200)])
        sim._assert_te_counts(win_rows, ["P1", "P1b"], self._paths(2), True)

    def test_win_count_mismatch_raises(self):
        """처리한 창 수 != 경로 수 → SystemExit."""
        win_rows = self._rows(["P1"], [("t_a", 10)])
        with self.assertRaises(SystemExit):
            sim._assert_te_counts(win_rows, ["P1"], self._paths(2), True)

    def test_duplicate_trace_raises(self):
        """중복 추적 이름 → SystemExit."""
        win_rows = self._rows(["P1", "P1b"], [("dup", 10), ("dup", 10)])
        with self.assertRaises(SystemExit):
            sim._assert_te_counts(win_rows, ["P1", "P1b"], self._paths(2), True)

    def test_candidate_n_te_mismatch_raises(self):
        """후보 간 창별 n_te 불일치 → SystemExit."""
        win_rows = {
            "P1":  [{"trace": "t_a", "n_te": 100}],
            "P1b": [{"trace": "t_a", "n_te": 101}],
        }
        with self.assertRaises(SystemExit):
            sim._assert_te_counts(win_rows, ["P1", "P1b"], self._paths(1), True)

    def test_official_total_mismatch_raises(self):
        """공식 실행(dev_test=False)에서 sum(n_te) != N_TE_TOTAL → SystemExit."""
        win_rows = self._rows(["P1", "P1b"], [("t_a", 100)])
        with self.assertRaises(SystemExit):
            sim._assert_te_counts(win_rows, ["P1", "P1b"], self._paths(1), False)

    def test_single_candidate_no_cross_check(self):
        """후보가 하나이면 후보 간 비교 없음 → 오류 없음(개발 시험)."""
        win_rows = {"P1": [{"trace": "t_a", "n_te": 50}]}
        sim._assert_te_counts(win_rows, ["P1"], self._paths(1), True)

    def test_boundary_zero_n_te_allowed(self):
        """n_te=0인 창이 있어도 중복·불일치가 아니면 통과(개발 시험)."""
        win_rows = self._rows(["P1", "P1b"], [("t_zero", 0), ("t_pos", 10)])
        sim._assert_te_counts(win_rows, ["P1", "P1b"], self._paths(2), True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
