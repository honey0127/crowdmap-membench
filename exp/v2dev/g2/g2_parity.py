#!/usr/bin/env python3
"""g2_parity.py — C++ 하니스 출력과 Python 기준선의 대칭 검증 (G2W2).

C++ 하니스(g2_harness parity)가 results/ 에 쓴 .bin 파일을
windows/ 의 Python 기준선과 대조한다. T_base 포함.

실행(저장소 루트)
  python3 exp/v2dev/g2/g2_parity.py [--win-dir ...] [--res-dir ...] [--tol ...]
"""
import argparse
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "exp/v2dev/g2"))
from g2_common import read_window, read_results

COST_TOL = 1e-9
T_TOL    = 0.0


def check_array(label, cpp, py, atol, issues):
    nan_py  = np.isnan(py)
    nan_cpp = np.isnan(cpp)
    if not np.array_equal(nan_py, nan_cpp):
        issues.append(f"{label}: NaN 위치 불일치 (py={nan_py.sum()}, cpp={nan_cpp.sum()})")
        return
    m = ~nan_py
    if not m.any():
        return
    diff = np.abs(cpp[m] - py[m])
    max_diff = float(diff.max())
    if max_diff > atol:
        worst_i = int(np.argmax(diff))
        issues.append(f"{label}: 최대 오차 {max_diff:.3e} > {atol} "
                      f"(py={py[m][worst_i]:.8g}, cpp={cpp[m][worst_i]:.8g})")


def parity_one(win_path, res_path, tol_cost):
    w = read_window(str(win_path))
    r = read_results(str(res_path), w["n"])

    issues = []

    check_array("T_base",   r["T_base"],   w["T_base_py"],   T_TOL,    issues)
    check_array("T_rule",   r["T_rule"],   w["T_rule_py"],   T_TOL,    issues)
    check_array("T_rb",     r["T_rb"],     w["T_rb_py"],     T_TOL,    issues)
    check_array("T_ai",     r["T_ai"],     w["T_ai_py"],     T_TOL,    issues)
    check_array("T_ai_fb",  r["T_ai_fb"],  w["T_ai_fb_py"],  T_TOL,    issues)
    check_array("T_c3",     r["T_c3"],     w["T_c3_py"],     T_TOL,    issues)

    for label, rv, pv in (("R_MiB", r["R"], w["R_py"]), ("H_MiB", r["H"], w["H_py"])):
        diff = abs(rv - pv)
        if diff > tol_cost:
            issues.append(f"{label}: |{rv:.8g} - {pv:.8g}| = {diff:.3e} > {tol_cost}")

    return issues


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--win-dir", default="exp/v2dev/g2/windows")
    ap.add_argument("--res-dir", default="exp/v2dev/g2/results")
    ap.add_argument("--tol-cost", type=float, default=COST_TOL)
    a = ap.parse_args()

    win_dir = Path(a.win_dir) if Path(a.win_dir).is_absolute() else ROOT / a.win_dir
    res_dir = Path(a.res_dir) if Path(a.res_dir).is_absolute() else ROOT / a.res_dir

    win_bins = sorted(win_dir.glob("*.bin"))
    if not win_bins:
        raise SystemExit(f"[오류] 창 파일이 없다: {win_dir}")

    total, passed, failed = 0, 0, 0
    all_issues = []

    for wb in win_bins:
        rb = res_dir / wb.name
        if not rb.exists():
            all_issues.append(f"[{wb.stem}] 결과 파일 없음: {rb}")
            failed += 1; total += 1
            continue
        total += 1
        issues = parity_one(wb, rb, a.tol_cost)
        if issues:
            failed += 1
            for iss in issues:
                all_issues.append(f"[{wb.stem}] {iss}")
        else:
            passed += 1

    print(f"\n대칭 검증: {total}창 중 {passed} PASS, {failed} FAIL")
    if all_issues:
        print("FAIL 목록:")
        for iss in all_issues[:20]:
            print(f"  - {iss}")
        if len(all_issues) > 20:
            print(f"  ... (+{len(all_issues)-20}개)")
        sys.exit(1)
    else:
        print("모든 창 PASS")


if __name__ == "__main__":
    main()
