#!/usr/bin/env python3
"""g2_measure.py — G2 공식 CPU 측정 실행기.

사전 점검 → CPU affinity/환경 설정 → C++ measure 모드 실행 → 결과 보고.

측정 경계 (C++ 타이머 기준, G2W3):
  CPU_rule 타이머 안: Ledger::build, compute_grp_r(bin_ids+unique-remap),
                     online_group_T(base+rule), run_fallback3, 비용 계산
  CPU_C3 타이머 안:  위 전체 + compute_p1b_features(pos/D0/mD→x0,x1,x2),
                     compute_Tk(encode+lookup+κ-scale), online_kappa_T,
                     fallback×2 추가, 비용 계산
  타이머 밖: 파일 읽기·메모리 적재, 출력 파일 쓰기

저장 파일:
  raw-dev.json / raw.json              — C++ 하니스 원시 출력
  raw-runs-dev.csv / raw-runs.csv      — 반복별 측정값
  run-environment-dev.json / run-environment.json — 실행 환경
  measure-summary-dev.json / measure-summary-official.json — 최종 요약

사용:
  python3 exp/v2dev/g2/g2_measure.py [--dev-test] [--force] [--out-dir measurement/]

이번 작업에서 --dev-test만 허용 (공식 420창 3회 측정은 별도 지시 필요).

G1 확정값:
  S_model * c = 1.5608601903 s
  ΔR = -906 MiB (rule - C3), c = 281.8 µs/MiB, T* = 2.6 s
"""
import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT  = Path(__file__).resolve().parents[3]
G2DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(G2DIR))
sys.path.insert(0, str(ROOT / "exp/v2dev/g1"))

# G1 확정값
G1_SMODEL_C = 1.5608601903    # S_model * c (초)
G1_DELTA_R  = -906.0          # ΔR = R_rule - R_C3 (MiB), 음수 = C3가 적은 재확보
G1_C        = 281.8e-6        # c (s/MiB)
G1_TSTAR    = 2.6             # T* (초)

N_EVAL      = 420
N_TE_TOTAL  = 2_003_785


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def preflight(win_dir, model_path, harness_path, dev_test):
    """기본 환경 점검."""
    errors = []

    def chk(cond, msg):
        if not cond: errors.append(msg)
        return cond

    chk(harness_path.exists(), f"g2_harness 없음: {harness_path}")
    chk(os.access(harness_path, os.X_OK), f"g2_harness 실행 권한 없음")
    chk(model_path.exists(), f"p1b_model.bin 없음: {model_path}")

    bins = sorted(Path(win_dir).glob("*.bin"))
    n_win = len(bins)
    if dev_test:
        chk(n_win >= 5, f"dev-test 창 부족: {n_win}")
    else:
        chk(n_win == N_EVAL, f"평가 창 수 오류: {n_win} ≠ {N_EVAL}")

    return errors, bins


def set_single_cpu():
    """taskset으로 CPU 1번 고정 (Linux only). 가능하면 적용."""
    if platform.system() != "Linux":
        return None
    taskset = shutil.which("taskset")
    if taskset:
        return [taskset, "-c", "1"]
    return None


def run_harness(harness_path, win_dir, model_path, out_json, dev_test, cpu_prefix):
    cmd = (cpu_prefix or []) + [
        str(harness_path), "measure",
        str(win_dir), str(model_path),
        "--out-json", str(out_json),
    ]
    if dev_test:
        cmd.append("--dev-test")

    env = os.environ.copy()
    env.update({
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "BLAS_NUM_THREADS": "1",
    })

    print(f"  실행: {' '.join(cmd)}")
    t0 = time.time()
    result = subprocess.run(cmd, env=env, capture_output=False)
    elapsed = time.time() - t0

    if result.returncode != 0:
        raise RuntimeError(f"g2_harness 실패: returncode={result.returncode}")
    print(f"  완료: {elapsed:.1f}초")
    return elapsed


def parse_json(out_json):
    with open(out_json) as f:
        return json.load(f)


def compute_metrics(data):
    """JSON 결과로부터 G_model, G_CPU, ΔI 등을 계산한다."""
    iters = data["iterations"]

    rows = []
    for it in iters:
        cpu_rule_s = it["cpu_rule_ns"] / 1e9
        cpu_c3_s   = it["cpu_c3_ns"]  / 1e9
        dI_s       = it["dI_ns"]       / 1e9
        # G_model = S_model*c - ΔI = G1_SMODEL_C - ΔI
        G_model = G1_SMODEL_C - dI_s
        # G_CPU = c*ΔR - ΔI  (ΔR = R_rule - R_C3 = -906 MiB → c*ΔR = 281.8e-6 * (-906))
        G_CPU   = G1_C * G1_DELTA_R - dI_s
        rows.append(dict(
            order=it["order"],
            cpu_rule_s=cpu_rule_s,
            cpu_c3_s=cpu_c3_s,
            dI_s=dI_s,
            dI_us_per_dec=dI_s * 1e6 / data.get("n_decisions_c3", 1),
            G_model=G_model,
            G_CPU=G_CPU,
            checksum_rule=it["checksum_rule"],
            checksum_c3=it["checksum_c3"],
        ))
    return rows


def save_results(out_dir, data, rows, run_env, dev_test, summ_path):
    """측정 결과를 dev/official 별 경로로 저장한다.

    summ_path: measure-summary-dev.json 또는 measure-summary-official.json
    raw-runs, run-environment도 동일 suffix를 사용.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "-dev" if dev_test else ""

    # raw-runs{suffix}.csv
    csv_path = out_dir / f"raw-runs{suffix}.csv"
    with open(csv_path, "w") as f:
        f.write("iter,order,cpu_rule_s,cpu_c3_s,dI_s,dI_us_per_dec,G_model_s,G_CPU_s,"
                "checksum_rule,checksum_c3\n")
        for i, r in enumerate(rows):
            f.write(f"{i+1},{r['order']},"
                    f"{r['cpu_rule_s']:.6f},{r['cpu_c3_s']:.6f},"
                    f"{r['dI_s']:.6f},{r['dI_us_per_dec']:.4f},"
                    f"{r['G_model']:.6f},{r['G_CPU']:.6f},"
                    f"{r['checksum_rule']},{r['checksum_c3']}\n")
    print(f"  raw-runs{suffix}.csv → {csv_path}")

    # measure-summary-dev.json 또는 measure-summary-official.json
    dI_vals   = [r["dI_s"] for r in rows]
    gm_vals   = [r["G_model"] for r in rows]
    gm_min    = min(gm_vals)
    all_pass  = all(g > 0 for g in gm_vals) if rows else False

    summary = dict(
        dev_test=dev_test,
        n_windows=data["n_windows"],
        n_decisions_rule=data.get("n_decisions_rule"),
        n_decisions_c3=data.get("n_decisions_c3"),
        g1_smodel_c_s=G1_SMODEL_C,
        g1_delta_r_mib=G1_DELTA_R,
        g1_c_us_per_mib=G1_C * 1e6,
        g1_tstar_s=G1_TSTAR,
        dI_s_min=min(dI_vals),
        dI_s_max=max(dI_vals),
        G_model_min=gm_min,
        G_model_max=max(gm_vals),
        G_CPU_min=min(r["G_CPU"] for r in rows),
        G_CPU_max=max(r["G_CPU"] for r in rows),
        g2_pass=(all_pass and not dev_test),
        g2_pass_dev=all_pass,
        note="dev_test: 비공식 5창 측정" if dev_test else "공식 420창 측정",
        iterations=rows,
    )
    with open(summ_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"  {summ_path.name} → {summ_path}")

    # run-environment{suffix}.json
    env_path = out_dir / f"run-environment{suffix}.json"
    with open(env_path, "w", encoding="utf-8") as f:
        json.dump(run_env, f, ensure_ascii=False, indent=2)
    print(f"  run-environment{suffix}.json → {env_path}")

    return summary


def print_report(summary, rows):
    print("\n" + "="*60)
    print("G2 측정 결과 요약")
    print("="*60)
    print(f"창 수: {summary['n_windows']}"
          f"  TE 결정 수: {summary['n_decisions_c3']}")
    print(f"G1 확정값: S_model*c = {G1_SMODEL_C:.4f}s  "
          f"c = {G1_C*1e6:.1f} µs/MiB  T* = {G1_TSTAR}s")
    print()
    print(f"{'반복':>4}  {'순서':>8}  {'CPU_rule':>10}  {'CPU_C3':>10}  "
          f"{'ΔI':>10}  {'G_model':>10}  {'G_CPU':>10}")
    for i, r in enumerate(rows):
        print(f"{i+1:>4}  {r['order']:>8}  "
              f"{r['cpu_rule_s']*1000:>9.3f}ms  "
              f"{r['cpu_c3_s']*1000:>9.3f}ms  "
              f"{r['dI_s']*1000:>9.3f}ms  "
              f"{r['G_model']*1000:>9.3f}ms  "
              f"{r['G_CPU']*1000:>9.3f}ms")
    print()
    print(f"ΔI 범위: {summary['dI_s_min']*1000:.3f}ms ~ {summary['dI_s_max']*1000:.3f}ms")
    print(f"G_model 범위: {summary['G_model_min']*1000:.3f}ms ~ "
          f"{summary['G_model_max']*1000:.3f}ms")
    if summary["dev_test"]:
        print("[dev-test] 비공식 5창 측정 — 공식 판정에 사용하지 않는다.")
        print(f"참고: G_model 최소 = {summary['G_model_min']*1000:.3f}ms "
              f"({'> 0' if summary['g2_pass_dev'] else '<= 0'})")
    else:
        verdict = "PASS" if summary["g2_pass"] else "FAIL"
        print(f"G2 판정: {verdict}  (3회 모두 G_model > 0 조건)")
    print("="*60)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dev-test", action="store_true",
                    help="5창 비공식 측정 (공식 420창 금지, 이번 세션)")
    ap.add_argument("--force", action="store_true",
                    help="기존 결과 파일 덮어쓰기 허용")
    ap.add_argument("--out-dir", default=str(G2DIR / "measurement"),
                    help="결과 저장 디렉터리")
    ap.add_argument("--win-dir", default=str(G2DIR / "windows"))
    ap.add_argument("--model", default=str(G2DIR / "p1b_model.bin"))
    ap.add_argument("--harness", default=str(G2DIR / "g2_harness"))
    a = ap.parse_args()

    out_dir     = Path(a.out_dir)
    win_dir     = Path(a.win_dir)
    model_path  = Path(a.model)
    harness_path= Path(a.harness)
    dev_test    = a.dev_test

    suffix = "-dev" if dev_test else ""
    out_json  = out_dir / f"raw{suffix}.json"
    # dev → measure-summary-dev.json, official → measure-summary-official.json
    summ_name = "measure-summary-dev.json" if dev_test else "measure-summary-official.json"
    summ_path = out_dir / summ_name

    # 공식 측정 중복 방지
    if not dev_test and summ_path.exists() and not a.force:
        print(f"[오류] 공식 측정 결과가 이미 존재: {summ_path}")
        print("       --force로 덮어쓰거나 별도 --out-dir을 지정한다.")
        sys.exit(1)

    print("=== G2 사전 점검 ===")
    errors, bins = preflight(win_dir, model_path, harness_path, dev_test)
    if errors:
        for e in errors: print(f"  ✗ {e}")
        sys.exit(1)
    print(f"  창 수: {len(bins)}  모델: {model_path.name}")

    cpu_prefix = set_single_cpu()
    if cpu_prefix:
        print(f"  CPU affinity: {' '.join(cpu_prefix)}")
    else:
        print("  CPU affinity: 미설정 (taskset 없음 또는 비Linux)")

    # 환경 정보 수집
    import subprocess as sp
    def cmd_out(cmd):
        try: return sp.check_output(cmd, text=True).strip()
        except: return "unknown"

    run_env = dict(
        platform=platform.platform(),
        python=sys.version,
        compiler_version=cmd_out(["g++", "--version"]),
        cpu_affinity=" ".join(cpu_prefix) if cpu_prefix else "none",
        omp_num_threads=os.environ.get("OMP_NUM_THREADS", "unset"),
        git_branch=cmd_out(["git", "branch", "--show-current"]),
        git_commit=cmd_out(["git", "rev-parse", "HEAD"]),
        git_dirty=cmd_out(["git", "status", "--porcelain"]) != "",
        harness_sha256=sha256_file(harness_path),
        model_sha256=sha256_file(model_path),
        n_windows=len(bins),
        dev_test=dev_test,
    )

    if run_env["git_dirty"]:
        print("  [경고] 미커밋 변경이 있다.")

    print("\n=== C++ measure 실행 ===")
    out_dir.mkdir(parents=True, exist_ok=True)
    run_harness(harness_path, win_dir, model_path, out_json, dev_test, cpu_prefix)

    print("\n=== 결과 계산 ===")
    data = parse_json(out_json)
    rows = compute_metrics(data)
    summary = save_results(out_dir, data, rows, run_env, dev_test, summ_path)
    print_report(summary, rows)


if __name__ == "__main__":
    main()
