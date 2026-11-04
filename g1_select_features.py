#!/usr/bin/env python3
"""G1 preregistered feature-pair selection on the 26 fixed development schedules.

Selection rule:
  * Reproduce paperexp2.rule_groups' 4-quantile cross-fit CV cost for every
    singleton and pair on each schedule at T*=2.6 s / 1 MiB chunks.
  * Verify A: the independently recomputed per-schedule winner equals
    rule_groups' selected feature set.
  * Verify B: that feature-set name equals the recorded `rule` value in
    exp/v2dev/chunk/res_q1.csv and res_q2.csv at Tstar=2.6.
  * P1: pair minimizing the mean across 26 schedules of
        R_s(fs) = cv_s(fs) / cv_s(empty_group).
  * P1b: keep P1 and add the one remaining feature minimizing the same mean
         relative CV cost for the 4x4x4 grouping.

--verify-only runs A and B but does not compute or reveal P1/P1b selections.
This script does not fit GBDTs, compile lookup tables, or run the 420-window
simulation. Run from the repository root:
  OMP_NUM_THREADS=1 python3 g1_select_features.py --verify-only
  OMP_NUM_THREADS=1 python3 g1_select_features.py
"""
from __future__ import annotations

# Thread limits must be set before NumPy / SciPy / scikit-learn are imported.
import os
for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_name] = "1"

import argparse
import csv
import hashlib
import itertools
import json
import math
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np

TSTAR = 2.6
EXPECTED_SCHEDULES = 26
RESULT_FILES_DEFAULT = (
    "exp/v2dev/chunk/res_q1.csv",
    "exp/v2dev/chunk/res_q2.csv",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_schedule_paths(repo: Path, list_path: Path) -> list[Path]:
    paths: list[Path] = []
    for raw in list_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        p = (repo / line).resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Schedule listed but not found: {line}")
        paths.append(p)
    if len(paths) != EXPECTED_SCHEDULES:
        raise ValueError(f"Expected exactly {EXPECTED_SCHEDULES} schedules, found {len(paths)}")
    rels = [p.relative_to(repo).as_posix() for p in paths]
    if len(set(rels)) != len(rels):
        raise ValueError("Duplicate schedule paths in development list")
    stems = [p.stem for p in paths]
    if len(set(stems)) != len(stems):
        raise ValueError("Schedule basenames are not unique; result CSV matching would be ambiguous")
    return paths


def read_recorded_rules(repo: Path, result_paths: list[Path], target_stems: set[str]) -> tuple[dict[str, set[str]], dict[str, str]]:
    """Collect unique recorded rule names per trace for Tstar=2.6 across q1/q2."""
    recorded: dict[str, set[str]] = {stem: set() for stem in target_stems}
    seen_rows: dict[str, int] = {stem: 0 for stem in target_stems}
    hashes: dict[str, str] = {}

    for path in result_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Recorded result CSV not found: {path}")
        hashes[path.relative_to(repo).as_posix()] = sha256_file(path)
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            required = {"trace", "Tstar", "rule"}
            if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
                raise ValueError(
                    f"{path}: expected columns {sorted(required)}, got {reader.fieldnames}"
                )
            for row in reader:
                stem = (row.get("trace") or "").strip()
                if stem not in target_stems:
                    continue
                try:
                    tstar = float(row.get("Tstar", "nan"))
                except (TypeError, ValueError):
                    continue
                if not math.isclose(tstar, TSTAR, rel_tol=0.0, abs_tol=1e-9):
                    continue
                rule = (row.get("rule") or "").strip()
                if not rule:
                    raise ValueError(f"{path}: empty rule for trace={stem}, Tstar={tstar}")
                recorded[stem].add(rule)
                seen_rows[stem] += 1

    missing = [stem for stem in sorted(target_stems) if seen_rows[stem] == 0]
    conflicts = {stem: sorted(rules) for stem, rules in recorded.items() if len(rules) > 1}
    if missing:
        raise ValueError(f"Recorded result CSVs have no Tstar=2.6 row for: {missing}")
    if conflicts:
        raise ValueError(f"Recorded rule values conflict across q1/q2 rows: {conflicts}")
    return recorded, hashes


def make_feature_bins(pe: Any, X_va: np.ndarray, n_features: int) -> list[np.ndarray]:
    """Match rule_groups: nan -> -1e9, 4 quantile bins, same bin_ids rule."""
    bins: list[np.ndarray] = []
    for f in range(n_features):
        x = np.nan_to_num(X_va[:, f], nan=-1e9)
        edges = pe.quant_edges(x, 4)
        bins.append(pe.bin_ids(edges, x).astype(np.int64, copy=False))
    return bins


def encode_groups(feature_bins: list[np.ndarray], fs: tuple[int, ...], n_rows: int) -> np.ndarray:
    """Match rule_groups' base-8 concatenation, including 3-feature extension."""
    gid = np.zeros(n_rows, dtype=np.int64)
    for f in fs:
        gid = gid * 8 + feature_bins[f]
    return gid


def crossfit_cv_cost(
    pe: Any,
    Cva: np.ndarray,
    gid_raw: np.ndarray,
    is1: np.ndarray,
    i_fix1: int,
    i_fix2: int,
) -> tuple[float, int]:
    """Reproduce rule_groups' two-fold cross-fit validation cost."""
    uniq, gid = np.unique(gid_raw, return_inverse=True)
    n_groups = len(uniq)
    cv = 0.0
    for fit_m, ev_m, fallback in (
        (is1, ~is1, i_fix1),
        (~is1, is1, i_fix2),
    ):
        idx = pe.fit_groups(Cva[:, fit_m], gid[fit_m], n_groups, fallback)
        ev_pos = np.where(ev_m)[0]
        cv += float(Cva[idx[gid[ev_m]], ev_pos].sum())
    return cv, n_groups


def schedule_scores(
    pe: Any,
    p2: Any,
    tr: Any,
    tstar: float,
) -> dict[str, Any]:
    """Compute baseline, all 1/2-feature scores, 3-feature scoring helper and A."""
    VA = tr.VA
    pos_va = np.where(VA)[0]
    if tr.TR.sum() == 0 or len(pos_va) == 0 or tr.TE.sum() == 0:
        raise ValueError(
            f"Empty partition: TR={int(tr.TR.sum())}, VA={len(pos_va)}, TE={int(tr.TE.sum())}"
        )
    is1 = tr.VA1[pos_va]
    if not is1.any() or is1.all():
        raise ValueError(
            f"Empty validation cross-fit half: VA1={int(is1.sum())}, VA2={int((~is1).sum())}"
        )

    Cva = pe.cost_matrix(tr.gva[VA], tr.h[VA], tr.rva[VA], tstar)
    if Cva.ndim != 2 or Cva.shape[1] != len(pos_va) or not np.isfinite(Cva).all():
        raise ValueError("Invalid or non-finite validation cost matrix")

    i_fix = int(np.argmin(Cva.sum(axis=1)))
    i_fix1 = int(np.argmin(Cva[:, is1].sum(axis=1)))
    i_fix2 = int(np.argmin(Cva[:, ~is1].sum(axis=1)))
    n_features = tr.X.shape[1]
    names = list(tr.feat_names)
    if len(names) != n_features:
        raise ValueError("Feature matrix/name mismatch")

    feature_bins = make_feature_bins(pe, tr.X[VA], n_features)
    cv_cache: dict[tuple[int, ...], tuple[float, int]] = {}

    def cv_for(fs: tuple[int, ...]) -> tuple[float, int]:
        if fs not in cv_cache:
            gid = encode_groups(feature_bins, fs, len(pos_va))
            cv_cache[fs] = crossfit_cv_cost(pe, Cva, gid, is1, i_fix1, i_fix2)
        return cv_cache[fs]

    # Empty feature set is one fixed-timeout group, selected only on the other fold.
    cv_empty, _ = cv_for(())
    if not math.isfinite(cv_empty) or cv_empty <= 0:
        raise ValueError(f"Invalid one-group baseline cross-fit cost: {cv_empty}")

    singles_pairs = [
        *( (i,) for i in range(n_features) ),
        *itertools.combinations(range(n_features), 2),
    ]
    candidate_costs = {fs: cv_for(fs)[0] for fs in singles_pairs}
    local_best: tuple[int, ...] | None = None
    local_best_cv = math.inf
    # The order and strict '<' are the same as rule_groups: singles then pairs;
    # exact ties keep the first feature combination in Trace2.feat_names order.
    for fs in singles_pairs:
        cv = candidate_costs[fs]
        if local_best is None or cv < local_best_cv:
            local_best, local_best_cv = fs, cv

    _, _, rule_name, rule_fs = p2.rule_groups(
        tr, tstar, Cva, pos_va, is1, i_fix, i_fix1, i_fix2
    )
    rule_fs_tuple = tuple(int(i) for i in rule_fs)
    a_ok = local_best == rule_fs_tuple

    # Candidate global P1/P1b scores are kept internally and only emitted in full mode.
    pair_costs = {fs: candidate_costs[fs] for fs in itertools.combinations(range(n_features), 2)}
    return {
        "names": names,
        "cv_empty": cv_empty,
        "cv_cache": cv_cache,
        "candidate_costs": candidate_costs,
        "pair_costs": pair_costs,
        "local_best": local_best,
        "local_best_cv": local_best_cv,
        "rule_fs": rule_fs_tuple,
        "rule_name": rule_name,
        "A_ok": a_ok,
        "n_TR": int(tr.TR.sum()),
        "n_VA": int(VA.sum()),
        "n_VA1": int(is1.sum()),
        "n_VA2": int((~is1).sum()),
        "n_TE": int(tr.TE.sum()),
        "rule_groups_count": int(len(np.unique(encode_groups(feature_bins, rule_fs_tuple, len(pos_va))))),
        "_cv_for": cv_for,
    }


def verify_and_collect(
    repo: Path,
    pe: Any,
    p2: Any,
    p3: Any,
    paths: list[Path],
    cfg: dict[str, Any],
    result_paths: list[Path],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    stems = {p.stem for p in paths}
    recorded, result_hashes = read_recorded_rules(repo, result_paths, stems)
    per_schedule: list[dict[str, Any]] = []
    for idx, schedule_path in enumerate(paths, start=1):
        tr = p2.Trace2(str(schedule_path), cfg["feats"])
        item = schedule_scores(pe, p2, tr, TSTAR)
        stem = schedule_path.stem
        record_values = sorted(recorded[stem])
        b_ok = record_values == [item["rule_name"]]
        row = {
            "schedule": schedule_path.relative_to(repo).as_posix(),
            "schedule_sha256": sha256_file(schedule_path),
            "n_TR": item["n_TR"],
            "n_VA": item["n_VA"],
            "n_VA1": item["n_VA1"],
            "n_VA2": item["n_VA2"],
            "n_TE": item["n_TE"],
            "rule_groups_count": item["rule_groups_count"],
            "local_best_feature_indices": list(item["local_best"] or ()),
            "local_best_features": [item["names"][i] for i in (item["local_best"] or ())],
            "local_best_cv": item["local_best_cv"],
            "rule_feature_indices": list(item["rule_fs"]),
            "rule_features": [item["names"][i] for i in item["rule_fs"]],
            "rule_name": item["rule_name"],
            "recorded_rule_values": record_values,
            "verification_A_rule_groups_reproduction": bool(item["A_ok"]),
            "verification_B_saved_result_match": bool(b_ok),
            "cv_empty_group": item["cv_empty"],
            "_cv_cache": item["cv_cache"],
            "_cv_for": item["_cv_for"],
            "_names": item["names"],
        }
        per_schedule.append(row)
        print(
            f"[{idx:02d}/26] {row['schedule']}: "
            f"A={'OK' if row['verification_A_rule_groups_reproduction'] else 'FAIL'} "
            f"B={'OK' if row['verification_B_saved_result_match'] else 'FAIL'} "
            f"rule={row['rule_name']}",
            flush=True,
        )
    return per_schedule, result_hashes


def serializable_schedule(row: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in row.items() if not k.startswith("_")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--schedule-list", default="exp/dev_sched.txt")
    ap.add_argument("--config", default="exp/frozen.json")
    ap.add_argument("--result-files", nargs="+", default=list(RESULT_FILES_DEFAULT))
    ap.add_argument("--out", default="exp/v2dev/g1/feature-selection.json")
    ap.add_argument("--verify-only", action="store_true", help="run A/B checks only; do not select or write P1/P1b")
    ap.add_argument("--force", action="store_true", help="overwrite existing JSON output in full-selection mode")
    args = ap.parse_args()

    repo = Path.cwd().resolve()
    list_path = (repo / args.schedule_list).resolve()
    config_path = (repo / args.config).resolve()
    out_path = (repo / args.out).resolve()
    result_paths = [(repo / value).resolve() for value in args.result_files]
    for path in (list_path, config_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not args.verify_only and out_path.exists() and not args.force:
        raise FileExistsError(f"Output exists; inspect it or pass --force: {out_path}")

    cfg: dict[str, Any] = json.loads(config_path.read_text(encoding="utf-8"))
    if cfg.get("feats") != "f15":
        raise ValueError(f"Expected frozen f15 feature set, got {cfg.get('feats')!r}")
    if int(cfg.get("seed", -1)) != 0:
        raise ValueError(f"Expected frozen seed 0, got {cfg.get('seed')!r}")

    sys.path.insert(0, str(repo))
    import paperexp as pe  # noqa: E402
    import paperexp2 as p2  # noqa: E402
    import paperexp3 as p3  # noqa: E402
    import sklearn  # noqa: E402

    p3.use_chunk_model()  # Must precede Trace2: installs 1 MiB chunk loader/slicer.
    paths = load_schedule_paths(repo, list_path)
    per_schedule, result_hashes = verify_and_collect(
        repo, pe, p2, p3, paths, cfg, result_paths
    )

    a_count = sum(row["verification_A_rule_groups_reproduction"] for row in per_schedule)
    b_count = sum(row["verification_B_saved_result_match"] for row in per_schedule)
    print(f"\nVERIFY A (independent single/pair CV winner == rule_groups): {a_count}/26")
    print(f"VERIFY B (rule_groups name == saved q1/q2 rule at T*=2.6): {b_count}/26")
    if a_count != EXPECTED_SCHEDULES or b_count != EXPECTED_SCHEDULES:
        print("ERROR: A/B verification failed; no feature selection or output JSON produced.", file=sys.stderr)
        return 2

    if args.verify_only:
        print("VERIFY-ONLY complete: P1/P1b were not selected or written.")
        return 0

    # Pair score: mean of schedule-normalized relative cross-fit costs.
    n_features = len(per_schedule[0]["_names"])
    pair_rows: list[dict[str, Any]] = []
    for fs in itertools.combinations(range(n_features), 2):
        ratios = [row["_cv_cache"][fs][0] / row["cv_empty_group"] for row in per_schedule]
        pair_rows.append({
            "feature_indices": list(fs),
            "features": [per_schedule[0]["_names"][i] for i in fs],
            "mean_relative_cv_cost": float(np.mean(np.asarray(ratios, dtype=np.float64))),
            "per_schedule_relative_cv_cost": [float(x) for x in ratios],
        })
    pair_rows.sort(key=lambda row: (row["mean_relative_cv_cost"], tuple(row["feature_indices"])))
    p1 = pair_rows[0]
    p1_fs = tuple(p1["feature_indices"])

    # P1b is the best 4x4x4 three-feature group containing the fixed P1 pair.
    third_rows: list[dict[str, Any]] = []
    for third in range(n_features):
        if third in p1_fs:
            continue
        fs = tuple(sorted((*p1_fs, third)))
        ratios: list[float] = []
        for row in per_schedule:
            cv, _ = row["_cv_for"](fs)
            ratios.append(cv / row["cv_empty_group"])
        third_rows.append({
            "feature_indices": list(fs),
            "features": [per_schedule[0]["_names"][i] for i in fs],
            "added_feature_index": third,
            "added_feature": per_schedule[0]["_names"][third],
            "mean_relative_cv_cost": float(np.mean(np.asarray(ratios, dtype=np.float64))),
            "per_schedule_relative_cv_cost": [float(x) for x in ratios],
        })
    third_rows.sort(key=lambda row: (row["mean_relative_cv_cost"], tuple(row["feature_indices"])))
    p1b = third_rows[0]

    # Frequency diagnostics only; they do not choose P1/P1b.
    names = per_schedule[0]["_names"]
    feature_counts = {name: 0 for name in names}
    winner_counts: dict[str, int] = {}
    for row in per_schedule:
        rule_name = row["rule_name"]
        winner_counts[rule_name] = winner_counts.get(rule_name, 0) + 1
        for name in row["rule_features"]:
            feature_counts[name] += 1

    result = {
        "status": "exploratory_feature_selection_only",
        "selection_rule": "mean over 26 schedules of cv_s(fs)/cv_s(empty_group)",
        "tstar_s": TSTAR,
        "chunk_size_mib": 1,
        "feature_order": names,
        "thread_environment": {k: os.environ.get(k) for k in (
            "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"
        )},
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "script_path": Path(__file__).resolve().relative_to(repo).as_posix(),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "code_sha256": {
            name: sha256_file(repo / name)
            for name in ("paperexp.py", "paperexp2.py", "paperexp3.py", "chunkslices.py")
        },
        "config_path": config_path.relative_to(repo).as_posix(),
        "config_sha256": sha256_file(config_path),
        "schedule_list_path": list_path.relative_to(repo).as_posix(),
        "schedule_list_sha256": sha256_file(list_path),
        "result_csv_sha256": result_hashes,
        "schedule_count": len(paths),
        "verification": {
            "A_rule_groups_reproduction": f"{a_count}/26",
            "B_saved_q1_q2_match": f"{b_count}/26",
            "A_all_pass": a_count == 26,
            "B_all_pass": b_count == 26,
        },
        "P1": p1,
        "P1b": p1b,
        "P1_pair_candidates_ranked": pair_rows,
        "P1b_third_feature_candidates_ranked": third_rows,
        "diagnostics_only": {
            "rule_winner_frequency": winner_counts,
            "selected_feature_frequency": feature_counts,
        },
        "per_schedule": [serializable_schedule(row) for row in per_schedule],
        "scope_note": (
            "Feature selection only. No GBDT fitting, lookup-table compilation, "
            "420-window simulation, sealed data, or Copilot data access. "
            "Feature and rule frequencies are diagnostics only; P1/P1b are chosen "
            "by mean schedule-normalized cross-fit cost."
        ),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp_path, out_path)

    print("\n===== GLOBAL FEATURE SELECTION =====")
    print(f"P1 pair: {p1['features']}  mean R={p1['mean_relative_cv_cost']:.12g}")
    print(f"P1b triple: {p1b['features']}  mean R={p1b['mean_relative_cv_cost']:.12g}")
    print("Rule/feature frequencies are diagnostics only; they did not select P1/P1b.")
    print(f"Saved: {out_path}")
    print("NOTE: This is exploratory development-data selection, not independent confirmation.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # concise CLI diagnostics; no silent recovery
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
