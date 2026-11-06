"""compare_grp_r.py — Verify rule group IDs across all 420 G2W3 windows.

Checks that the Python replica of C++ compute_grp_r exactly matches
the original p2.rule_groups() output stored in each window.
"""
import sys
sys.path.insert(0, 'exp/v2dev/g1')
sys.path.insert(0, 'exp/v2dev/g2')
import g1_common as G   # MUST be before numpy
import numpy as np
import g2_common as gc
from g2_common import read_window

pe, p2, p3, ma = G.research_modules()
import ailearn as al

from pathlib import Path

REPO = G.REPO
WINDOWS_DIR = REPO / "exp/v2dev/g2/windows"


# ---------------------------------------------------------------- Python replica of C++ compute_grp_r
def compute_grp_r_py(X_rule, rule_edges_list):
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


# ---------------------------------------------------------------- Load eval list → map name → path
paths = ma.read_list(G.EVAL_LIST, '창')
if len(paths) != G.N_EVAL:
    raise SystemExit(f"평가 창이 {G.N_EVAL}개가 아니다: {len(paths)}")

name_to_path = {ma.name_of(p): p for p in paths}

# ---------------------------------------------------------------- Main loop
total_slices = 0
total_grp_r_mismatch = 0
total_Gr_mismatch = 0
failed_windows = []

window_files = sorted(WINDOWS_DIR.glob("*.bin"))
n_windows = len(window_files)
print(f"Found {n_windows} window files in {WINDOWS_DIR}")

b = G.TSTAR

for i, win_path in enumerate(window_files):
    win_name = win_path.stem  # e.g. "bg2_h0157"

    # Load G2W3 window
    w = read_window(win_path)
    X_rule = w['X_rule']
    rule_edges = w['rule_edges']
    G_r_header = w['G_r']
    n = w['n']

    # Find the original trace path
    if win_name not in name_to_path:
        print(f"  [WARN] Window {win_name} not found in eval list, skipping")
        continue

    trace_path = name_to_path[win_name]

    # Load original trace
    tr = p2.Trace2(trace_path, G.FEATS)

    # Run original rule_groups
    cap = p2.tcap(b)
    VA, VA1 = tr.VA, tr.VA1
    h = tr.h
    Cva = pe.cost_matrix(tr.gva[VA], h[VA], tr.rva[VA], b)
    pos_va = np.where(VA)[0]
    is1 = VA1[pos_va]
    i_fix  = int(np.argmin(Cva.sum(1)))
    i_fix1 = int(np.argmin(Cva[:, is1].sum(1)))
    i_fix2 = int(np.argmin(Cva[:, ~is1].sum(1)))
    grp_r_orig, G_r_orig, _, _ = p2.rule_groups(tr, b, Cva, pos_va, is1, i_fix, i_fix1, i_fix2)

    # Recompute grp_r from X_rule + rule_edges
    grp_r_recomp, G_r_recomp = compute_grp_r_py(X_rule, rule_edges)

    # Compare
    n_slices = len(grp_r_orig)
    total_slices += n_slices

    mismatch_rows = int(np.sum(grp_r_orig != grp_r_recomp))
    total_grp_r_mismatch += mismatch_rows

    Gr_match = (G_r_orig == G_r_recomp) and (G_r_orig == G_r_header)
    if not Gr_match:
        total_Gr_mismatch += 1

    if mismatch_rows > 0 or not Gr_match:
        failed_windows.append(dict(
            name=win_name,
            mismatch_rows=mismatch_rows,
            G_r_orig=G_r_orig,
            G_r_recomp=G_r_recomp,
            G_r_header=G_r_header,
        ))

    if (i + 1) % 50 == 0:
        print(f"  Progress: {i+1}/{n_windows} windows processed  "
              f"(grp_r mismatches so far: {total_grp_r_mismatch}, "
              f"G_r mismatches: {total_Gr_mismatch})",
              flush=True)

# ---------------------------------------------------------------- Report
print()
print("=" * 60)
print("RESULTS")
print("=" * 60)
print(f"Total windows processed : {n_windows}")
print(f"Total slices            : {total_slices}")
print(f"grp_r mismatched rows   : {total_grp_r_mismatch}")
print(f"G_r mismatches          : {total_Gr_mismatch}")

if failed_windows:
    print()
    print("Failed windows:")
    for fw in failed_windows:
        print(f"  {fw['name']}: grp_r_mismatch={fw['mismatch_rows']}, "
              f"G_r_orig={fw['G_r_orig']}, G_r_recomp={fw['G_r_recomp']}, "
              f"G_r_header={fw['G_r_header']}")

print()
if total_grp_r_mismatch == 0 and total_Gr_mismatch == 0:
    print("PASS")
else:
    print("FAIL")
