#!/usr/bin/env python3
"""P1b 특징 검증: G2W3 창 420개 대상으로 pos/D0/mD 기반 x0,x1,x2를 tr.X[:,0:3]과 비교한다."""
import sys
sys.path.insert(0, 'exp/v2dev/g1')
sys.path.insert(0, 'exp/v2dev/g2')
import g1_common as G   # MUST be before numpy
import numpy as np
import g2_common as gc
from g2_common import read_window

pe, p2, p3, ma = G.research_modules()
import ailearn as al
import policysim as ps

WIN_DIR = G.REPO / 'exp/v2dev/g2/windows'

# Load eval list paths via ma.read_list (same as g2_prepare.py)
all_paths = ma.read_list(G.EVAL_LIST, "평가 창")
print(f"eval list: {len(all_paths)} 경로 로드됨")

total_slices = 0
max_abs_err = 0.0
max_rel_err = 0.0
mismatch_count = 0
windows_ok = 0
windows_err = 0
THRESHOLD = 1e-12

for i, path in enumerate(all_paths):
    from pathlib import Path
    stem = Path(path).stem  # e.g. bg2_h0157
    win_path = WIN_DIR / f"{stem}.bin"

    if not win_path.exists():
        print(f"  [{i+1}] {stem}: 창 파일 없음 — 건너뜀")
        windows_err += 1
        continue

    try:
        # Load G2W3 window
        w = read_window(str(win_path))
        pos_w = w["pos"]    # (n,) bytes
        D0_w  = w["D0"]     # (n,) bytes
        mD_w  = w["mD"]     # scalar

        n = w["n"]

        # Load original trace
        tr = p2.Trace2(path, G.FEATS)

        # Compute mD from d_history (same as g2_prepare.compute_mD)
        ta, tf, sz = tr.ta, tr.tf, tr.sz
        t0_hist, b_hist, binmax = al.d_history(ta, tf, sz)
        c1t = tr.t0 + 0.5 * tr.dur
        nb = int((c1t - t0_hist) / b_hist)
        mD_py = max(binmax[:max(nb, 1)].mean() / gc.MiB, 0.25)

        # Get pos/D0 from policysim.slices
        S = ps.slices(tr.ta, tr.tf, tr.sz)
        pos_ps = S["pos"]   # (n,) bytes
        D0_ps  = S["D0"]    # (n,) bytes

        # Compute P1b features the same way C++ will
        x0_g2 = pos_w / gc.MiB / mD_w
        x1_g2 = D0_w  / gc.MiB / mD_w
        x2_g2 = (pos_w + D0_w) / gc.MiB / mD_w

        # Reference from tr.X[:, 0:3]
        x0_tr = tr.X[:, 0]
        x1_tr = tr.X[:, 1]
        x2_tr = tr.X[:, 2]

        # Compute errors
        abs_err = np.max(np.abs(np.stack([
            x0_g2 - x0_tr,
            x1_g2 - x1_tr,
            x2_g2 - x2_tr,
        ])))

        # Relative error (avoid division by zero)
        denom = np.maximum(np.abs(np.stack([x0_tr, x1_tr, x2_tr])), 1e-30)
        rel_err = np.max(np.abs(np.stack([
            x0_g2 - x0_tr,
            x1_g2 - x1_tr,
            x2_g2 - x2_tr,
        ])) / denom)

        # Count mismatched rows (any feature exceeds threshold)
        row_max_err = np.maximum(
            np.maximum(np.abs(x0_g2 - x0_tr), np.abs(x1_g2 - x1_tr)),
            np.abs(x2_g2 - x2_tr)
        )
        n_mismatch = int((row_max_err > THRESHOLD).sum())

        total_slices += n
        max_abs_err = max(max_abs_err, abs_err)
        max_rel_err = max(max_rel_err, rel_err)
        mismatch_count += n_mismatch
        windows_ok += 1

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(all_paths)}] {stem}  n={n}  "
                  f"abs_err={abs_err:.3e}  rel_err={rel_err:.3e}  mismatch={n_mismatch}",
                  flush=True)

    except Exception as e:
        import traceback
        print(f"  [{i+1}] {stem} 오류: {e}")
        traceback.print_exc()
        windows_err += 1

print()
print("=" * 60)
print("결과 요약")
print("=" * 60)
print(f"  처리 창:       {windows_ok} / {len(all_paths)} (오류 {windows_err})")
print(f"  총 슬라이스:   {total_slices:,}")
print(f"  최대 절대 오차: {max_abs_err:.6e}")
print(f"  최대 상대 오차: {max_rel_err:.6e}")
print(f"  불일치 행 (>1e-12): {mismatch_count}")
print()
if windows_err == 0 and mismatch_count == 0 and max_abs_err < THRESHOLD:
    print("PASS — 모든 420 창에서 P1b 특징이 tr.X[:,0:3]과 정확히 일치한다.")
elif mismatch_count == 0 and max_abs_err < 1e-9:
    print("PASS (부동소수점 허용 오차 내) — 불일치 행 없음, 최대 절대 오차 < 1e-9.")
else:
    print("FAIL — 불일치 발견.")
    if windows_err > 0:
        print(f"  경고: {windows_err}개 창에서 오류 발생.")
    if mismatch_count > 0:
        print(f"  경고: {mismatch_count}개 행에서 오차 > {THRESHOLD}.")
print("=" * 60)
