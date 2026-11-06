#!/usr/bin/env python3
"""g2_preflight.py — G2 환경 및 소스 무결성 점검.

다음을 확인한다:
  1. g2_harness 바이너리 존재 여부
  2. G2 소스 파일 SHA-256 체크섬 (SHA256SUMS.txt 대조)
  3. G1 아티팩트 (table, grid) 로드 가능 여부
  4. frozen.json 상수 일치 여부
  5. p1b_model.bin 모델 파일 존재 여부
  6. 창 파일 G2W3 매직 바이트 검사 (windows/ 디렉터리)

실행(저장소 루트)
  python3 exp/v2dev/g2/g2_preflight.py [--win-dir exp/v2dev/g2/windows]
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
G2_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "exp/v2dev/g1"))

ERRORS = []

def check(cond, msg):
    if not cond:
        ERRORS.append(msg)
    return cond


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def check_shasums():
    sums_path = G2_DIR / "SHA256SUMS.txt"
    if not check(sums_path.exists(), f"SHA256SUMS.txt 없음: {sums_path}"):
        return
    ok = 0
    for line in sums_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        expected, rel = line.split(None, 1)
        abs_path = ROOT / rel
        if not check(abs_path.exists(), f"파일 없음: {rel}"):
            continue
        actual = sha256_file(abs_path)
        if check(actual == expected, f"체크섬 불일치: {rel}\n  expected {expected}\n  actual   {actual}"):
            ok += 1
    print(f"  체크섬: {ok} 파일 OK")


def check_harness():
    harness = G2_DIR / "g2_harness"
    check(harness.exists(), f"g2_harness 바이너리 없음: {harness}")
    if harness.exists():
        check(os.access(harness, os.X_OK), f"g2_harness 실행 권한 없음: {harness}")
        print(f"  g2_harness: {harness} OK")


def check_g1_artifacts():
    try:
        import g1_common as G
    except ImportError as e:
        ERRORS.append(f"g1_common 임포트 실패: {e}")
        return

    man_path = G.G1_DIR / G.MANIFEST_NAME
    if not check(man_path.exists(), f"G1 manifest 없음: {man_path}"):
        return

    man = G.load_manifest(G.G1_DIR)
    b_man = man["build"]["P1b"]
    c_man = man["compile"]["P1b"]

    grid_path = ROOT / b_man["grid"]["path"]
    table_path = ROOT / c_man["table"]["path"]

    check(grid_path.exists(), f"P1b grid 없음: {grid_path}")
    check(table_path.exists(), f"P1b table 없음: {table_path}")

    if grid_path.exists() and table_path.exists():
        import numpy as np
        grid_obj = G.grid_from_json(G.read_json(grid_path))
        table = np.load(table_path, allow_pickle=False)
        check(table.dtype == np.uint8, f"table dtype 오류: {table.dtype}")
        check(table.shape == (65, 65, 65), f"table shape 오류: {table.shape}")
        n_cuts = [len(c) for c in grid_obj.get("cutpoints", [])]
        print(f"  P1b 아티팩트: 컷포인트={n_cuts}, table={table.shape} OK")


def check_frozen():
    frozen_path = ROOT / "exp/frozen.json"
    if not check(frozen_path.exists(), f"frozen.json 없음: {frozen_path}"):
        return

    with open(frozen_path) as f:
        cfg = json.load(f)

    from g2_common import TSTAR, CAP, W_CAL, U_PERIOD, W_FB, KGRID

    checks = [
        ("wcal",    cfg.get("wcal"),    W_CAL,    "W_CAL"),
        ("period",  cfg.get("period"),  U_PERIOD, "U_PERIOD"),
        ("fb_W",    cfg.get("fb_W"),    W_FB,     "W_FB"),
        ("kgrid[0]",cfg.get("kgrid",[])[0] if cfg.get("kgrid") else None, KGRID[0], "KGRID[0]"),
        ("kgrid[-1]",cfg.get("kgrid",[])[- 1] if cfg.get("kgrid") else None, KGRID[-1], "KGRID[-1]"),
    ]
    for name, frozen_val, const_val, label in checks:
        check(frozen_val == const_val,
              f"frozen.json[{name}]={frozen_val} ≠ g2_common.{label}={const_val}")

    cap_expected = min(300.0, cfg.get("cap_mult", 20.0) * TSTAR)
    check(abs(CAP - cap_expected) < 1e-12,
          f"CAP={CAP} ≠ min(300, cap_mult×TSTAR)={cap_expected}")

    print("  frozen.json 상수 일치 OK")


def check_model_file():
    model_path = G2_DIR / "p1b_model.bin"
    if not check(model_path.exists(), f"p1b_model.bin 없음: {model_path}"):
        return
    sys.path.insert(0, str(G2_DIR))
    from g2_common import read_model, TABLE_DIM
    try:
        m = read_model(str(model_path))
        n_cuts = [len(c) for c in m["cuts"]]
        n_knots = len(m["knots"])
        t_shape = m["table"].shape
        check(t_shape == (TABLE_DIM, TABLE_DIM, TABLE_DIM),
              f"table shape 오류: {t_shape}")
        print(f"  p1b_model.bin: cuts={n_cuts}  knots={n_knots}  table={t_shape} OK")
    except Exception as e:
        ERRORS.append(f"p1b_model.bin 읽기 오류: {e}")


MAGIC_G2W3 = b"G2W3"
N_EVAL = 420


def check_window_magic(win_dir):
    """windows/ 의 모든 .bin 파일이 G2W3 매직 바이트를 갖는지 확인한다."""
    win_path = Path(win_dir)
    if not check(win_path.exists(), f"창 디렉터리 없음: {win_path}"):
        return

    bins = sorted(win_path.glob("*.bin"))
    if not check(len(bins) > 0, f"창 파일 없음: {win_path}"):
        return

    bad = []
    for f in bins:
        try:
            with open(f, "rb") as fp:
                magic = fp.read(4)
        except OSError as e:
            bad.append(f"{f.name}: 읽기 오류 ({e})")
            continue
        if magic != MAGIC_G2W3:
            bad.append(f"{f.name}: 매직={magic!r} (기대 {MAGIC_G2W3!r})")

    if bad:
        for msg in bad[:10]:
            ERRORS.append(f"G2W3 매직 불일치: {msg}")
        if len(bad) > 10:
            ERRORS.append(f"  ... (+{len(bad)-10}개)")
    else:
        print(f"  창 매직 G2W3: {len(bins)}개 전체 OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--win-dir", default=str(G2_DIR / "windows"),
                    help="창 파일 디렉터리 (기본: exp/v2dev/g2/windows)")
    a = ap.parse_args()

    print("=== G2 프리플라이트 점검 ===")

    sys.path.insert(0, str(G2_DIR))

    print("[1] g2_harness 바이너리")
    check_harness()

    print("[2] SHA-256 체크섬")
    check_shasums()

    print("[3] G1 아티팩트")
    check_g1_artifacts()

    print("[4] frozen.json 상수")
    check_frozen()

    print("[5] p1b_model.bin")
    check_model_file()

    print("[6] 창 파일 G2W3 매직 바이트")
    check_window_magic(a.win_dir)

    print()
    if ERRORS:
        print(f"실패 {len(ERRORS)}건:")
        for e in ERRORS:
            print(f"  ✗ {e}")
        sys.exit(1)
    else:
        print("모든 점검 PASS")


if __name__ == "__main__":
    main()
