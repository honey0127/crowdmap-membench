#!/usr/bin/env python3
"""g1_compile_tables.py — G1 3단계: 학습한 저차원 생존 GBDT를 모든 코드 조합에서 평가해 uint8 조회표로 컴파일한다.

표 칸 = 그 코드 조합에서 10개 매듭 모델 → measure_a_cpu.predict_knots(단조 감소 보정) →
paperexp.survival_timeouts(Sk, T* = 2.6, κ = 1)이 고른 기본 반환 시점의 KNOTS 위치(0 = 즉시, 1..10 = τ1..τ10).
조합은 C 순서로 COMPILE_BATCH개씩 만들고, 결측 축 값(sentinel)은 모델 입력에서 NaN으로 되돌린다.
15특징 모델을 근사하는 학생 모델이 아니다. 코드화한 저차원 모델 출력을 그대로 담는다.
산출(공식): exp/v2dev/g1/tables/lookup-<후보>.npy, model-table-manifest.json의 compile 절.

실행(저장소 루트)
  taskset -c 1 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \\
    ~/.venvs/membench-g1/bin/python exp/v2dev/g1/g1_compile_tables.py
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import g1_common as G  # noqa: E402  (numpy보다 먼저: 스레드 환경 변수)

import numpy as np  # noqa: E402

UNFILLED = 255  # 채우지 않은 칸 표시(유효 값은 0..10)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    G.add_common_args(ap)
    a = ap.parse_args()
    cands = G.parse_candidates(a.candidates)
    env, issues = G.preflight(a, "compile")
    out = G.out_dir(a)
    man = G.load_manifest(out)
    if man.get("dev_test", False) != bool(a.dev_test):
        raise SystemExit("매니페스트의 개발 시험 여부와 이번 실행이 다르다")
    import joblib
    pe, p2, p3, ma = G.research_modules()
    K = G.knots(pe)
    if len(K) != 11:
        raise SystemExit(f"기본 반환 시점 선택지가 11개가 아니다: {len(K)}")
    man.setdefault("compile", {})
    for cand in cands:
        b = man.get("build", {}).get(cand)
        if b is None:
            raise SystemExit(f"{cand} 학습 기록이 없다. g1_build_models.py를 먼저 돌린다")
        if cand in man["compile"] and not a.force:
            raise SystemExit(f"매니페스트에 {cand} 표 기록이 이미 있다. 다시 만들려면 --force")
        grid_path, model_path = G.REPO / b["grid"]["path"], G.REPO / b["model"]["path"]
        G.check_sha(grid_path, b["grid"]["sha256"], "격자")
        G.check_sha(model_path, b["model"]["sha256"], "모델")
        grid = G.grid_from_json(G.read_json(grid_path))
        bundle = joblib.load(model_path)
        if bundle["grid_sha256"] != b["grid"]["sha256"] or len(bundle["models"]) != len(pe.TAU):
            raise SystemExit(f"[{cand}] 모델 묶음이 격자·매듭 수와 맞지 않는다")
        shape = G.table_shape(grid)
        total = int(np.prod(shape))
        table = np.full(shape, UNFILLED, dtype=np.uint8)
        flat = table.reshape(-1)
        with G.Timer() as t:
            for start in range(0, total, G.COMPILE_BATCH):
                stop = min(total, start + G.COMPILE_BATCH)
                codes = G.all_codes(shape, start, stop)
                flat[start:stop] = G.predict_index(pe, ma, bundle["models"], G.model_input(codes, grid["sentinel"]))
        if (flat == UNFILLED).any():
            raise SystemExit(f"[{cand}] 채우지 않은 칸이 {int((flat == UNFILLED).sum())}개 있다")
        path = out / "tables" / f"lookup-{cand}.npy"
        sha = G.write_npy(path, table, a.force)
        counts = np.bincount(flat, minlength=len(K)).tolist()
        man["compile"][cand] = dict(
            table=dict(path=G.rel(path), sha256=sha, array_sha256=G.sha256_array(table), dtype=str(table.dtype),
                       shape=list(shape), nbytes=int(table.nbytes)),
            n_states=total, batch=G.COMPILE_BATCH, index_meaning="KNOTS 위치: 0 = 즉시(0 s), k = τk",
            knots_s=K.tolist(), tstar=G.TSTAR, value_counts=counts, cpu_s=t.cpu_s, wall_s=t.wall_s,
            grid_sha256=b["grid"]["sha256"], model_sha256=b["model"]["sha256"])
        print(f"[{cand}] 표 {shape} uint8 {table.nbytes:,} B, 칸 {total:,}, 값 분포 {counts}, CPU {t.cpu_s:.1f}s "
              f"→ {G.rel(path)}", flush=True)
    man["compile_env"] = env
    G.save_manifest(out, man)
    return 0


if __name__ == "__main__":
    sys.exit(main())
