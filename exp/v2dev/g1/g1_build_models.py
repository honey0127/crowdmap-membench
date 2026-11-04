#!/usr/bin/env python3
"""g1_build_models.py — G1 1·2단계: 공통 분위 격자 → 정수 코드 → P1·P1b 생존 GBDT(매듭 10개) 학습.

계획 docs/additional-analysis-plan-v4.md §2.3·§2.4를 따른다.
  학습원  26개 개발 일정(exp/dev_sched.txt)의 TR 행. 목록 순서·각 파일의 원래 행 순서로 붙이고, 가중치는 일정마다
          h / h.sum() × 1000(measure_a_cpu.prepare_pre, run_frozen.Pretrained와 같다). 조각 수는 820,340이어야 한다.
  격자    후보마다 따로, 선택 특징마다 TR 유한값 전체(가중치 없음)의 q = k/max_bins 분위수(numpy.quantile linear),
          정렬 후 중복 제거. P1 max_bins 255(결측 코드 255), P1b max_bins 64(결측 코드 64). ±Inf가 있으면 멈춘다.
  학습    measure_a_cpu.fit_knots와 같은 규칙(한 종류 정답 → 상수 y.mean(), 학습 오류는 적은 쪽 정답이 1개뿐일 때만 상수)에
          max_bins만 후보별로 준다. 모델 입력은 정수 코드이고 결측은 실제 NaN이다(sentinel 정수를 넣지 않는다).
산출(공식): exp/v2dev/g1/grids/grid-<후보>.json, models/model-<후보>.joblib, model-table-manifest.json의 build 절.
--dev-test면 exp/v2dev/g1/_devtest/ 아래에 쓰고 공식 결과로 쓰지 않는다.

실행(저장소 루트)
  taskset -c 1 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \\
    ~/.venvs/membench-g1/bin/python exp/v2dev/g1/g1_build_models.py
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import g1_common as G  # noqa: E402  (numpy보다 먼저: 스레드 환경 변수)

import numpy as np  # noqa: E402


def fit_knots_g1(pe, Xm, g, w, max_bins, rows_sha):
    """measure_a_cpu.fit_knots와 같은 학습 규칙에 max_bins만 더한다. 매듭별 기록을 함께 돌려준다."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import train_test_split

    models, rows = [], []
    for k, tau in enumerate(pe.TAU):
        t0 = G.time.process_time_ns()
        y = (g > tau).astype(int)
        kind, n_iter = "gbdt", 0
        if len(np.unique(y)) < 2:
            m, kind = (float(y.mean()) if len(y) else 0.0), "const_one_class"
        else:
            try:
                m = HistGradientBoostingClassifier(**G.HGB_PARAMS, max_bins=max_bins).fit(Xm, y, sample_weight=w)
                n_iter = int(m.n_iter_)
            except ValueError as e:
                if min(int(y.sum()), len(y) - int(y.sum())) >= 2:
                    raise SystemExit(f"매듭 {k}(τ = {tau}) 학습 불가: {type(e).__name__}: {e}") from e
                m, kind = float(y.mean()), "const_fit_error"
        dt = G.time.process_time_ns() - t0
        row = dict(knot=k, tau=float(tau), kind=kind, n_iter=n_iter, fit_cpu_ns=int(dt), n_rows=int(len(y)),
                   n_pos=int(y.sum()), n_neg=int(len(y) - y.sum()), w_pos=float(w[y == 1].sum()),
                   w_neg=float(w[y == 0].sum()), y_sha256=G.sha256_array(y.astype(np.int8)), rows_sha256=rows_sha)
        if kind == "gbdt":
            row["params"] = G.jsonable(m.get_params())
            row["do_early_stopping"] = bool(m.do_early_stopping_)
            row["internal_bin_thresholds"] = [dict(n=int(len(t)), sha256=G.sha256_array(np.asarray(t, np.float64)))
                                              for t in m._bin_mapper.bin_thresholds_]
            if m.do_early_stopping_:  # sklearn 내부 조기 종료 분할을 같은 시드로 재현한 진단값(정답으로 층화해 매듭마다 다르다)
                tr_idx, _ = train_test_split(np.arange(len(y)), test_size=m.validation_fraction, stratify=y,
                                             random_state=m._random_seed)
                row["internal_train_rows"] = dict(n=int(len(tr_idx)), sha256=G.sha256_array(np.sort(tr_idx)))
        else:
            row["const_value"] = float(m)
        models.append(m)
        rows.append(row)
        print(f"  매듭 {k} τ={tau:g}: {kind} n_iter={n_iter} 양성 {row['n_pos']}/{row['n_rows']} "
              f"{dt / 1e9:.1f}s", flush=True)
    return models, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    G.add_common_args(ap)
    a = ap.parse_args()
    cands = G.parse_candidates(a.candidates)
    if a.selection != G.SELECTION and not a.dev_test:
        raise SystemExit("--selection은 --dev-test에서만 바꾼다")
    env, issues = G.preflight(a, "build")
    out = G.out_dir(a)
    man = G.load_manifest(out)
    for cand in cands:
        if cand in man.get("build", {}) and not a.force:
            raise SystemExit(f"매니페스트에 {cand} 학습 기록이 이미 있다. 다시 만들려면 --force")

    pe, p2, p3, ma = G.research_modules()
    with G.Timer() as t_data:
        data = G.load_pretrain_rows(pe, p2, ma)
    X, g, w, row_ids = data["X"], data["g"], data["w"], data["row_ids"]
    sel, sel_sha = G.load_selection(a.selection, data["feat_names"])
    rows_sha = G.sha256_array(row_ids)
    print(f"학습원: 일정 {len(data['per_schedule'])}개, 조각 {len(g)}개, 가중치 합 {w.sum():.6f} "
          f"(준비 CPU {t_data.cpu_s:.1f}s)", flush=True)

    man.update(dict(
        dev_test=bool(a.dev_test), preflight_issues=issues, plan=dict(path=G.PLAN, sha256=G.sha256_file(G.REPO / G.PLAN)),
        selection=dict(path=a.selection, sha256=sel_sha, candidates=sel),
        runtime_environment_file=(dict(path=G.RUNTIME_ENV, sha256=G.sha256_file(G.REPO / G.RUNTIME_ENV))
                                  if (G.REPO / G.RUNTIME_ENV).is_file() else None),
        training_data=dict(list=G.DEV_SCHED, list_sha256=data["list_sha256"], per_schedule=data["per_schedule"],
                           n_rows=int(len(g)), weight_sum=float(w.sum()), weight_rule="h[TR] / h[TR].sum() * 1000 (일정마다)",
                           order="exp/dev_sched.txt 순서, 각 파일 TR 원래 행 순서", row_ids_sha256=rows_sha,
                           g_sha256=G.sha256_array(g), w_sha256=G.sha256_array(w), prep_cpu_s=t_data.cpu_s,
                           prep_wall_s=t_data.wall_s),
    ))
    man.setdefault("build", {})
    for cand in cands:
        spec = G.CANDIDATES[cand]
        nb, idxs = spec["max_bins"], sel[cand]["indices"]
        print(f"[{cand}] 특징 {sel[cand]['features']} max_bins={nb}", flush=True)
        with G.Timer() as t_grid:
            cuts, stats = zip(*[G.build_grid(X[:, fi], nb) for fi in idxs])
        grid = dict(candidate=cand, features=sel[cand]["features"], indices=idxs, max_bins=nb, sentinel=nb,
                    source="26개 개발 일정 TR 행을 목록 순서·원래 행 순서로 붙인 유한값 전체(가중치 없음)",
                    quantile=f"numpy.quantile(q = k/{nb}, k = 1..{nb - 1}, method='linear'), 정렬 후 중복 제거",
                    code_rule="numpy.searchsorted(cutpoints, x, side='right'); NaN → sentinel; ±Inf → 오류",
                    model_input_rule="유효 코드는 그 정수 값(float64), sentinel은 np.nan",
                    table_index_rule=f"C 순서, 축 = 특징 순서 {sel[cand]['features']}, 축 길이 {nb + 1}",
                    cutpoints=[c.tolist() for c in cuts], cutpoints_sha256=[G.sha256_array(c) for c in cuts],
                    stats=list(stats), source_data_sha256=G.sha256_array(X[:, idxs]))
        gobj = G.grid_from_json(grid)
        with G.Timer() as t_enc:
            codes = G.encode_matrix(X, gobj)
        for j, st in enumerate(stats):  # 코드화 점검: 유한값은 유효 범위, NaN은 sentinel
            cj = codes[:, j]
            fin = np.isfinite(X[:, idxs[j]])
            if not ((cj[fin] >= 0) & (cj[fin] < st["n_codes"])).all() or not (cj[~fin] == nb).all():
                raise SystemExit(f"[{cand}] 코드 범위 오류: 특징 {grid['features'][j]}")
            st["code_counts"] = np.bincount(cj, minlength=nb + 1).tolist()
        Xm = G.model_input(codes, nb)
        with G.Timer() as t_fit:
            models, knot_rows = fit_knots_g1(pe, Xm, g, w, nb, rows_sha)
        if len({r["rows_sha256"] for r in knot_rows}) != 1:
            raise SystemExit("매듭마다 학습 행 집합이 다르다")
        grid_path = out / "grids" / f"grid-{cand}.json"
        model_path = out / "models" / f"model-{cand}.joblib"
        grid_sha = G.write_json(grid_path, grid, a.force)
        bundle = dict(candidate=cand, tstar=G.TSTAR, knots=[float(t) for t in pe.TAU], models=models,
                      grid_sha256=grid_sha, features=sel[cand]["features"], indices=idxs, max_bins=nb,
                      sklearn=env["runtime"]["sklearn"], joblib=env["runtime"]["joblib"])
        model_sha = G.write_joblib(model_path, bundle, a.force)
        man["build"][cand] = dict(
            features=sel[cand]["features"], indices=idxs, max_bins=nb, sentinel=nb,
            grid=dict(path=G.rel(grid_path), sha256=grid_sha, stats=stats),
            model=dict(path=G.rel(model_path), sha256=model_sha),
            train_codes_sha256=G.sha256_array(codes), model_input_sha256=G.sha256_array(Xm),
            knots=knot_rows, n_gbdt=sum(r["kind"] == "gbdt" for r in knot_rows),
            n_const=sum(r["kind"] != "gbdt" for r in knot_rows),
            cpu_s=dict(grid=t_grid.cpu_s, encode=t_enc.cpu_s, fit=t_fit.cpu_s),
            wall_s=dict(grid=t_grid.wall_s, encode=t_enc.wall_s, fit=t_fit.wall_s))
        print(f"[{cand}] 격자 칸 수 {[s['n_codes'] for s in stats]}, 학습 {man['build'][cand]['n_gbdt']}개·상수 "
              f"{man['build'][cand]['n_const']}개, fit CPU {t_fit.cpu_s:.1f}s → {G.rel(model_path)}", flush=True)
    man["build_env"] = env
    G.save_manifest(out, man)
    print(f"매니페스트 → {G.rel(out / G.MANIFEST_NAME)}" + ("  [개발 시험: 공식 결과 아님]" if a.dev_test else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
