#!/usr/bin/env python3
"""g1_verify_tables.py — G1 4단계: 조회표와 저차원 GBDT 직접 계산의 동등성 검증. 비용 시뮬레이션은 하지 않는다.

검증 E  코드화 경계: 각 cutpoint 값, 바로 아래·위(nextafter), 학습 최솟값·최댓값과 그 바깥, NaN → sentinel,
        ±Inf → 오류. 기대값은 searchsorted와 따로 '그 값 이하 cutpoint 수'로 계산한다. 학습 TR 전체를 한 번에·일정별로
        코드화한 결과가 같고, 학습 단계의 코드 빈도·해시와 같은지도 본다.
검증 A  모든 코드 조합(P1 256², P1b 65³): 디스크에서 다시 읽은 모델로 직접 계산한 색인 = 디스크에서 읽은 표 값.
        묶음 크기(VERIFY_BATCH)를 생성 때와 다르게 둔다.
검증 B  420개 개발 창(exp/v2dev/daypool/all.txt, 작업 A 목록 해시와 대조)의 모든 청크 행: 같은 청크 모델·Trace2 특징 →
        공통 격자 코드 → 표 값 = 모델 직접 계산. 시험 구간(TE)·결정 구간(VA∪TE) 행 수를 따로 센다.
목록  검증 전에 학습 26일정(exp/dev_sched.txt)과 평가 창의 이름(measure_a_cpu.name_of) 교집합이 0인지 본다.
        겹치면 공식 검증은 BLOCKED로 멈추고, 겹치지 않으면 학습·평가·교집합 개수를 매니페스트에 적는다.
판정  후보마다 E·A·B 불일치 0이면 PASS, 하나라도 있으면 FAIL. --dev-test 실행은 판정하지 않는다.
산출(공식): model-table-manifest.json의 verify 절, exp/v2dev/g1/verify/actual-windows.csv.

실행(저장소 루트)
  taskset -c 1 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \\
    ~/.venvs/membench-g1/bin/python exp/v2dev/g1/g1_verify_tables.py
"""
import argparse
import csv
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import g1_common as G  # noqa: E402  (numpy보다 먼저: 스레드 환경 변수)

import numpy as np  # noqa: E402

MAX_EXAMPLES = 20


def boundary_tests(grid):
    sent = grid["sentinel"]
    n, fails, ex = 0, 0, []
    for j, cut in enumerate(grid["cutpoints"]):
        st = grid["stats"][j]
        lo, hi = st["train_min"], st["train_max"]
        v = np.concatenate([cut, np.nextafter(cut, -np.inf), np.nextafter(cut, np.inf),
                            [lo, hi, np.nextafter(lo, -np.inf), np.nextafter(hi, np.inf), lo - 1.0, hi + 1.0,
                             -1e300, 1e300]])
        got = G.encode(v, cut, sent).astype(np.int64)
        exp = (cut[None, :] <= v[:, None]).sum(1)
        bad = np.nonzero(got != exp)[0]
        n, fails = n + len(v), fails + len(bad)
        ex += [dict(feature=grid["features"][j], value=float(v[i]), got=int(got[i]), expected=int(exp[i]))
               for i in bad[:MAX_EXAMPLES]]
        special = [(G.encode([lo - 1.0], cut, sent)[0] == 0, "학습 최솟값 아래 → 0"),
                   (G.encode([hi + 1.0], cut, sent)[0] == len(cut), "학습 최댓값 위 → 마지막 코드"),
                   (G.encode([np.nan], cut, sent)[0] == sent, "NaN → sentinel")]
        for x in (np.inf, -np.inf):
            try:
                G.encode([x], cut, sent)
                special.append((False, f"{x} → 오류여야 한다"))
            except ValueError:
                special.append((True, f"{x} → 오류"))
        n += len(special)
        for ok, what in special:
            if not ok:
                fails += 1
                ex.append(dict(feature=grid["features"][j], check=what))
    return dict(n_tests=n, failures=fails, examples=ex[:MAX_EXAMPLES])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    G.add_common_args(ap)
    ap.add_argument("--eval-list", default=G.EVAL_LIST, help="검증 B 입력 목록(공식 실행은 바꾸지 않는다)")
    a = ap.parse_args()
    cands = G.parse_candidates(a.candidates)
    if a.eval_list != G.EVAL_LIST and not a.dev_test:
        raise SystemExit("--eval-list는 --dev-test에서만 바꾼다")
    env, issues = G.preflight(a, "verify")
    pe, p2, p3, ma = G.research_modules()

    # 학습·평가 일정 중복 차단(measure_a_cpu와 같은 목록 규칙). 겹치면 공식 검증을 하지 않는다(어떤 계산·기록보다 먼저).
    overlap = G.check_list_overlap(ma, a.eval_list)
    has_overlap = overlap["n_intersection_stem"] > 0 or overlap["n_intersection_basename"] > 0
    if has_overlap:
        parts = []
        if overlap["n_intersection_stem"]:
            parts.append(f"일정이름 {overlap['n_intersection_stem']}개(예: {overlap['intersection_stem'][:3]})")
        if overlap["n_intersection_basename"]:
            parts.append(f"파일이름 {overlap['n_intersection_basename']}개(예: {overlap['intersection_basename'][:3]})")
        msg = "학습·평가 일정이 겹친다: " + ", ".join(parts)
        if not a.dev_test:
            raise SystemExit(f"[verify] BLOCKED: {msg}")
        print(f"[개발 시험] {msg} — 공식 실행이면 BLOCKED다", flush=True)
    elif not a.dev_test and (overlap["n_train"] != G.N_DEV or overlap["n_eval"] != G.N_EVAL):
        raise SystemExit(f"[verify] BLOCKED: 학습 {overlap['n_train']}개·평가 {overlap['n_eval']}개(기대 {G.N_DEV}·{G.N_EVAL})")
    print(f"목록: 학습 {overlap['n_train']}개, 평가 {overlap['n_eval']}개, "
          f"교집합(일정이름) {overlap['n_intersection_stem']}개 / (파일이름) {overlap['n_intersection_basename']}개",
          flush=True)

    out = G.out_dir(a)
    man = G.load_manifest(out)
    if man.get("dev_test", False) != bool(a.dev_test):
        raise SystemExit("매니페스트의 개발 시험 여부와 이번 실행이 다르다")
    for cand in cands:
        if cand not in man.get("compile", {}):
            raise SystemExit(f"{cand} 표 기록이 없다. g1_compile_tables.py를 먼저 돌린다")
        if cand in man.get("verify", {}) and not a.force:
            raise SystemExit(f"매니페스트에 {cand} 검증 기록이 이미 있다. 다시 하려면 --force")
    import joblib

    # 산출물을 디스크에서 다시 읽는다(해시 대조)
    art = {}
    for cand in cands:
        b, c = man["build"][cand], man["compile"][cand]
        G.check_sha(G.REPO / b["grid"]["path"], b["grid"]["sha256"], "격자")
        G.check_sha(G.REPO / b["model"]["path"], b["model"]["sha256"], "모델")
        G.check_sha(G.REPO / c["table"]["path"], c["table"]["sha256"], "조회표")
        grid = G.grid_from_json(G.read_json(G.REPO / b["grid"]["path"]))
        table = np.load(G.REPO / c["table"]["path"], allow_pickle=False)
        if table.dtype != np.uint8 or list(table.shape) != c["table"]["shape"] or table.max() > 10:
            raise SystemExit(f"[{cand}] 표 dtype·shape·값 범위가 기록과 다르다")
        art[cand] = dict(grid=grid, table=table, models=joblib.load(G.REPO / b["model"]["path"])["models"])

    res = {cand: {} for cand in cands}
    # 검증 E: 경계 시험과 학습 코드 일관성
    data = G.load_pretrain_rows(pe, p2, ma)
    starts = np.cumsum([0] + [p["n_tr"] for p in data["per_schedule"] if p["n_tr"] > 0])
    for cand in cands:
        grid = art[cand]["grid"]
        bt = boundary_tests(grid)
        whole = G.encode_matrix(data["X"], grid)
        pieces = np.vstack([G.encode_matrix(data["X"][s:e], grid) for s, e in zip(starts[:-1], starts[1:])])
        same_split = bool((whole == pieces).all())
        same_hash = G.sha256_array(whole) == man["build"][cand]["train_codes_sha256"]
        same_counts = all(np.bincount(whole[:, j], minlength=grid["sentinel"] + 1).tolist() == st["code_counts"]
                          for j, st in enumerate(man["build"][cand]["grid"]["stats"]))
        bt["train_codes_whole_vs_per_schedule"] = same_split
        bt["train_codes_match_build"] = bool(same_hash and same_counts)
        bt["failures"] += int(not same_split) + int(not (same_hash and same_counts))
        res[cand]["boundary"] = bt
        print(f"[{cand}] 경계 시험 {bt['n_tests']}건, 실패 {bt['failures']}건", flush=True)
    del data

    # 검증 A: 모든 코드 조합
    for cand in cands:
        grid, table, models = art[cand]["grid"], art[cand]["table"], art[cand]["models"]
        shape, flat = table.shape, table.reshape(-1)
        total, mism, ex = int(flat.size), 0, []
        with G.Timer() as t:
            for start in range(0, total, G.VERIFY_BATCH):
                stop = min(total, start + G.VERIFY_BATCH)
                codes = G.all_codes(shape, start, stop)
                idx = G.predict_index(pe, ma, models, G.model_input(codes, grid["sentinel"]))
                bad = np.nonzero(idx != flat[start:stop])[0]
                mism += len(bad)
                ex += [dict(codes=codes[i].tolist(), table=int(flat[start + i]), model=int(idx[i]))
                       for i in bad[:MAX_EXAMPLES - len(ex)]]
        res[cand]["exhaustive"] = dict(n_states=total, mismatches=int(mism), batch=G.VERIFY_BATCH, examples=ex,
                                       cpu_s=t.cpu_s, wall_s=t.wall_s)
        print(f"[{cand}] 전 조합 {total:,}개, 불일치 {mism}건 ({t.cpu_s:.1f}s)", flush=True)

    # 검증 B: 실제 평가 입력
    paths = ma.read_list(a.eval_list, "평가 창")
    list_sha = G.sha256_file(G.REPO / a.eval_list)
    ref = G.read_json(G.REPO / G.CPU_A_ENV)["targets"] if (G.REPO / G.CPU_A_ENV).is_file() else None
    if not a.dev_test:
        if len(paths) != G.N_EVAL:
            raise SystemExit(f"평가 창이 {G.N_EVAL}개가 아니다: {len(paths)}")
        if ref is None or ref.get("sha256") != list_sha or ref.get("count_used") != G.N_EVAL:
            raise SystemExit(f"평가 목록이 작업 A 목록({G.CPU_A_ENV})과 다르다")
    acc = {cand: dict(n_rows=0, n_dec=0, n_te=0, mismatches=0, mismatches_te=0, sentinel_rows=0,
                      nan_by_feature={f: 0 for f in art[cand]["grid"]["features"]}, inf_rows=0, examples=[])
           for cand in cands}
    rows_csv = []
    with G.Timer() as t_b:
        for wi, path in enumerate(paths):
            tr = p2.Trace2(path, G.FEATS)
            dec, te = tr.VA | tr.TE, tr.TE
            for cand in cands:
                grid, table, models, A = art[cand]["grid"], art[cand]["table"], art[cand]["models"], acc[cand]
                Xs = tr.X[:, grid["indices"]]
                inf = np.isinf(Xs).any(1)
                if inf.any():
                    A["inf_rows"] += int(inf.sum())
                    continue
                codes = G.encode_matrix(tr.X, grid)
                tab = table[tuple(codes.T)]
                mod = G.predict_index(pe, ma, models, G.model_input(codes, grid["sentinel"]))
                bad = tab != mod
                A["n_rows"] += len(tab)
                A["n_dec"] += int(dec.sum())
                A["n_te"] += int(te.sum())
                A["mismatches"] += int(bad.sum())
                A["mismatches_te"] += int((bad & te).sum())
                A["sentinel_rows"] += int((codes == grid["sentinel"]).any(1).sum())
                for j, f in enumerate(grid["features"]):
                    A["nan_by_feature"][f] += int(np.isnan(Xs[:, j]).sum())
                for i in np.nonzero(bad)[0][:max(0, MAX_EXAMPLES - len(A["examples"]))]:
                    A["examples"].append(dict(window=tr.name, row=int(i), codes=codes[i].tolist(),
                                              raw=Xs[i].tolist(), table=int(tab[i]), model=int(mod[i])))
                rows_csv.append(dict(candidate=cand, window=tr.name, n_rows=len(tab), n_dec=int(dec.sum()),
                                     n_te=int(te.sum()), mismatches=int(bad.sum()),
                                     sentinel_rows=int((codes == grid["sentinel"]).any(1).sum())))
            if (wi + 1) % 20 == 0 or wi + 1 == len(paths):
                print(f"  창 {wi + 1}/{len(paths)} " + ", ".join(
                    f"{c} 행 {acc[c]['n_rows']:,} 불일치 {acc[c]['mismatches']}" for c in cands), flush=True)
    for cand in cands:
        res[cand]["actual"] = dict(list=a.eval_list, list_sha256=list_sha, cpu_a_list_sha256=(ref or {}).get("sha256"),
                                   n_windows=len(paths), **acc[cand], cpu_s=t_b.cpu_s, wall_s=t_b.wall_s)

    verify_dir = out / "verify"
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=["candidate", "window", "n_rows", "n_dec", "n_te", "mismatches", "sentinel_rows"])
    w.writeheader()
    w.writerows(rows_csv)
    csv_path = verify_dir / ("actual-windows.csv" if len(cands) == 2 else f"actual-windows-{'-'.join(cands)}.csv")
    G.write_text(csv_path, buf.getvalue(), a.force)

    man.setdefault("verify", {})
    verdicts = {}
    for cand in cands:
        r = res[cand]
        ok = (r["boundary"]["failures"] == 0 and r["exhaustive"]["mismatches"] == 0
              and r["actual"]["mismatches"] == 0 and r["actual"]["inf_rows"] == 0)
        verdicts[cand] = "PASS" if ok else "FAIL"
        man["verify"][cand] = dict(**r, verdict=verdicts[cand])
    overall = ("DEV-TEST(판정 없음)" if a.dev_test else
               "PASS" if all(v == "PASS" for v in verdicts.values()) else "FAIL")
    man["verify_summary"] = dict(candidates=verdicts, overall=overall, actual_windows_csv=dict(
        path=G.rel(csv_path), sha256=G.sha256_file(csv_path)))
    man["list_overlap"] = overlap
    man["verify_env"] = env
    G.save_manifest(out, man)
    print("===== G1 TABLE EQUIVALENCE =====")
    for cand in cands:
        r = res[cand]
        print(f"[{cand}] 경계 실패 {r['boundary']['failures']} / 전 조합 {r['exhaustive']['n_states']:,}개 불일치 "
              f"{r['exhaustive']['mismatches']} / 실제 {r['actual']['n_windows']}창 {r['actual']['n_rows']:,}행"
              f"(TE {r['actual']['n_te']:,}) 불일치 {r['actual']['mismatches']} → {verdicts[cand]}")
    print(f"최종: {overall}")
    return 0 if overall != "FAIL" else 1


if __name__ == "__main__":
    sys.exit(main())
