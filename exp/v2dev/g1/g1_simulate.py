#!/usr/bin/env python3
"""g1_simulate.py — G1 경제성 평가 시뮬레이터: P1/P1b 조회표를 C3 정책에 연결해 평가 창에서
ΔR, ΔH, S_model을 계산한다.

계획: docs/additional-analysis-plan-v4.md §3.6–§3.7.
절차:
  1. G1 조회표(P1/P1b)를 로드하고 매니페스트의 verify PASS 상태를 확인한다.
  2. 평가 창마다 lookup-rec → κ→C3 경로를 실행한다.
     κ·대체·전환 설정은 기존 C3와 동일하다(변경하지 않는다).
     rec = KNOTS[table[codes]]를 pe.survival_timeouts(Sk, T*, 1.0) 대신 쓴다.
  3. ΔR, ΔH, S_model, S_model×c를 창별·후보별로 집계한다.
  4. 사전 고정 통과 기준(S_model×c ≥ 1.0 s)으로 후보를 선별한다.

산출(공식):
  exp/v2dev/g1/simulation/summary.json
  exp/v2dev/g1/simulation/windows-P1.csv
  exp/v2dev/g1/simulation/windows-P1b.csv
  exp/v2dev/g1/simulation/run-environment.json

실행(저장소 루트)
  taskset -c 1 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \\
    ~/.venvs/membench-g1/bin/python exp/v2dev/g1/g1_simulate.py
"""
import argparse
import csv
import io
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import g1_common as G  # noqa: E402  (numpy보다 먼저: 스레드 환경 변수)

import numpy as np  # noqa: E402

C_NS = 281_800          # 비용 1단위(재확보 1 MiB)의 CPU 등가 (ns/MiB), measure_a_cpu.C_NS와 같다
THRESHOLD_SC_S = 1.0    # 통과 기준: S_model × c ≥ 1.0 s (계획 v4 §3)
SIM_DIR = "simulation"

W_FIELDS = [
    "trace", "n_te", "n_dec", "tau_s",
    "rule_refault_MiB", "rule_waste_MiB",
    "c3_refault_MiB", "c3_waste_MiB",
    "dR_MiB", "dH_MiB", "S_units", "Sc_ns",
    "c3_rule_mode_frac",
    "oracle_refault_MiB", "oracle_waste_MiB",
    "rule_name",
]


def select_g2_candidate(summaries, cands):
    """통과 후보 목록에서 G2로 보낼 후보를 고른다(계획 v4 §3).
    반환: (cand_or_None, verdict_str, reason_str)"""
    passing = [c for c in cands if summaries[c]["verdict_en"] == "PASS"]
    if not passing:
        return None, "없음", "두 후보 모두 S_model×c < 1.0 s"
    if len(passing) == 1:
        c = passing[0]
        return c, c, f"{c}만 통과"
    # 둘 다 통과: S_model×c가 큰 쪽 선택. 값이 같으면 더 작은 표 P1 선택(계획 v4 §3)
    sc_p1 = summaries.get("P1", {}).get("S_total_times_c_s", -math.inf)
    sc_p1b = summaries.get("P1b", {}).get("S_total_times_c_s", -math.inf)
    if sc_p1 >= sc_p1b:
        return "P1", "P1", f"둘 다 통과, P1의 S_model×c가 같거나 더 크다 (P1={sc_p1:.4f}s)"
    return "P1b", "P1b", f"둘 다 통과, P1b의 S_model×c가 더 크다 (P1b={sc_p1b:.4f}s > P1={sc_p1:.4f}s)"


def _assert_te_counts(win_rows, cands, paths, dev_test):
    """GAP-8 보수 검사: 420창 전체 TE 행 수·중복·후보 간 일치를 확인한다.
    실패하면 결과 파일 저장 전에 멈춘다."""
    # 1. 창 수 확인
    n_wins = len(list(win_rows.values())[0])
    if n_wins != len(paths):
        raise SystemExit(f"[GAP-8] 처리한 창 수 {n_wins} != 경로 수 {len(paths)}")

    # 2. 추적 이름 중복 없음
    first_cand = cands[0]
    trace_names = [r["trace"] for r in win_rows[first_cand]]
    seen: set = set()
    dups = []
    for t in trace_names:
        if t in seen:
            dups.append(t)
        seen.add(t)
    if dups:
        raise SystemExit(f"[GAP-8] 추적 이름 중복: {dups}")

    # 3. 후보 간 창별 n_te 동일
    if len(cands) >= 2:
        for i in range(n_wins):
            n_te_values = {c: win_rows[c][i]["n_te"] for c in cands}
            if len(set(n_te_values.values())) > 1:
                raise SystemExit(
                    f"[GAP-8] 창 {i} ({win_rows[cands[0]][i]['trace']}): "
                    f"후보 간 n_te 불일치 {n_te_values}"
                )

    # 4. pre_sim_costs.csv와 창별 n_te 대조
    pre_path = G.REPO / G.PRE_SIM_COSTS
    if pre_path.exists():
        with open(pre_path, newline="", encoding="utf-8") as f:
            pre_map = {r["trace"]: int(r["n_te"]) for r in csv.DictReader(f)}
        for cand in cands:
            for row in win_rows[cand]:
                name = row["trace"]
                if name in pre_map and row["n_te"] != pre_map[name]:
                    raise SystemExit(
                        f"[GAP-8] [{cand}] {name}: n_te={row['n_te']} "
                        f"!= pre_sim_costs.csv={pre_map[name]}"
                    )

    # 5. 공식 실행 시 총 TE 행 수 == N_TE_TOTAL
    for cand in cands:
        total = sum(r["n_te"] for r in win_rows[cand])
        if not dev_test and total != G.N_TE_TOTAL:
            raise SystemExit(
                f"[GAP-8] [{cand}] sum(n_te)={total:,} != "
                f"N_TE_TOTAL={G.N_TE_TOTAL:,}"
            )

    if not dev_test:
        totals = {c: sum(r["n_te"] for r in win_rows[c]) for c in cands}
        print("[GAP-8] TE 행 수 단언 통과: "
              + ", ".join(f"{c}={v:,}" for c, v in totals.items())
              + f" (기대 {G.N_TE_TOTAL:,})", flush=True)


def _sk_none_coverage():
    """within_runs.csv에서 ai=none 창을 찾아 커버리지 메타데이터를 반환한다.
    파일이 없으면 None을 반환한다."""
    within_path = G.REPO / "exp/v2dev/cpu_a/within_runs.csv"
    pre_path = G.REPO / G.PRE_SIM_COSTS
    if not within_path.exists() or not pre_path.exists():
        return None

    with open(within_path, newline="", encoding="utf-8") as f:
        none_traces = {r["trace"] for r in csv.DictReader(f) if r["ai"] == "none"}

    if not none_traces:
        return dict(n_windows=0, traces=[], n_te=0, n_te_total=G.N_TE_TOTAL,
                    fraction=0.0,
                    note="모든 창에서 AI 모델이 사용됨(Sk=None 창 없음).")

    with open(pre_path, newline="", encoding="utf-8") as f:
        pre_map = {r["trace"]: int(r["n_te"]) for r in csv.DictReader(f)}

    none_te = sum(pre_map.get(t, 0) for t in none_traces)
    total_te = sum(pre_map.values())
    return dict(
        n_windows=len(none_traces),
        traces=sorted(none_traces),
        n_te=none_te,
        n_te_total=total_te,
        fraction=none_te / total_te if total_te > 0 else 0.0,
        note=(
            "evaluate3()에서 이 창들은 Sk=None → T_rb(규칙)를 C3로 썼다. "
            "G1은 사전 학습 조회표를 항상 쓰므로 이 창들에서도 C3 경로가 작동한다. "
            "ΔR·ΔH·S_model은 G1 조회표 기반으로 계산한 값이다."
        ),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    G.add_common_args(ap)
    ap.add_argument("--eval-list", default=G.EVAL_LIST, help="평가 창 목록(공식 실행은 바꾸지 않는다)")
    a = ap.parse_args()
    cands = G.parse_candidates(a.candidates)
    if a.eval_list != G.EVAL_LIST and not a.dev_test:
        raise SystemExit("--eval-list는 --dev-test에서만 바꾼다")

    env, issues = G.preflight(a, "simulate")
    pe, p2, p3, ma = G.research_modules()

    out = G.out_dir(a)
    man = G.load_manifest(out)
    if man.get("dev_test", False) != bool(a.dev_test):
        raise SystemExit("매니페스트의 개발 시험 여부와 이번 실행이 다르다")

    sim_dir = out / SIM_DIR
    sentinel = sim_dir / "summary.json"
    if sentinel.exists() and not a.force:
        raise SystemExit(f"시뮬레이션 결과가 이미 있다({G.rel(sentinel)}). "
                         "다시 실행하려면 --force.")

    # 조회표 로드 (verify PASS 확인)
    art = {}
    for cand in cands:
        v = man.get("verify", {}).get(cand, {})
        if v.get("verdict") != "PASS" and not a.dev_test:
            raise SystemExit(f"[{cand}] 조회표 검증이 PASS가 아니다({v.get('verdict', '없음')}). "
                             "g1_verify_tables.py를 먼저 성공적으로 실행한다.")
        b_man = man["build"][cand]
        c_man = man["compile"][cand]
        G.check_sha(G.REPO / b_man["grid"]["path"], b_man["grid"]["sha256"], f"{cand} 격자")
        G.check_sha(G.REPO / c_man["table"]["path"], c_man["table"]["sha256"], f"{cand} 조회표")
        grid = G.grid_from_json(G.read_json(G.REPO / b_man["grid"]["path"]))
        table = np.load(G.REPO / c_man["table"]["path"], allow_pickle=False)
        art[cand] = dict(grid=grid, table=table)
        print(f"[{cand}] 격자 {grid['features']} 조회표 {table.shape}", flush=True)

    cfg = G.read_json(G.REPO / "exp/frozen.json")
    paths = ma.read_list(a.eval_list, "평가 창")
    if not a.dev_test and len(paths) != G.N_EVAL:
        raise SystemExit(f"평가 창이 {G.N_EVAL}개가 아니다: {len(paths)}")

    K = G.knots(pe)   # KNOTS = [0, τ1..τ10], 길이 11
    win_rows = {cand: [] for cand in cands}
    t_start = time.time()

    for wi, path in enumerate(paths):
        tr = p2.Trace2(path, G.FEATS)
        led = p2.Ledger(tr)
        b = G.TSTAR
        cap = p2.tcap(b)
        VA, VA1, TE, h = tr.VA, tr.VA1, tr.TE, tr.h
        dec = VA | TE
        te_dur = float(tr.t0 + tr.dur - tr.c2)
        gt, ht, rt = tr.g[TE], h[TE], tr.r[TE]
        n_dec = int(dec.sum())

        # 고정-온라인 (FALLBACK의 기준, evaluate3와 같다)
        base = p2.online_group_T(
            tr, led, b, np.zeros(len(tr.g), dtype=np.int64), 1,
            dec, cfg["wcal"], cfg["period"])

        # 규칙-온라인+대체 (두 후보에서 공통, evaluate3와 같은 순서)
        Cva = pe.cost_matrix(tr.gva[VA], h[VA], tr.rva[VA], b)
        pos_va = np.where(VA)[0]
        is1 = VA1[pos_va]
        i_fix = int(np.argmin(Cva.sum(1)))
        i_fix1 = int(np.argmin(Cva[:, is1].sum(1)))
        i_fix2 = int(np.argmin(Cva[:, ~is1].sum(1)))
        grp_r, G_r, rname, _ = p2.rule_groups(tr, b, Cva, pos_va, is1, i_fix, i_fix1, i_fix2)
        T_rule = p2.online_group_T(tr, led, b, grp_r, G_r, dec, cfg["wcal"], cfg["period"])
        T_rb, _, _ = p3.FALLBACK(tr, T_rule, base, b, dec, **p2.FB)

        rule_ref, rule_wst = pe.parts(gt, ht, rt, T_rb[TE], b)
        orc_T = np.where(rt & (gt < b), np.inf, 0.0)
        orc_ref, orc_wst = pe.parts(gt, ht, rt, orc_T, b)

        for cand in cands:
            grid = art[cand]["grid"]
            table = art[cand]["table"]

            # 조회표로 반환 시점(rec) 얻기: 원시 특징 → 코드 → 표 인덱스 → KNOTS 값
            # evaluate3의 rec = pe.survival_timeouts(Sk, b, 1.0)를 대체한다
            codes = G.encode_matrix(tr.X, grid)   # (n, k) int16
            idx = table[tuple(codes.T)]             # uint8 index into KNOTS
            rec = K[idx]                            # (n,) float64 recommended timeout

            # AI생존κ-온라인+대체 (evaluate3와 같은 순서)
            Tk = np.stack([np.minimum(rec * k, cap) for k in p2.KGRID])
            T_ai, _ = p2.online_kappa_T(tr, led, b, Tk, dec, cfg["wcal"], cfg["period"])
            T_ai_fb, _, _ = p3.FALLBACK(tr, T_ai, base, b, dec, **p2.FB)

            # C3 AI→규칙대체: AI생존κ-온라인+대체 ↔ 규칙-온라인+대체 (evaluate3와 같은 순서)
            T_c3, mode_c3, idx_fb = p3.FALLBACK(tr, T_ai_fb, T_rb, b, dec, **p2.FB)
            te_m = TE[idx_fb]
            c3_rule_frac = float(mode_c3[te_m].mean()) if te_m.any() else math.nan

            c3_ref, c3_wst = pe.parts(gt, ht, rt, T_c3[TE], b)
            MiB = pe.MiB
            R_rule, H_rule = rule_ref / MiB, rule_wst / MiB
            R_c3, H_c3 = c3_ref / MiB, c3_wst / MiB
            dR = R_rule - R_c3
            dH = H_rule - H_c3
            S_model = dR + dH / G.TSTAR  # = (rule_cost - c3_cost) / MiB

            win_rows[cand].append(dict(
                trace=tr.name, n_te=int(TE.sum()), n_dec=n_dec, tau_s=te_dur,
                rule_refault_MiB=R_rule, rule_waste_MiB=H_rule,
                c3_refault_MiB=R_c3, c3_waste_MiB=H_c3,
                dR_MiB=dR, dH_MiB=dH, S_units=S_model, Sc_ns=S_model * C_NS,
                c3_rule_mode_frac=c3_rule_frac,
                oracle_refault_MiB=orc_ref / MiB, oracle_waste_MiB=orc_wst / MiB,
                rule_name=rname,
            ))

        if (wi + 1) % 20 == 0 or wi + 1 == len(paths):
            elapsed = time.time() - t_start
            print(f"  창 {wi + 1}/{len(paths)} " +
                  ", ".join(f"{c} cumS={sum(r['S_units'] for r in win_rows[c]):.1f}" for c in cands) +
                  f" ({elapsed:.0f}s)", flush=True)

    # GAP-8: 결과 파일 저장 전 TE 행 수 전역 단언
    _assert_te_counts(win_rows, cands, paths, bool(a.dev_test))

    # 집계·판정
    summaries = {}
    for cand in cands:
        rows = win_rows[cand]
        S_tot = sum(r["S_units"] for r in rows)
        sc_ns = S_tot * C_NS
        sc_s = sc_ns / 1e9
        verdict_en = "PASS" if sc_s >= THRESHOLD_SC_S else "FAIL"
        summaries[cand] = dict(
            features=art[cand]["grid"]["features"],
            table_path=man["compile"][cand]["table"]["path"],
            table_sha256=man["compile"][cand]["table"]["sha256"],
            S_total_units=S_tot,
            dR_total_MiB=sum(r["dR_MiB"] for r in rows),
            dH_total_MiB=sum(r["dH_MiB"] for r in rows),
            S_total_times_c_ns=sc_ns,
            S_total_times_c_s=sc_s,
            threshold_sc_s=THRESHOLD_SC_S,
            n_windows=len(rows),
            n_te_total=sum(r["n_te"] for r in rows),
            verdict="통과" if verdict_en == "PASS" else "실패",
            verdict_en=verdict_en,
        )

    g2_cand, sel_verdict, sel_reason = select_g2_candidate(summaries, cands)
    sim_started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    sk_none = _sk_none_coverage()

    # 창별 CSV 쓰기
    sim_dir.mkdir(parents=True, exist_ok=True)
    for cand in cands:
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=W_FIELDS)
        w.writeheader()
        w.writerows(win_rows[cand])
        G.write_text(sim_dir / f"windows-{cand}.csv", buf.getvalue(), a.force)

    # 환경 정보 쓰기
    renv = dict(
        schema="g1-simulation-env/1",
        started_at=sim_started_at,
        eval_list=a.eval_list, eval_list_sha256=G.sha256_file(G.REPO / a.eval_list),
        n_windows=len(paths),
        runtime=env["runtime"], git=env["git"], provenance=env["provenance"],
        preflight_issues=issues,
        manifest_sha256=G.sha256_file(out / G.MANIFEST_NAME),
    )
    G.write_json(sim_dir / "run-environment.json", renv, a.force)

    # 요약 쓰기
    summary = dict(
        schema="g1-simulation/1",
        dev_test=bool(a.dev_test),
        tstar_s=G.TSTAR, cpu_cost_unit_ns=C_NS, threshold_sc_s=THRESHOLD_SC_S,
        n_windows=len(paths),
        candidates={c: summaries[c] for c in cands},
        selection=dict(verdict=sel_verdict, reason=sel_reason, g2_candidate=g2_cand),
        sk_none_coverage=sk_none,
        manifest_sha256=G.sha256_file(out / G.MANIFEST_NAME),
        finished_at=sim_started_at,
    )
    G.write_json(sim_dir / "summary.json", summary, a.force)

    # 매니페스트에 simulate_summary 절 추가
    man["simulate_summary"] = dict(
        candidates={c: summaries[c]["verdict_en"] for c in cands},
        g2_candidate=g2_cand,
        selection_reason=sel_reason,
        summary_path=G.rel(sim_dir / "summary.json"),
        summary_sha256=G.sha256_file(sim_dir / "summary.json"),
    )
    man["simulate_env"] = env
    G.save_manifest(out, man)

    print("===== G1 시뮬레이션 결과 =====")
    for cand in cands:
        s = summaries[cand]
        print(f"[{cand}] S_model×c = {s['S_total_times_c_s']:.4f} s "
              f"(ΔR {s['dR_total_MiB']:.1f} MiB, ΔH {s['dH_total_MiB']:.1f} MiB·s, "
              f"ΔR+ΔH/T*={s['S_total_units']:.1f}) → {s['verdict']}")
    print(f"선택: {sel_verdict} ({sel_reason})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
