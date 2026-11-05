#!/usr/bin/env python3
"""g2_prepare.py — G2 창별 이진 파일 생성 (G2W3 형식).

Python 파이프라인(g1_simulate와 동일한 경로)을 실행해 창별 이진 입력과 Python 기준선을 함께 저장한다.
모델 아티팩트(cuts, table, KNOTS)는 p1b_model.bin으로 별도 저장한다.

측정 경계 (G2W3):
  타이머 밖: 파일 I/O, 창 파싱, 배열 로드, 정적 설정 로드
  CPU_rule 타이머 안:
    - Ledger 정렬
    - grp_r 계산: bin_ids(X_rule, rule_edges) + unique-remap
    - online_group_T(base, G=1) + online_group_T(rule, grp_r)
    - run_fallback3
  CPU_C3 타이머 안 (CPU_rule과 공통 연산 포함):
    - P1b 특징: x0=pos/MiB/mD, x1=D0/MiB/mD, x2=(pos+D0)/MiB/mD
    - encode + table lookup + κ-scale
    - online_kappa_T, fallback, 비용 계산

Python 사전계산 (C++ 타이머 밖):
  - mD: d_history 훈련 구간 평균/MiB (스칼라, b/cap/W_cal과 동일 성격)
  - X_rule[n, n_rf]: 규칙 선택 특징값 (NaN→-1e9; 복잡한 이력 계산 필요)
  - rule_edges: 검증 구간 분위 경계 (config 성격)
  - T_base, T_rule, ...: Python 기준선 (parity 검증용)

실행(저장소 루트)
  ~/.venvs/membench-g1/bin/python exp/v2dev/g2/g2_prepare.py [--dev-test] [--out-dir ...]
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "exp/v2dev/g1"))

import g1_common as G
import numpy as np

MiB = 1024.0 * 1024.0

N_DEV_PREP = 5
G2_DIR = Path(__file__).resolve().parent


def build_grid(p2, b):
    return p2.grid_on(b)


def compute_mD(tr, al):
    """P1b 정규화 상수 mD 계산.

    Trace._features()와 동일한 로직:
      mD = max(d_history 훈련 구간 binmax 평균 / MiB, 0.25)
    """
    ta, tf, sz = tr.ta, tr.tf, tr.sz
    t0_hist, b_hist, binmax = al.d_history(ta, tf, sz)
    c1t = tr.t0 + 0.5 * tr.dur
    nb = int((c1t - t0_hist) / b_hist)
    return max(binmax[:max(nb, 1)].mean() / MiB, 0.25)


def compute_rule_data(tr, grp_r, fs, pe):
    """G2W3용 규칙 데이터 계산.

    반환:
      X_rule: (n, n_rf) float64, NaN→-1e9 처리
      rule_ne: (n_rf,) uint32
      rule_edges_flat: (sum(rule_ne),) float64
    """
    VA = tr.VA
    n_rf = len(fs)
    # 검증 구간 특징값으로 경계 계산 (rule_groups 내부와 동일 로직)
    rule_edges_list = []
    for f in fs:
        x_va = np.nan_to_num(tr.X[VA, f].astype(float), nan=-1e9)
        e = pe.quant_edges(x_va, 4)
        rule_edges_list.append(e)

    # 전체 슬라이스 특징값 (NaN→-1e9)
    X_rule = np.nan_to_num(tr.X[:, list(fs)].astype(float), nan=-1e9)  # (n, n_rf)

    rule_ne = np.array([len(e) for e in rule_edges_list], dtype=np.uint32)
    rule_edges_flat = np.concatenate(rule_edges_list) if rule_edges_list else np.array([], dtype=np.float64)

    return X_rule, rule_ne, rule_edges_flat


def prepare_window(path, pe, p2, p3, al, ps, cfg, grid_p1b, table_p1b, K):
    """한 트레이스를 처리해 WindowData를 반환한다."""
    tr = p2.Trace2(path, G.FEATS)
    led = p2.Ledger(tr)
    b = G.TSTAR
    cap = p2.tcap(b)

    VA, VA1, TE, h = tr.VA, tr.VA1, tr.TE, tr.h
    dec = VA | TE
    n = len(tr.g)

    # 고정-온라인 (기준)
    T_base = p2.online_group_T(tr, led, b, np.zeros(n, dtype=np.int64), 1,
                               dec, cfg["wcal"], cfg["period"])

    # 규칙-온라인 그룹
    Cva = pe.cost_matrix(tr.gva[VA], h[VA], tr.rva[VA], b)
    pos_va = np.where(VA)[0]
    is1 = VA1[pos_va]
    i_fix  = int(np.argmin(Cva.sum(1)))
    i_fix1 = int(np.argmin(Cva[:, is1].sum(1)))
    i_fix2 = int(np.argmin(Cva[:, ~is1].sum(1)))
    grp_r, G_r, rname, fs = p2.rule_groups(tr, b, Cva, pos_va, is1, i_fix, i_fix1, i_fix2)

    T_rule = p2.online_group_T(tr, led, b, grp_r, G_r, dec, cfg["wcal"], cfg["period"])
    T_rb, _, _ = p3.FALLBACK(tr, T_rule, T_base, b, dec, **p2.FB)

    # P1b 특징: Python 기준선 계산 (parity 검증용)
    # G2W3: C++가 pos/D0/mD로부터 직접 계산 → X_p1b 미저장
    feat_indices = grid_p1b["indices"]   # [0, 1, 2]
    codes = G.encode_matrix(tr.X, grid_p1b)   # (n, 3) int16
    idx_lut = table_p1b[tuple(codes.T)]        # (n,) uint8
    rec = K[idx_lut]                           # (n,) float64
    Tk = np.stack([np.minimum(rec * k, cap) for k in p2.KGRID])  # (9, n)

    T_ai, _ = p2.online_kappa_T(tr, led, b, Tk, dec, cfg["wcal"], cfg["period"])
    T_ai_fb, _, _ = p3.FALLBACK(tr, T_ai, T_base, b, dec, **p2.FB)
    T_c3, _, _ = p3.FALLBACK(tr, T_ai_fb, T_rb, b, dec, **p2.FB)

    gt, ht, rt = tr.g[TE], h[TE], tr.r[TE]
    c3_ref, c3_wst = pe.parts(gt, ht, rt, T_c3[TE], b)
    R_MiB = c3_ref / pe.MiB
    H_MiB = c3_wst / pe.MiB

    # G2W3용 원시 슬라이스 속성 계산
    # pos, D0: policysim.slices()에서 직접 추출
    S_raw = ps.slices(tr.ta, tr.tf, tr.sz)
    pos_raw = S_raw["pos"]   # (n,) bytes 단위
    D0_raw  = S_raw["D0"]    # (n,) bytes 단위

    # G2W3용 P1b 정규화 상수
    mD = compute_mD(tr, al)

    # G2W3용 규칙 데이터 (X_rule, edges)
    X_rule, rule_ne, rule_edges_flat = compute_rule_data(tr, grp_r, fs, pe)

    grid_arr = build_grid(p2, b)

    return dict(
        name=tr.name, n=n, G_r=G_r,
        b=b, cap=cap, W_cal=cfg["wcal"], U=cfg["period"], W_fb=p2.FB["W"],
        mD=mD,
        g=tr.g, h=h, r=tr.r.astype(np.uint8), ts=tr.ts,
        dec_mask=dec.astype(np.uint8), te_mask=TE.astype(np.uint8),
        pos=pos_raw, D0=D0_raw,
        X_rule=X_rule, rule_ne=rule_ne, rule_edges_flat=rule_edges_flat,
        rule_n_rf=len(fs), rname=rname,
        grid=grid_arr,
        T_base=T_base, T_rule=T_rule, T_rb=T_rb,
        T_ai=T_ai, T_ai_fb=T_ai_fb, T_c3=T_c3,
        R_MiB=R_MiB, H_MiB=H_MiB,
    )


def write_model_file(out_dir, grid_p1b, table_p1b, K, p2):
    """G2MB 모델 파일을 한 번만 생성한다."""
    from g2_common import write_model
    model_path = out_dir.parent / "p1b_model.bin"
    cuts_list = grid_p1b["cutpoints"]   # list of 3 numpy arrays
    write_model(str(model_path), cuts_list, K, table_p1b)
    print(f"  모델 파일: {model_path}  "
          f"cuts={[len(c) for c in cuts_list]}  "
          f"knots={len(K)}  table={table_p1b.shape}")
    return model_path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dev-test", action="store_true")
    ap.add_argument("--out-dir", default=str(G2_DIR / "windows"))
    ap.add_argument("--eval-list", default=G.EVAL_LIST)
    a = ap.parse_args()

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pe, p2, p3, ma = G.research_modules()
    # ailearn, policysim은 직접 임포트
    import ailearn as al
    import policysim as ps
    cfg = G.read_json(G.REPO / "exp/frozen.json")

    man = G.load_manifest(G.G1_DIR)
    b_man = man["build"]["P1b"]
    c_man = man["compile"]["P1b"]
    grid_p1b  = G.grid_from_json(G.read_json(G.REPO / b_man["grid"]["path"]))
    table_p1b = np.load(G.REPO / c_man["table"]["path"], allow_pickle=False)
    K = G.knots(pe)

    if a.dev_test:
        all_paths = ma.read_list(G.EVAL_LIST, "평가 창")
        paths = all_paths[:N_DEV_PREP]
        print(f"[dev-test] 첫 {len(paths)}/{len(all_paths)} 창만 처리")
    else:
        paths = ma.read_list(a.eval_list, "평가 창")
        if len(paths) != G.N_EVAL:
            raise SystemExit(f"평가 창이 {G.N_EVAL}개가 아니다: {len(paths)}")
        print(f"공식 모드: {len(paths)} 창 처리")

    import time
    sys.path.insert(0, str(G2_DIR))
    from g2_common import write_window

    # 모델 파일 생성 (창 파일과 별도)
    write_model_file(out_dir, grid_p1b, table_p1b, K, p2)

    t0 = time.time()
    ok, err = 0, 0
    for i, path in enumerate(paths):
        try:
            d = prepare_window(path, pe, p2, p3, al, ps, cfg, grid_p1b, table_p1b, K)
        except Exception as e:
            print(f"  [{i+1}/{len(paths)}] {Path(path).stem} 오류: {e}", flush=True)
            err += 1
            continue

        out_path = out_dir / f"{d['name']}.bin"
        write_window(
            str(out_path),
            d["n"], d["G_r"], len(d["grid"]), d["rule_n_rf"],
            d["b"], d["cap"], d["W_cal"], d["U"], d["W_fb"], d["mD"],
            d["g"], d["h"], d["r"], d["ts"],
            d["dec_mask"], d["te_mask"],
            d["pos"], d["D0"],
            d["X_rule"], d["rule_ne"], d["rule_edges_flat"],
            d["grid"],
            d["T_base"], d["T_rule"], d["T_rb"],
            d["T_ai"], d["T_ai_fb"], d["T_c3"],
            d["R_MiB"], d["H_MiB"],
        )
        ok += 1
        if (i + 1) % 20 == 0 or (i + 1) == len(paths):
            print(f"  [{i+1}/{len(paths)}] {d['name']}  n={d['n']}  "
                  f"G_r={d['G_r']}  규칙={d['rname']}  ({time.time()-t0:.0f}s)", flush=True)

    print(f"\n완료: {ok} 창 성공, {err} 창 오류 → {out_dir}")


if __name__ == "__main__":
    main()
