#!/usr/bin/env python3
"""sealed_pool.py — 봉인 시험용 창 고르기. 입력 통계만 쓰고 정책은 돌리지 않는다.

  python3 sealed_pool.py stats  --out exp/sealed/pool.csv
  python3 sealed_pool.py sample --pool exp/sealed/pool.csv --out exp/sealed/windows.csv [--per-cell 6] [--seed 20261024]
  python3 sealed_pool.py extract --windows exp/sealed/windows.csv --outdir exp/sealed

창: 파일 첫 요청 시각부터 1시간 단위로 자른다(traces.py --start k*3600 --duration 3600과 같다).
변환: traces.py와 같다(BurstGPT는 실패 응답 제외 + 정수 초 시각에 U(0,1) 흩뿌림(seed 1), 크기는 창마다 중앙값 2MiB로
      비례 환산 후 64KiB~32MiB로 자름).
통계: 창의 뒤 절반(1800~3600초) 도착으로 계산한다(paperexp2.irregularity).
자격: 창 전체 1,000건 이상, 뒤 절반 400건 이상, 파일 범위 안에 온전히 들어가는 창.
표본: Azure 2024(code+conv)는 자격 창에서 무작위 24개. BurstGPT(2+3)는 자격 창의 size_cv·fano10 삼분위로
      아래·위 칸을 만들어 네 칸(sa, sA, Sa, SA)에서 12개씩(가운데 삼분위 제외).
워커 수: c = max(2, ceil(λ̄·E[S] / 0.45)), E[S] = 10ms + 20ms/MiB × 평균 크기(창 평균 이용률이 0.45 이하가 되게).
솎기(extract): 창 평균 도착이 8건/s를 넘으면 요청마다 확률 p = 8·3600/n으로 남긴다(창마다 고정 시드).
      Azure 2024 창(시간당 9천~30만 건)을 개발 트레이스와 같은 규모로 맞추려는 것이다. 크기 분포와 분 단위 수준 변화는
      유지되고 짧은 몰림은 포아송 쪽으로 약해진다(Fano 1 + p(F − 1)). 워커 수는 솎은 뒤 다시 계산한다.
"""
import argparse
import contextlib
import csv
import io
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paperexp2 as p2  # noqa: E402
import traces as T  # noqa: E402

MiB, KiB = 1 << 20, 1 << 10
SRC = {
    "az24_code": ("azure", "/home/claude/dl/azure2024/AzureLLMInferenceTrace_code_1week.csv"),
    "az24_conv": ("azure", "/home/claude/dl/azure2024/AzureLLMInferenceTrace_conv_1week.csv"),
    "bg2": ("burstgpt", "/home/claude/dl/burstgpt_v2/BurstGPT_2.csv"),
    "bg3": ("burstgpt", "/home/claude/dl/burstgpt_v2/BurstGPT_3.csv"),
}
GROUP = {"az24_code": "Azure2024", "az24_conv": "Azure2024", "bg2": "BurstGPT", "bg3": "BurstGPT"}
HOUR = 3600.0
RHO_TARGET = 0.45
RATE_CAP = 8.0
THIN_SEED = 20261024


def load(kind, path):
    """traces.py와 같은 변환: (상대 시각 t − t.min(), 크기 대용값(토큰), work)"""
    import pandas as pd
    if kind == "azure":
        ts, ctx, gen = [], [], []
        for ch in pd.read_csv(path, usecols=["TIMESTAMP", "ContextTokens", "GeneratedTokens"], chunksize=2_000_000):
            dt = pd.to_datetime(ch["TIMESTAMP"], utc=True, format="ISO8601")
            ns = dt.astype("int64").to_numpy()
            unit = {"datetime64[ns, UTC]": 1_000_000_000, "datetime64[us, UTC]": 1_000_000}.get(str(dt.dtype))
            if unit is None:
                raise SystemExit(f"알 수 없는 시각 형식 {dt.dtype}")
            ts.append((ns // unit).astype(np.float64) + (ns % unit).astype(np.float64) / unit)
            ctx.append(ch["ContextTokens"].to_numpy(np.float64))
            gen.append(ch["GeneratedTokens"].to_numpy(np.float64))
        t, prox, work = np.concatenate(ts), np.concatenate(ctx), np.concatenate(gen)
    else:
        df = pd.read_csv(path, usecols=["Timestamp", "Request tokens", "Response tokens"])
        resp = df["Response tokens"].to_numpy(np.float64)
        keep = resp > 0
        t = df["Timestamp"].to_numpy(np.float64)[keep]
        prox = df["Request tokens"].to_numpy(np.float64)[keep]
        work = resp[keep]
        rng = np.random.default_rng(1)
        if len(t) and np.all(t == np.round(t)):
            t = t + rng.uniform(0, 1, len(t))
    return t - t.min(), prox, work


def window_arrays(rel, prox, work, k):
    m = (rel >= k * HOUR) & (rel < (k + 1) * HOUR)
    t = rel[m] - k * HOUR
    size = T.scale_sizes(prox[m], 2.0, 64, 32) if m.any() else np.array([])
    return t, size, work[m]


def stats_one(t, size):
    """창(1시간) 입력 통계. 시각은 창 시작 기준."""
    s2 = p2.irregularity(t, size, 1800.0, 3600.0)
    va = p2.irregularity(t, size, 1800.0, 2700.0)
    te = p2.irregularity(t, size, 2700.0, 3600.0)
    drift = math.log2(te["rate"] / va["rate"]) if va["rate"] > 0 and te["rate"] > 0 else math.nan
    lam = len(t) / HOUR
    ES = 0.010 + 0.020 * (size.mean() / MiB if len(size) else 0.0)
    workers = max(2, math.ceil(lam * ES / RHO_TARGET))
    return dict(n=len(t), n2=s2["n"], size_cv=s2["size_cv"], fano10=s2["fano10"], burst=s2["burst"],
                level=s2["level"], drift=drift, mean_MiB=size.mean() / MiB if len(size) else 0.0, workers=workers)


def cmd_stats(a):
    import contextlib
    import io
    rows = []
    for src, (kind, path) in SRC.items():
        rel, prox, work = load(kind, path)
        nh = int(rel.max() // HOUR)  # 온전히 들어가는 창만
        k_all = (rel // HOUR).astype(np.int64)
        order = np.argsort(k_all, kind="stable")
        bounds = np.searchsorted(k_all[order], np.arange(nh + 1))
        for k in range(nh):
            idx = order[bounds[k]:bounds[k + 1]]
            if len(idx) < 1000:
                rows.append(dict(source=src, group=GROUP[src], start=k * HOUR, n=len(idx), eligible=0))
                continue
            t = rel[idx] - k * HOUR
            with contextlib.redirect_stdout(io.StringIO()):
                size = T.scale_sizes(prox[idx], 2.0, 64, 32)
            st = stats_one(t, size)
            elig = int(st["n"] >= 1000 and st["n2"] >= 400)
            rows.append(dict(source=src, group=GROUP[src], start=k * HOUR, eligible=elig, **st))
        print(f"{src}: 요청 {len(rel)}건, 온전한 창 {nh}개, 자격 {sum(r.get('eligible', 0) for r in rows if r['source'] == src)}개",
              flush=True)
    keys = ["source", "group", "start", "eligible", "n", "n2", "size_cv", "fano10", "burst", "level", "drift", "mean_MiB",
            "workers"]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, restval="")
        w.writeheader()
        w.writerows(rows)
    print(f"→ {a.out}")


def cmd_sample(a):
    """표본 설계(사전 등록 v1.1). 분석 단위는 군집 = (파일, 파일 시작부터 24시간 단위 날짜)이고, 한 군집에서 창을 하나만 뽑는다.
    (1) 대표 집합 R, Azure 2024: 자격 창이 있는 군집(파일 2개 × 7일 = 14개) 모두에서 창 하나씩 무작위
    (2) 대표 집합 R, BurstGPT: 자격 창이 있는 군집 중 n_bg_days개를 무작위로 고르고 군집마다 창 하나씩 무작위
    (3) 조건 집합 S, BurstGPT: size_cv·fano10 삼분위의 네 모서리 칸(sa, sA, Sa, SA)마다, 그 칸 창이 있는 군집 중
        per_cell개를 무작위로 고르고 군집마다 그 칸의 창 하나씩 무작위(가운데 삼분위는 쓰지 않는다)
    (4) 솎기 민감도(탐색): R의 Azure 창 중 무작위 n_sens개
    R과 S에 같은 창이 뽑히면 한 번만 계산하고 두 집합에 다 쓴다."""
    rows = [r for r in csv.DictReader(open(a.pool)) if r["eligible"] == "1"]
    rng = np.random.default_rng(a.seed)
    out = {}

    def key(r):
        return f"{r['source']}_h{int(float(r['start']) // HOUR):04d}"

    def clus(r):
        return f"{r['source']}_d{int(float(r['start']) // 86400):03d}"

    def add(r, sset, cell, **kw):
        k = key(r)
        if k in out:
            out[k]["set"] = "".join(sorted(set(out[k]["set"]) | set(sset)))
            if cell not in ("bg", "az"):
                out[k]["cell"] = cell
            return
        out[k] = dict(r, name=k, set=sset, cell=cell, cluster=clus(r), **kw)

    def one_per_cluster(cands, n_clusters):
        cl = sorted({clus(r) for r in cands})
        chosen = cl if n_clusters is None or n_clusters >= len(cl) else sorted(rng.choice(cl, size=n_clusters, replace=False))
        picks = []
        for c in chosen:
            members = [r for r in cands if clus(r) == c]
            picks.append(members[int(rng.integers(len(members)))])
        return picks, len(cl)

    AZ = [r for r in rows if r["group"] == "Azure2024"]
    picks, ncl = one_per_cluster(AZ, None)
    for r in picks:
        add(r, "R", "az", thr_size="", thr_fano="")
    print(f"Azure2024: 자격 {len(AZ)}개, 군집 {ncl}개 → 군집마다 1개 = {len(picks)}개 (R)")
    BG = [r for r in rows if r["group"] == "BurstGPT"]
    sc = np.array([float(r["size_cv"]) for r in BG])
    fa = np.array([float(r["fano10"]) for r in BG])
    s1, s2 = np.percentile(sc, [100 / 3, 200 / 3])
    f1, f2 = np.percentile(fa, [100 / 3, 200 / 3])
    thr = dict(thr_size=f"{s1:.4f}/{s2:.4f}", thr_fano=f"{f1:.3f}/{f2:.3f}")
    picks, ncl = one_per_cluster(BG, a.n_bg_days)
    for r in picks:
        add(r, "R", "bg", **thr)
    print(f"BurstGPT: 자격 {len(BG)}개, 군집 {ncl}개 → 무작위 군집 {len(picks)}개에서 1개씩 (R)")
    print(f"  size_cv 삼분위 {s1:.3f}/{s2:.3f}, fano10 삼분위 {f1:.2f}/{f2:.2f}")
    cells = {"sa": (sc <= s1) & (fa <= f1), "sA": (sc <= s1) & (fa > f2),
             "Sa": (sc > s2) & (fa <= f1), "SA": (sc > s2) & (fa > f2)}
    for cell, m in cells.items():
        cands = [BG[i] for i in np.where(m)[0]]
        picks, ncl = one_per_cluster(cands, a.per_cell)
        for r in picks:
            add(r, "S", cell, **thr)
        print(f"  {cell}: 후보 창 {len(cands)}개·군집 {ncl}개 → {len(picks)}개 (S)")
    az_r = sorted(k for k, v in out.items() if v["cell"] == "az")
    sens = sorted(str(x) for x in rng.choice(az_r, size=min(a.n_sens, len(az_r)), replace=False))
    print(f"솎기 민감도(Azure): {sens}")
    W = sorted(out.values(), key=lambda v: (v["source"], float(v["start"])))
    keys = list(W[0])
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(W)
    with open(a.out.replace(".csv", "_sens.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows([out[k] for k in sens])
    nR = sum("R" in v["set"] for v in W)
    nS = sum("S" in v["set"] for v in W)
    print(f"→ {a.out}: 창 {len(W)}개 (R {nR}개, S {nS}개, 둘 다 {sum(v['set'] == 'RS' for v in W)}개)")


def cmd_extract(a):
    W = list(csv.DictReader(open(a.windows)))
    os.makedirs(os.path.join(a.outdir, "traces"), exist_ok=True)
    os.makedirs(os.path.join(a.outdir, "sched"), exist_ok=True)
    with open(os.path.join(a.outdir, "workers.txt"), "w") as fw:
        for src in SRC:  # 파일마다 한 번만 읽는다
            rs = [r for r in W if r["source"] == src]
            if not rs:
                continue
            rel, prox, work = load(*SRC[src])
            for r in rs:
                k = int(float(r["start"]) // HOUR)
                m = (rel >= k * HOUR) & (rel < (k + 1) * HOUR)
                idx = np.where(m)[0]
                # 솎기: 창 평균 도착이 RATE_CAP건/s를 넘으면 요청마다 확률 p로 남긴다(창마다 고정 시드)
                p_keep = min(1.0, a.rate_cap * HOUR / max(len(idx), 1))
                if p_keep < 1.0:
                    rng = np.random.default_rng(THIN_SEED + 10000 * list(SRC).index(src) + k)
                    idx = idx[rng.uniform(0, 1, len(idx)) < p_keep]
                t = rel[idx] - k * HOUR
                with contextlib.redirect_stdout(io.StringIO()):
                    size = T.scale_sizes(prox[idx], 2.0, 64, 32)
                ES = 0.010 + 0.020 * size.mean() / MiB
                workers = max(2, math.ceil(len(t) / HOUR * ES / RHO_TARGET))
                name = f"{src}_h{k:04d}"
                T.write_trace(os.path.join(a.outdir, "traces", name + ".csv"), t, size, work[idx])
                fw.write(f"{name} {workers} {r['cell']} {p_keep:.6f}\n")
                print(f"{name}: 남김 비율 {p_keep:.3f}, 워커 {workers}, 칸 {r['cell']}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("stats")
    q.add_argument("--out", required=True)
    q.set_defaults(fn=cmd_stats)
    q = sub.add_parser("sample")
    q.add_argument("--pool", required=True)
    q.add_argument("--out", required=True)
    q.add_argument("--per-cell", type=int, default=20)
    q.add_argument("--n-az", type=int, default=24)
    q.add_argument("--n-bg-days", type=int, default=40)
    q.add_argument("--n-sens", type=int, default=6)
    q.add_argument("--seed", type=int, default=20261024)
    q.set_defaults(fn=cmd_sample)
    q = sub.add_parser("extract")
    q.add_argument("--windows", required=True)
    q.add_argument("--outdir", required=True)
    q.add_argument("--rate-cap", type=float, default=RATE_CAP, help="솎기 상한(건/s), 기본 8")
    q.set_defaults(fn=cmd_extract)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
