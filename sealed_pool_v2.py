#!/usr/bin/env python3
"""sealed_pool_v2.py — v2 봉인 창 표본(입력만). v1에서 정책을 한 번도 돌리지 않은 BurstGPT 군집에서 창을 하나씩 뽑는다.

  python3 sealed_pool_v2.py --pool exp/sealed/pool.csv --v1 exp/sealed/windows.csv --out exp/sealed_v2/windows.csv
  python3 sealed_pool.py extract --windows exp/sealed_v2/windows.csv --outdir exp/sealed_v2   # 창 추출(입력만, v1과 같은 변환·솎기)

- 군집 = (파일, 파일 시작부터 24시간 단위 날짜). 자격 조건은 v1과 같다(pool.csv의 eligible).
- v1 창 목록(R·S·민감도)에 들어간 군집은 모두 뺀다. Azure 2024는 14개 군집을 v1에서 모두 썼으므로 남지 않는다.
- 남은 군집마다 자격 창 하나를 무작위로 고른다(시드 20261025). 창 하나가 독립 단위 하나다.
- 이 단계에서는 정책을 돌리지 않는다. 일정·결과 변수(D(t), 비용)는 v2 사전 등록 승인 뒤에 만든다.
"""
import argparse
import csv
import os

import numpy as np

HOUR = 3600.0


def clus(r):
    return f"{r['source']}_d{int(float(r['start']) // 86400):03d}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--v1", required=True, nargs="+", help="v1 창 목록(windows.csv, windows_sens.csv)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=20261025)
    a = ap.parse_args()
    rows = [r for r in csv.DictReader(open(a.pool)) if r["eligible"] == "1" and r["group"] == "BurstGPT"]
    used = set()
    for p in a.v1:
        used |= {r["cluster"] if r.get("cluster") else clus(r) for r in csv.DictReader(open(p))}
    cl_all = sorted({clus(r) for r in rows})
    free = [c for c in cl_all if c not in used]
    rng = np.random.default_rng(a.seed)
    picks = []
    for c in free:
        members = [r for r in rows if clus(r) == c]
        r = members[int(rng.integers(len(members)))]
        picks.append(dict(r, name=f"{r['source']}_h{int(float(r['start']) // HOUR):04d}", set="V2", cell="bg", cluster=c))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(picks[0]))
        w.writeheader()
        w.writerows(picks)
    by_src = {}
    for p in picks:
        by_src[p["source"]] = by_src.get(p["source"], 0) + 1
    print(f"BurstGPT 자격 군집 {len(cl_all)}개, v1에서 쓴 군집 {len(cl_all) - len(free)}개, 남은 군집 {len(free)}개 → 창 {len(picks)}개 {by_src}")
    for k in ("size_cv", "fano10", "burst", "level"):
        v = np.array([float(p[k]) for p in picks])
        print(f"  {k}: 중앙 {np.median(v):.3f} (사분위 {np.percentile(v, 25):.3f}~{np.percentile(v, 75):.3f})")
    print(f"→ {a.out}")


if __name__ == "__main__":
    main()
