#!/usr/bin/env python3
"""sealed_pool_v2day.py — v2 봉인 날 41개의 자격 창 전부(입력만). 판정 단위는 날이고, 날마다 자격 창을 모두 쓴다.

  python3 sealed_pool_v2day.py --pool exp/sealed/pool.csv --picks exp/sealed_v2/windows.csv --out exp/sealed_v2/day/windows_day.csv \
      --blocks exp/sealed_v2/day/blocks.csv
  python3 sealed_pool.py extract --windows exp/sealed_v2/day/windows_day.csv --outdir exp/sealed_v2/day   # 입력만

- 날(군집)과 자격 조건은 sealed_pool_v2.py와 같다(pool.csv의 eligible, 파일 시작부터 24시간 단위).
- picks(날마다 무작위 창 하나, 시드 20261025)의 날 41개에서 자격 창을 모두 고른다. picked = 1은 그 무작위 창이다(탐색 분석에 쓴다).
- 날 번호(day_index) = 파일의 첫 요청 시각을 0으로 둔 경과 시간 ÷ 86,400초의 정수 부분이다(달력 날짜도, 자격 날짜의 순번도 아니다).
  주 번호(week) = day_index // 7, 주 블록(block) = 파일_w주번호. 블록은 파일을 넘지 않는다. 블록별 날 목록은 --blocks 파일에 쓴다.
- 이 단계에서는 정책을 돌리지 않는다. 일정(D(t))과 정책 비용은 사전 등록 v2 승인 뒤 run_sealed_v2.sh가 만든다.
"""
import argparse
import csv
import os

HOUR = 3600.0


def clus(r):
    return f"{r['source']}_d{int(float(r['start']) // 86400):03d}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--picks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--blocks", required=True, help="주 블록별 날 목록(판정 전에 고정한다)")
    a = ap.parse_args()
    picks = list(csv.DictReader(open(a.picks)))
    days = {r["cluster"] for r in picks}
    picked = {r["name"] for r in picks}
    rows = [r for r in csv.DictReader(open(a.pool)) if r["eligible"] == "1" and r["group"] == "BurstGPT" and clus(r) in days]
    out = []
    for r in sorted(rows, key=lambda r: (r["source"], float(r["start"]))):
        name = f"{r['source']}_h{int(float(r['start']) // HOUR):04d}"
        di = int(float(r["start"]) // 86400)
        out.append(dict(r, name=name, set="V2D", cell="bg", cluster=clus(r), picked=int(name in picked),
                        day_index=di, week=di // 7, block=f"{r['source']}_w{di // 7:02d}"))
    assert {r["cluster"] for r in out} == days and sum(r["picked"] for r in out) == len(picks)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    blk = {}
    for r in out:
        b = blk.setdefault(r["block"], dict(block=r["block"], source=r["source"], week=r["week"], days=set(), n_windows=0))
        b["days"].add(r["cluster"])
        b["n_windows"] += 1
    with open(a.blocks, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["block", "source", "week", "n_days", "n_windows", "days"])
        w.writeheader()
        for k in sorted(blk):
            b = blk[k]
            w.writerow(dict(block=k, source=b["source"], week=b["week"], n_days=len(b["days"]), n_windows=b["n_windows"],
                            days=" ".join(sorted(b["days"]))))
    per = {}
    for r in out:
        per[r["cluster"]] = per.get(r["cluster"], 0) + 1
    v = sorted(per.values())
    print(f"날 {len(days)}개, 자격 창 {len(out)}개(날마다 {v[0]}~{v[-1]}개, 중앙 {v[len(v) // 2]}), 주 블록 {len(blk)}개 → {a.out}, {a.blocks}")


if __name__ == "__main__":
    main()
