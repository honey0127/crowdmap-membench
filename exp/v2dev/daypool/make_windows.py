#!/usr/bin/env python3
"""make_windows.py — 개발용 '하루 전체' 창 목록. v1 R 집합의 BurstGPT 40일(개발 데이터로 전환된 날)에서 자격 창을 모두 고른다.

  python3 exp/v2dev/daypool/make_windows.py > exp/v2dev/daypool/windows.csv

- 군집(날) 정의와 자격 조건은 sealed_pool_v2.py와 같다(pool.csv의 eligible, 파일 시작부터 24시간 단위).
- v2 봉인 창(exp/sealed_v2/windows.csv)의 날은 쓰지 않는다(겹치면 멈춘다).
"""
import csv
import sys

HOUR = 3600.0


def clus(r):
    return f"{r['source']}_d{int(float(r['start']) // 86400):03d}"


v1 = list(csv.DictReader(open("exp/sealed/windows.csv")))
days = sorted({r["cluster"] for r in v1 if "R" in r["set"] and r["group"] == "BurstGPT"})
sealed = {r["cluster"] for r in csv.DictReader(open("exp/sealed_v2/windows.csv"))}
assert not (set(days) & sealed), "v2 봉인 날과 겹친다"
rows = [r for r in csv.DictReader(open("exp/sealed/pool.csv")) if r["eligible"] == "1" and r["group"] == "BurstGPT" and clus(r) in days]
w = csv.DictWriter(sys.stdout, fieldnames=list(rows[0]) + ["name", "set", "cell", "cluster"])
w.writeheader()
for r in sorted(rows, key=lambda r: (r["source"], float(r["start"]))):
    w.writerow(dict(r, name=f"{r['source']}_h{int(float(r['start']) // HOUR):04d}", set="DEVDAY", cell="bg", cluster=clus(r)))
print(f"날 {len(days)}개, 창 {len(rows)}개", file=sys.stderr)
