#!/usr/bin/env python3
"""check_parts_v2.py — run_sealed_v2.sh의 묶음별 결과가 빠짐없는지 확인한다(사전 등록 v2 개정 6).

  python3 check_parts_v2.py <묶음 폴더> <T* 목록(쉼표)>   # 결과가 빠진 묶음 id를 한 줄에 하나씩 출력한다

묶음마다 계획(plan.tsv)의 일정 × T* 목록 × 정책 8개 조합이 정확히 한 번씩 있어야 한다. 묶음 CSV가 없거나 읽을 수 없거나,
일정·T*·정책 조합이 빠지거나 더 있으면 그 묶음을 출력한다. run_sealed_v2.sh는 출력된 묶음을 같은 코드·설정으로 다시 돌린다.
"""
import csv
import sys

POLICIES = {"고정-온라인", "규칙-온라인+대체", "AI생존κ-온라인+대체", "C1 AI세분규칙-온라인+대체", "C2 AI생존κ칸별-온라인+대체",
            "C4 AI생존세밀κ-온라인+대체", "C3 AI→규칙대체", "오라클"}


def main():
    part_dir, tl = sys.argv[1], {round(float(x), 9) for x in sys.argv[2].split(",")}
    for line in open(f"{part_dir}/plan.tsv"):
        pid, _kind, scheds = line.rstrip("\n").split("\t")
        names = {s.rsplit("/", 1)[1][:-4] for s in scheds.split()}
        need = {(n, t, p) for n in names for t in tl for p in POLICIES}
        rows = []
        try:
            with open(f"{part_dir}/{pid}.csv") as f:
                rows = [(r["trace"], round(float(r["Tstar"]), 9), r["policy"]) for r in csv.DictReader(f)]
        except (OSError, KeyError, ValueError):
            rows = []
        if len(rows) != len(set(rows)) or set(rows) != need:
            print(pid)


if __name__ == "__main__":
    main()
