#!/usr/bin/env python3
"""dev_summary.py — 개발 실행(res2_*.csv) 비교표. 봉인 시험 분석과 같은 기준(±0.5%p 무승부, 부호 검정)을 쓴다.
  python3 dev_summary.py exp/res2/res2_f15_w900.csv [더 많은 파일] [--exclude bg_q50,bg_q75]
"""
import argparse
import csv
import statistics as st
from collections import defaultdict

import prereg_analysis as pa

POLS = ["고정(정적)", "AI분류(정적)", "규칙-온라인", "규칙-온라인+대체", "AI분류-온라인", "AI분류-온라인+대체",
        "AI생존그룹-온라인", "AI생존그룹-온라인+대체", "AI생존κ-온라인", "AI생존κ-온라인+대체", "오라클"]
AIS = ["AI분류-온라인+대체", "AI생존그룹-온라인+대체", "AI생존κ-온라인+대체"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--exclude", default="")
    ap.add_argument("--proto", default="within")
    ap.add_argument("--col", default="rel_fixon", help="rel_fixon(고정-온라인 대비) 또는 rel_fixst(고정(정적) 대비)")
    a = ap.parse_args()
    excl = set(x for x in a.exclude.split(",") if x)
    for path in a.files:
        by = defaultdict(dict)
        for r in csv.DictReader(open(path)):
            if r["proto"] != a.proto or r["trace"] in excl:
                continue
            by[(r["trace"], float(r["Tstar"]))][r["policy"]] = float(r[a.col])
        traces = sorted({t for t, _ in by})
        print(f"\n===== {path} ({a.proto}, {a.col}, 트레이스 {len(traces)}개{', 제외 ' + ','.join(sorted(excl)) if excl else ''})")
        for b in sorted({b for _, b in by}):
            ks = [k for k in by if k[1] == b]
            print(f"T*={b:g} (n={len(ks)})  중앙값/최악: " + "  ".join(
                f"{p} {st.median([by[k][p] for k in ks]):+.1f}/{max(by[k][p] for k in ks):+.1f}" for p in POLS))
            for ai in AIS:
                d_r = [by[k][ai] - by[k]["규칙-온라인+대체"] for k in ks]
                d_f = [by[k][ai] - by[k].get("고정-온라인", 0.0) for k in ks]
                d_s = [by[k][ai] - by[k]["고정(정적)"] for k in ks]
                w, t, l = pa.wtl(d_r)
                w2, t2, l2 = pa.wtl(d_f)
                print(f"   {ai}: vs 규칙 {w}/{t}/{l} (p={pa.sign_p(w, l):.3f}, 중앙 {st.median(d_r):+.1f}) | "
                      f"vs 고정-온라인 {w2}/{t2}/{l2} (p={pa.sign_p(w2, l2):.3f}, 중앙 {st.median(d_f):+.1f}) | "
                      f"vs 고정(정적) 중앙 {st.median(d_s):+.1f}")
            d = [by[k]["규칙-온라인+대체"] - by[k].get("고정-온라인", 0.0) for k in ks]
            w, t, l = pa.wtl(d)
            print(f"   규칙-온라인+대체 vs 고정-온라인 {w}/{t}/{l} (p={pa.sign_p(w, l):.3f}, 중앙 {st.median(d):+.1f})")


if __name__ == "__main__":
    main()
