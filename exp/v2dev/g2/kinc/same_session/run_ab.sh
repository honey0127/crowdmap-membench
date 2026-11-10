#!/usr/bin/env bash
# 같은 세션 원본/증분 measure 교대 비교 (orig → inc → orig → inc). 공식 measurement/ 는 건드리지 않는다.
set -euo pipefail
cd "$(dirname "$0")/../../../../.."
G2=exp/v2dev/g2; O=$G2/kinc/same_session
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
{ echo "date: $(date -Iseconds)"; echo "git: $(git rev-parse HEAD)"; sha256sum $G2/g2_harness $G2/g2_harness_kinc $G2/g2_harness.cpp $G2/g2_harness_kinc.cpp; } > $O/env.txt
for r in 1 2; do
  for v in orig inc; do
    B=$G2/g2_harness; [ $v = inc ] && B=$G2/g2_harness_kinc
    echo "=== round $r $v $(date +%T) ==="
    taskset -c 1 $B measure $G2/windows $G2/p1b_model.bin --out-json $O/measure_${v}_r$r.json
  done
done
python3 - $O <<'PY' | tee $O/summary.txt
import json, statistics as st, sys
O=sys.argv[1]; V=1560.860
for v in ("orig","inc"):
    its=[it for r in (1,2) for it in json.load(open(f"{O}/measure_{v}_r{r}.json"))["iterations"]]
    dI=[it["dI_ns"]/1e6 for it in its]; ru=[it["cpu_rule_ns"]/1e9 for it in its]
    print(f"{v}: n={len(dI)} CPU_rule {min(ru):.3f}-{max(ru):.3f} s, ΔI 평균 {st.mean(dI):.1f} ms (SD {st.stdev(dI):.1f}, {min(dI):.1f}-{max(dI):.1f}), G 평균 {V-st.mean(dI):+.1f} ms, 체크섬 {sorted({it['checksum_c3'] for it in its})}")
PY
