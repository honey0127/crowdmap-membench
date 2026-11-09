#!/usr/bin/env bash
# run_g2_kinc.sh — κ 보정 증분 갱신판(g2_harness_kinc.cpp)의 동등성 검증과 CPU 재측정(사후 최적화).
# 공식 G2(g2_measure.py)와 같은 조건: 420창, taskset -c 1, OMP_NUM_THREADS=1, 워밍업 1회 + 3회(순서 교대).
# 원본 g2_harness.cpp는 바꾸지 않는다. 공식 결과(−845.072 ms)는 덮어쓰지 않는다. 출력: exp/v2dev/g2/kinc/
#
# 통과 조건(1순위): equiv에서 420창 불일치 0, 원본 C3 체크섬 = 공식 측정 체크섬(ba240bd4f8369334).
#   이 조건이 깨지면 측정 결과를 쓰지 않는다.
# 사용: ./run_g2_kinc.sh          (전체)
#       ./run_g2_kinc.sh --smoke  (첫 5창: 빌드·equiv만)
set -euo pipefail
cd "$(dirname "$0")"
G2=exp/v2dev/g2
SMOKE=0; [ "${1:-}" = "--smoke" ] && SMOKE=1
OUT=$G2/kinc; [ $SMOKE = 1 ] && OUT=$G2/kinc/smoke
mkdir -p "$OUT"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
OFFICIAL_CK=ba240bd4f8369334
V_POOL_MS=1560.860

# 1. 원본 지문
echo "51dc713e6c1d2139452979d8407db9bbc1b25245ce631cfa436e6e52899219aa  $G2/g2_harness.cpp" | sha256sum -c -

# 2. 창 파일 — 공식 측정에 쓴 창을 그대로 쓴다(새로 만들면 체크섬이 달라질 수 있다)
N=$(ls $G2/windows/*.bin 2>/dev/null | wc -l || true)
[ "$N" -eq 420 ] || { echo "창 파일이 420개가 아니다: $N (공식 측정의 $G2/windows 가 필요하다)"; exit 1; }
WIN=$G2/windows
if [ $SMOKE = 1 ]; then
  WIN=$OUT/windows; mkdir -p $WIN
  ls $G2/windows/*.bin | sort | head -5 | xargs -I{} cp {} $WIN/
fi

# 3. 빌드
g++ -O2 -std=c++17 -Wall -Wextra -Wno-unused-parameter -o $G2/g2_harness_kinc $G2/g2_harness_kinc.cpp -lrt

# 4. 환경 기록
{
  echo "date: $(date -Iseconds)"
  echo "git: $(git rev-parse HEAD) $(git rev-parse --abbrev-ref HEAD) dirty=$(git status --porcelain | wc -l)"
  echo "kernel: $(uname -r)"
  echo "cpu: $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2-)"
  echo "gxx: $(g++ --version | head -1)"
  echo "taskset: -c 1  OMP_NUM_THREADS=$OMP_NUM_THREADS"
  sha256sum $G2/g2_harness.cpp $G2/g2_harness_kinc.cpp $G2/p1b_model.bin
  echo "n_windows: $(ls $WIN/*.bin | wc -l)"
} > $OUT/env.txt
cat $OUT/env.txt

# 5. 동등성 검증 (통과해야 다음 단계로 간다)
$G2/g2_harness_kinc equiv $WIN $G2/p1b_model.bin --out-json $OUT/equiv.json | tee $OUT/equiv.log
if [ $SMOKE = 0 ]; then
  CK=$(python3 -c "import json;print(json.load(open('$OUT/equiv.json'))['checksum_c3_orig'])")
  [ "$CK" = "$OFFICIAL_CK" ] || { echo "원본 C3 체크섬 $CK ≠ 공식 $OFFICIAL_CK: 창이 공식 측정과 다르다. 중단"; exit 1; }
  echo "[확인] 원본 C3 체크섬 = 공식 측정 체크섬"
fi
[ $SMOKE = 1 ] && { echo "smoke 끝"; exit 0; }

# 6. ③만 비교 (원본 online_kappa_T vs 증분)
taskset -c 1 $G2/g2_harness_kinc kbench $WIN $G2/p1b_model.bin --out-json $OUT/kbench.json | tee $OUT/kbench.log

# 7. 전체 ΔI 재측정 (공식과 같은 measure 모드, κ만 증분)
taskset -c 1 $G2/g2_harness_kinc measure $WIN $G2/p1b_model.bin --out-json $OUT/measure.json | tee $OUT/measure.log

# 8. 요약
python3 - "$OUT" "$V_POOL_MS" "$OFFICIAL_CK" <<'EOF' | tee $OUT/summary.txt
import json, statistics as st, sys
out, V, ck = sys.argv[1], float(sys.argv[2]), sys.argv[3]
m = json.load(open(f"{out}/measure.json")); kb = json.load(open(f"{out}/kbench.json"))
dI = [it["dI_ns"] / 1e6 for it in m["iterations"]]
cks = {it["checksum_c3"] for it in m["iterations"]}
print("사후 최적화: κ 보정 증분 갱신 (실측, 이 실행)")
print(f"C3 체크섬: {sorted(cks)} {'= 공식' if cks == {ck} else '≠ 공식 — 결과 사용 금지'}")
for it, d in zip(m["iterations"], dI):
    print(f"  {it['order']}: CPU_rule {it['cpu_rule_ns']/1e9:.4f} s, CPU_C3 {it['cpu_c3_ns']/1e9:.4f} s, ΔI {d:.1f} ms, G {V - d:+.1f} ms")
print(f"ΔI 평균 {st.mean(dI):.1f} ms (SD {st.stdev(dI):.1f}), G 평균 {V - st.mean(dI):+.1f} ms (V_pool {V} ms, 결정이 같아 V_pool 불변)")
print(f"세 회 모두 G > 0: {all(V - d > 0 for d in dI)}")
o = [x['orig_ns'] / 1e6 for x in kb['iterations']]; i = [x['inc_ns'] / 1e6 for x in kb['iterations']]
print(f"③만: 원본 {st.mean(o):.1f} ms → 증분 {st.mean(i):.1f} ms ({st.mean(o)/st.mean(i):.1f}배)")
EOF
