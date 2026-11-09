#!/usr/bin/env bash
# run_g2_breakdown.sh — C3-P1b 추가 CPU(ΔI)를 C3 전용 연산 5개로 분해 측정한다(사후 분석).
# 기존 공식 측정(g2_measure.py)과 같은 조건: 420창, taskset -c 1, OMP_NUM_THREADS=1, 워밍업 1회 + 3회(순서 교대).
# 원본 g2_harness.cpp는 건드리지 않는다. 분해판 g2_harness_breakdown.cpp를 따로 빌드한다.
# 사용: ./run_g2_breakdown.sh            (전체 420창)
#       ./run_g2_breakdown.sh --smoke    (첫 5창으로 빌드·체크섬만 확인, 결과는 smoke/에)
set -euo pipefail
cd "$(dirname "$0")"
G2=exp/v2dev/g2
SMOKE=0; [ "${1:-}" = "--smoke" ] && SMOKE=1
OUT=$G2/breakdown; [ $SMOKE = 1 ] && OUT=$G2/breakdown/smoke
mkdir -p "$OUT"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

# 1. 원본 지문 확인
echo "51dc713e6c1d2139452979d8407db9bbc1b25245ce631cfa436e6e52899219aa  $G2/g2_harness.cpp" | sha256sum -c -

# 2. 창 파일(없으면 만든다)
N=$(ls $G2/windows/*.bin 2>/dev/null | wc -l || true)
if [ "$N" -ne 420 ]; then
  echo "[준비] 창 파일 $N개 → g2_prepare.py로 420개를 만든다"
  python3 $G2/g2_prepare.py
  N=$(ls $G2/windows/*.bin | wc -l)
  [ "$N" -eq 420 ] || { echo "창 파일이 420개가 아니다: $N"; exit 1; }
fi
WIN=$G2/windows
if [ $SMOKE = 1 ]; then
  WIN=$OUT/windows; mkdir -p $WIN
  ls $G2/windows/*.bin | sort | head -5 | xargs -I{} cp {} $WIN/
fi

# 3. 빌드
g++ -O2 -std=c++17 -Wall -Wextra -Wno-unused-parameter -o $G2/g2_harness_breakdown $G2/g2_harness_breakdown.cpp -lrt

# 4. 환경 기록
{
  echo "date: $(date -Iseconds)"
  echo "git: $(git rev-parse HEAD) $(git rev-parse --abbrev-ref HEAD) dirty=$(git status --porcelain | wc -l)"
  echo "kernel: $(uname -r)"
  echo "cpu: $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2-)"
  echo "gxx: $(g++ --version | head -1)"
  echo "taskset: -c 1  OMP_NUM_THREADS=$OMP_NUM_THREADS"
  sha256sum $G2/g2_harness.cpp $G2/g2_harness_breakdown.cpp $G2/p1b_model.bin
  echo "n_windows: $(ls $WIN/*.bin | wc -l)"
} > $OUT/env.txt
cat $OUT/env.txt

# 5. 측정
taskset -c 1 $G2/g2_harness_breakdown breakdown $WIN $G2/p1b_model.bin --out-json $OUT/breakdown.json | tee $OUT/run.log

# 6. 요약
python3 $G2/g2_breakdown_summary.py $OUT/breakdown.json | tee $OUT/summary.txt
