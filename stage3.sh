#!/bin/bash
# stage3.sh — 3단계 측정: 실효 T*(청크 풀의 반납·재확보 CPU 비용)와 정책 결정 재생. WSL·보드에서 실행한다.
#   ./stage3.sh             # 전체(약 30~60분, 추정)
#   QUICK=1 ./stage3.sh     # 동작 확인(수 분)
#   CPU=2 ./stage3.sh       # 측정 스레드를 고정할 코어(기본 1)
# 결과: exp/stage3/{env.txt, tstar.csv, tstar_summary.txt, tstar_eval.txt, replay.csv, replay_summary.txt, dec/}
set -euo pipefail
trap 'echo "오류로 멈췄다(stage3.sh 줄 $LINENO). 위의 메시지와 exp/stage3/의 기록을 확인한다" >&2' ERR
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1
OUT=exp/stage3
CPU=${CPU:-1}
T0=$(date +%s)
el() { echo "$(( $(date +%s) - T0 ))초"; }

# 필요한 것 확인(없으면 무엇을 설치할지 알려 주고 멈춘다)
miss=0
command -v g++ >/dev/null 2>&1 || { echo "필요: g++  →  sudo apt install -y g++" >&2; miss=1; }
command -v python3 >/dev/null 2>&1 || { echo "필요: python3  →  sudo apt install -y python3" >&2; miss=1; }
if [ $miss = 1 ]; then echo "설치한 뒤 다시 실행한다" >&2; exit 2; fi
# 2단계(정책 결정 재생)에만 numpy·scikit-learn 1.0 이상이 필요하다. 없으면 1단계(실효 T*)만 하고 알려 준다.
HAVE_ML=1
python3 -c "import numpy; from sklearn.ensemble import HistGradientBoostingClassifier" 2>/dev/null || HAVE_ML=0

mkdir -p $OUT/dec
echo "[준비] chunkpool 빌드"
g++ -O2 -std=c++17 -o chunkpool chunkpool.cpp

# 0) 환경 기록(실패해도 멈추지 않는다)
set +e
{
  date -Is; uname -a; nproc; free -m
  echo "thp enabled: $(cat /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null)"
  echo "thp defrag:  $(cat /sys/kernel/mm/transparent_hugepage/defrag 2>/dev/null)"
  lscpu 2>/dev/null | grep -m1 "Model name"
  python3 -c "import sys,numpy,sklearn; print('python', sys.version.split()[0], 'numpy', numpy.__version__, 'sklearn', sklearn.__version__)"
} > $OUT/env.txt 2>&1
set -e

# 1) 실효 T*: 조건별로 반납 후 바로 다시 쓰는 비용을 잰다(전체 판은 3회 반복)
if [ "${QUICK:-0}" = 1 ]; then REG=128M; REPS=40; RUNS=1; else REG=512M; REPS=300; RUNS=3; fi
echo "[1/2] 실효 T* 측정: 조건 36개 × ${RUNS}회 (전체 판 약 5~10분, 추정)"
: > $OUT/tstar.csv
for k in $(seq 1 $RUNS); do
  echo "  회차 $k/$RUNS 시작 ($(el))"
  ./chunkpool tstar --chunk 1M,2M --method dontneed,free,remap --thp default,on,off --batch 1,16 \
      --region $REG --reps $REPS --cpu $CPU > $OUT/tstar_run$k.csv
  if [ $k = 1 ]; then cat $OUT/tstar_run$k.csv >> $OUT/tstar.csv; else grep -v '^#' $OUT/tstar_run$k.csv | tail -n +2 >> $OUT/tstar.csv; fi
done
echo "  측정 끝 ($(el))"
python3 stage3_summary.py tstar $OUT/tstar.csv | tee $OUT/tstar_summary.txt
# 판정용 T*: 1 MiB 청크, MADV_DONTNEED, 시스템 기본 THP, batch 16의 실효 T* 중앙값을 유효숫자 2자리로
TS=$(python3 - "$OUT/tstar.csv" <<'EOF'
import csv, statistics, sys
rows = [r for r in csv.DictReader(l for l in open(sys.argv[1]) if not l.startswith("#"))
        if r["chunk_bytes"] == "1048576" and r["method"] == "dontneed" and r["thp"] == "default" and r["batch"] == "16"]
v = statistics.median(float(r["tstar_eff_s"]) for r in rows)
print(f"{float(f'{v:.2g}'):g}")
EOF
)
echo "$TS" > $OUT/tstar_eval.txt
echo "판정용 T* = $TS 초"

# 2) 정책 결정 재생: 개발 트레이스(재실험 v1 데이터). 봉인 미사용 창은 쓰지 않는다.
if [ $HAVE_ML = 0 ]; then
  echo "[2/2] 건너뜀: numpy·scikit-learn(1.0 이상)이 없다. 1단계 결과(실효 T*)는 $OUT/에 있다." >&2
  echo "  설치(우분투 버전과 상관없이 되는 방법):" >&2
  echo "    sudo apt install -y python3-venv && python3 -m venv ~/venv-mb && ~/venv-mb/bin/pip install numpy scikit-learn" >&2
  echo "    source ~/venv-mb/bin/activate && ./stage3.sh" >&2
  exit 0
fi
if [ "${QUICK:-0}" = 1 ]; then TRACES="exp/sched/bg_q90.csv"; CHUNKS="1M"; else
  TRACES="exp/sched/az_code.csv exp/sched/az_conv.csv exp/sched/bg_q90.csv exp/sched/syn_periodic.csv"; CHUNKS="1M 2M"; fi
echo "[2/2] 정책 결정 재생: 트레이스 $(wc -w <<< "$TRACES")개 × 청크 $(wc -w <<< "$CHUNKS")종 × T* 2개 (전체 판 약 20~40분, 추정)"
HDR=""
: > $OUT/replay_rows.csv
for tr in $TRACES; do
  name=$(basename $tr .csv)
  for c in $CHUNKS; do
    for b in 1 $TS; do
      b=$(python3 -c "print(f'{float(\"$b\"):g}')")
      echo "  $name 청크 $c T* $b: 결정 내보내기 ($(el))"
      if ! python3 export_decisions.py $tr --chunk $c --tstar $b --outdir $OUT/dec > $OUT/dec/${name}_c${c}_t${b}.log 2>&1; then
        echo "건너뜀: $name c$c t$b ($(tail -1 $OUT/dec/${name}_c${c}_t${b}.log))"; continue
      fi
      for thp in default on; do
        for p in fixed rule ai never now; do
          res=$(./chunkpool replay --in $OUT/dec/${name}_c${c}_t${b}_${p}.csv --method dontneed --thp $thp --cpu $CPU)
          [ -z "$HDR" ] && HDR="trace,chunk,tstar,$(echo "$res" | head -1)"
          echo "$name,$c,$b,$(echo "$res" | tail -1)" >> $OUT/replay_rows.csv
        done
      done
    done
  done
done
{ echo "$HDR"; cat $OUT/replay_rows.csv; } > $OUT/replay.csv
python3 stage3_summary.py replay $OUT/replay.csv $OUT/dec | tee $OUT/replay_summary.txt
echo "끝: $OUT ($(el))"
