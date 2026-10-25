#!/bin/bash
# stage3.sh — 3단계 측정: 실효 T*(청크 풀의 반납·재확보 CPU 비용)와 정책 결정 재생. WSL·보드에서 실행한다.
#   ./stage3.sh             # 전체(약 30~60분, 추정)
#   QUICK=1 ./stage3.sh     # 동작 확인(수 분)
#   CPU=2 ./stage3.sh       # 측정 스레드를 고정할 코어(기본 1)
# 결과: exp/stage3/{env.txt, tstar.csv, tstar_summary.txt, tstar_eval.txt, replay.csv, replay_summary.txt, dec/}
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1
OUT=exp/stage3
CPU=${CPU:-1}
mkdir -p $OUT/dec
g++ -O2 -std=c++17 -o chunkpool chunkpool.cpp

# 0) 환경 기록
{
  date -Is; uname -a; nproc; free -m
  echo "thp enabled: $(cat /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null)"
  echo "thp defrag:  $(cat /sys/kernel/mm/transparent_hugepage/defrag 2>/dev/null)"
  lscpu 2>/dev/null | grep -m1 "Model name" || true
  python3 -c "import sys,numpy,sklearn; print('python', sys.version.split()[0], 'numpy', numpy.__version__, 'sklearn', sklearn.__version__)"
} > $OUT/env.txt 2>&1

# 1) 실효 T*: 조건별로 반납 후 바로 다시 쓰는 비용을 잰다(전체 판은 3회 반복)
if [ "${QUICK:-0}" = 1 ]; then REG=128M; REPS=40; RUNS=1; else REG=512M; REPS=300; RUNS=3; fi
: > $OUT/tstar.csv
for k in $(seq 1 $RUNS); do
  ./chunkpool tstar --chunk 1M,2M --method dontneed,free,remap --thp default,on,off --batch 1,16 \
      --region $REG --reps $REPS --cpu $CPU > $OUT/tstar_run$k.csv
  if [ $k = 1 ]; then cat $OUT/tstar_run$k.csv >> $OUT/tstar.csv; else grep -v '^#' $OUT/tstar_run$k.csv | tail -n +2 >> $OUT/tstar.csv; fi
done
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
if [ "${QUICK:-0}" = 1 ]; then TRACES="exp/sched/bg_q90.csv"; CHUNKS="1M"; else
  TRACES="exp/sched/az_code.csv exp/sched/az_conv.csv exp/sched/bg_q90.csv exp/sched/syn_periodic.csv"; CHUNKS="1M 2M"; fi
HDR=""
: > $OUT/replay_rows.csv
for tr in $TRACES; do
  name=$(basename $tr .csv)
  for c in $CHUNKS; do
    for b in 1 $TS; do
      b=$(python3 -c "print(f'{float(\"$b\"):g}')")
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
echo "끝: $OUT"
