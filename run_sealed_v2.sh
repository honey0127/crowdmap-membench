#!/bin/bash
# run_sealed_v2.sh — 사전 등록 v2(초안) 승인 뒤 한 번 실행한다. 다시 실행하면 끝난 단계·묶음은 건너뛴다.
# 판정 T*는 exp/sealed_v2/tstar_eval.txt(3단계 실측으로 정해 승인 커밋에 넣는다), 주 후보는 exp/sealed_v2/primary.txt다.
# (1) v2 봉인 창 일정 (2) 합성 확인 40개(시드 201~205) (3) paperexp3.py를 묶음으로 실행(새 파일, 종료 코드 확인)
# (4) 이어 붙이기 (5) prereg_v2_analysis.py 판정
# 종료 코드: 0 = 판정까지 끝, 3 = 남은 묶음이 있어 다시 실행해야 한다
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1
export LC_ALL=C
LAUNCH_UNTIL=${LAUNCH_UNTIL:-4500}
SLOTS=${SLOTS:-2}
PER=${PER:-6}
T0=$(date +%s)
V=${V:-exp/sealed_v2}           # 시험 실행용으로만 바꾼다
FX_SEEDS=${FX_SEEDS:-"201 202 203 204 205"}
R=$V/res
P=$R/parts
if [ ! -s $V/tstar_eval.txt ] || [ ! -s $V/primary.txt ]; then
  echo "판정 T* 또는 주 후보가 없다: 3단계 측정 뒤 $V/tstar_eval.txt를 적고 사전 등록 v2를 승인한 다음 실행한다" >&2
  exit 2
fi
TS=$(cat $V/tstar_eval.txt)
PRIMARY=$(cat $V/primary.txt)
TLIST=$(python3 -c "v=[float('$TS'),0.3,1.0,3.0,10.0]; print(','.join(f'{x:g}' for x in dict.fromkeys(v)))")
mkdir -p $V/sched $V/fx/traces $V/fx/sched $P
log() { echo "[$(date '+%F %T')] $*" | tee -a $P/driver.log; }
log "시작 (커밋 $(git rev-parse --short HEAD 2>/dev/null || echo ?), T*=$TS, 주 후보=$PRIMARY, T* 목록=$TLIST)"

if [ ! -f $P/inputs.done ]; then
  while read -r name workers cell p; do
    python3 traces.py schedule --in $V/traces/$name.csv --out $V/sched/$name.csv --workers "$workers" > $V/sched/$name.log
  done < $V/workers.txt
  for seed in $FX_SEEDS; do for S in s S; do for A in p b; do for L in n w; do
    if [ $S = s ]; then sig=0.3; rate=9.77; else sig=1.2; rate=4.98; fi
    if [ $A = p ]; then arr="--pattern poisson"; else arr="--pattern random --burst-frac 0.5 --burst-width 3 --period 30"; fi
    if [ $L = n ]; then lev=""; else lev="--day-period 1800 --day-amp 0.6 --wave-all"; fi
    n=fx_${S}${A}${L}_$seed
    python3 traces.py synth $arr $lev --sigma $sig --rate $rate --duration 3600 --seed $seed --out $V/fx/traces/$n.csv > /dev/null
    python3 traces.py schedule --in $V/fx/traces/$n.csv --out $V/fx/sched/$n.csv --workers 2 > $V/fx/sched/$n.log
  done; done; done; done
  touch $P/inputs.done
  log "입력 만들기 끝"
fi

if [ ! -f $P/plan.tsv ]; then
  python3 - "$PER" "$V" > $P/plan.tsv <<'EOF'
import glob, sys
k, V = int(sys.argv[1]), sys.argv[2]
real = [f"{V}/sched/" + l.split()[0] + ".csv" for l in open(f"{V}/workers.txt") if l.strip()]
fx = sorted(glob.glob(f"{V}/fx/sched/*.csv"))
for kind, L in (("real", real), ("fx", fx)):
    for i in range(0, len(L), k):
        print(f"{kind}_{i // k + 1:02d}\t{kind}\t{' '.join(L[i:i + k])}")
EOF
  log "묶음 계획: $(cut -f2 $P/plan.tsv | sort | uniq -c | tr '\n' ' ')"
fi

run_part() {
  local id=$1; shift
  local tmp=$P/tmp_${id}_$T0
  if python3 paperexp3.py --tstars "$TLIST" --out $tmp.csv "$@" > $tmp.log 2>&1; then
    mv $tmp.csv $P/$id.csv && mv $tmp.log $P/$id.log && touch $P/$id.done && log "끝 $id"
  else
    log "실패 $id (기록 $tmp.log)"
    return 1
  fi
}
while IFS=$'\t' read -r id kind scheds; do
  [ -f $P/$id.done ] && continue
  while [ "$(jobs -rp | wc -l)" -ge "$SLOTS" ]; do wait -n || true; done
  if [ $(( $(date +%s) - T0 )) -ge "$LAUNCH_UNTIL" ]; then break; fi
  log "시작 $id ($kind, 일정 $(wc -w <<< "$scheds")개)"
  run_part "$id" $scheds &
done < $P/plan.tsv
wait
left=$(cut -f1 $P/plan.tsv | while read -r id; do [ -f $P/$id.done ] || echo "$id"; done | wc -l)
if [ "$left" -gt 0 ]; then log "남은 묶음 $left개. 다시 실행한다"; exit 3; fi

for kind in real fx; do
  first=1; : > $R/res_$kind.csv
  for id in $(awk -F'\t' -v k=$kind '$2==k{print $1}' $P/plan.tsv); do
    if [ $first = 1 ]; then cat $P/$id.csv >> $R/res_$kind.csv; first=0; else tail -n +2 $P/$id.csv >> $R/res_$kind.csv; fi
  done
done
python3 prereg_v2_analysis.py --res $R/res_real.csv --fx $R/res_fx.csv --primary "$PRIMARY" --tstar "$TS" > $R/verdict.txt
log "판정 끝 → $R/verdict.txt"
cat $R/verdict.txt
