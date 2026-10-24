#!/bin/bash
# run_sealed_parts.sh — run_sealed.sh와 같은 명령을 나눠 실행한다(코드·설정·입력·명령 인자는 같다).
# 이유: 봉인 시험을 돌린 환경은 백그라운드 작업 하나가 2시간까지라, (4) 정책 실행을 일정 묶음(part)으로
#       나눠 여러 번에 걸쳐 돌린다. 각 묶음은 run_sealed.sh (4)의 run_frozen.py 명령을 일정 일부에만 돌린 것이고,
#       (5) 직전에 원래 순서대로 이어 붙인다(머리줄 한 번). 분할 실행이 단일 실행과 바이트 단위로 같음을
#       개발 데이터로 먼저 확인했다(exp/sealed/res/EXECUTION.md).
# 다시 실행하면 끝난 단계·묶음은 건너뛴다. 새 묶음은 이번 실행 시작 후 LAUNCH_UNTIL초 안에만 띄운다.
# 종료 코드: 0 = 판정까지 끝, 3 = 남은 묶음이 있어 다시 실행해야 한다, 그 밖 = 오류.
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1
export LC_ALL=C   # 이 환경의 기본 로캘(POSIX)과 같은 정렬. run_sealed.sh의 glob 순서와 같게 한다
LAUNCH_UNTIL=${LAUNCH_UNTIL:-4800}
SLOTS=${SLOTS:-2}
TARGET_MAIN=${TARGET_MAIN:-80000}   # 실제 창 묶음 하나의 요청 수 목표(묶음 계획을 처음 만들 때만 쓴다)
FX_PER=${FX_PER:-5}
SENS_PER=${SENS_PER:-2}
T0=$(date +%s)
R=exp/sealed/res
P=$R/parts
mkdir -p exp/sealed/sched exp/sealed/sens_sched $R exp/sealed/fx/traces exp/sealed/fx/sched $P
log() { echo "[$(date '+%F %T')] $*" | tee -a $P/driver.log; }
log "시작 (커밋 $(git rev-parse --short HEAD 2>/dev/null || echo ?), LAUNCH_UNTIL=$LAUNCH_UNTIL, SLOTS=$SLOTS)"

# ---- (1)~(3) 입력: run_sealed.sh와 같은 명령 ----
if [ ! -f $P/inputs.done ]; then
  log "입력 만들기 시작"
  # (1) 창별 워커 수는 exp/sealed/workers.txt(솎은 뒤 계산)를 따른다
  while read -r name workers cell p; do
    python3 traces.py schedule --in exp/sealed/traces/$name.csv --out exp/sealed/sched/$name.csv --workers "$workers" > exp/sealed/sched/$name.log
  done < exp/sealed/workers.txt
  # (2) 솎기 민감도(탐색): 같은 Azure 창 6개를 남김 상한 16·32건/s로 추출한 판
  for r in 16 32; do
    while read -r name workers cell p; do
      python3 traces.py schedule --in exp/sealed/sens$r/traces/$name.csv --out exp/sealed/sens_sched/${name}_r$r.csv --workers "$workers" > exp/sealed/sens_sched/${name}_r$r.log
    done < exp/sealed/sens$r/workers.txt
  done
  # (3) 합성 확인 요인 실험: 개발 요인 실험(exp/fx, 시드 11·12)과 같은 조건, 시드만 101~105
  for seed in 101 102 103 104 105; do for S in s S; do for A in p b; do for L in n w; do
    if [ $S = s ]; then sig=0.3; rate=9.77; else sig=1.2; rate=4.98; fi
    if [ $A = p ]; then arr="--pattern poisson"; else arr="--pattern random --burst-frac 0.5 --burst-width 3 --period 30"; fi
    if [ $L = n ]; then lev=""; else lev="--day-period 1800 --day-amp 0.6 --wave-all"; fi
    n=fx_${S}${A}${L}_$seed
    python3 traces.py synth $arr $lev --sigma $sig --rate $rate --duration 3600 --seed $seed --out exp/sealed/fx/traces/$n.csv > /dev/null
    python3 traces.py schedule --in exp/sealed/fx/traces/$n.csv --out exp/sealed/fx/sched/$n.csv --workers 2 > exp/sealed/fx/sched/$n.log
  done; done; done; done
  touch $P/inputs.done
  log "입력 만들기 끝"
fi

# ---- 묶음 계획(처음 한 번만 만들고 고정한다) ----
if [ ! -f $P/plan.tsv ]; then
  REAL=$(awk '{print "exp/sealed/sched/"$1".csv"}' exp/sealed/workers.txt)
  FX=$(ls exp/sealed/fx/sched/*.csv)
  SENS=$(ls exp/sealed/sens_sched/*.csv)
  python3 - "$TARGET_MAIN" "$FX_PER" "$SENS_PER" > $P/plan.tsv <<EOF
import sys
target, fx_per, sens_per = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
real = """$REAL""".split(); fx = """$FX""".split(); sens = """$SENS""".split()
rows = lambda p: sum(1 for _ in open(p)) - 1
parts, cur, n = [], [], 0
for s in real:                      # 원래 순서를 지키며 요청 수 기준으로 자른다
    cur.append(s); n += rows(s)
    if n >= target:
        parts.append(cur); cur, n = [], 0
if cur:
    parts.append(cur)
for i, c in enumerate(parts, 1):
    print(f"main_{i:02d}\tmain\t{' '.join(c)}")
for i in range(0, len(fx), fx_per):
    print(f"fx_{i // fx_per + 1:02d}\tfx\t{' '.join(fx[i:i + fx_per])}")
for i in range(0, len(sens), sens_per):
    print(f"sens_{i // sens_per + 1:02d}\tsens\t{' '.join(sens[i:i + sens_per])}")
EOF
  log "묶음 계획: $(cut -f2 $P/plan.tsv | sort | uniq -c | tr '\n' ' ')"
fi

# ---- (4) 정책 실행: 묶음마다 run_sealed.sh (4)와 같은 run_frozen.py 명령 ----
run_part() {
  local id=$1 kind=$2; shift 2
  local tmp=$P/tmp_${id}_$T0
  local rc=0
  case $kind in
    main) python3 run_frozen.py --config exp/frozen.json --out $tmp.csv --pretrain exp/dev_sched.txt "$@" > $tmp.log 2>&1 || rc=$? ;;
    fx)   python3 run_frozen.py --config exp/frozen.json --out $tmp.csv "$@" > $tmp.log 2>&1 || rc=$? ;;
    sens) python3 run_frozen.py --config exp/frozen.json --no-ablate --out $tmp.csv "$@" > $tmp.log 2>&1 || rc=$? ;;
  esac
  if [ $rc != 0 ]; then log "실패 $id (종료 코드 $rc, 기록 $tmp.log)"; return $rc; fi
  if [ -f $tmp.csv ]; then mv $tmp.csv $P/$id.csv; fi   # 결과 행이 하나도 없는 묶음은 csv가 없다
  mv ${tmp}_irregularity.csv $P/${id}_irregularity.csv
  mv $tmp.log $P/$id.log
  touch $P/$id.done
  log "끝 $id ($(( $(date +%s) - T0 ))s 시점)"
}

stopped=0
while IFS=$'\t' read -r id kind scheds; do
  [ -f $P/$id.done ] && continue
  while [ "$(jobs -rp | wc -l)" -ge "$SLOTS" ]; do wait -n || true; done
  if [ $(( $(date +%s) - T0 )) -ge "$LAUNCH_UNTIL" ]; then stopped=1; break; fi
  log "시작 $id ($kind, 일정 $(wc -w <<< "$scheds")개)"
  run_part "$id" "$kind" $scheds &
done < $P/plan.tsv
wait
left=$(cut -f1 $P/plan.tsv | while read -r id; do [ -f $P/$id.done ] || echo "$id"; done | wc -l)
if [ "$left" -gt 0 ]; then
  log "남은 묶음 $left개(시간 예산 끝=$stopped). 다시 실행한다"
  exit 3
fi

# ---- 이어 붙이기: 묶음 계획 순서 = run_sealed.sh의 일정 순서 ----
cat_parts() {
  local out=$1 kind=$2 suffix=$3 first=1
  : > $out
  for id in $(awk -F'\t' -v k=$kind '$2==k{print $1}' $P/plan.tsv); do
    f=$P/$id$suffix
    [ -f $f ] || continue
    if [ $first = 1 ]; then cat $f >> $out; first=0; else tail -n +2 $f >> $out; fi
  done
}
for kind in main fx sens; do
  cat_parts $R/res_$kind.csv $kind .csv
  cat_parts $R/res_${kind}_irregularity.csv $kind _irregularity.csv
  : > $R/run_$kind.log
  for id in $(awk -F'\t' -v k=$kind '$2==k{print $1}' $P/plan.tsv); do echo "### $id" >> $R/run_$kind.log; cat $P/$id.log >> $R/run_$kind.log; done
done
log "이어 붙이기 끝: main $(($(wc -l < $R/res_main.csv)-1))행, fx $(($(wc -l < $R/res_fx.csv)-1))행, sens $(($(wc -l < $R/res_sens.csv)-1))행"

# ---- (5) 판정: run_sealed.sh와 같은 명령 ----
PRIMARY=$(python3 -c "import json;print(json.load(open('exp/frozen.json'))['primary'])")
python3 prereg_analysis.py --res exp/sealed/res/res_main.csv --windows exp/sealed/windows.csv \
  --factorial exp/sealed/res/res_fx.csv --sens exp/sealed/res/res_sens.csv \
  --irr exp/sealed/res/res_main_irregularity.csv --ai "$PRIMARY" > exp/sealed/res/verdict.txt
log "판정 끝 → exp/sealed/res/verdict.txt"
