#!/bin/bash
# run_sealed.sh — 사전 등록 v1.1 승인 뒤 한 번만 실행한다. 설정은 exp/frozen.json, 판정은 prereg_analysis.py.
# 순서: (1) 봉인 창 일정 (2) 솎기 민감도 일정 (3) 합성 확인 요인 실험(시드 101~105) (4) 정책 실행 (5) 판정
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1
mkdir -p exp/sealed/sched exp/sealed/sens_sched exp/sealed/res exp/sealed/fx/traces exp/sealed/fx/sched
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
# (4) 정책 실행: 코어 하나에 실제 창(within + 특징 통제 + 사전 학습), 다른 하나에 합성 → 솎기 민감도
REAL=$(awk '{print "exp/sealed/sched/"$1".csv"}' exp/sealed/workers.txt)
python3 run_frozen.py --config exp/frozen.json --out exp/sealed/res/res_main.csv --pretrain exp/dev_sched.txt $REAL > exp/sealed/res/run_main.log 2>&1 &
( python3 run_frozen.py --config exp/frozen.json --out exp/sealed/res/res_fx.csv exp/sealed/fx/sched/*.csv > exp/sealed/res/run_fx.log 2>&1
  python3 run_frozen.py --config exp/frozen.json --no-ablate --out exp/sealed/res/res_sens.csv exp/sealed/sens_sched/*.csv > exp/sealed/res/run_sens.log 2>&1 ) &
wait
# (5) 판정
PRIMARY=$(python3 -c "import json;print(json.load(open('exp/frozen.json'))['primary'])")
python3 prereg_analysis.py --res exp/sealed/res/res_main.csv --windows exp/sealed/windows.csv \
  --factorial exp/sealed/res/res_fx.csv --sens exp/sealed/res/res_sens.csv \
  --irr exp/sealed/res/res_main_irregularity.csv --ai "$PRIMARY" > exp/sealed/res/verdict.txt
cat exp/sealed/res/verdict.txt
