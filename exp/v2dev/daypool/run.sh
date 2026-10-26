#!/bin/bash
# 개발용 '하루 전체' 평가: v1 R의 BurstGPT 40일, 자격 창 420개(입력 추출 → 일정 → 청크 모델, T* = 2.6초).
# v2 봉인 날은 쓰지 않는다(make_windows.py가 확인한다). 다시 실행하면 끝난 일정은 건너뛴다.
set -euo pipefail
cd "$(dirname "$0")/../../.."
export OMP_NUM_THREADS=1
D=exp/v2dev/daypool
[ -s $D/windows.csv ] || python3 $D/make_windows.py > $D/windows.csv
if [ ! -f $D/inputs.done ]; then
  python3 sealed_pool.py extract --windows $D/windows.csv --outdir $D > $D/extract.log
  while read -r name workers cell p; do
    python3 traces.py schedule --in $D/traces/$name.csv --out $D/sched/$name.csv --workers "$workers" > $D/sched/$name.log
  done < $D/workers.txt
  touch $D/inputs.done
fi
awk '{print "'$D'/sched/" $1 ".csv"}' $D/workers.txt > $D/all.txt
awk 'NR % 2 == 1' $D/all.txt > $D/list_q1.txt
awk 'NR % 2 == 0' $D/all.txt > $D/list_q2.txt
for q in q1 q2; do
  python3 paperexp3.py --model chunk --tstars 2.6 --out $D/res_$q.csv $(cat $D/list_$q.txt) >> $D/run_$q.log 2>&1 &
done
wait
echo "끝"
