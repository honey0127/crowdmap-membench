#!/bin/bash
# 청크 모델로 v2 개발 데이터를 다시 돌린다(v2 봉인 창은 쓰지 않는다). 두 줄을 나란히 돌리고, 끝난 일정은 건너뛴다(이어 하기).
set -euo pipefail
cd "$(dirname "$0")/../../.."
export OMP_NUM_THREADS=1
D=exp/v2dev/chunk
for q in q1 q2; do
  python3 paperexp3.py --model chunk --tstars 0.3,1,2.6,3,10 --out $D/res_$q.csv $(cat $D/list_$q.txt) >> $D/run_$q.log 2>&1 &
done
wait
echo "끝"
