#!/bin/bash
# 분할 실행 = 단일 실행인지 개발 데이터로 확인한다(봉인 데이터는 쓰지 않는다).
# A: 일정 4개를 한 번에. B: [1] [2,3] [4]로 나눠 따로 돌린 뒤 머리줄을 빼고 이어 붙인다.
set -euo pipefail
cd /home/claude/sealed_run/repo
export OMP_NUM_THREADS=1 LC_ALL=C
O=/tmp/claude-0/-home-claude/a2c14167-10d1-56bf-8297-4516d77ce920/scratchpad/split_test
rm -rf $O && mkdir -p $O
S1=exp/sched/bg_q50.csv; S2=exp/sched/bg_q75.csv; S3=exp/sched/bg_q90.csv; S4=exp/sched/az_code.csv
t0=$(date +%s)
( python3 run_frozen.py --config exp/frozen.json --out $O/A.csv --pretrain exp/dev_sched.txt $S1 $S2 $S3 $S4 > $O/A.log 2>&1; echo "A done $(( $(date +%s) - t0 ))s" ) &
( python3 run_frozen.py --config exp/frozen.json --out $O/B1.csv --pretrain exp/dev_sched.txt $S1 > $O/B1.log 2>&1
  python3 run_frozen.py --config exp/frozen.json --out $O/B2.csv --pretrain exp/dev_sched.txt $S2 $S3 > $O/B2.log 2>&1
  python3 run_frozen.py --config exp/frozen.json --out $O/B3.csv --pretrain exp/dev_sched.txt $S4 > $O/B3.log 2>&1
  echo "B done $(( $(date +%s) - t0 ))s" ) &
wait
# 이어 붙이기: 있는 파일 중 첫 파일의 머리줄 + 모든 파일의 본문
cat_parts() { local out=$1; shift; local first=1; : > $out
  for f in "$@"; do [ -f "$f" ] || continue
    if [ $first = 1 ]; then cat "$f" >> $out; first=0; else tail -n +2 "$f" >> $out; fi; done; }
cat_parts $O/B.csv $O/B1.csv $O/B2.csv $O/B3.csv
cat_parts $O/B_irregularity.csv $O/B1_irregularity.csv $O/B2_irregularity.csv $O/B3_irregularity.csv
ls -l $O/*.csv
cmp $O/A.csv $O/B.csv && echo "res: 같다 ($(($(wc -l < $O/A.csv)-1))행)"
cmp $O/A_irregularity.csv $O/B_irregularity.csv && echo "irregularity: 같다 ($(($(wc -l < $O/A_irregularity.csv)-1))행)"
grep -h "사전 학습\|끝" $O/*.log
