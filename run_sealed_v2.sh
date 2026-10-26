#!/bin/bash
# run_sealed_v2.sh — 사전 등록 v2 승인 뒤 한 번 실행한다. 다시 실행하면 끝난 단계·묶음은 건너뛴다.
# 판정 T*는 exp/sealed_v2/tstar_eval.txt(3단계 실측), 주 후보는 exp/sealed_v2/primary.txt다. 평가 모델은 청크 모델(--model chunk)이다.
# (1) v2 봉인 날 41개의 자격 창 405개 일정(입력은 exp/sealed_v2/day, 승인 전에 추출) (2) 합성 확인 40개(조건 8 × 반복 5, 일정마다 다른 시드 201~240)
# (3) paperexp3.py를 묶음으로 실행(새 파일, 종료 코드 확인) (4) 묶음마다 결과가 빠짐없는지 확인(빠지면 같은 코드·설정으로 다시 실행)
# (5) 이어 붙이기 (6) prereg_v2_analysis.py 판정(완전성 확인 → 41일 총비용 관측 판정 + 주 블록 방향 참고)
# 실행 관문(개정 7, 무엇이든 쓰기 전에 확인한다):
#   정식 실행 = V가 봉인 폴더(실제 경로로 비교)이고 FX_BASE = 201, FX_REPS = 5. 지문(코드·설정·입력·사전 등록 문서)이 맞고,
#     지금 커밋 = 태그 prereg-v2의 커밋이고 추적 파일 수정이 없으며, origin이 승인 저장소(아래 REPO_URLS)이고
#     그 저장소(GitHub)의 태그 prereg-v2가 같은 커밋일 때만 실행한다.
#   시험 실행 = V가 봉인 폴더 밖. 봉인 폴더(또는 그 안)에서 시험 설정을 쓰거나, 시험 입력이 봉인 트레이스와 내용이 같으면 멈춘다.
# 종료 코드: 0 = 판정까지 끝, 3 = 남은(또는 다시 돌릴) 묶음이 있어 같은 명령을 다시 실행한다,
#           4 = 판정 중단(완전성 확인 실패, res/verdict_stopped.txt), 5 = 같은 묶음이 세 번 실패해 멈춤(코드 문제, 연구자에게 보고),
#           6 = 지문이 사전 등록과 다르다, 7 = 시험 설정·시험 입력이 봉인 데이터를 가리킨다, 8 = 승인 태그의 커밋이 아니다
#           (6·7·8은 아무것도 쓰지 않고 끝낸다), 그 밖의 0이 아닌 값 = 스크립트 오류(판정 없음)
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1
export LC_ALL=C
LAUNCH_UNTIL=${LAUNCH_UNTIL:-4500}
SLOTS=${SLOTS:-2}
PER=${PER:-6}
T0=$(date +%s)
V=${V:-exp/sealed_v2}           # 정식 실행은 기본값(봉인 폴더). 시험 실행은 봉인 폴더 밖의 폴더를 따로 지정한다
FX_BASE=${FX_BASE:-201}          # 합성 시드 = FX_BASE + 8 × (반복 − 1) + 조건 번호(0~7). 시험 실행에서만 바꾼다
FX_REPS=${FX_REPS:-5}            # 조건마다 반복 수(사전 등록 5). 시험 실행에서만 바꾼다
TAG=prereg-v2                    # 정식 실행은 GitHub에 올린 이 태그의 커밋에서만 한다
# 승인 저장소 github.com/honey0127/crowdmap-membench로 허용하는 origin 주소(이 목록과 정확히 같아야 한다. https의 사용자 정보는 지우고 비교한다)
REPO_URLS="https://github.com/honey0127/crowdmap-membench https://github.com/honey0127/crowdmap-membench.git https://github.com/honey0127/crowdmap-membench/ git@github.com:honey0127/crowdmap-membench git@github.com:honey0127/crowdmap-membench.git ssh://git@github.com/honey0127/crowdmap-membench ssh://git@github.com/honey0127/crowdmap-membench.git"
R=$V/res
P=$R/parts
# ── 실행 관문(개정 7). 여기를 지나기 전에는 아무것도 쓰지 않는다 ──
# 봉인 폴더는 실제 경로로 정규화해 비교한다(./exp/sealed_v2/, 절대 경로, 심볼릭 링크도 같은 폴더다).
SEALED=$(realpath -e exp/sealed_v2) || { echo "봉인 폴더 exp/sealed_v2가 없다" >&2; exit 2; }
VR=$(realpath -m -- "$V")
case "$VR/" in
  "$SEALED"/*)
    # V가 봉인 폴더이거나 그 안이면 정식 설정만 허용한다
    if [ "$VR" != "$SEALED" ] || [ "$FX_BASE" != 201 ] || [ "$FX_REPS" != 5 ]; then
      echo "봉인 폴더($SEALED)에서는 정식 실행(V·FX_BASE·FX_REPS 기본값)만 한다. 시험 실행은 봉인 폴더 밖의 V를 따로 지정한다. 아무것도 쓰지 않고 끝낸다" >&2
      exit 7
    fi
    TESTRUN=0 ;;
  *) TESTRUN=1 ;;                # 시험 실행: 판정 코드에 --test-mode를 넘기고 판정 출력 첫 줄에 '판정이 아니다'를 적는다
esac
if [ $TESTRUN = 0 ]; then
  # 정식 실행 (가) 지문: 코드·설정·입력·사전 등록 문서(docs/prereg-v2.md)
  if ! sha256sum -c --quiet exp/sealed_v2/SHA256SUMS.txt; then
    echo "지문이 사전 등록(exp/sealed_v2/SHA256SUMS.txt)과 다르다. 실행하지 않는다" >&2
    exit 6
  fi
  # (나) 승인 커밋: 지금 커밋 = 태그의 커밋, 추적 파일 수정 없음, origin = 승인 저장소, 그 저장소의 태그 = 같은 커밋
  head=$(git rev-parse -q --verify HEAD 2>/dev/null || true)
  tagc=$(git rev-parse -q --verify "refs/tags/$TAG^{commit}" 2>/dev/null || true)
  # origin의 실제 주소: git remote get-url은 insteadOf를 풀어 쓴 주소를 준다. 주소가 둘 이상이면 허용하지 않는다
  ourl=$(git remote get-url --all origin 2>/dev/null || true)
  nurl=$(printf '%s\n' "$ourl" | sed -E 's#^https://[^/@]*@#https://#')
  url_ok=0
  if [ -n "$nurl" ] && [ "$(printf '%s\n' "$nurl" | wc -l)" -eq 1 ]; then
    for u in $REPO_URLS; do if [ "$nurl" = "$u" ]; then url_ok=1; fi; done
  fi
  why=""
  if [ -z "$head" ]; then why="git 저장소가 아니다(승인 커밋을 git clone으로 받아 실행한다)"
  elif [ -z "$tagc" ]; then why="태그 $TAG가 없다(승인 전이다)"
  elif [ "$head" != "$tagc" ]; then why="지금 커밋($head)이 태그 $TAG의 커밋($tagc)과 다르다"
  elif ! git diff --quiet HEAD --; then why="추적 파일이 태그 $TAG의 커밋과 다르다"
  elif [ $url_ok = 0 ]; then why="origin($(echo $nurl))이 승인 저장소가 아니다. 허용 주소: $REPO_URLS"
  else
    rtag=$(GIT_TERMINAL_PROMPT=0 timeout 60 git ls-remote origin "refs/tags/$TAG*" < /dev/null 2>/dev/null || true)
    rc=$(printf '%s\n' "$rtag" | awk -v t="refs/tags/$TAG" '$2 == t "^{}" {p = $1} $2 == t {q = $1} END {print (p != "" ? p : q)}')
    if [ -z "$rc" ]; then why="승인 저장소(origin)에서 태그 $TAG를 확인하지 못했다(태그를 올리지 않았거나 연결이 없다)"
    elif [ "$rc" != "$tagc" ]; then why="승인 저장소(origin)의 태그 $TAG($rc)가 이 저장소의 태그($tagc)와 다르다"
    fi
  fi
  if [ -n "$why" ]; then
    echo "$why. 정식 실행은 승인 저장소(GitHub)에 올린 태그 $TAG의 커밋에서만 한다. 실행하지 않는다" >&2
    exit 8
  fi
else
  # 시험 실행: 입력 트레이스가 봉인 트레이스(exp/sealed_v2/traces, exp/sealed_v2/day/traces)와 내용이 같으면 멈춘다(봉인 폴더를 복사한 경우 등)
  hit=$(awk 'NR == FNR {h[$1] = 1; next} ($1 in h) && n++ < 3 {print $2}' \
          <(sha256sum exp/sealed_v2/traces/*.csv exp/sealed_v2/day/traces/*.csv 2>/dev/null) \
          <(sha256sum "$V"/day/traces/*.csv 2>/dev/null))
  if [ -n "$hit" ]; then
    echo "시험 실행 입력에 봉인 트레이스와 내용이 같은 파일이 있다($(echo $hit) …). 봉인 데이터로는 시험 실행을 하지 않는다. 아무것도 쓰지 않고 끝낸다" >&2
    exit 7
  fi
fi
if [ ! -s $V/tstar_eval.txt ] || [ ! -s $V/primary.txt ] || [ ! -s $V/day/workers.txt ] || [ ! -s $V/day/blocks.csv ]; then
  echo "판정 T*·주 후보·하루 전체 입력·블록 목록 중 없는 것이 있다($V/tstar_eval.txt, $V/primary.txt, $V/day/workers.txt, $V/day/blocks.csv). 사전 등록 v2를 승인한 다음 실행한다" >&2
  exit 2
fi
TS=$(cat $V/tstar_eval.txt)
PRIMARY=$(cat $V/primary.txt)
TLIST=$(python3 -c "v=[float('$TS'),0.3,1.0,3.0,10.0]; print(','.join(f'{x:g}' for x in dict.fromkeys(v)))")
mkdir -p $V/day/sched $V/fx/traces $V/fx/sched $P
log() { echo "[$(date '+%F %T')] $*" | tee -a $P/driver.log; }
log "시작 (커밋 $(git rev-parse HEAD 2>/dev/null || echo ?), $([ $TESTRUN = 0 ] && echo "승인 태그 $TAG 확인" || echo "시험 실행이라 태그 확인 안 함"), T*=$TS, 주 후보=$PRIMARY, T* 목록=$TLIST, 시험 실행=$TESTRUN)"

if [ ! -f $P/inputs.done ]; then
  while read -r name workers cell p; do
    python3 traces.py schedule --in $V/day/traces/$name.csv --out $V/day/sched/$name.csv --workers "$workers" > $V/day/sched/$name.log
  done < $V/day/workers.txt
  for rep in $(seq 1 "$FX_REPS"); do k=0; for S in s S; do for A in p b; do for L in n w; do
    seed=$(( FX_BASE + 8 * (rep - 1) + k )); k=$(( k + 1 ))
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
real = [f"{V}/day/sched/" + l.split()[0] + ".csv" for l in open(f"{V}/day/workers.txt") if l.strip()]
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
  if python3 paperexp3.py --model chunk --tstars "$TLIST" --out $tmp.csv "$@" > $tmp.log 2>&1; then
    [ -f $tmp.csv ] || : > $tmp.csv  # 결과 행이 없으면 빈 파일(아래 묶음 확인에서 다시 돌린다)
    mv $tmp.csv $P/$id.csv && mv $tmp.log $P/$id.log && touch $P/$id.done && log "끝 $id"
  else
    echo "$id" >> $P/retries.txt
    log "실패 $id (기록 $tmp.log)"
    return 1
  fi
}
while IFS=$'\t' read -r id kind scheds; do
  [ -f $P/$id.done ] && continue
  if [ -f $P/retries.txt ] && [ "$(grep -cx "$id" $P/retries.txt)" -ge 3 ]; then
    log "묶음 $id가 세 번 실패했다. 코드 문제로 보고 멈춘다(사전 등록 이탈로 연구자에게 보고)"
    exit 5
  fi
  while [ "$(jobs -rp | wc -l)" -ge "$SLOTS" ]; do wait -n || true; done
  if [ $(( $(date +%s) - T0 )) -ge "$LAUNCH_UNTIL" ]; then break; fi
  log "시작 $id ($kind, 일정 $(wc -w <<< "$scheds")개)"
  run_part "$id" $scheds &
done < $P/plan.tsv
wait
left=$(cut -f1 $P/plan.tsv | while read -r id; do [ -f $P/$id.done ] || echo "$id"; done | wc -l)
if [ "$left" -gt 0 ]; then log "남은 묶음 $left개. 다시 실행한다"; exit 3; fi

# 묶음마다 (일정 × T* 목록 × 정책 8개) 결과가 빠짐없는지 확인한다. 빠진 묶음은 완료 표시를 지우고 같은 코드·설정으로 다시 돌린다
python3 check_parts_v2.py "$P" "$TLIST" > $P/incomplete.txt
if [ -s $P/incomplete.txt ]; then
  for id in $(cat $P/incomplete.txt); do
    echo "$id" >> $P/retries.txt
    rm -f "${P:?}/${id:?}.done"
    log "결과가 빠진 묶음 $id: 같은 코드·설정으로 다시 실행한다"
  done
  exit 3
fi

for kind in real fx; do
  first=1; : > $R/res_$kind.csv
  for id in $(awk -F'\t' -v k=$kind '$2==k{print $1}' $P/plan.tsv); do
    [ -s $P/$id.csv ] || continue
    if [ $first = 1 ]; then cat $P/$id.csv >> $R/res_$kind.csv; first=0; else tail -n +2 $P/$id.csv >> $R/res_$kind.csv; fi
  done
done
# AI를 쓸 수 없어 대체 정책으로 계산한 창(paperexp3.py의 'AI 학습 불가' 줄)을 기록으로 모은다. 판정은 결과 파일의 ai 열을 쓴다
for id in $(cut -f1 $P/plan.tsv); do grep -h "AI 학습 불가" $P/$id.log || true; done > $R/ai_unavailable.txt
MODE=""; [ $TESTRUN = 1 ] && MODE="--test-mode"
# 이전 실행의 판정 파일은 지우지 않고 prev_<시작 시각>/으로 옮긴다. 이번 판정이 중단되면 verdict.txt가 남지 않는다
for f in verdict.txt verdict_stopped.txt; do
  if [ -f $R/$f ]; then mkdir -p $R/prev_$T0 && mv $R/$f $R/prev_$T0/$f && log "이전 $f → $R/prev_$T0/$f"; fi
done
if python3 prereg_v2_analysis.py --res $R/res_real.csv --fx $R/res_fx.csv --windows $V/day/windows_day.csv --blocks $V/day/blocks.csv \
     --primary "$PRIMARY" --tstar "$TS" --tlist "$TLIST" $MODE > $R/verdict.tmp 2> $R/verdict.err; then
  mv $R/verdict.tmp $R/verdict.txt
  log "판정 끝 → $R/verdict.txt"
  cat $R/verdict.txt
else
  mv $R/verdict.tmp $R/verdict_stopped.txt
  log "판정 중단 → $R/verdict_stopped.txt: $(cat $R/verdict.err)"
  exit 4
fi
