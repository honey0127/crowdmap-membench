#!/bin/bash
# run_guard_test.sh — run_sealed_v2.sh의 실행 관문(사전 등록 v2 개정 8)을 확인한다.
#
#   bash checks/run_guard_test.sh      # 저장소 루트에서. 모두 맞으면 'run guard test OK'를 출력하고 0으로 끝난다
#
# 안전장치: 임시 폴더에 지문 목록의 파일(실행 코드·봉인 입력·사전 등록 문서)을 복사해 그 사본에서만 실행한다.
#   python3는 대역으로 바꿔 'python3 -c'(T* 목록 계산)만 실제 파이썬으로 돌리고, 스크립트 실행(traces.py·paperexp3.py 등)은
#   종료 코드 97로 막는다. 그래서 관문이 잘못돼도 일정·정책 계산은 돌지 않는다. 이 저장소의 파일은 읽기만 한다.
#   GitHub에는 쓰지 않는다. T19만 승인 저장소의 태그를 읽기 전용으로 조회한다(연결이 없어도 결과는 같다).
# 확인하는 경우(기대 종료 코드; 6·7·8은 아무것도 쓰지 않고 끝나야 한다)
#  시험 설정
#   T1  FX_BASE=901만 지정(V 기본값)                          → 7, 봉인 폴더에 새 파일·폴더가 없다
#   T2  FX_REPS=1만 지정                                        → 7
#   T3  V=./exp/sealed_v2/ FX_BASE=901(경로 표기만 다름)        → 7
#   T4  V=<봉인 폴더 절대 경로> FX_BASE=901                     → 7
#   T5  V=<봉인 폴더를 가리키는 심볼릭 링크> FX_BASE=901        → 7
#   T6  V=exp/sealed_v2/day(봉인 폴더 안) FX_BASE=901           → 7
#   T7  V=<봉인 폴더를 복사한 폴더> FX_BASE=901                 → 7(봉인 트레이스와 내용이 같다), 그 폴더에도 새 파일이 없다
#   T8  V=<봉인이 아닌 시험 입력> FX_BASE=901                   → 97(관문을 지나 첫 스크립트 실행에서 대역이 막는다)
#  정식 설정(사본의 실행 스크립트·지문은 이 저장소와 같다)
#   T9  git 저장소가 아님                                        → 8
#   T10 태그 prereg-v2 없음(승인 전)                             → 8
#   T11 지금 커밋이 태그의 커밋과 다름                           → 8
#   T12 지문에 없는 추적 파일 수정                               → 8
#   T13 사전 등록 문서(docs/prereg-v2.md) 한 줄 수정             → 6
#   T14 실행 코드(paperexp3.py) 한 줄 수정                       → 6
#   T15 origin = 임시 bare 저장소(커밋·태그는 모두 맞음)          → 8(승인 저장소가 아니다)
#   T16 origin = 다른 GitHub 저장소                              → 8
#   T17 origin = 승인 주소인데 insteadOf로 임시 저장소를 가리킴  → 8(실제 주소로 비교한다)
#   T18 origin 주소가 둘(승인 주소 + 임시 저장소)                 → 8
#   T19 origin = 승인 주소(https)                                → 8. 주소 확인은 지나고 원격 태그 단계에서 멈춘다:
#       승인 전이거나 연결이 없으면 '태그를 확인하지 못했다', 승인 뒤면 '태그가 다르다'(사본의 커밋은 승인 커밋이 아니다)
#  시험용 변형 사본(REPO_URLS만 임시 bare 저장소로 바꾸고 그 사본의 지문 목록을 다시 만든 사본. 원격 태그 단계의 논리 확인용)
#   T20 원격에 태그 없음                                         → 8
#   T21 원격 태그가 다른 커밋                                    → 8
#   T22 지금 커밋 = 태그 = 원격 태그, 수정 없음                  → 97(관문 통과)
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PY=$(command -v python3)
T=$(mktemp -d)
trap 'rm -rf -- "${T:?}"' EXIT
export GIT_CEILING_DIRECTORIES="$T"
fails=0
n=0

mkdir -p "$T/stub"
printf '#!/bin/sh\nif [ "$1" = "-c" ]; then exec %s "$@"; fi\necho "대역이 막음: python3 $*" >&2\nexit 97\n' "$PY" > "$T/stub/python3"
chmod +x "$T/stub/python3"

# 사본: 지문 목록의 파일과 지문 목록
mkdir -p "$T/repo"
(cd "$ROOT" && awk '{print $2}' exp/sealed_v2/SHA256SUMS.txt | xargs cp --parents -t "$T/repo" && cp --parents -t "$T/repo" exp/sealed_v2/SHA256SUMS.txt) || { echo "사본을 만들지 못했다"; exit 1; }
echo "메모(지문 목록에 없는 추적 파일)" > "$T/repo/NOTE.txt"
(cd "$T/repo" && sha256sum -c --quiet exp/sealed_v2/SHA256SUMS.txt) || { echo "사본의 지문이 맞지 않는다(저장소 상태를 먼저 확인한다)"; exit 1; }

snap() { (cd "$1" && find . -printf '%P\t%y\t%s\t%T@\n' | LC_ALL=C sort); }

check() {  # check <이름> <기대 종료 코드> <실제 종료 코드> <출력> <기대 문구> [<다른 기대 문구>]
  local name=$1 want=$2 rc=$3 out=$4 msg=$5 alt=${6:-}
  n=$((n + 1))
  if [ "$rc" = "$want" ] && { grep -qF -- "$msg" <<< "$out" || { [ -n "$alt" ] && grep -qF -- "$alt" <<< "$out"; }; }; then
    echo "통과 $name: 종료 코드 $rc"
  else
    echo "실패 $name: 종료 코드 $rc(기대 $want), 기대 문구 '$msg' / 출력: $(head -c 300 <<< "$out")"
    fails=$((fails + 1))
  fi
}

run() {  # run <이름> <기대 종료 코드> <기대 문구> <폴더> [환경 변수...]
  local name=$1 want=$2 msg=$3 dir=$4; shift 4
  local out rc
  out=$(cd "$dir" && env PATH="$T/stub:$PATH" "$@" ./run_sealed_v2.sh 2>&1)
  rc=$?
  check "$name" "$want" "$rc" "$out" "$msg"
}

same() {  # same <이름> <폴더> <이전 목록 파일>
  if ! diff -q <(snap "$2") "$3" > /dev/null; then
    echo "실패 $1: 폴더가 바뀌었다 — $(diff <(snap "$2") "$3" | head -3 | tr '\n' ' ')"
    fails=$((fails + 1))
  fi
}

S0="$T/sealed_before.txt"
snap "$T/repo/exp/sealed_v2" > "$S0"
SEALED_ABS="$T/repo/exp/sealed_v2"
ln -s "$SEALED_ABS" "$T/link"
NOTAPPROVED="정식 실행(V·FX_BASE·FX_REPS 기본값)만"

# ── 시험 설정 ──
run T1 7 "$NOTAPPROVED" "$T/repo" FX_BASE=901
same T1 "$SEALED_ABS" "$S0"
run T2 7 "$NOTAPPROVED" "$T/repo" FX_REPS=1
run T3 7 "$NOTAPPROVED" "$T/repo" V=./exp/sealed_v2/ FX_BASE=901
run T4 7 "$NOTAPPROVED" "$T/repo" V="$SEALED_ABS" FX_BASE=901
run T5 7 "$NOTAPPROVED" "$T/repo" V="$T/link" FX_BASE=901
run T6 7 "$NOTAPPROVED" "$T/repo" V=exp/sealed_v2/day FX_BASE=901
same "T2~T6" "$SEALED_ABS" "$S0"

cp -a "$SEALED_ABS" "$T/copy"
C0="$T/copy_before.txt"
snap "$T/copy" > "$C0"
run T7 7 "봉인 트레이스와 내용이 같은 파일이 있다" "$T/repo" V="$T/copy" FX_BASE=901 FX_REPS=1
same T7 "$T/copy" "$C0"

mkdir -p "$T/testin/day/traces"
printf 't,size\n0.0,1048576\n' > "$T/testin/day/traces/x0001.csv"
echo "x0001 2 test 1.000000" > "$T/testin/day/workers.txt"
printf 'block,source,week,n_days,n_windows,days\nx_w00,x,0,1,1,x_d000\n' > "$T/testin/day/blocks.csv"
echo 2.6 > "$T/testin/tstar_eval.txt"
cp "$SEALED_ABS/primary.txt" "$T/testin/primary.txt"
run T8 97 "대역이 막음: python3 traces.py schedule --in $T/testin/day/traces/x0001.csv" "$T/repo" V="$T/testin" FX_BASE=901 FX_REPS=1
same "T1~T8" "$SEALED_ABS" "$S0"

# ── 정식 설정: git 저장소·태그·origin을 차례로 갖춘다 ──
run T9 8 "git 저장소가 아니다" "$T/repo"
# 시험용 git: 사용자·시스템 git 설정(서명, 줄 끝 변환 등)을 쓰지 않는다
gx() { GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_NOSYSTEM=1 git -c user.name=guard -c user.email=guard@example.invalid \
         -c commit.gpgsign=false -c tag.gpgsign=false -c push.negotiate=false "$@"; }
g() { gx -C "$T/repo" "$@"; }
g init -q && g add -A && g commit -q -m "승인 커밋(시험)" || { echo "git 준비 실패"; exit 1; }
run T10 8 "태그 prereg-v2가 없다" "$T/repo"
g tag -a prereg-v2 -m "사전 등록 v2(시험)"
echo "다음 커밋" > "$T/repo/NOTE2.txt" && g add NOTE2.txt && g commit -q -m "태그 뒤 커밋"
run T11 8 "지금 커밋(" "$T/repo"
g reset -q --hard prereg-v2
echo "수정" >> "$T/repo/NOTE.txt"
run T12 8 "추적 파일이 태그 prereg-v2의 커밋과 다르다" "$T/repo"
g checkout -q -- NOTE.txt
echo "수정" >> "$T/repo/docs/prereg-v2.md"
run T13 6 "지문이 사전 등록" "$T/repo"
g checkout -q -- docs/prereg-v2.md
echo "# 수정" >> "$T/repo/paperexp3.py"
run T14 6 "지문이 사전 등록" "$T/repo"
g checkout -q -- paperexp3.py

APPROVED=https://github.com/honey0127/crowdmap-membench.git
gx init -q --bare "$T/origin.git"
g remote add origin "$T/origin.git" && g push -q origin HEAD:refs/heads/main refs/tags/prereg-v2
run T15 8 "이 승인 저장소가 아니다" "$T/repo"
g remote set-url origin https://github.com/someone-else/crowdmap-membench.git
run T16 8 "이 승인 저장소가 아니다" "$T/repo"
g remote set-url origin "$APPROVED"
g config "url.$T/origin.git.insteadOf" "$APPROVED"
run T17 8 "이 승인 저장소가 아니다" "$T/repo"
g config --unset "url.$T/origin.git.insteadOf"
g remote set-url --add origin "$T/origin.git"
run T18 8 "이 승인 저장소가 아니다" "$T/repo"
g remote set-url --delete origin "$T/origin.git"
out=$(cd "$T/repo" && env PATH="$T/stub:$PATH" ./run_sealed_v2.sh 2>&1); rc=$?
check T19 8 "$rc" "$out" "승인 저장소(origin)에서 태그 prereg-v2를 확인하지 못했다" "승인 저장소(origin)의 태그 prereg-v2("
same "T9~T19" "$SEALED_ABS" "$S0"

# ── 시험용 변형 사본: REPO_URLS만 임시 bare 저장소로 바꾸고, 그 사본의 지문 목록을 다시 만든다 ──
mkdir -p "$T/vrepo"
(cd "$T/repo" && awk '{print $2}' exp/sealed_v2/SHA256SUMS.txt | xargs cp --parents -t "$T/vrepo" && cp --parents -t "$T/vrepo" exp/sealed_v2/SHA256SUMS.txt NOTE.txt) || { echo "변형 사본을 만들지 못했다"; exit 1; }
sed -i "s#^REPO_URLS=.*#REPO_URLS=\"$T/origin2.git\"#" "$T/vrepo/run_sealed_v2.sh"
(cd "$T/vrepo" && awk '{print $2}' exp/sealed_v2/SHA256SUMS.txt | xargs sha256sum > "$T/sums.new" && mv "$T/sums.new" exp/sealed_v2/SHA256SUMS.txt)
v() { gx -C "$T/vrepo" "$@"; }
v init -q && v add -A && v commit -q -m "승인 커밋(변형 사본)" && v tag -a prereg-v2 -m "시험" || { echo "변형 사본 git 준비 실패"; exit 1; }
gx init -q --bare "$T/origin2.git"
v remote add origin "$T/origin2.git" && v push -q origin HEAD:refs/heads/main
run T20 8 "승인 저장소(origin)에서 태그 prereg-v2를 확인하지 못했다" "$T/vrepo"
v commit -q --allow-empty -m "다른 커밋" && v push -q origin HEAD:refs/tags/prereg-v2 && v reset -q --hard prereg-v2
run T21 8 "승인 저장소(origin)의 태그 prereg-v2(" "$T/vrepo"
v push -q -f origin refs/tags/prereg-v2
run T22 97 "대역이 막음: python3 traces.py schedule --in exp/sealed_v2/day/traces/" "$T/vrepo"

if [ "$fails" -gt 0 ]; then
  echo "run guard test 실패: $fails개"
  exit 1
fi
echo "run guard test OK (경우 $n개, 6·7·8로 끝난 경우 봉인 폴더 변화 없음)"
