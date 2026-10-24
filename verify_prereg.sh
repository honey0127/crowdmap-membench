#!/bin/bash
# verify_prereg.sh [원본 폴더] — 사전 등록 지문(exp/sealed/SHA256SUMS.txt)을 확인한다.
#   코드·설정·창 목록 17개 + 봉인 창 목록 3개. 원본 폴더(fetch_data.sh의 대상 폴더)를 주면 원본 4개도 확인한다.
set -uo pipefail
cd "$(dirname "$0")"
export LC_ALL=C   # 목록 지문은 파일 이름 정렬 순서에 달려 있다. 정렬을 바이트 순서로 고정한다
S=exp/sealed/SHA256SUMS.txt
bad=0

n=$(grep -vc -e '(파일별' -e '  원본/' "$S")
if grep -v -e '(파일별' -e '  원본/' "$S" | sha256sum -c --quiet; then echo "파일 $n개: OK"; else bad=1; fi

while read -r h p _; do
  d=${p%/\*.csv}
  got=$( (cd "$d" 2>/dev/null && sha256sum *.csv | sha256sum | cut -d' ' -f1) || true)   # 지문을 만든 명령과 같다
  if [ "$got" = "$h" ]; then echo "$p 목록: OK"; else echo "$p 목록: FAILED"; bad=1; fi
done < <(grep '(파일별' "$S")

if [ $# -ge 1 ]; then
  if grep '  원본/' "$S" | sed "s#  원본/#  $1/#" | sha256sum -c; then :; else bad=1; fi
fi

if [ $bad = 0 ]; then echo "사전 등록 지문: 모두 맞다"; else echo "사전 등록 지문: 다른 파일이 있다" >&2; exit 1; fi
