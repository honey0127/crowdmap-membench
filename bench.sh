#!/usr/bin/env bash
# bench.sh — 측정 실행기 (PC·서버·보드 공통)
#   ./bench.sh build                         컴파일
#   ./bench.sh info                          장비 정보 기록 → out/sysinfo.txt
#   ./bench.sh L                             크기별 확보 지연 L → out/L.csv, out/L.png
#   ./bench.sh Lrep [회차]                   L을 회차(기본 3)만큼 THP 조건(기본 default off on)별로 재고 표 하나로 정리
#                                            → out/L_run1..N/L.csv, out/L_thp.png
#   ./bench.sh sweep <schedule.csv> [반복]   같은 일정 재생: glibc + jemalloc 감쇠 스윕 → out/results.csv, out/curve.png
# 환경변수(선택)
#   JEMALLOC=/경로/libjemalloc.so.2          자동으로 못 찾을 때
#   SIZES="64K 256K 1M 2M 4M 8M 16M 64M"     L 측정 크기
#   THP_MODES="default"                      L 측정 THP 조건 (예: "default off on"; Lrep 기본은 "default off on")
#   CPU=2                                    L 측정 때 고정할 코어
#   DECAYS="0 10 50 200 1000 5000 10000 30000 -1"   jemalloc dirty_decay_ms 스윕 값(-1 = 반납 안 함)
#   OUT=out                                  결과 폴더
set -euo pipefail
cd "$(dirname "$0")"
OUT=${OUT:-out}
SIZES=${SIZES:-"64K 256K 1M 2M 4M 8M 16M 64M"}
THP_MODES_USER=${THP_MODES:-}
THP_MODES=${THP_MODES:-"default"}
DECAYS=${DECAYS:-"0 10 50 200 1000 5000 10000 30000 -1"}
# 반납은 백그라운드 스레드가, MADV_FREE(muzzy) 단계 없이, 8MiB 이상도 감쇠 규칙을 따르게 한다
JE_BASE=${JE_BASE:-"background_thread:true,muzzy_decay_ms:0,oversize_threshold:0"}
mkdir -p "$OUT"

find_je() {
  local p
  for p in "${JEMALLOC:-}" /usr/lib/aarch64-linux-gnu/libjemalloc.so.2 /usr/lib/x86_64-linux-gnu/libjemalloc.so.2 \
           /usr/local/lib/libjemalloc.so.2 /usr/lib/libjemalloc.so.2 /usr/lib64/libjemalloc.so.2; do
    if [[ -n "$p" && -f "$p" ]]; then echo "$p"; return 0; fi
  done
  return 1
}

build() {
  g++ -O2 -std=c++17 -o alloc_bench alloc_bench.cpp -ldl
  g++ -O2 -std=c++17 -pthread -o replay replay.cpp -ldl
  echo "빌드 완료: ./alloc_bench ./replay"
}
need_build() { [[ -x ./alloc_bench && -x ./replay ]] || build; }

check_je() {  # jemalloc이 실제로 로드되고 설정이 먹는지 확인
  local je="$1" out
  out=$(MALLOC_CONF="$JE_BASE,dirty_decay_ms:1234" LD_PRELOAD="$je" ./replay --probe 2>&1 || true)
  echo "jemalloc 확인: $out"
  if ! grep -q "allocator=jemalloc" <<<"$out"; then
    echo "[오류] jemalloc이 로드되지 않았다. 'Unsupported system page size'가 보이면 README의 페이지 크기 항목을 본다." >&2
    return 1
  fi
  grep -q "dirty_decay_ms=1234" <<<"$out" || { echo "[오류] MALLOC_CONF가 적용되지 않았다" >&2; return 1; }
  grep -q "background_thread=true" <<<"$out" || echo "[경고] background_thread가 꺼져 있다 → 반납이 할당 경로에서 일어난다" >&2
  grep -q "oversize=off" <<<"$out" || echo "[경고] oversize 문턱이 남아 있다 → 그 크기 이상은 감쇠와 무관하게 즉시 반납된다" >&2
  return 0
}

info() {
  need_build
  {
    echo "date: $(date -Is)"
    echo "uname: $(uname -a)"
    echo "page_size: $(getconf PAGESIZE)"
    echo "thp_enabled: $(cat /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null || echo NA)"
    echo "thp_defrag: $(cat /sys/kernel/mm/transparent_hugepage/defrag 2>/dev/null || echo NA)"
    echo "cpu: $(grep -m1 -E 'model name|Model' /proc/cpuinfo | cut -d: -f2- | xargs || true)"
    echo "nproc: $(nproc)"
    echo "governor: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo NA)"
    if command -v vcgencmd >/dev/null; then echo "temp: $(vcgencmd measure_temp) throttled: $(vcgencmd get_throttled)"; fi
    free -m
    echo "glibc: $(./replay --probe)"
    if je=$(find_je); then
      echo "jemalloc_path: $je"
      check_je "$je" || true
    else
      echo "jemalloc: 없음 (sudo apt install libjemalloc2)"
    fi
  } 2>&1 | tee "$OUT/sysinfo.txt"
}

measure_L() {
  need_build
  local f="$OUT/L.csv" je="" first=1 s thp
  local args=()
  [[ -n "${CPU:-}" ]] && args+=(--cpu "$CPU")
  je=$(find_je) || echo "[경고] jemalloc을 찾지 못해 glibc만 잰다 (JEMALLOC=경로 로 지정)" >&2
  [[ -n "$je" ]] && { check_je "$je" || je=""; }
  : > "$f"
  for thp in $THP_MODES; do
    for s in $SIZES; do  # 크기마다 새 프로세스: 할당기 상태가 앞 크기에 끌려가지 않게
      if (( first )); then
        ./alloc_bench --sizes "$s" --thp "$thp" ${args[@]+"${args[@]}"} >> "$f"
        first=0
      else
        ./alloc_bench --sizes "$s" --thp "$thp" --no-header ${args[@]+"${args[@]}"} >> "$f"
      fi
      if [[ -n "$je" ]]; then  # jemalloc은 malloc 계열 행만 추가(os_* 행은 할당기와 무관)
        MALLOC_CONF="$JE_BASE" LD_PRELOAD="$je" ./alloc_bench --sizes "$s" --thp "$thp" --no-header ${args[@]+"${args[@]}"} \
          | grep -E ',(malloc_first|malloc_touch|free),' >> "$f" || true
      fi
      echo "L 측정: $s (THP $thp) 완료"
    done
  done
  echo "→ $f"
  python3 plot.py L "$f" "$OUT/L.png" || echo "(matplotlib이 없으면 그림만 건너뛴다: pip install matplotlib)"
}

measure_L_rep() {
  need_build
  local reps="${1:-3}" base="$OUT" i files=()
  local modes="${THP_MODES_USER:-default off on}"
  echo "L 반복 측정: ${reps}회 × THP {${modes}} × 크기 {${SIZES}}"
  for (( i = 1; i <= reps; i++ )); do
    ( OUT="$base/L_run$i"; THP_MODES="$modes"; mkdir -p "$OUT"; measure_L ) > /dev/null
    files+=("$base/L_run$i/L.csv")
    echo "회차 $i/$reps 끝"
  done
  python3 plot.py Lsum "${files[@]}" --out "$base/L_thp.png" | tee "$base/L_summary.txt"
}

sweep() {
  need_build
  local sched="${1:-}" reps="${2:-3}" je res="$OUT/results.csv" dur nconf hdr="" r cfg
  [[ -f "$sched" ]] || { echo "사용법: ./bench.sh sweep <schedule.csv> [반복]" >&2; exit 1; }
  je=$(find_je) || { echo "[오류] jemalloc 없음: sudo apt install libjemalloc2 또는 JEMALLOC=경로" >&2; exit 1; }
  check_je "$je"
  dur=$(awk -F, 'NR>1 && $2+0>m {m=$2+0} END {printf "%d", m+1}' "$sched")
  nconf=$(( 1 + $(wc -w <<<"$DECAYS") ))
  echo "일정 길이 약 ${dur}s × 설정 ${nconf}개 × 반복 ${reps} → 약 $(( (dur + 2) * nconf * reps / 60 ))분"
  mkdir -p "$OUT/series"
  [[ -s "$res" ]] || hdr="--header"
  for (( r = 1; r <= reps; r++ )); do
    # 설정 순서를 매번 섞는다(보드 온도·시간에 따른 치우침 방지)
    for cfg in $(printf '%s\n' glibc $DECAYS | shuf); do
      local ser=()
      if [[ "$cfg" == glibc ]]; then
        (( r == 1 )) && ser=(--series "$OUT/series/glibc.csv")
        ./replay "$sched" glibc --rep "$r" $hdr ${ser[@]+"${ser[@]}"} >> "$res"
      else
        (( r == 1 )) && ser=(--series "$OUT/series/je_$cfg.csv")
        MALLOC_CONF="$JE_BASE,dirty_decay_ms:$cfg" LD_PRELOAD="$je" \
          ./replay "$sched" "je_$cfg" --rep "$r" $hdr ${ser[@]+"${ser[@]}"} >> "$res"
      fi
      hdr=""
      echo "반복 $r/$reps: $cfg 완료"
    done
  done
  echo "→ $res"
  python3 plot.py curve "$res" "$OUT/curve.png" || echo "(matplotlib이 없으면 그림만 건너뛴다)"
}

case "${1:-}" in
  build) build ;;
  info) info ;;
  L) measure_L ;;
  Lrep) shift; measure_L_rep "$@" ;;
  sweep) shift; sweep "$@" ;;
  *) sed -n '2,15p' "$0"; exit 1 ;;
esac
