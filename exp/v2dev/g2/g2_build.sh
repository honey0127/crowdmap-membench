#!/usr/bin/env bash
# g2_build.sh — g2_harness.cpp 컴파일
# 실행: bash exp/v2dev/g2/g2_build.sh [출력 경로]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${1:-"${SCRIPT_DIR}/g2_harness"}"

echo "[build] g2_harness.cpp → ${OUT}"
g++ -O2 -std=c++17 \
    -Wall -Wextra -Wno-unused-parameter \
    -o "${OUT}" \
    "${SCRIPT_DIR}/g2_harness.cpp" \
    -lrt
echo "[build] 완료: ${OUT}"
