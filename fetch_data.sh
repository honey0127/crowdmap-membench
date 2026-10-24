#!/bin/bash
# fetch_data.sh [대상 폴더=data] [dev|sealed|all=all]
# 원본 트레이스를 받아 SHA-256으로 확인한다. 이미 있고 지문이 맞으면 건너뛴다.
#   dev    = Azure 2023(code·conv) + BurstGPT_1      약 52MB  (재실험 v1·방법 확정 개발 데이터)
#   sealed = Azure 2024 1주(code·conv) + BurstGPT_2·3 약 2.2GB (봉인 창의 원본)
# 봉인 시험을 다시 돌리는 데는 원본이 필요 없다(봉인 창이 exp/sealed/traces에 있다).
set -euo pipefail
DEST=${1:-data}
WHICH=${2:-all}
case "$WHICH" in dev|sealed|all) ;; *) echo "두 번째 인자는 dev, sealed, all 중 하나다" >&2; exit 2 ;; esac
while read -r set sub name sha url; do
  [ "$WHICH" = all ] || [ "$WHICH" = "$set" ] || continue
  f="$DEST/$sub/$name"
  mkdir -p "$DEST/$sub"
  if [ -f "$f" ] && echo "$sha  $f" | sha256sum -c --status 2>/dev/null; then
    echo "$f: 이미 있음(OK)"
    continue
  fi
  curl -fL --retry 3 -o "$f.part" "$url"
  mv "$f.part" "$f"
  echo "$sha  $f" | sha256sum -c
done <<'EOF'
dev azure2023 AzureLLMInferenceTrace_code.csv 54e9a6d2a4bd06ba1e060304b900abbc74cbea53de96506e60fe5bb4f2277fb6 https://raw.githubusercontent.com/Azure/AzurePublicDataset/master/data/AzureLLMInferenceTrace_code.csv
dev azure2023 AzureLLMInferenceTrace_conv.csv 2f1e5b666d4e3055fdbba98598ce2ec307767b9064e03e2fa46676dbcc7d0bf8 https://raw.githubusercontent.com/Azure/AzurePublicDataset/master/data/AzureLLMInferenceTrace_conv.csv
dev burstgpt BurstGPT_1.csv 46fc9480ef0b748ecb2b51d512ff08c196b031782cbe6f78e28044d768e86d5a https://raw.githubusercontent.com/HPMLL/BurstGPT/main/data/BurstGPT_1.csv
sealed azure2024 AzureLLMInferenceTrace_code_1week.csv 71de5c55cbc35f8f1ed0b6b7806b4cd1e9764b0058469725a6aac98023a1448f https://github.com/Azure/AzurePublicDataset/releases/download/dataset-llm-2024/AzureLLMInferenceTrace_code_1week.csv
sealed azure2024 AzureLLMInferenceTrace_conv_1week.csv a0cc9b969a9bbf0fd811802cbf4323edd3a209ace791e3799ad4f9207f213941 https://github.com/Azure/AzurePublicDataset/releases/download/dataset-llm-2024/AzureLLMInferenceTrace_conv_1week.csv
sealed burstgpt_v2 BurstGPT_2.csv 44bf5942b03fca42c01a226a545ad3a750ec62688faecd67b6831c56a8928bf7 https://github.com/HPMLL/BurstGPT/releases/download/v2.0/BurstGPT_2.csv
sealed burstgpt_v2 BurstGPT_3.csv 2299986a07388aa303ec2c41d1131e756db650a39ed6ef9dfe7cc3d7f9a43b8f https://github.com/HPMLL/BurstGPT/releases/download/v2.0/BurstGPT_3.csv
EOF
