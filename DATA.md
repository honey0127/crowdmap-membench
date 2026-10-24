# 데이터

원본 트레이스는 저장소에 넣지 않는다. 일부 파일이 GitHub의 파일당 100MB 제한을 넘고, 모두 공개 주소에서 다시 받을 수 있기 때문이다. 받은 파일은 아래 지문(SHA-256)으로 같은 파일인지 확인한다.
봉인 시험을 다시 돌리는 데는 원본이 필요 없다. 봉인 창은 `exp/sealed/traces`에 있다.

## 1. 원본

| 용도 | 파일(`fetch_data.sh` 대상 폴더 기준) | 크기(바이트) | 주소 |
|---|---|---|---|
| 개발 | azure2023/AzureLLMInferenceTrace_code.csv | 320,117 | https://raw.githubusercontent.com/Azure/AzurePublicDataset/master/data/AzureLLMInferenceTrace_code.csv |
| 개발 | azure2023/AzureLLMInferenceTrace_conv.csv | 719,188 | https://raw.githubusercontent.com/Azure/AzurePublicDataset/master/data/AzureLLMInferenceTrace_conv.csv |
| 개발 | burstgpt/BurstGPT_1.csv | 50,853,373 | https://raw.githubusercontent.com/HPMLL/BurstGPT/main/data/BurstGPT_1.csv |
| 봉인 | azure2024/AzureLLMInferenceTrace_code_1week.csv | 691,989,454 | https://github.com/Azure/AzurePublicDataset/releases/download/dataset-llm-2024/AzureLLMInferenceTrace_code_1week.csv |
| 봉인 | azure2024/AzureLLMInferenceTrace_conv_1week.csv | 1,135,195,393 | https://github.com/Azure/AzurePublicDataset/releases/download/dataset-llm-2024/AzureLLMInferenceTrace_conv_1week.csv |
| 봉인 | burstgpt_v2/BurstGPT_2.csv | 144,819,209 | https://github.com/HPMLL/BurstGPT/releases/download/v2.0/BurstGPT_2.csv |
| 봉인 | burstgpt_v2/BurstGPT_3.csv | 231,682,327 | https://github.com/HPMLL/BurstGPT/releases/download/v2.0/BurstGPT_3.csv |

```
54e9a6d2a4bd06ba1e060304b900abbc74cbea53de96506e60fe5bb4f2277fb6  azure2023/AzureLLMInferenceTrace_code.csv
2f1e5b666d4e3055fdbba98598ce2ec307767b9064e03e2fa46676dbcc7d0bf8  azure2023/AzureLLMInferenceTrace_conv.csv
46fc9480ef0b748ecb2b51d512ff08c196b031782cbe6f78e28044d768e86d5a  burstgpt/BurstGPT_1.csv
71de5c55cbc35f8f1ed0b6b7806b4cd1e9764b0058469725a6aac98023a1448f  azure2024/AzureLLMInferenceTrace_code_1week.csv
a0cc9b969a9bbf0fd811802cbf4323edd3a209ace791e3799ad4f9207f213941  azure2024/AzureLLMInferenceTrace_conv_1week.csv
44bf5942b03fca42c01a226a545ad3a750ec62688faecd67b6831c56a8928bf7  burstgpt_v2/BurstGPT_2.csv
2299986a07388aa303ec2c41d1131e756db650a39ed6ef9dfe7cc3d7f9a43b8f  burstgpt_v2/BurstGPT_3.csv
```

받기와 확인:

```bash
./fetch_data.sh                 # data/ 아래에 전부 받는다(약 2.3GB). 이미 있고 지문이 맞으면 건너뛴다
./fetch_data.sh data dev        # 개발 원본만(약 52MB)
./verify_prereg.sh data         # 사전 등록 지문 + 봉인 원본 4개 지문
```

- 봉인 원본의 지문은 사전 등록(`docs/prereg-v1.md` §7, `exp/sealed/SHA256SUMS.txt`)에도 적혀 있다.
- `sealed_pool.py`는 봉인을 준비한 환경의 경로 `/home/claude/dl/…`에서 원본을 읽는다. 다른 곳에 받았으면 그 폴더를 이 경로로 링크한다(`sudo mkdir -p /home/claude && sudo ln -s "$PWD/data" /home/claude/dl`). 코드를 고치지 않으므로 코드 지문이 그대로 맞는다.

## 2. 저장소에 있는 파생 데이터

| 경로 | 내용 | 만든 방법 |
|---|---|---|
| `exp/sealed/traces/*.csv` (131개) | 봉인 창(R 54 + S 80, 겹침 3) | `sealed_pool.py sample` → `extract --windows exp/sealed/windows.csv --outdir exp/sealed` (표본 시드 20261024, 솎기 상한 8건/s) |
| `exp/sealed/sens16/traces`, `sens32/traces` (6개씩) | 솎기 민감도 창 | `extract --windows exp/sealed/windows_sens.csv --outdir exp/sealed/sens16 --rate-cap 16`(32도 같다) |
| `exp/sched/*.csv` (10개) | 개발 트레이스 일정 | README §6 |
| `exp/fx/sched/*.csv` (16개) | 개발 요인 실험 일정 | `run_sealed.sh` (3)과 같은 명령, 시드 11·12 |

- 트레이스 형식은 `t_s,size_bytes,work`(도착 시각 초, 본문 크기 바이트, 출력 토큰 수)다. 일정의 처리시간은 `work`를 쓰지 않는다(`--svc-ms-per-work` 기본 0).
- 일정은 `traces.py schedule`의 출력이다. 실험 코드(`paperexp*.py`, `run_frozen.py`)는 일정을 읽는다.
- 저장소에 넣지 않은 중간 파일(`exp/traces`, `exp/fx/traces`, 봉인 실행이 만드는 일정·합성 입력)은 같은 명령으로 바이트 단위까지 다시 만들어진다. 개발 요인 실험 2개(fx_Sbn_11, fx_spw_12)로 트레이스·일정이 같음을 확인했다.

## 3. 원본에서 바꾼 점

- 파일 첫 요청부터 1시간 단위로 창을 잘랐다. 창의 시각은 창 시작을 0으로 둔다.
- 요청 크기는 토큰 수(Azure: 입력 토큰, BurstGPT: 요청 토큰)를 바이트로 환산했다. 분포 모양을 유지하고 창마다 중앙값을 2 MiB에 맞춘 뒤 64 KiB~32 MiB로 잘랐다(`traces.py scale_sizes`).
- BurstGPT는 실패 응답을 빼고, 정수 초 시각에 U(0,1)을 더해 흩뿌렸다.
- 봉인 창은 평균 도착이 8건/s를 넘으면 요청마다 확률 p = 8·3600 ÷ n으로 남겼다(창마다 고정 시드). 자세한 내용은 `docs/prereg-v1.md` §4에 있다.

## 4. 라이선스와 인용

두 데이터셋 모두 CC BY 4.0이다. 파생 데이터(2절)를 쓸 때도 원본의 출처를 밝힌다.

- Azure LLM 추론 트레이스(Microsoft Azure Public Dataset, https://github.com/Azure/AzurePublicDataset)
  - 2023: Pratyush Patel, Esha Choukse, Chaojie Zhang, Aashaka Shah, Íñigo Goiri, Saeed Maleki, Ricardo Bianchini. "Splitwise: Efficient generative LLM inference using phase splitting." ISCA 2024.
  - 2024: Jovan Stojkovic, Chaojie Zhang, Íñigo Goiri, Josep Torrellas, Esha Choukse. "DynamoLLM: Designing LLM Inference Clusters for Performance and Energy Efficiency." HPCA 2025.
- BurstGPT(https://github.com/HPMLL/BurstGPT): Yuxin Wang et al. "BurstGPT: A Real-World Workload Dataset to Optimize LLM Serving Systems." KDD '25. https://doi.org/10.1145/3711896.3737413
