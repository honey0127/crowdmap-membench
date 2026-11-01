# membench — 예측 기반 메모리 반납 정책의 순비용 평가

요청 본문을 담는 1 MiB 청크 풀에서, 놀고 있는 청크를 언제 운영체제에 돌려줄지 학습 예측으로 정하는 정책을 평가한다.
핵심 질문은 이 정책이 시뮬레이션에서 줄인 비용이 자신의 학습·추론 CPU 비용보다 큰가이다.

> **먼저 읽을 문서**: `docs/current-research-brief.md`(현행 연구 지침). 결과의 보고 순서, 쓰는 표현과 쓰지 않는 표현을 정한다.
> 이 저장소의 초기 목표(큐 본문 버퍼의 확보 지연 L, 낭비–지연 곡선)는 현재 연구가 아니다. 그때의 README는 `docs/archive/README-v0.md`에 보존했다.

## 현재 상태 (2026-10-31)

| 순서 | 단계 | 결과 요약 | 문서 |
|---|---|---|---|
| 1 | v1 봉인 시험(사전 등록 v1.1, T* = 1초) | AI가 고정 타임아웃보다 비용이 낮았다(H1 충족). 2특징 규칙보다는 비용이 높았다(H2 미충족) | `docs/sealed-results-v1.md` |
| 2 | 모델 점검·수정, C3 개발 | 결정 시각 뒤 정보가 섞인 특징 등 결함 3개를 고쳐 청크 모델을 만들었다. 개발 데이터에서 C3(AI↔규칙 장부 전환)를 골랐다 | `docs/causality-check.md`, `docs/v2-dev-results.md` |
| 3 | 실효 T* 측정 | 1 MiB 청크·4 KiB 페이지에서 실효 T*를 2.6초로 쟀다(WSL 실측 + Fargate 가격 비율 가정) | `docs/stage3-results.md` |
| 4 | v2 봉인 시험(사전 등록 v2, T* = 2.6초) | BurstGPT 41일에서 C3의 총비용이 규칙보다 2.34%p, 고정-온라인보다 9.02%p 낮았다(시뮬, H1v2~H3v2 충족) | `docs/sealed-results-v2.md` |
| 5 | 작업 A: 순비용 | 개발 40일에서 within·사전 학습 C3 모두 학습·예측 CPU가 시뮬레이션 이득보다 컸다(탈락) | `docs/a-cpu-results.md` |
| 6 | 작업 B: 위약 대조(사후) | 같은 30초 구간 안에서 예측값과 조각의 대응을 끊으면 C3의 시뮬레이션 이득이 거의 사라졌다 | `docs/b-placebo-results.md` |

- 모든 정책 비용은 청크 풀 유체 모델 시뮬레이션 값이다. CPU 시간은 Python/scikit-learn 구현의 실측이다.
- 분석 계획과 판정 기준은 `docs/additional-analysis-plan-v3.md`(커밋 `a4a480e`)에 있다.
- 논문 초안과 그림은 `docs/paper/`에 둔다. 제출처는 ICAIIC 2027(영어, IEEE 6쪽, 마감 2026-11-15)이다.

## 다시 확인하고 실행하기

- 데이터
  - 원본 트레이스는 넣지 않았다. 주소·지문·라이선스는 `DATA.md`, 받기는 `./fetch_data.sh`다.
  - 봉인 창은 `exp/sealed/traces`에 있다. 개발 40일의 일정 420개(`exp/v2dev/daypool/sched/`)는 원본에서 `exp/v2dev/daypool/run.sh`로 다시 만든다.
- 사전 등록 지문
  - v1: `./verify_prereg.sh`
  - v2: `sha256sum -c exp/sealed_v2/SHA256SUMS.txt`
- v2 봉인 판정 재실행
  - 태그 `prereg-v2`로 받은 사본에서 `./run_sealed_v2.sh`를 실행한다.
  - 판정 출력은 `exp/sealed_v2/res/verdict.txt`다. 다른 환경에서 재실행해 비교한 내용은 `docs/sealed-results-v2.md` §5와 보충표 §8에 있다. 재실행의 원시 결과는 저장소에 없다.
- 실효 T*(3단계)
  - `./stage3.sh`로 실행하고, `QUICK=1 ./stage3.sh`로 동작만 확인한다. 방법은 `docs/stage3-guide.md`에 있다.
- 작업 A(CPU 계측)
  - 단일 스레드로 CPU 하나에 고정해 실행한다.
  - 명령: `taskset -c 1 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python measure_a_cpu.py --targets exp/v2dev/daypool/all.txt --mode both --outdir <새 폴더>`
  - 스모크 시험은 `--limit 3 --reps 1`이다. 공식 결과는 `exp/v2dev/cpu_a/`에 있다.
- 작업 B(위약 대조)
  - 실행: `MEMBENCH_REPO=$PWD python3 measure_b_placebo.py --targets exp/v2dev/daypool/list_q1.txt --out <폴더>/part_q1.csv`. `list_q2`도 같게 돌린다.
  - 요약: `--summarize --targets exp/v2dev/daypool/all.txt --parts <폴더>/part_q1.csv,<폴더>/part_q2.csv --outdir <폴더>`
  - 공식 결과는 `exp/v2dev/b_placebo/`에 있다.

## 파일

### 현행 연구(v2 이후)에 쓰는 코드

| 파일 | 하는 일 |
|---|---|
| `chunkslices.py` | v2 평가 모델의 청크 분해(1 MiB 올림, 청크마다 조각 하나, 인과적 위치 특징) |
| `paperexp.py`, `paperexp2.py` | 비용 계산, 생존 모델, 그림자 장부·온라인 보정, 규칙 정책의 공통 함수 |
| `paperexp3.py` | v2 후보 정책 평가. `--model chunk`가 v2 평가 모델이고 `--model v1`은 v1 모델 재현이다 |
| `causality_check.py` | 미래 교란 시험: 자르는 시각 뒤 도착 크기를 섞어도 그 전 결정이 그대로인지 확인한다 |
| `analyze_v2dev.py`, `exp/v2dev/daypool/` | v2 후보 선택, 개발 40일 '하루 전체' 평가와 검정력 |
| `sealed_pool_v2.py`, `sealed_pool_v2day.py` | v2 봉인 창 표본과 41일 자격 창(입력만) |
| `run_sealed_v2.sh`, `prereg_v2_analysis.py`, `check_parts_v2.py` | v2 봉인 실행·판정과 결과 완전성 확인 |
| `checks/` | v2 판정 계산과 실행 관문의 자체 시험 |
| `chunkpool.cpp`, `export_decisions.py`, `stage3.sh`, `stage3_summary.py` | 3단계: 청크 풀 반납·재확보 CPU(실효 T*)와 정책 결정 재생 |
| `measure_a_cpu.py` | 작업 A: within·사전 학습 C3의 학습·예측 CPU와 순비용 판정 |
| `measure_b_placebo.py` | 작업 B: 위약 3종 × 시드 5의 사후 분해 |
| `docs/paper/` | 논문 초안과 그림(작성 중). 그림은 커밋된 결과 파일만 읽어 만든다 |

### 초기 단계 도구(보존, 현행 연구의 판정에는 쓰지 않는다)

| 파일 | 하는 일 |
|---|---|
| `alloc_bench.cpp`, `replay.cpp`, `bench.sh`, `plot.py` | 확보 지연 L과 낭비–지연 곡선 측정(초기 목표, `docs/archive/README-v0.md`) |
| `traces.py` | 합성·Azure·BurstGPT 시계열과 일정 만들기(v2에서도 일정 생성에 쓴다) |
| `policysim.py`, `ailearn.py` | 초기 반납 정책 시뮬레이션과 AI 반납 정책 초기판 |
| `summarize.py`, `dev_summary.py` | 초기 재실험 결과 표 |
| `run_frozen.py`, `sealed_pool.py`, `prereg_analysis.py`, `run_sealed.sh`, `run_sealed_parts.sh`, `verify_prereg.sh` | v1 사전 등록·봉인 시험(판정 기록 보존) |
| `fetch_data.sh` | 원본 트레이스 받기와 지문 확인 |

- `traces.py`는 v2 일정 생성(`exp/v2dev/daypool/run.sh`)에도 쓰인다.
- `sealed_pool.py`는 개발 40일의 창 추출에도 쓰인다.
