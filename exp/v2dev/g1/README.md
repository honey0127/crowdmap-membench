# G1 구현 소스 (검토용 전달본)

계획 `docs/additional-analysis-plan-v4.md` §2.3·§2.4의 G1 단계(공통 격자 → 저차원 생존 GBDT 학습 → uint8 조회표 컴파일 → 동등성 검증)를 수행하는 스크립트 4개다.
이 묶음에는 소스만 담았다. 개발 시험의 산출물(격자·모델·표·매니페스트)과 수치는 넣지 않았다. 개발 시험 결과는 공식 G1 결과가 아니다.

| 파일 | 역할 |
|---|---|
| `g1_common.py` | 스레드 환경(numpy보다 먼저 1로 고정), 사전 점검(버전·CPU affinity·스레드 풀·Git·특징 선택 출처 해시), 학습·평가 목록 중복 검사, 학습 행 적재, 공통 격자·정수 코드화, 원자적 저장, 해시. 직접 실행하면 점검만 한다 |
| `g1_build_models.py` | 26개 개발 일정 TR로 후보별 공통 분위 격자를 만들고, 정수 코드 입력으로 매듭 10개 생존 GBDT를 학습한다 |
| `g1_compile_tables.py` | 모든 코드 조합(P1 256², P1b 65³)을 C 순서로 평가해 uint8 조회표를 만든다 |
| `g1_verify_tables.py` | 학습·평가 일정 중복 검사 뒤 코드화 경계 시험, 모든 코드 조합 검증, 420개 개발 창 실제 입력 검증을 하고 PASS/FAIL을 낸다 |

SHA-256은 `SHA256SUMS.txt`에 있다. 기존 연구 코드(`paperexp*.py`, `chunkslices.py`, `measure_a_cpu.py`)는 고치지 않고 불러 쓴다.

## 1. 풀기와 무결성 확인

```bash
cd ~/research/membench-v2sealed
unzip g1_implementation_review.zip          # exp/v2dev/g1/ 아래에만 풀린다
sha256sum -c exp/v2dev/g1/SHA256SUMS.txt     # 네 파일 모두 OK여야 한다
```

묶음에는 `feature-selection.json`·`runtime-environment.json`이 없으므로 기존 두 파일을 덮어쓰지 않는다.

## 2. 필요한 것

- 추가 설치는 필요 없다. 기존 가상환경 `~/.venvs/membench-g1/`을 그대로 쓴다.
- 공식 실행 사전 점검이 확인하는 판: Python 3.12.3, NumPy 2.5.4, SciPy 1.18.1, scikit-learn 1.9.1(`requirements-g1.txt`).
- 함께 쓰는 라이브러리: joblib(모델 저장), threadpoolctl(스레드 풀 확인). 둘 다 scikit-learn의 의존성이라 이미 설치돼 있다. 판은 점검하지 않고 매니페스트에 기록만 한다.
- 시스템 명령: `git`(사전 점검), `taskset`(CPU 1 고정). Linux 전용이다(`os.sched_getaffinity`).
- 저장소 모듈: `paperexp.py`, `paperexp2.py`, `paperexp3.py`, `chunkslices.py`, `policysim.py`, `ailearn.py`, `measure_a_cpu.py`.
- 입력 파일: `exp/dev_sched.txt`와 26개 일정, `exp/v2dev/daypool/all.txt`와 420개 일정(`exp/v2dev/daypool/sched/`), `exp/v2dev/cpu_a/environment.json`, `docs/additional-analysis-plan-v4.md`, `exp/v2dev/g1/feature-selection.json`, `exp/v2dev/g1/runtime-environment.json`.

## 3. 공식 실행 순서

```bash
R="taskset -c 1 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 $HOME/.venvs/membench-g1/bin/python"
$R -B exp/v2dev/g1/g1_common.py         # 점검만 한다(아무것도 쓰지 않는다). 통과 0, BLOCKED 2
                                        # -B: 연구 모듈 import 시 __pycache__ 작성 방지(코드 자체도 차단하지만 명시 권장)
$R exp/v2dev/g1/g1_build_models.py      # grids/, models/, model-table-manifest.json(build 절)
$R exp/v2dev/g1/g1_compile_tables.py    # tables/, 매니페스트 compile 절
$R exp/v2dev/g1/g1_verify_tables.py     # verify/actual-windows.csv, 매니페스트 verify 절, PASS/FAIL
```

- 각 스크립트는 `--help`를 지원한다. 결과가 이미 있으면 덮어쓰지 않고 멈춘다(다시 만들 때만 `--force`).
- 이 단계는 420창 정책 비용 시뮬레이션을 하지 않는다.

## 4. 공식 실행의 사전 조건

하나라도 어긋나면 결과를 쓰지 않고 `BLOCKED`로 멈춘다.

- 소프트웨어 판이 2절과 같고, CPU affinity가 `[1]`이며, 모든 스레드 풀이 1이다.
- 브랜치가 `exp/a-cpu`다.
- `exp/v2dev/g1/feature-selection.json`과 `runtime-environment.json`이 커밋돼 있고 HEAD와 같다.
- 연구 코드(`paperexp*.py`, `chunkslices.py`, `policysim.py`, `ailearn.py`, `measure_a_cpu.py`, `g1_select_features.py`), 계획 v4, `exp/dev_sched.txt`, `exp/v2dev/daypool/all.txt`, `exp/frozen.json`, `exp/v2dev/chunk/res_q1.csv`·`res_q2.csv`가 존재하고, Git에서 추적되며, HEAD와 같다. 파일이 없거나 미추적이거나 미커밋 변경이 있으면 `BLOCKED`다.
- 이 묶음의 스크립트 4개가 커밋돼 있다(계획 v4: G1 실행 전 코드 지문 커밋).
- 특징 선택 결과의 출처 해시가 현재 파일과 같다(`check_provenance`, 세 스크립트 모두 실행 전에 확인한다).

  | 기록 위치 | 필드 | 대조하는 현재 파일 |
  |---|---|---|
  | `feature-selection.json` | `script_sha256` | `g1_select_features.py` |
  | `feature-selection.json` | `schedule_list_sha256` | `exp/dev_sched.txt` |
  | `feature-selection.json` | `config_sha256` | `exp/frozen.json` |
  | `feature-selection.json` | `code_sha256` | `paperexp.py`, `paperexp2.py`, `paperexp3.py`, `chunkslices.py`(정확히 이 4개) |
  | `feature-selection.json` | `result_csv_sha256` | `exp/v2dev/chunk/res_q1.csv`, `res_q2.csv`(두 파일을 정확히 가리켜야 한다) |
  | `runtime-environment.json` | `result_json_sha256` | `exp/v2dev/g1/feature-selection.json` |
  | `runtime-environment.json` | `script_sha256` | `g1_select_features.py` |

  필드는 이름으로 찾고 중첩 위치는 가정하지 않는다. 필드가 없거나, 같은 이름이 서로 다른 값으로 여러 곳에 있거나, 값이 SHA-256 16진 문자열이 아니거나, 기록된 경로를 저장소 안 파일로 해석할 수 없거나, 해시가 다르면 `BLOCKED`다. `code_sha256`·`result_csv_sha256`은 `{경로: 해시}` 또는 `[{path(또는 file), sha256}]` 형식만 받는다. 다른 이름이나 형식으로 바꿔 맞추지 않는다. `code_sha256`에는 위 표의 4개 파일이 정확히 모두 있어야 하며, 누락·추가·중복(경로 표기가 달라도 같은 파일이면 중복)은 모두 `BLOCKED`다. JSON 같은 수준의 중복 키도 `BLOCKED`다.
- 학습 조각 수가 820,340이다(작업 A 사전 학습과 같아야 한다).
- 420창 목록의 SHA-256과 개수가 `exp/v2dev/cpu_a/environment.json`의 작업 A 목록과 같다.
- 학습 26일정과 평가 420창의 일정 이름이 겹치지 않는다(`g1_verify_tables.py`가 검증 전에 확인한다). 규칙은 `measure_a_cpu.py`와 같다: `read_list`(파일 존재, 목록 안 이름 중복 금지), `name_of`(파일 이름에서 `.csv`를 뗀 것)의 교집합 금지. 확장자를 포함한 파일 이름 교집합도 별도로 검사한다. 두 종류 중 하나라도 있으면 `BLOCKED`다. 겹치지 않으면 `n_intersection_stem`·`intersection_stem`·`n_intersection_basename`·`intersection_basename`을 포함한 `list_overlap`을 매니페스트에 적는다.

`feature-selection.json`은 최상위 `P1`·`P1b` 항목에 `features`와 `feature_indices`(또는 `index`·`indices`)가 있어야 한다. 특징 이름이 `Trace2.feat_names`의 해당 위치와 다르거나 형식이 다르면, 추측하지 않고 키 목록을 보여 주며 멈춘다. 표 축 순서는 이 파일의 특징 순서를 따른다.

## 5. 개발 시험과 공식 실행의 차이

개발 시험(2026-11-04)은 코드 경로가 끝까지 도는지 확인하려고 돌렸다. 공식 G1 결과로 쓰지 않는다.

| 항목 | 개발 시험 | 공식 실행 |
|---|---|---|
| 실행 기계 | 클라우드 컨테이너 | 연구자 WSL |
| 소프트웨어 | Python 3.13.18, NumPy 2.5.3, SciPy 1.18.1, scikit-learn 1.9.1, joblib 1.6.0 | Python 3.12.3, NumPy 2.5.4, SciPy 1.18.1, scikit-learn 1.9.1(`~/.venvs/membench-g1/`) |
| 코드 기준 | 공개 원격 `exp/a-cpu`(a18e905)의 사본. 로컬 작업 사본이 아니다 | 로컬 `exp/a-cpu` HEAD |
| 특징 선택 입력 | 손으로 만든 대체 파일(P1 = 사용량·수준 [1, 2], P1b = 높이·사용량·수준 [0, 1, 2]). `--selection`으로 넣었다 | 커밋된 `exp/v2dev/g1/feature-selection.json` |
| 검증 B 입력 | 26개 개발 일정(`--eval-list exp/dev_sched.txt`). 420창 일정이 공개 저장소에 없어 코드 경로만 시험했다 | `exp/v2dev/daypool/all.txt`의 420창. 작업 A 목록 해시와 대조한다 |
| 사전 점검 | 어긋남(판, 브랜치, 없는 파일, 커밋되지 않은 코드)을 기록만 하고 계속했다(`--dev-test`) | 하나라도 있으면 멈춘다 |
| CPU·스레드 | `taskset -c 1`, 스레드 풀 1 | 같다 |
| 산출 위치 | `exp/v2dev/g1/_devtest/` | `exp/v2dev/g1/{grids,models,tables,verify}/`, `model-table-manifest.json` |
| 판정 출력 | `DEV-TEST(판정 없음)` | `PASS` 또는 `FAIL` |

`--dev-test`, `--selection`, `--eval-list`는 개발 시험용이다. 공식 실행에서는 주지 않는다. `g1_build_models.py`에 `--selection`을, `g1_verify_tables.py`에 `--eval-list`를 기본값과 다르게 주면 `--dev-test` 없이는 멈춘다.

## 6. 개발 시험 뒤 바뀐 코드

학습·격자·코드화·조회표 계산과 PASS/FAIL 판정 규칙은 바꾸지 않았다. `g1_build_models.py`와 `g1_compile_tables.py`는 개발 시험 때와 같은 파일이다.

1. `g1_verify_tables.py`: 실제 입력 CSV(`verify/actual-windows.csv`)의 머리글을 첫 행에서 가져오던 것을 고정 열 목록으로 바꿨다. 모든 창이 건너뛰어져 행이 없을 때 멈추지 않게 하려는 것이다. 26일정 전체 개발 시험은 바꾸기 전 판으로 돌렸고, 바꾼 판은 짧은 개발 시험(P1, 일정 2개)만 다시 돌렸다.
2. `g1_common.py`: 공식 실행 사전 점검에 특징 선택 출처 해시 대조(`check_provenance`)를 더했다. 점검 함수를 `collect_issues`(읽기만)와 `preflight`(공식 실행이면 멈춤)로 나눴다. 파일을 직접 실행하면 점검만 하는 진입점을 더했다. `exp/frozen.json`과 `res_q1·q2.csv`를 커밋 여부 점검 목록에 더했다.
3. `g1_verify_tables.py`: 검증 전에 학습·평가 일정 중복을 검사한다(`check_list_overlap`). 결과를 매니페스트 `list_overlap`에 적는다.

2·3의 점검은 실제 `feature-selection.json`·`runtime-environment.json`으로는 아직 돌리지 않았다. 개발 시험 환경에 두 파일이 없어, 같은 필드 이름으로 만든 대체 JSON으로만 시험했다. 공식 실행 전에 `g1_common.py`를 직접 실행해 실제 파일로 확인한다.

## 7. 특징 선택 규칙의 사전 고정 이력

- `a18e905`(2026-11-04 11:13:51)의 계획 v4 §2.2는 특징별 빈도 상위 2·3개를 고르는 규칙이었다.
- 특징 선택을 실행하기 전에 규칙을 바꿔 커밋했고, 그다음 스크립트를 고정한 뒤 선택을 실행해 결과를 동결했다. 로컬 `exp/a-cpu` 이력은 다음과 같다.

  | 커밋 | 시각(2026-11-04) | 내용 |
  |---|---|---|
  | `3912a9a` | 11:42:48 | 전역 평균 상대 검증 비용 기반 특징 선택 규칙 고정 |
  | `7070ca3` | 11:47:08 | 특징 선택 스크립트(`g1_select_features.py`) 고정 |
  | `0754efe` | 11:50:05 | 특징 선택 결과(`feature-selection.json`, `runtime-environment.json`) 동결 |

- 따라서 a18e905의 빈도 규칙과 실제 선택 규칙의 충돌은 3912a9a에서 해결됐다. 선택은 바뀐 규칙이 커밋된 뒤에 실행됐다.
- 결과 파일이 고정된 스크립트·목록·설정·연구 코드·기록 CSV에서 나왔다는 것은 실행할 때마다 §4의 출처 해시 대조로 확인한다.
- 이력은 다음 명령으로 다시 볼 수 있다.

  ```bash
  git log --format='%h %ad %s' --date=iso -- docs/additional-analysis-plan-v4.md g1_select_features.py exp/v2dev/g1/
  ```

- 이 스크립트들은 `feature-selection.json`의 특징을 그대로 쓰며 특징을 다시 고르지 않는다.
