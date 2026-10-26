# membench — 큐 본문 버퍼의 확보 지연 L과 낭비–지연 곡선 측정

면접 대기장 비유: 버퍼 = 의자. 의자를 새로 펴는 시간이 L, 펴 놓고 아무도 안 앉은 의자 × 시간이 낭비 W다.
같은 손님 명단(일정)을 의자 치우는 규칙(할당기 설정)만 바꿔 여러 번 돌려, 규칙마다 (낭비, 지연) 점을 찍는다.

## 저장소 안내
- 사전 등록: `docs/prereg-v1.md`(v1.1). 판정 코드·설정·창 목록의 지문은 `exp/sealed/SHA256SUMS.txt`에 있고 `./verify_prereg.sh`로 확인한다.
- 사전 등록 v2: `docs/prereg-v2.md`. 지문은 `exp/sealed_v2/SHA256SUMS.txt`(`sha256sum -c`)다. v1 유체 모델의 미래 정보 점검과 v2 평가 모델은 `docs/causality-check.md`에 있다.
- 데이터: 원본은 넣지 않았다. 주소·지문·라이선스·인용은 `DATA.md`, 받기는 `./fetch_data.sh`다.
- 결과: `exp/res*`(재실험 v1), `exp/res2`(방법 확정 개발), `exp/sealed/res`(봉인 시험: 판정 `verdict.txt`, 실행 기록 `EXECUTION.md`). 해석은 `docs/sim-results-v1.md`, `docs/paper-design-v1.md`, `docs/sealed-results-v1.md`에 있다.

| 파일 | 하는 일 |
|---|---|
| `alloc_bench.cpp` | 본문 크기별 확보 지연 L (새 페이지 폴트·0 채우기 vs 이미 펴 둔 버퍼) |
| `replay.cpp` | 일정(schedule.csv)을 실시간 재생해 낭비 W, 할당 지연 A, 반납 시간, 페이지 폴트 수를 잰다 |
| `traces.py` | 시계열 만들기(합성·Azure·BurstGPT) → 예측 가능성 검사 → 할당/반납 일정 변환 |
| `plot.py` | 표와 그림(L, 낭비–지연 곡선, B·D 시계열) |
| `policysim.py` | 반납 정책 시뮬레이션: 빈 메모리 ↔ 새 폴트 곡선, 오라클 상한, 학습 가능 몫 |
| `ailearn.py` | AI(그래디언트 부스팅) 반납 정책 + 대체 장치 평가(초기판, 앞 절반 학습·뒤 절반 평가) |
| `paperexp.py` | 논문용 재실험: 학습/검증/시험 3분할, 2특징 규칙·AI·AI 생존·PSK·오라클, 대체 장치 민감도, 트레이스 단위 일반화 |
| `summarize.py` | paperexp.py 결과 CSV → 표 |
| `paperexp2.py` | 방법 확정용: 온라인 보정(그림자 장부), 같은 결정 단계를 쓰는 고정·규칙·AI 비교, 입력 패턴 특징, 불규칙성 지표 |
| `run_frozen.py` | 고정 설정(frozen.json)으로 창을 하나씩 처리(봉인 시험용), 사전 학습 판 포함 |
| `sealed_pool.py` | 봉인 후보 창의 입력 통계, 층별 표본, 창 추출(솎기 포함) |
| `prereg_analysis.py` | 사전 등록 판정(H1~H4) 계산 |
| `dev_summary.py` | 개발 실행 비교표 |
| `run_sealed.sh` | 봉인 시험 한 번 실행(승인 뒤) |
| `run_sealed_parts.sh` | run_sealed.sh와 같은 명령을 묶음으로 나눠 실행(실제 봉인 시험에 쓴 판, `exp/sealed/res/EXECUTION.md`) |
| `verify_prereg.sh` | 사전 등록 지문(코드·설정·창 목록·봉인 창, 선택: 원본) 확인 |
| `fetch_data.sh` | 원본 트레이스 받기 + 지문 확인(`DATA.md`) |
| `paperexp3.py` | v2 후보 정책 평가: 규칙 칸×AI 세분, 칸별 κ, 규칙 대체, 고운 격자 생존. `--model chunk`가 v2 평가 모델(청크 분해 + 고친 장부), `--model v1`은 v1 모델 재현 |
| `chunkslices.py` | v2 평가 모델의 청크 분해(1 MiB 올림, 청크마다 조각 하나, 인과적 위치 특징, `docs/causality-check.md`) |
| `causality_check.py` | 미래 교란 시험: 자르는 시각 뒤 도착 크기를 섞어도 그 전 결정이 그대로인지 확인 |
| `analyze_v2dev.py` | v2 후보 개발 평가 요약과 선택 규칙 적용 |
| `exp/v2dev/daypool/` | 개발 데이터로 잰 '하루 전체' 판정 설계의 검정력(`run.sh`, `analyze_daypool.py`) |
| `sealed_pool_v2.py` | v2 봉인 창 표본(v1에서 안 쓴 BurstGPT 군집, 입력만) |
| `sealed_pool_v2day.py` | v2 봉인 날 41개의 자격 창 전부(판정에 쓰는 405창, 입력만) |
| `run_sealed_v2.sh`, `prereg_v2_analysis.py` | v2 봉인 확인 실행·판정(사전 등록 v2 승인 뒤). 실행 관문(정식 실행은 지문·GitHub 승인 태그 확인, 시험 설정은 봉인 폴더 사용 불가), 결과가 하나라도 빠지면 판정하지 않고 멈춤 |
| `check_parts_v2.py` | 묶음마다 (일정 × T* × 정책 8개) 결과가 빠짐없는지 확인(빠진 묶음은 `run_sealed_v2.sh`가 다시 실행) |
| `checks/prereg_v2_selftest.py` | 판정 계산(주 블록·기준 비용 0·AI 대체 창·H3 조건별 보고)과 판정 중단 경우 19개(누락 창·정책·T*, 잘못된 시드 등)를 답을 아는 가짜 입력으로 확인 |
| `checks/run_guard_test.sh` | 실행 관문 22개 경우(시험 설정의 봉인 폴더 사용, 승인 태그·승인 저장소 확인)를 임시 사본과 파이썬 대역으로 확인(봉인 폴더를 읽기만 하고 일정·정책 계산은 돌지 않음) |
| `chunkpool.cpp` | 3단계: 청크 풀 반납·재확보 실제 CPU 비용(실효 T*)과 정책 결정 재생 |
| `export_decisions.py` | 3단계: 유체 모델 정책 결정 → 청크 풀 사건 목록 |
| `stage3.sh`, `stage3_summary.py` | 3단계 측정 실행·요약(`docs/stage3-guide.md`) |
| `bench.sh` | build / info / L / sweep 실행기 |

## 0. 준비 (한 번)
```bash
sudo apt install -y g++ python3-numpy python3-matplotlib libjemalloc2
./bench.sh build
./bench.sh info        # out/sysinfo.txt: 페이지 크기, THP, 온도, jemalloc 로드 확인
```
- `jemalloc 확인`에 `allocator=jemalloc ... oversize=off`가 나와야 한다.
- `<jemalloc>: Unsupported system page size`가 나오면(라즈베리파이 5 기본 커널은 16KB 페이지) 둘 중 하나:
  - 4KB 커널로 부팅: `/boot/firmware/config.txt`에 `kernel=kernel8.img` 추가 후 재부팅
  - jemalloc 직접 빌드: `sudo apt install -y autoconf && git clone --depth 1 --branch 5.3.0 https://github.com/jemalloc/jemalloc && cd jemalloc && ./autogen.sh --with-lg-page=14 && make -j4` 후 `export JEMALLOC=$PWD/lib/libjemalloc.so.2`

## 1. L 재기 (수 분)
```bash
./bench.sh L                 # 1회: out/L.csv, out/L.png, 크기별 표
CPU=2 ./bench.sh Lrep 3      # 3회 × THP {default, off, on}: out/L_summary.txt, out/L_thp.png
```
- 표의 `L p50/p99`(ms)를 본문 크기 분포의 p99 크기에서 읽고, 처리시간 S·대기 예산 D_max·본문 수신 시간과 비교한다.
- THP on은 이 도구가 버퍼에 MADV_HUGEPAGE를 거는 조건이다. 서버에서 쓰려면 할당기 설정이 따로 필요하다
  (glibc: `GLIBC_TUNABLES=glibc.malloc.hugetlb=1`, jemalloc: `MALLOC_CONF=thp:always`).

## 2. 시계열 만들기
```bash
python3 traces.py synth --pattern periodic --out syn_periodic.csv   # 30초마다 같은 위상에서 몰림
python3 traces.py synth --pattern random   --out syn_random.csv     # 같은 양, 시각만 무작위(대조군)
python3 traces.py check --in syn_periodic.csv --png out/check_periodic.png

# 실제 트레이스(PC에서 받아 변환한 뒤 작은 CSV만 보드로 옮겨도 된다)
curl -LO https://raw.githubusercontent.com/Azure/AzurePublicDataset/master/data/AzureLLMInferenceTrace_conv.csv
python3 traces.py azure --in AzureLLMInferenceTrace_conv.csv --out azure_conv.csv
python3 traces.py check --in azure_conv.csv --png out/check_azure.png
curl -LO https://raw.githubusercontent.com/HPMLL/BurstGPT/main/data/BurstGPT_1.csv          # 50MB, 2개월
python3 traces.py burstgpt --in BurstGPT_1.csv --out burstgpt.csv
python3 traces.py check --in burstgpt.csv --bin 60 --horizon 60 --max-lag 700000 --lags 86400,604800
```

## 3. 일정 → 낭비–지연 곡선
```bash
python3 traces.py schedule --in syn_periodic.csv --out sched_periodic.csv --workers 2 --max-duration 300
./bench.sh sweep sched_periodic.csv 1            # 시험: 설정 10개 × 300초 ≈ 50분
nohup ./bench.sh sweep sched_periodic.csv 3 > sweep.log 2>&1 &   # 본 측정 ≈ 2.5시간
python3 plot.py series out/series/je_0.csv out/series/je_1000.csv out/series/je_-1.csv --out out/series.png
```
- 결과: `out/results.csv`(회차별 한 줄), `out/curve.png`(파란 선 = jemalloc 감쇠 스윕 = AI가 넘어야 할 선)
- jemalloc 설정: `background_thread:true,muzzy_decay_ms:0,oversize_threshold:0` + `dirty_decay_ms` 스윕(-1 = 반납 안 함)
  - jemalloc 5.3.0은 8MiB 이상 요청을 감쇠 0인 전용 arena로 보내 즉시 반납한다. `oversize_threshold:0`으로 끄고,
    메인 스레드 arena 0에 남는 8MiB 문턱은 replay·alloc_bench가 mallctl로 끈다(probe의 `oversize=off`로 확인).
- 재생 지연이 50ms를 넘은 회차는 plot.py가 빼고 알려 준다. 그 설정만 다시 돌린다.

## 4. 반납 정책 시뮬레이션 (메모리 ↔ CPU, AI 이득 상한)
```bash
python3 traces.py schedule --in syn_periodic.csv --out s_periodic.csv --workers 2
python3 policysim.py s_periodic.csv --learn --period 30 --out out/pol_periodic
python3 policysim.py sched60.csv --meas out/results.csv --out out/pol60   # 같은 일정의 실측 점과 비교
```
- 유체 모델(단편화 없음): 정책 = 놀고 있는 메모리 조각을 언제 반납하나. 선확보는 폴트 수를 줄이지 못한다.
- timeout(T) ≈ jemalloc 감쇠, oracle = 조각마다 다시 쓰일 시각을 앎(느슨한 상한).
- `--learn`: 놀기 시작할 때 알 수 있는 특징별로 타임아웃을 따로 골라(앞 절반) 뒤 절반에서 평가 = 학습 가능 몫.

## 5. AI 반납 정책 평가 (그래디언트 부스팅)
```bash
sudo apt install -y python3-sklearn          # scikit-learn + scipy
python3 ailearn.py s_periodic.csv --period 30
python3 ailearn.py s_azconv.csv --tstars 0.3,1,3,10
```
- 조각이 놀기 시작할 때 아는 특징으로 '손익분기 시간 T*보다 오래 놀까'를 예측(앞 절반 학습) → 점수 10칸별 타임아웃.
- AI+대체: 최근 5분 동안 AI의 실제 비용이 고정 타임아웃보다 크면 고정 타임아웃으로 바꾼다(분포 변화 대비).
- 기준 = 앞 절반으로 고른 고정 타임아웃(≈ jemalloc 감쇠). 뒤 절반에서 비용 감소율을 출력한다.

## 6. 논문용 재실험 (paperexp.py, 약 15분 / 2코어)
```bash
./fetch_data.sh data dev     # 개발 원본(Azure 2023, BurstGPT_1)을 data/ 아래에 받는다
mkdir -p exp/traces exp/sched
for p in periodic random poisson; do python3 traces.py synth --pattern $p --duration 3600 --out exp/traces/syn_$p.csv; done
python3 traces.py azure --in data/azure2023/AzureLLMInferenceTrace_conv.csv --out exp/traces/az_conv.csv
python3 traces.py azure --in data/azure2023/AzureLLMInferenceTrace_code.csv --out exp/traces/az_code.csv
# BurstGPT_1.csv의 시간당 요청 수 50·75·90·99·100번째 백분위 1시간 창(시작 초는 exp/bg_windows.txt)
while read q st n; do python3 traces.py burstgpt --in data/burstgpt/BurstGPT_1.csv --start $st --duration 3600 --out exp/traces/bg_q$q.csv; done < exp/bg_windows.txt
for f in exp/traces/*.csv; do python3 traces.py schedule --in $f --out exp/sched/$(basename $f) --workers 2; done
S=""; for t in syn_periodic syn_random syn_poisson az_conv az_code bg_q50 bg_q75 bg_q90 bg_q99 bg_q100; do S="$S exp/sched/$t.csv"; done
python3 paperexp.py $S --out exp/res          # within·within_parts·fallback·loto.csv (이 순서여야 loto가 똑같이 나온다)
python3 summarize.py exp/res > exp/summary.txt
```
- 동봉한 `exp/res/`는 위 명령(시드 0)의 결과다. `exp/res_seed1`, `exp/res_seed2`는 시드만 바꾼 트레이스 안 결과다.
- 결과 해석은 프로젝트 문서 `sim-results-v1.md`에 있다.

## 7. 방법 확정과 봉인 시험 (paperexp2.py, run_frozen.py)
```bash
# 개발: 온라인 보정 + 같은 결정 단계 비교(설정 비교)
python3 paperexp2.py $S --out exp/res2 --feats f15 --wcal 900 --tag f15_w900
python3 dev_summary.py exp/res2/res2_f15_w900.csv
# 봉인 시험 준비(입력 통계만): 후보 창 통계 → 층별 표본 → 창 추출
python3 sealed_pool.py stats --out exp/sealed/pool.csv
python3 sealed_pool.py sample --pool exp/sealed/pool.csv --out exp/sealed/windows.csv
python3 sealed_pool.py extract --windows exp/sealed/windows.csv --outdir exp/sealed
for r in 16 32; do python3 sealed_pool.py extract --windows exp/sealed/windows_sens.csv --outdir exp/sealed/sens$r --rate-cap $r; done
./verify_prereg.sh           # 지문 확인
# 사전 등록 승인 뒤 한 번만: 일정 → 합성 확인 실험 → 정책 실행 → 판정
./run_sealed.sh
```
- 설정은 `exp/frozen.json`, 판정 기준은 `docs/prereg-v1.md`와 `prereg_analysis.py`에 고정했다.
- 봉인 창(`exp/sealed/traces`, `sens16`, `sens32`)은 저장소에 있으므로 원본 없이 `./run_sealed.sh`를 돌릴 수 있다. 창을 원본에서 다시 뽑으려면 `DATA.md`대로 원본을 받는다(`sealed_pool.py`는 `/home/claude/dl/…`을 읽는다).
- 개발 요인 실험 입력(`exp/fx`)은 `run_sealed.sh` (3)과 같은 명령을 시드 11·12로 돌려 만들었다.

## 측정 범위(단순화)
- 할당·반납을 한 작업 스레드에서 일정대로 실행한다. 본문 복사 대신 페이지마다 1바이트를 쓴다.
- 큐 대기·처리 시간은 traces.py schedule이 워커 c개 FIFO로 미리 계산한다(할당 지연이 큐에 되먹임되지 않는다).
- NPU 입력처럼 별도 API로 잡는 버퍼(DMA 매핑 등)는 이 도구가 재지 않는다. 쓰게 되면 그 API로 같은 방식의 측정을 더한다.
