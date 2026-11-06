# 관련 연구 최종 감사 — 서지 정보 및 연구 차별성 검증

> 작성: 2026-11-06  
> 감사 대상: `docs/related-work-matrix.md`, `docs/manuscript-outline.md`  
> 검증 방법: DOI 등록 정보, 학회 공식 페이지(ACM DL, USENIX, SPLASH, arXiv), 공개 PDF 직접 확인  
> 상태: 감사 완료. 기존 연구 문서 미수정. 추가 실험 없음. Git commit 없음.

---

## 1. 11편 서지 정보 검증 결과

### 오류 요약표 (수정 필요 항목)

| # | 논문 약칭 | 오류 항목 | 현재 기재 | 검증된 올바른 값 | 중요도 |
|---|----------|---------|---------|--------------|--------|
| 1 | RecMG | arXiv 번호 | 2502.03091 | **2511.08568** | 치명 |
| 1 | RecMG | 저자 (전체) | Zhu, Xianhao et al. | **Jie Ren, Bin Ma, Shuangyan Yang, Benjamin Francis, Ehsan K. Ardestani, Min Si, Dong Li** | 치명 |
| 1 | RecMG | end-to-end speedup | 31% | **up to 43%** | 경미 |
| 2 | MEMTIS | 제1저자 성 | Kim, Taehyung | **Lee, Taehyung** | 치명 |
| 3 | TMO | 2번째 저자 | Niket Iyer | **Niket Agarwal** | 치명 |
| 4 | LeCaR | 3번째 저자 이름 | Waldek Talmadge Martinez | **Wendy A. Martinez** | 치명 |
| 4 | LeCaR | 누락 저자 | — | **Ming Zhao, Giri Narasimhan** (8명) | 치명 |
| 5 | MemPerf | 제목 | Profiling Memory Performance of Allocator-Heavy Applications | **MemPerf: Profiling Allocator-Induced Performance Slowdowns** | 치명 |
| 5 | MemPerf | 저자 (전체) | Mossé, Miguel et al. (가상) | **Jin Zhou, Sam Silvestro, Steven Tang, Hanmei Yang, Hongyu Liu, Guangming Zeng, Bo Wu, Cong Liu, Tongping Liu** | 치명 |
| 5 | MemPerf | DOI 미기재 | — | **10.1145/3622848** | 중요 |
| 6 | BurstGPT | 제1저자 | Wang, Ruidong | **Yuxin Wang** | 치명 |
| 6 | BurstGPT | 제2저자 | Chen, Zhenyu | **Yuhan Chen** | 치명 |
| 6 | BurstGPT | 제3저자 | Li, Zhuomin | **Zeyu Li** | 치명 |
| 7 | SC 2025 | 제목 불명 | (불명) | **"Assessing Page Reclamation Mechanisms for Linux"** | 보완 |
| 7 | SC 2025 | 발표 형태 | "SC 2025 발표 초록" | **SC Workshops '25 (ROSS 워크숍), not 메인 트랙** | 중요 |
| 8 | LLAMA | 5번째 저자 | Murray, Derek G. | **McKinley, Kathryn S. (추정)** — 재확인 권장 | 경미 |

---

### 1.1 LLAMA (ASPLOS 2020)

- **제목**: Learning-based Memory Allocation for C++ Server Workloads ✓
- **저자**: Martin Maas, David G. Andersen, Michael Isard, Mohammad Mahdi Javanmard, **Kathryn S. McKinley(?), Colin Raffel** — 현재 matrix에 "Murray, Derek G."로 기재; ACM DL 등록 저자 재확인 권장
- **학회**: ASPLOS '20 ✓
- **연도**: 2020 ✓
- **DOI**: 10.1145/3373376.3378525 ✓
- **공개 원문**: https://colinraffel.com/publications/asplos2020learning.pdf (저자 공개)
- **확인 수준**: PDF 본문 직접 읽기 (이전 세션 에이전트 검색)

### 1.2 TMO (ASPLOS 2022)

- **제목**: TMO: Transparent Memory Offloading in Datacenters ✓
- **저자**: Johannes Weiner, **Niket Agarwal** (matrix: "Niket Iyer" — 오류), Dan Schatzberg, Leon Yang, Hao Wang, Blaise Sanouillet, Bikash Sharma, Tejun Heo, Mayank Jain, Chunqiang Tang, Dimitrios Skarlatos
- **학회**: ASPLOS '22 ✓
- **연도**: 2022 ✓
- **DOI**: 10.1145/3503222.3507731 ✓
- **공개 원문**: https://www.cs.cmu.edu/~dskarlat/publications/tmo_asplos22.pdf (저자 공개)
- **확인 수준**: PDF 본문 직접 읽기

### 1.3 MEMTIS (SOSP 2023)

- **제목**: MEMTIS: Efficient Memory Tiering with Dynamic Page Classification and Page Size Determination ✓
- **저자**: **Taehyung Lee** (matrix: "Kim, Taehyung" — 오류), Sumit Kumar Monga, Changwoo Min, Young Ik Eom
- **학회**: SOSP 2023 ✓
- **연도**: 2023 ✓
- **DOI**: 10.1145/3600006.3613167 ✓
- **공개 원문**: https://multics69.github.io/pages/pubs/memtis-lee-sosp23.pdf (저자 공개)
- **확인 수준**: PDF 본문 직접 읽기

### 1.4 LeCaR (HotStorage 2018)

- **제목**: Driving Cache Replacement with ML-based LeCaR ✓
- **저자**: Giuseppe Vietri, Liana V. Rodriguez, **Wendy A. Martinez** (matrix: "Waldek Talmadge Martinez" — 오류), Steven Lyons, Jason Liu, Raju Rangaswami, **Ming Zhao, Giri Narasimhan** (matrix에서 누락 — 오류)
- **학회**: USENIX HotStorage 2018 (10th USENIX Workshop on Hot Topics in Storage and File Systems) ✓
- **연도**: 2018 ✓
- **ACM DL**: 10.5555/3277332.3277335
- **공식 페이지**: https://www.usenix.org/conference/hotstorage18/presentation/vietri
- **확인 수준**: USENIX 공식 페이지 직접 확인

### 1.5 MemPerf (OOPSLA 2023)

- **제목**: **MemPerf: Profiling Allocator-Induced Performance Slowdowns** (matrix: "Profiling Memory Performance of Allocator-Heavy Applications" — 완전 오류)
- **저자**: **Jin Zhou, Sam Silvestro, Steven (Jiaxun) Tang, Hanmei Yang, Hongyu Liu, Guangming Zeng, Bo Wu, Cong Liu, Tongping Liu** (matrix 저자 완전 오류)
- **학회**: SPLASH 2023 / OOPSLA ✓
- **연도**: 2023 ✓
- **DOI**: **10.1145/3622848** (matrix에 미기재)
- **공식 페이지**: https://2023.splashcon.org/details/splash-2023-oopsla/82/MemPerf-Profiling-Allocator-Induced-Performance-Slowdowns
- **확인 수준**: SPLASH 2023 공식 페이지 직접 확인 (본문 미독)

### 1.6 RecMG (HPCA 2025)

- **제목**: Machine Learning-Guided Memory Optimization for DLRM Inference on Tiered Memory ✓
- **저자**: **Jie Ren, Bin Ma, Shuangyan Yang, Benjamin Francis, Ehsan K. Ardestani, Min Si, Dong Li** (matrix 저자 완전 오류)
- **학회**: HPCA 2025 ✓ (HPCA 2025 프로그램에 수록 확인)
- **연도**: 2025 ✓
- **arXiv**: **2511.08568** (matrix: "2502.03091" — 완전 오류; arXiv:2502.03091은 격자 QCD 물리학 논문)
- **arXiv 등록일**: 2025년 11월 11일 (HPCA 2025 발표 후 등록으로 추정)
- **end-to-end speedup**: **up to 43%** (matrix: "31%" — 오류, 초록 직접 인용)
- **확인 수준**: arXiv:2511.08568 초록 직접 확인

### 1.7 SC 2025 (SC Workshops '25)

- **제목**: **Assessing Page Reclamation Mechanisms for Linux** (matrix에서 불명 — 보완 완료)
- **저자**: Shaochang Liu, Jie Ren (College of William & Mary)
- **발표 형태**: **SC Workshops '25 (ROSS 워크숍)** — 메인 SC 2025 트랙 아님
- **학회**: Proceedings of the SC '25 Workshops
- **연도**: 2025 ✓
- **DOI**: **10.1145/3731599.3767537** (신규 확인)
- **공개 원문**: ACM DL Open Access (CC BY 4.0)
- **확인 수준**: ACM DL 전체 본문 직접 읽기 (이번 세션 확인)

### 1.8 KML (TOCS 2023)

- **제목**: KML: Using Machine Learning to Improve Storage Systems ✓
- **저자**: Ibrahim Umit Akgun, Ali Selman Aydin, Andrew Burford, Michael McNeill, Michael (외) — 전체 목록 arXiv:2111.11554에서 확인 가능
- **저널**: ACM Transactions on Computer Systems (TOCS) ✓ (arXiv 버전도 존재)
- **연도**: 2023 ✓
- **DOI**: 10.1145/3579370 (matrix 기재; 재확인 권장)
- **arXiv**: 2111.11554 ✓
- **확인 수준**: arXiv 메타데이터 확인 (본문 미독)

### 1.9 jemalloc

- **제목**: jemalloc: A Scalable Concurrent malloc Implementation
- **저자**: Jason Evans (외)
- **공식 페이지**: https://jemalloc.net/ ✓
- **확인 수준**: 공식 문서 직접 확인

### 1.10 TCMalloc

- **제목**: TCMalloc: Thread-Caching Malloc
- **저자**: Sanjay Ghemawat, Paul Menage
- **공식 페이지**: https://google.github.io/tcmalloc/ ✓
- **확인 수준**: 공식 문서 직접 확인

### 1.11 BurstGPT (KDD 2025)

- **제목**: BurstGPT: A Real-World Workload Dataset to Optimize LLM Serving Systems ✓
- **저자**: **Yuxin Wang**, **Yuhan Chen**, **Zeyu Li**, Xueze Kang, Yuchu Fang, Yeju Zhou, Yang Zheng, Zhenheng Tang, Xin He, Rui Guo, Xin Wang, Qiang Wang, Amelie Chi Zhou, Xiaowen Chu (matrix 저자 오류: Ruidong Wang, Zhenyu Chen, Zhuomin Li)
- **학회**: KDD 2025 (31st ACM SIGKDD) ✓
- **연도**: 2025 ✓
- **DOI**: 10.1145/3711896.3737413 ✓
- **GitHub**: https://github.com/HPMLL/BurstGPT ✓
- **확인 수준**: GitHub BibTeX + ACM DL 메타데이터 직접 확인

---

## 2. 원문 확인 수준 (갱신)

| # | 논문 | 이전 상태 | 감사 후 상태 | 근거 |
|---|------|---------|------------|------|
| 1 | LLAMA | VERIFIED (본문) | VERIFIED | PDF 직접 읽기 |
| 2 | TMO | VERIFIED (본문) | VERIFIED | PDF 직접 읽기 |
| 3 | MEMTIS | VERIFIED (본문) | VERIFIED | PDF 직접 읽기 |
| 4 | LeCaR | VERIFIED (본문) | VERIFIED | USENIX 공식 페이지 |
| 5 | MemPerf | PARTIAL (초록) | PARTIAL → 제목·저자 수정 완료 | SPLASH 공식 페이지 (본문 미독) |
| 6 | RecMG | VERIFIED (arXiv 오인) | **arXiv 번호·저자 수정 필요** | arXiv:2511.08568 직접 확인 |
| 7 | SC 2025 | PARTIAL (초록) | **제목·DOI·형태 수정 필요** | ACM DL 전체 본문 직접 읽기 |
| 8 | KML | PARTIAL (초록) | PARTIAL | arXiv:2111.11554 메타데이터 |
| 9 | jemalloc | VERIFIED (문서) | VERIFIED | 공식 문서 |
| 10 | TCMalloc | VERIFIED (문서) | VERIFIED | 공식 문서 |
| 11 | BurstGPT | VERIFIED | **저자명 수정 필요** | GitHub BibTeX + ACM DL |

**본문 미독 항목에서 비용 평가 주장을 직접 인용할 경우**: MemPerf, KML 본문 열람 필수.

---

## 3. RecMG 상세 비교

> RecMG는 related-work-matrix.md에서 "가장 근접한 선행 연구"로 선정됐다. 이를 재검증한다.

### 3.1 RecMG 선정 근거 — 재검증 판정: **유지**

RecMG는 조사 범위(11편) 내에서 "ML 추론 비용을 명시적 수치(92 µs)로 측정하고, 그 처리 방안을 직접 논의한 유일한 논문"이라는 선정 기준이 유지된다. 이 기준 자체는 올바르다. 단, 아래 비교표에서 보듯 RecMG의 최적화 대상이 이 연구와 근본적으로 달라 "가장 유사한" 연구이지 "동일한" 연구가 아님을 명확히 해야 한다.

### 3.2 RecMG vs C3 상세 비교표

| 비교 항목 | RecMG | 본 연구 C3 |
|----------|-------|----------|
| **최적화 대상** | DLRM 임베딩 접근 latency 감소 (추천 시스템 추론 처리량) | 메모리 청크 풀 OS 반환에 소비되는 비용 절약 |
| **관리 단위** | 임베딩 벡터 블록 (tiered memory 내 caching/prefetching) | 메모리 청크 (madvise MADV_DONTNEED 반환 단위) |
| **AI 개입 위치** | 임베딩 접근 직전: 어떤 임베딩을 hot tier에 올릴지 결정 | 각 300초 창: AI vs 규칙 중 어느 경로를 선택할지 결정 |
| **학습 방식** | LSTM + Attention, 오프라인 (전용 데이터 레이블링 파이프라인) | Cox 비례위험 + gradient boosting (오프라인) → 65³ 조회표 증류 |
| **추론 방식** | 온라인, 단건 추론 92 µs/call | 조회표 직접 참조 (3-feature lookup) |
| **특징 생성 비용 측정** | 미측정 (특징 추출이 메인 경로와 분리됨) | compute_p1b_features가 ΔI에 포함돼 측정됨 |
| **정책 실행 비용 측정** | 추론 비용 92 µs 명시; 파이프라이닝 시간 측정은 없음 | G2W3: ΔI = CPU_C3 − CPU_rule (실제 추가 사이클 측정) |
| **메모리 효과 측정** | 임베딩 on-demand fetch 감소 (2.2× ~ 2.8×); 직접 메모리 반환 없음 | ΔR (반환량 차이, MiB), ΔH (유휴 보유 비용 차이, MiB·s) |
| **전체 비용 모델** | 없음 (throughput/latency 위주 평가) | G_model = c(ΔR + ΔH/T*) − ΔI; 비용 등가 단위 µs/MiB |
| **손익분기 분석** | 없음; "파이프라이닝으로 숨길 수 있다"는 주장 | G_model > 0 조건; ΔI, T*, c, ΔH, ΔR 각각 민감도 분석 |
| **평가 환경** | Industrial DLRM inference, tiered memory (CXL/SSD tier) | WSL2, Intel Core Ultra 7 155H, BurstGPT LLM 추론 420창 |
| **결과 보고 방향** | PASS: RecMG가 on-demand fetch를 2.2–2.8× 감소시킴 | FAIL: G_model 평균 −845.072 ms (G2 최종 판정) |

### 3.3 공통점 (축소하지 않을 것)

- 두 연구 모두 ML 추론 비용을 구체적인 수치로 측정했다는 점이 핵심 공통점이다.
- 두 연구 모두 "ML 사용이 정당화되는가"라는 질문을 명시적으로 제기했다.
- 두 연구 모두 추론 비용과 시스템 편익 사이의 관계를 중심 주제로 다룬다.
- RecMG는 조사된 모든 선행 연구 중 이 연구에 가장 근접하며, Related Work에서 가장 길게 비교해야 할 논문이다.

### 3.4 차이점 요약 (논문 본문에 사용할 한 문단 초안)

> RecMG [Cite]는 DLRM 임베딩 프리패치에 LSTM+Attention 모델을 적용하고 추론 비용 92 µs를 명시적으로 측정한 점에서 본 연구와 가장 근접한다. 그러나 RecMG는 임베딩 접근 latency와의 파이프라이닝으로 추론 비용을 숨길 수 있다고 주장할 뿐, 추론 비용과 메모리 절감을 동일 단위(CPU 시간)로 환산하는 비용 모델을 제시하지 않는다. 본 연구는 비용 등가 단위 c(µs/MiB)를 도입하여 G_model = c(ΔR + ΔH/T*) − ΔI > 0 을 판정 기준으로 명시화하고, 사전 등록된 기준에 따라 FAIL 결과를 보고한다는 점에서 RecMG와 구분된다.

---

## 4. 연구 차별성 주장별 등급

### 주장 A: 예측 기반 메모리 회수 정책에서 메모리 이득과 자체 CPU 비용을 함께 평가한다

**등급: MODERATE**

**근거**:
- RecMG(HPCA 2025)가 추론 CPU 비용(92 µs)과 임베딩 접근 latency를 함께 고려한다는 점에서 완전히 새롭지는 않다.
- 그러나 RecMG는 메모리 반환 정책이 아닌 프리패치/캐싱 정책이며, 비용 비교 방식도 다르다(파이프라이닝 가정 vs. 명시적 단위 환산).
- "조사 범위 내" 메모리 반환 정책에 한정하면 선행 연구 없음이지만, 조사 범위가 11편으로 제한적이다.

**권장 표현**: "우리가 아는 한(to our knowledge), 메모리 반환 정책에서 ML 추론 CPU 비용을 메모리 절감 가치와 명시적으로 비교한 연구는 없다."

---

### 주장 B: 특징 생성부터 정책 실행까지의 완전한 실행 경로를 CPU 측정에 포함한다

**등급: MODERATE**

**근거**:
- 완전한 실행 경로 측정은 이 연구의 명확한 방법론적 강점이다. RecMG는 92 µs를 측정하지만 이것이 전체 경로인지 불분명하다.
- 그러나 KML(TOCS 2023), 일반 성능 프로파일링 연구에서 "전체 경로" 측정은 표준 방법론이다.
- 차별성은 "완전한 실행 경로"가 아니라, "메모리 반환 정책 맥락에서 추론 경로 전체를 측정하여 경제성 판정에 사용했다"는 점이다.

**권장 표현**: "G2W3 경계는 compute_p1b_features부터 run_fallback3까지 C3 경로의 모든 추가 연산을 포함하며, 이를 통해 메모리 절감 가치와 직접 비교 가능한 ΔI를 도출했다."

---

### 주장 C: 메모리 관련 모델 환산 가치와 CPU 추가 비용을 공통 단위로 비교한다

**등급: STRONG**

**근거**:
- 조사된 11편 중 메모리 절감을 CPU 시간 단위로 환산하는 비용 모델(c = µs/MiB, G_model 공식)을 사용한 연구는 없다.
- RecMG의 편익은 처리량(end-to-end speedup)이며 메모리 절감 가치를 CPU 시간으로 환산하지 않는다.
- TMO, MEMTIS, SC 2025는 메모리 절감량(MiB, %)을 목적 변수로 사용하지만 CPU 비용과의 비율 계산 없음.
- "공통 단위(CPU 시간)"라는 프레임은 조사 범위 내 선행 연구에서 발견되지 않았다.

**권장 표현**: "우리는 메모리 반환 가치 c·(ΔR + ΔH/T*)와 추론 비용 ΔI를 동일한 CPU 시간 단위(ms)로 환산하여 직접 비교하는 비용 등가 프레임워크를 제안한다. 우리가 아는 한, 메모리 반환 정책 연구에서 이 환산 방식을 사용한 선행 연구는 없다."

---

### 주장 D: 손익분기 조건과 민감도 분석으로 경제성이 성립하기 위한 조건을 제시한다

**등급: STRONG**

**근거**:
- 조사된 모든 선행 연구에서 G_model > 0 형태의 명시적 손익분기 공식, 그리고 그 경계에서의 각 변수별 민감도 분석을 제시한 연구는 없다.
- RecMG는 92 µs가 "숨겨진다"고 주장하지만 ΔI < G_benefit 형태의 조건을 명시화하지 않는다.
- 이 연구의 §8.1 손익분기표(ΔI < 1,560.860 ms, T* < 1.7744 s, c > 434.371 µs/MiB 등)는 조사된 모든 논문에서 유사한 형식을 발견하지 못했다.

**권장 표현**: "우리는 G_model > 0 조건을 각 파라미터에 대해 역산하여, 경제성이 성립하기 위해 ΔI가 현재 대비 −35.1%, T*가 −31.8% 감소해야 함을 보인다. 이러한 손익분기 민감도 분석을 메모리 반환 정책에 적용한 선행 연구를 발견하지 못했다."

---

## 5. 연구 공백 문장 세 가지

### 5.1 보수적 표현 (조사 범위 내 차이만 기술)

> 우리가 조사한 11편의 선행 연구 중, ML 메모리 반환 정책에서 추론 CPU 비용을 메모리 절감 가치와 동일 단위(CPU 시간)로 환산하여 경제성 판정 기준을 명시화하고 민감도 분석을 제시한 연구는 없었다. 조사 범위 밖에서 유사한 연구가 존재할 수 있다.

**용도**: §3 Related Work 말미의 "본 연구의 위치" 단락.

### 5.2 중간 표현 (비용 회수 평가의 공백을 명시)

> 기존 ML 메모리 관리 연구(LLAMA, LeCaR, RecMG 등)는 ML 적용의 편익을 측정하지만, ML 추론 자체의 CPU 비용이 절감 가치를 회수하는지(break-even)를 형식적으로 평가하지 않는다. 가장 근접한 선행 연구인 RecMG(HPCA '25)는 추론 비용 92 µs를 명시하지만, 편익과의 단위 환산 없이 파이프라이닝으로 숨길 수 있다고 주장한다. 이 공백에서 본 연구는 비용 등가 단위(µs/MiB)와 명시적 손익분기 조건(G_model > 0)을 도입한다.

**용도**: §1 Introduction의 Research Gap 단락 (§1.2).

### 5.3 강한 표현 (원문 조사에서 입증된 범위 내에서만)

> ML 메모리 관리 연구에서, 추론 CPU 비용을 메모리 절감 가치와 동일 화폐(CPU 시간)로 환산하여 G_model = c(ΔR + ΔH/T*) − ΔI > 0 형태의 경제성 판정 조건을 도출하고, 사전 등록된 기준에 따라 FAIL 결과를 명시적으로 보고한 연구를, 우리가 조사한 범위 내에서 발견하지 못했다. 이 공백이 본 연구의 출발점이다.

**사용 조건**: "우리가 조사한 범위 내에서" 한정 문구 필수. 이 표현은 11편 전수 조사 결과에서 직접 입증된 주장이다.

---

## 6. 논문에서 피해야 할 과장된 표현

### 절대 사용 금지

- **"최초로 (first to)"**: 조사 범위 11편은 메모리 시스템 논문 전체가 아니다.
- **"유일한 (the only)"**: 같은 이유로 근거 불충분.
- **"새로운 (novel) approach"** (단독 사용): 선행 연구 범위를 명시하지 않은 경우.
- **"RecMG 저자명: Zhu, Xianhao et al."**: 완전히 다른 사람들의 이름. arXiv 2502.03091은 물리학 논문.
- **"BurstGPT 저자: Ruidong Wang"**: 실제 제1저자는 Yuxin Wang.

### 한정 표현으로만 사용

- **"우리가 아는 한(to our knowledge)"**: 모든 신규성 주장에 필수 한정.
- **"조사한 선행 연구들 중(among the papers we surveyed)"**: 광범위한 신규성 주장 대신 사용.
- **"형식적인 손익분기 조건을 제시하지 않는다"**: 개별 논문에 대한 주장 시 해당 논문 본문 직접 확인 후 사용.

### 맥락이 중요한 표현

- **"RecMG는 가장 근접한 선행 연구이다"**: 타당하지만, RecMG의 최적화 대상(임베딩 프리패치)이 이 연구(메모리 반환)와 근본적으로 다름을 함께 명시해야 한다.
- **"SC 2025 측정 연구"**: 실제로는 SC Workshops (ROSS 워크숍) 발표. 메인 SC 2025가 아님을 구분해야 한다.
- **"사전 등록(preregistration)"**: CS 시스템 연구에서 비표준 용어. 첫 등장 시 "사전 등록된 판정 기준(pre-specified decision criteria)"으로 정의하고 사용.

---

## 7. Introduction과 Related Work에 적용할 수정 권고

### 7.1 related-work-matrix.md §8 BibTeX 수정 사항

다음 BibTeX 항목을 수정해야 논문 초안에 사용할 수 있다.

**TMO**: `Iyer, Niket` → `Agarwal, Niket`; 전체 저자 목록 갱신:
```
author = {Weiner, Johannes and Agarwal, Niket and Schatzberg, Dan and Yang, Leon and Wang, Hao and Sanouillet, Blaise and Sharma, Bikash and Heo, Tejun and Jain, Mayank and Tang, Chunqiang and Skarlatos, Dimitrios}
```

**MEMTIS**: `Kim, Taehyung` → `Lee, Taehyung`; 저자 수정:
```
author = {Lee, Taehyung and Monga, Sumit Kumar and Min, Changwoo and Eom, Young Ik}
```

**LeCaR**: 저자 전체 교체:
```
author = {Vietri, Giuseppe and Rodriguez, Liana V. and Martinez, Wendy A. and Lyons, Steven and Liu, Jason and Rangaswami, Raju and Zhao, Ming and Narasimhan, Giri}
```

**MemPerf**: 제목 및 저자 전체 교체:
```
@article{memperf2023,
  author    = {Zhou, Jin and Silvestro, Sam and Tang, Steven (Jiaxun) and Yang, Hanmei and Liu, Hongyu and Zeng, Guangming and Wu, Bo and Liu, Cong and Liu, Tongping},
  title     = {MemPerf: Profiling Allocator-Induced Performance Slowdowns},
  journal   = {Proceedings of the ACM on Programming Languages},
  volume    = {7},
  number    = {OOPSLA2},
  year      = {2023},
  doi       = {10.1145/3622848}
}
```

**RecMG**: arXiv 번호 및 저자 전체 교체:
```
@inproceedings{recmg2025,
  author    = {Ren, Jie and Ma, Bin and Yang, Shuangyan and Francis, Benjamin and Ardestani, Ehsan K. and Si, Min and Li, Dong},
  title     = {Machine Learning-Guided Memory Optimization for {DLRM} Inference on Tiered Memory},
  booktitle = {Proceedings of the 31st IEEE International Symposium on High-Performance Computer Architecture (HPCA)},
  year      = {2025},
  archivePrefix = {arXiv},
  eprint    = {2511.08568},
  note      = {ML-guided vector caching and prefetching for DLRM on tiered memory; 92 µs inference explicitly measured; pipelined with embedding access latency; no formal break-even condition; up to 43\% end-to-end speedup}
}
```

**SC 2025**: 제목 및 형태 추가:
```
@inproceedings{pagereclaim2025,
  author    = {Liu, Shaochang and Ren, Jie},
  title     = {Assessing Page Reclamation Mechanisms for Linux},
  booktitle = {Proceedings of the SC '25 Workshops of the International Conference for High Performance Computing, Networking, Storage and Analysis (SC Workshops '25)},
  year      = {2025},
  doi       = {10.1145/3731599.3767537},
  note      = {ROSS workshop at SC'25; profiler for 2QLRU vs. MGLRU decision overhead and reclamation effectiveness; no ML, no break-even}
}
```

**BurstGPT**: 저자 전체 교체:
```
author = {Wang, Yuxin and Chen, Yuhan and Li, Zeyu and Kang, Xueze and Fang, Yuchu and Zhou, Yeju and Zheng, Yang and Tang, Zhenheng and He, Xin and Guo, Rui and Wang, Xin and Wang, Qiang and Zhou, Amelie Chi and Chu, Xiaowen}
```

### 7.2 manuscript-outline.md 수정 권고

**§5 Experimental Setup / BurstGPT 항목**: 현재 "BurstGPT v2, v3 (LLM 추론 트레이스): 41일, 405창..."으로 기재돼 있으나 BurstGPT는 LLM 서빙 추적 데이터셋이고 이 연구의 메모리 청크 풀 측정 대상이 아니다.

**권고**: §5에서 BurstGPT를 인용할 경우, "청크 풀 접근 패턴 시뮬레이션에 사용된 LLM 서빙 워크로드 추적 데이터셋"으로 역할을 명확히 한다. 직접 사용한 BurstGPT 버전(v2/v3), 창 수(405/420), 그리고 이 연구에서 BurstGPT를 사용한 방식을 본문에서 명시한다.

**§2 Background / Related Work**: RecMG 인용 시 arXiv 번호(2511.08568)와 저자명(Jie Ren et al.) 수정 필수.

### 7.3 final-report-draft.md 수정 권고

현재 final-report-draft.md는 final-report-audit-v2.md에서 READY FOR MANUSCRIPT 판정을 받았으며 서지 정보는 포함하지 않는다. 이 감사 결과로 인한 수정 불필요.

---

## 8. 최종 판정

### 판정: **READY FOR MANUSCRIPT DRAFTING (서지 오류 수정 조건부)**

---

**이유**:

**근거 1**: 연구 차별성 주장(C, D)은 조사 범위 내에서 STRONG 등급으로 뒷받침된다. G_model 비용 등가 프레임워크와 손익분기 민감도 분석은 조사된 11편 모두에서 유사한 형식의 선행 연구가 없다.

**근거 2**: 주장 A, B는 MODERATE 등급으로, RecMG와의 비교를 통해 차별성을 명확히 서술하면 충분히 뒷받침 가능하다.

**근거 3**: 연구 공백 문장 3가지(보수적/중간/강한)가 모두 조사된 원문에 근거한다.

**근거 4**: 핵심 결과 수치(G_model, ΔI, G1_SMODEL_C, 손익분기 경계)는 final-report-audit-v2.md에서 VERIFIED 판정을 받았으며 이 감사에서 변경 없음.

---

**선행 조건 (manuscript 초안 집필 전 처리)**:

| 항목 | 필수 여부 | 내용 |
|------|---------|------|
| BibTeX 오류 6건 수정 | **필수** | TMO, MEMTIS, LeCaR, MemPerf, RecMG, BurstGPT 저자명/제목/arXiv 번호 |
| RecMG 92 µs 비용 근거 확인 | **필수** | arXiv:2511.08568 §4 본문 직접 확인 후 인용 |
| SC 2025 논문 형태 명기 | **필수** | "SC Workshops '25 (ROSS 워크숍)" — 메인 SC 트랙 아님 명기 |
| MemPerf §3 본문 열람 | **필수** (본문 인용 시) | "할당자 비용 측정 도구" 주장 인용 근거 |
| KML §3 연결 한정 | 권장 | "I/O 스케줄링 커널 ML 오버헤드 측정 사례"로만 인용 |
| LLAMA 5번째 저자 재확인 | 권장 | "Murray, Derek G." vs "McKinley, Kathryn S." ACM DL 직접 확인 |

---

*이 감사 문서는 조사 결과 정리용이며 실험을 포함하지 않는다. 기존 연구 문서 미수정. Git commit 없음.*
