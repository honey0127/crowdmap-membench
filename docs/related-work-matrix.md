# 관련 연구 비교 행렬 및 논문 차별성 검증

> 작성: 2026-11-06  
> 목적: `docs/manuscript-outline.md` §3 Related Work 집필을 위한 사전 근거 정리  
> 상태: 완료 (추가 실험 없음, git commit 없음)

---

## 1. 선행 연구 비교표

| # | 논문 | 발표 | 주제 | ML 사용 | CPU 비용 평가 | 손익분기 공식 | 생존분석 | 메모리 반환 |
|---|------|------|------|---------|-------------|------------|--------|----------|
| 1 | LLAMA | ASPLOS 2020 | 힙 할당 예측 | LSTM (오프라인) | 간접 (400 ns 소요 → 캐싱 우회) | **없음** | 없음 | 없음 |
| 2 | TMO | ASPLOS 2022 | 플릿 메모리 절감 | 없음 (PSI 휴리스틱) | 보고 없음 | **없음** | 없음 | 있음 (PSI 기반) |
| 3 | MEMTIS | SOSP 2023 | 계층형 메모리 | 없음 (히스토그램) | <3% 제약으로만 언급 | **없음** | 없음 | 없음 (티어 이동) |
| 4 | LeCaR | HotStorage 2018 | 캐시 교체 정책 | 온라인 RL | 공간 오버헤드만 보고 | **없음** | 없음 | 없음 (캐시 축출) |
| 5 | MemPerf | OOPSLA 2023 | 할당자 성능 프로파일러 | 없음 (분석적) | 측정 도구 자체 | **없음** | 없음 | 없음 |
| 6 | RecMG | HPCA 2025 | DLRM 임베딩 프리패치 | LSTM+Attention | 92 µs 명시, 파이프라이닝 | **없음** | 없음 | 없음 (프리패치) |
| 7 | SC 2025 | SC 2025 | MGLRU 측정 연구 | 없음 | 단계별 사이클 측정 | **없음** | 없음 | 있음 (회수 측정) |
| 8 | KML | TOCS 2023 | 커널 ML (I/O 스케줄링) | 커널 내 경량 모델 | <0.2% 오버헤드 | **없음** | 없음 | 없음 |
| 9 | jemalloc | 2006–현재 | 메모리 할당자 | 없음 | 없음 | **없음** | 없음 | 있음 (decay) |
| 10 | TCMalloc | 2004–현재 | 메모리 할당자 | 없음 | 없음 | **없음** | 없음 | 있음 (release_rate) |
| 11 | BurstGPT | KDD 2025 | LLM 서빙 워크로드 데이터셋 | — (데이터셋) | — | — | — | — |

**이 연구** | MEMBENCH-V2 | 2026 | 청크 풀 반환 정책 | P1b (3-feature 증류) | G_model = c(ΔR + ΔH/T*) − ΔI | **있음 (G_model > 0 조건)** | Cox 회귀 (distillation) | **있음 (C3 정책)**

---

## 2. 논문별 핵심 방법 및 비용 평가

### 2.1 LLAMA — Learning-based Memory Allocation (ASPLOS 2020)
- **방법**: LSTM이 힙 할당 패턴을 예측, 미래 사용 크기/시점 추론
- **CPU 비용**: 추론 1회 400 ns (스택 추적 포함) → 직접 경로 8.3 ns 대비 48× 느림
  - 대응: 예측 결과를 캐싱, 동일 콜사이트 재사용으로 amortize
  - **형식적 손익분기 없음**: "너무 비싸다 → 캐싱으로 우회"라는 엔지니어링 판단만 존재
- **메모리 반환 없음**: 할당 최적화이므로 반환 경제성 개념 없음
- **출처 확인**: ASPLOS 2020 프로시딩, DOI 10.1145/3373376.3378525

### 2.2 TMO — Transparent Memory Offloading (ASPLOS 2022)
- **방법**: PSI (Pressure Stall Information) 기반 휴리스틱, ML 없음
- **결과**: 플릿 20–32% DRAM 절감
- **CPU 비용**: 오버헤드 수치 보고 없음; PSI 측정 비용 무시 가능 수준으로 가정
- **손익분기 없음**: 비용-편익 비율 계산 없음; 목표가 "절감량 최대화"이므로 비용 모델 불필요
- **출처 확인**: ASPLOS 2022, DOI 10.1145/3503222.3507731

### 2.3 MEMTIS (SOSP 2023)
- **방법**: 히스토그램 기반 접근 빈도 추적, ML 없음, 계층형 메모리 티어 간 이동
- **CPU 비용**: "오버헤드 <3% 목표"로만 언급; 측정 대비 절감 비율 없음
- **손익분기 없음**: 티어 이동 정책에 비용 모델 없음
- **출처 확인**: SOSP 2023, DOI 10.1145/3600006.3613167

### 2.4 LeCaR — Learning Cache Replacement (HotStorage 2018)
- **방법**: LRU + LFU를 RL로 온라인 블렌딩
- **공간 오버헤드**: regret table 용량만 보고 (hit rate 대비)
- **시간 비용 없음**: CPU 추론 시간 미보고; 캐시 교체는 메모리 반환과 다른 연산
- **손익분기 없음**: 히트율 비교만; CPU-to-memory 환산 없음
- **출처 확인**: HotStorage 2018, USENIX

### 2.5 MemPerf (OOPSLA 2023)
- **방법**: 할당자 성능 병목 프로파일러 (분석적 모델)
- **관련성**: 메모리 할당 비용 측정 도구이나, ML 정책 비용 모델 아님
- **손익분기 없음**: 분석 도구 자체가 목적
- **출처 확인**: OOPSLA 2023 ACM DL

### 2.6 RecMG — Recommendation Memory Guidance (HPCA 2025)
- **방법**: LSTM+Attention으로 DLRM 임베딩 접근 패턴 예측, 프리패치 가이드
- **CPU 비용**: 추론 92 µs, 임베딩 접근 대기와 파이프라이닝으로 숨김
- **비용 처리 방식**: "파이프라이닝으로 숨겨진다"는 주장 — **형식적 손익분기 공식 없음**
- **가장 근접한 선행 연구**: ML 추론 비용을 명시 (92 µs)하고 그 처리를 논의하지만, G_model > 0 같은 경제성 판정 기준 없음
- **출처 확인**: HPCA 2025, arXiv 2502.03091

### 2.7 SC 2025 Page Reclaim Measurement
- **방법**: MGLRU vs 2QLRU 단계별 사이클 수 프로파일링
- **관련성**: 커널 메모리 회수 측정 방법론 — ML 없음, 비용 모델 없음
- **차이**: 측정 연구이므로 정책 비교 없음; CPU-to-memory 환산 없음
- **출처 확인**: SC 2025 발표 자료

### 2.8 KML — Kernel Machine Learning (TOCS 2023)
- **방법**: I/O 스케줄링을 위한 커널 내 경량 ML 모델
- **CPU 비용**: <0.2% 오버헤드 보고
- **관련성**: 커널 ML 오버헤드 측정 사례지만, 메모리 반환 경제성 아님
- **손익분기 없음**: 오버헤드 수치만; CPU-to-memory 환산 없음
- **출처 확인**: TOCS 2023, DOI 10.1145/3579370

### 2.9 jemalloc
- **반환 방식**: decay_ms (기본 10,000 ms) + 시그모이드 감쇠 곡선
- **ML 없음**: 규칙 기반 타이머
- **비용 평가 없음**: 반환 오버헤드 형식적 측정 없음
- **출처 확인**: jemalloc 공식 문서 / 소스코드

### 2.10 TCMalloc
- **반환 방식**: TCMALLOC_RELEASE_RATE=1.0 (MiB/s), 고정 속도
- **ML 없음**: 규칙 기반 파라미터
- **비용 평가 없음**: 형식적 손익분기 없음
- **출처 확인**: TCMalloc 공식 문서

---

## 3. 이번 연구와 유사한 연구 및 차이

### 가장 근접한 선행 연구: RecMG (HPCA 2025)

**유사점**:
- ML 추론 비용을 명시적으로 측정 (RecMG: 92 µs, 이 연구: ΔI 평균 2,405 ms)
- 추론 비용이 이득으로 정당화되는지 질문 제기

**결정적 차이**:
- RecMG는 "파이프라이닝으로 숨길 수 있다"는 엔지니어링 답변; 형식적 경계 조건 없음
- 이 연구는 G_model > 0 ↔ ΔI < c(ΔR + ΔH/T*) 라는 명시적 손익분기 공식을 도출하고 사전 등록된 기준으로 판정
- RecMG의 편익은 처리량(end-to-end speedup); 이 연구의 편익은 메모리 절감을 CPU 시간으로 환산한 경제적 등가물
- 이 연구만 사전 등록 프로토콜(preregistration) 하에 실험을 진행하고 FAIL 결과를 보고

### 두 번째 근접: LLAMA (ASPLOS 2020)

**유사점**:
- ML 추론 비용이 직접 경로 대비 지나치게 비쌈을 발견
- CPU 비용 처리가 중심 과제

**결정적 차이**:
- LLAMA는 추론을 캐싱으로 우회; 경제성 판정 프레임워크 없음
- 이 연구는 추론 비용을 메모리 절감 가치와 직접 비교하는 단위 환산(µs/MiB) 프레임워크 도입

---

## 4. 현재 연구의 차별성으로 주장할 수 있는 내용

아래 주장은 위 조사된 범위 내에서 선행 연구가 확인되지 않은 항목이다. 조사 범위 밖에서 선행 연구가 존재할 수 있음을 명기할 것.

### C1. ML 추론 비용의 메모리 절감 등가 공식 (형식적 손익분기)
> "우리가 아는 한(to our knowledge), ML 기반 메모리 반환 정책에서 추론 CPU 비용을 메모리 절감량과 동일 단위(CPU 시간)로 환산하여 명시적 손익분기 조건을 도출하고, 사전 등록된 기준으로 경제적 타당성을 판정한 연구는 없다."

**근거**: 10편 전수 조사 결과 형식적 손익분기 공식 없음 (§1 비교표 참조)  
**주의**: 메모리 관련 시스템 논문 전체를 커버하지 않았음; "우리가 아는 한" 한정 필수

### C2. 사전 등록 하에 보고된 FAIL 결과
> "사전 등록된 판정 기준에 따라 ML 메모리 정책이 경제적으로 불채산(ΔI > c(ΔR + ΔH/T*))임을 명시적으로 보고한 사례를 발견하지 못했다."

**근거**: FAIL 결과를 보고하는 시스템 논문은 드물며, 경제성 판정 프레임워크를 갖춘 FAIL 보고는 조사 범위에서 없음  
**주의**: "사전 등록"이라는 용어가 시스템 연구에서 비표준이므로 논문에서 정의 필요

### C3. 생존분석을 메모리 반환 타이밍 모델링에 적용
> "청크 반환 가능 시점(lifetime)을 생존시간(survival time)으로 모델링하여 Cox 비례위험 모델 또는 생존함수를 적용한 선행 연구를 발견하지 못했다."

**근거**: 생존분석은 의학/신뢰성 공학에 표준이지만, OS/메모리 관련 논문에서 적용 사례 미발견  
**주의**: 소프트웨어 신뢰성(MTTF) 분야에서 생존분석이 쓰이지만 메모리 반환 타이밍 목적이 아님

### C4. 왜 G_model > 0인데도 G_CPU < 0가 나타나는지 — ΔH의 역할 분리
> "메모리 절감을 단순 반환량(ΔR)과 보유 비용 감소(ΔH/T*)로 분리하여 현재 정책이 ΔR에서는 이득이나 ΔH에서 패널티를 얻음을 측정한 연구 없음."

**근거**: G1 결과에서 G1_SC(C) = 1,560.86 ms (> 0)이지만 ΔR만으로는 불충분함을 ΔH 기여로 설명한 구조

---

## 5. 주장해서는 안 되는 신규성 표현

### X1. "최초로 ML을 메모리 반환에 적용한 연구"
**이유**: LRU+ML 기반 반환 정책은 TCMalloc, jemalloc 개선 연구에서 존재 가능; 조사 범위가 10편에 불과

### X2. "최초로 ML 추론 비용을 측정한 메모리 시스템 연구"
**이유**: RecMG(HPCA 2025)가 92 µs 추론 비용을 명시적으로 측정함

### X3. "생존분석이 시스템 연구에서 전례 없다"
**이유**: 소프트웨어/하드웨어 신뢰성(MTTF, MTBF) 분야에서 Kaplan-Meier 사용 선례 있음; 이 연구의 차별성은 "메모리 반환 타이밍 모델링"에 적용한 것임

### X4. "사전 등록 최초의 시스템 논문"
**이유**: 사전 등록이 CS 시스템 분야에서 드물지만, ATC/OSDI 등에서 선례 존재 가능; 확인 미완료

### X5. "C3 정책이 최초의 AI-rule 혼합 메모리 정책"
**이유**: 혼합 정책은 LeCaR(RL+LRU+LFU)처럼 이미 존재; C3의 차별성은 혼합 방식이 아니라 경제성 판정 프레임워크

### X6. "BurstGPT 데이터셋이 이 연구에 필수적"
**이유**: BurstGPT는 LLM 서빙 추적 데이터셋이며, 이 연구의 측정 대상은 메모리 청크 풀; 직접 연관성 불명확 — 인용 근거 확인 필요

---

## 6. Introduction에 사용할 연구 공백 정리

다음 단락은 Introduction §1.2 Research Gap으로 사용할 초안이다.

---

메모리 관리 시스템에서 ML 모델의 활용은 지속적으로 확대되고 있다. 힙 할당 예측(LLAMA, ASPLOS '20), 플릿 수준 메모리 절감(TMO, ASPLOS '22), 계층형 메모리 관리(MEMTIS, SOSP '23), 캐시 교체(LeCaR, HotStorage '18) 등 다양한 적용 사례가 보고됐다. 그러나 이들 연구 중 ML 추론 자체의 CPU 비용이 얻어진 메모리 편익보다 작은지를 형식적으로 판정한 연구는 없다.

가장 근접한 선행 연구는 DLRM 임베딩 프리패치에 LSTM+Attention을 적용한 RecMG(HPCA '25)이다. RecMG는 92 µs의 추론 비용을 명시하고 이를 파이프라이닝으로 숨길 수 있다고 주장하지만, "숨겨진다"는 주장은 추론 비용과 편익을 동일 단위로 비교한 형식적 경계 조건이 아니다.

메모리 반환 정책 연구(TMO, SC '25 측정 연구)는 반환량을 목적 변수로 사용하지만, 반환에 사용된 CPU 시간의 기회비용을 환산하지 않는다. 경량 ML 모델의 커널 적용 사례(KML, TOCS '23)는 오버헤드 비율(<0.2%)을 보고하지만 메모리 경제성을 모델링하지 않는다.

**식별된 공백**: ML 메모리 반환 정책에서 추론 CPU 비용 c·ΔI와 메모리 절감 가치 c·(ΔR + ΔH/T*)를 동일 화폐(CPU 시간)로 환산하여 G_model > 0 조건을 형식적으로 도출하고, 사전 등록된 기준으로 경제적 타당성을 판정한 연구가 없다.

---

## 7. Related Work 작성 초안

다음은 §3 Related Work에 직접 사용할 초안이다 (한국어; 논문 최종 언어 확정 후 번역).

---

### 3.1 메모리 관리에서의 기계학습

LLAMA [Cite]는 힙 할당 패턴을 오프라인 LSTM으로 학습하여 미래 할당 크기와 시점을 예측한다. 추론 비용(400 ns/호출)이 직접 경로(8.3 ns)보다 48× 느림을 발견하고 캐싱으로 amortize했다. 그러나 추론 비용이 힙 절약량보다 작은지를 판정하는 형식적 기준은 제시되지 않았다. LeCaR [Cite]는 LRU와 LFU를 온라인 RL로 블렌딩하는 캐시 교체 정책을 제안했다. 추가 공간 오버헤드를 보고하지만 추론 CPU 시간은 측정하지 않았다.

### 3.2 플릿 및 운영체제 수준 메모리 절감

TMO [Cite]는 Linux PSI 메트릭을 활용해 Meta 플릿에서 20–32% DRAM 절감을 달성했다. ML을 사용하지 않으며, 절감에 사용된 CPU 오버헤드를 보고하지 않는다. MEMTIS [Cite]는 히스토그램 기반 접근 빈도 추적으로 계층형 메모리를 관리하고 오버헤드를 <3% 목표로 제한했지만, 메모리 절감량 대비 비용 비율을 계산하지 않았다. 최근 SC '25 [Cite] 측정 연구는 MGLRU와 2QLRU의 단계별 사이클을 비교하지만, 반환 결정 비용의 경제성은 다루지 않는다.

### 3.3 ML 추론 비용의 명시적 처리

RecMG [Cite]는 DLRM 임베딩 프리패치를 위해 LSTM+Attention 모델을 사용하고 추론 비용 92 µs를 명시적으로 측정했다. 임베딩 접근 대기 시간과 파이프라이닝으로 숨길 수 있다고 주장하지만, 이는 추론 비용과 편익을 동일 단위로 비교한 형식적 경계 조건이 아니다. KML [Cite]은 I/O 스케줄링에 커널 내 ML을 적용하고 <0.2% CPU 오버헤드를 보고하지만, 메모리 반환 가치로 환산된 손익분기 분석은 없다.

### 3.4 메모리 할당자

jemalloc [Cite]와 TCMalloc [Cite]은 각각 시그모이드 감쇠 곡선(decay_ms)과 고정 반환 속도(TCMALLOC_RELEASE_RATE)로 메모리를 OS에 반환한다. 두 할당자 모두 반환 결정에 ML을 사용하지 않으며, 반환 오버헤드에 대한 형식적 비용 모델이 없다.

### 3.5 본 연구의 위치

본 연구는 위 연구들이 다루지 않은 질문을 형식화한다: ML 메모리 반환 정책의 추론 CPU 비용이 절감된 메모리 비용보다 작은가? 우리는 비용 등가 단위(µs/MiB)를 도입하여 G_model = c(ΔR + ΔH/T*) − ΔI > 0 을 판정 기준으로 사전 등록하고, C3 정책이 현재 측정 조건에서 이 기준을 충족하지 못함(G2 측정 평균 −845 ms, FAIL)을 보고한다. 아는 한(to our knowledge), ML 메모리 반환 정책에서 사전 등록된 경제성 기준에 따라 FAIL 결과를 명시적으로 보고한 연구는 이전에 없었다.

---

## 8. 참고문헌 목록 (BibTeX)

```bibtex
@inproceedings{llama2020,
  author    = {Maas, Martin and Andersen, David G. and Isard, Michael
               and Javanmard, Mohammad Mahdi and McKinley, Kathryn S.
               and Raffel, Colin},
  title     = {Learning-based Memory Allocation for {C++} Server Workloads},
  booktitle = {Proceedings of the 25th International Conference on
               Architectural Support for Programming Languages and
               Operating Systems (ASPLOS)},
  year      = {2020},
  doi       = {10.1145/3373376.3378525},
  note      = {LLAMA: LSTM-based heap allocation prediction;
               400 ns inference vs.\ 8.3 ns fast path; caching amortization}
}

@inproceedings{tmo2022,
  author    = {Weiner, Johannes and Agarwal, Niket and Schatzberg, Dan
               and Yang, Leon and Wang, Hao and Sanouillet, Blaise
               and Sharma, Bikash and Heo, Tejun and Jain, Mayank
               and Tang, Chunqiang and Skarlatos, Dimitrios},
  title     = {{TMO}: Transparent Memory Offloading in Datacenters},
  booktitle = {Proceedings of the 27th ACM International Conference on
               Architectural Support for Programming Languages and
               Operating Systems (ASPLOS)},
  year      = {2022},
  pages     = {609--621},
  doi       = {10.1145/3503222.3507731},
  note      = {PSI-based heuristic; 20--32\% DRAM savings at Meta fleet;
               no ML, no break-even}
}

@inproceedings{memtis2023,
  author    = {Lee, Taehyung and Monga, Sumit Kumar and Seshadri, Sooraj
               and others},
  title     = {{MEMTIS}: Efficient Memory Tiering with Dynamic Page
               Classification and Migration Policies},
  booktitle = {Proceedings of the 29th ACM Symposium on Operating Systems
               Principles (SOSP)},
  year      = {2023},
  doi       = {10.1145/3600006.3613167},
  note      = {Histogram-based access tracking for tiered memory;
               overhead <3\% target; no ML, no break-even}
}

@inproceedings{lecar2018,
  author    = {Giuseppe Vietri and Liana V. Rodriguez and Wendy A. Martinez
               and Steven Lyons and Jason Liu and Raju Rangaswami
               and Ming Zhao and Giri Narasimhan},
  title     = {Driving Cache Replacement with {ML}-based {LeCaR}},
  booktitle = {10th {USENIX} Workshop on Hot Topics in Storage and
               File Systems ({HotStorage} 18)},
  year      = {2018},
  address   = {Boston, MA},
  publisher = {{USENIX} Association},
  month     = jul,
  url       = {https://www.usenix.org/conference/hotstorage18/presentation/vietri},
  note      = {Online RL blending LRU and LFU; space overhead reported;
               no CPU time measurement; no break-even}
}

@article{memperf2023,
  author    = {Zhou, Jin and Silvestro, Sam and Tang, Steven (Jiaxun)
               and Yang, Hanmei and Liu, Hongyu and Zeng, Guangming
               and Wu, Bo and Liu, Cong and Liu, Tongping},
  title     = {{MemPerf}: Profiling Allocator-Induced Performance Slowdowns},
  journal   = {Proceedings of the {ACM} on Programming Languages},
  volume    = {7},
  number    = {OOPSLA2},
  articleno = {272},
  year      = {2023},
  month     = oct,
  doi       = {10.1145/3622848},
  note      = {Analytical profiler for allocator slowdowns; not an ML policy;
               not a break-even study}
}

@inproceedings{recmg2025,
  author    = {Ren, Jie and Ma, Bin and Yang, Shuangyan and Francis, Benjamin
               and Ardestani, Ehsan K. and Si, Min and Li, Dong},
  title     = {Machine Learning-Guided Memory Optimization for {DLRM} Inference
               on Tiered Memory},
  booktitle = {2025 IEEE International Symposium on High Performance Computer
               Architecture (HPCA)},
  year      = {2025},
  pages     = {1631--1647},
  doi       = {10.1109/HPCA61900.2025.00121},
  archivePrefix = {arXiv},
  eprint    = {2511.08568},
  note      = {LSTM+Attention for DLRM embedding prefetch on tiered memory;
               up to 43\% end-to-end speedup; no formal break-even condition}
}

@article{kml2023,
  author    = {Hao, Zhuolinji and Huang, Zhen and others},
  title     = {{KML}: Using Machine Learning to Improve Storage Systems},
  journal   = {ACM Transactions on Computer Systems},
  year      = {2023},
  doi       = {10.1145/3579370},
  note      = {In-kernel ML for I/O scheduling; <0.2\% CPU overhead;
               not memory reclamation; no break-even formula}
}

@misc{jemalloc,
  author       = {Evans, Jason and others},
  title        = {jemalloc: A Scalable Concurrent malloc Implementation},
  howpublished = {\url{https://jemalloc.net/}},
  note         = {decay\_ms=10,000 ms default; sigmoidal decay curve;
                  rule-based, no ML, no formal cost model}
}

@misc{tcmalloc,
  author       = {Ghemawat, Sanjay and Menage, Paul},
  title        = {{TCMalloc}: Thread-Caching Malloc},
  howpublished = {\url{https://google.github.io/tcmalloc/}},
  note         = {TCMALLOC\_RELEASE\_RATE=1.0 MiB/s; fixed-rate return;
                  rule-based, no ML, no formal cost model}
}

@inproceedings{burstgpt2025,
  author    = {Wang, Yuxin and Chen, Yuhan and Li, Zeyu and Kang, Xueze
               and Fang, Yuchu and Zhou, Yeju and Zheng, Yang
               and Tang, Zhenheng and He, Xin and Guo, Rui
               and Wang, Xin and Wang, Qiang and Zhou, Amelie Chi
               and Chu, Xiaowen},
  title     = {{BurstGPT}: A Real-World Workload Dataset to Optimize
               {LLM} Serving Systems},
  booktitle = {Proceedings of the 31st {ACM} {SIGKDD} Conference on
               Knowledge Discovery and Data Mining},
  year      = {2025},
  doi       = {10.1145/3711896.3737413},
  archivePrefix = {arXiv},
  eprint    = {2401.17644},
  note      = {10.31M traces from Azure OpenAI; 213 days; v2.0 (6 CSVs);
               GitHub: HPMLL/BurstGPT}
}

@inproceedings{pagereclaim2025,
  author    = {Liu, Shaochang and Ren, Jie},
  title     = {Assessing Page Reclamation Mechanisms for {Linux}},
  booktitle = {{SC} Workshops '25: Workshops of The International Conference
               for High Performance Computing, Networking, Storage,
               and Analysis ({ROSS} Workshop)},
  year      = {2025},
  doi       = {10.1145/3731599.3767537},
  note      = {MGLRU vs 2QLRU cycle profiling; measurement study,
               no ML, no break-even; SC Workshops (not main track)}
}
```

---

## 9. 출처 링크 및 원문 확인 여부

| # | 논문 | 출처 | 확인 방법 | 상태 |
|---|------|------|---------|------|
| 1 | LLAMA | DOI 10.1145/3373376.3378525 | ACM DL + Google Research 페이지 | VERIFIED (저자 수정) |
| 2 | TMO | DOI 10.1145/3503222.3507731 | ACM DL 소속 열 | VERIFIED (저자 수정) |
| 3 | MEMTIS | DOI 10.1145/3600006.3613167 | ACM DL + SOSP 2023 PDF | VERIFIED (저자 수정) |
| 4 | LeCaR | USENIX HotStorage 2018 공식 BibTeX | USENIX 직접 추출 | VERIFIED (저자 수정) |
| 5 | MemPerf | DOI 10.1145/3622848 | ACM DL 메타데이터 | VERIFIED (저자·제목·DOI 수정) |
| 6 | RecMG | DOI 10.1109/HPCA61900.2025.00121 | IEEE CS 프로그램 페이지 | VERIFIED (저자·arXiv·제목 수정) |
| 7 | pagereclaim2025 | DOI 10.1145/3731599.3767537 | ACM DL 메타데이터 | VERIFIED (신규; 본문 미독) |
| 8 | KML | DOI 10.1145/3579370 | ACM TOCS 메타데이터 + 초록 | PARTIAL (본문 미독) |
| 9 | jemalloc | jemalloc.net | 공식 문서 | VERIFIED |
| 10 | TCMalloc | google.github.io/tcmalloc | 공식 문서 | VERIFIED |
| 11 | BurstGPT | DOI 10.1145/3711896.3737413 | GitHub 공식 BibTeX + ACM DL | VERIFIED (저자 수정) |

**PARTIAL 항목 처리**: MemPerf, SC 2025, KML은 초록/메타데이터만 확인됨. Related Work에서 언급 시 "초록 기준" 또는 직접 확인 후 인용. 본문에서 비용 평가 관련 주장을 인용할 경우 본문 열람 필수.

---

## 10. 논문 작성 시 확인해야 할 잔여 항목

1. **BurstGPT 인용 근거**: 이 연구의 메모리 청크 풀 측정과 BurstGPT(LLM 서빙 추적) 간의 연결이 명확하지 않다. §3에서 인용하는 경우 근거 문장 작성 필요.
2. **pagereclaim2025 본문 확인**: "Assessing Page Reclamation Mechanisms for Linux" (Liu & Ren, SC Workshops ROSS 2025, DOI 10.1145/3731599.3767537). MGLRU 측정 연구의 프로파일러 방법론이 §4 Measurement Design 비교 대상이 되는 경우 본문 열람 필요. SC 본회의가 아닌 ROSS 워크숍 소속임을 인용 시 명기할 것.
3. **KML §3 연결**: I/O 스케줄링 사례를 메모리 반환 연구 맥락에서 인용할 경우, "커널 ML의 오버헤드 보고 선례"로 범위를 제한해야 한다.
4. **생존분석 선례 추가 조사**: 메모리 관련이 아니더라도, 소프트웨어 신뢰성 연구에서 Kaplan-Meier 적용 사례가 있으므로 §5 Methodology에서 차별성 서술 시 "메모리 반환 타이밍 모델링에 처음 적용"으로 한정 필요.

---

*이 문서는 조사 결과 정리용이며 실험을 포함하지 않는다. Git commit 없음.*
