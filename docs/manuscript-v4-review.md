# Manuscript v4 최종 검토 보고서

> 작성: 2026-11-06  
> 대상: `docs/manuscript-v4.md`  
> 참조: `docs/manuscript-v3-review.md`, `docs/final-report-audit-v2.md`,
>       `docs/related-work-audit.md`, `docs/bibliography-verification.md`,
>       `exp/v2dev/g1/simulation/summary.json`,
>       `exp/v2dev/g2/measurement/measure-summary-official.json`  
> 상태: 검토 완료.

---

## 1. 검토 요약

| 범주 | 건수 | 판정 |
|------|------|------|
| 수치 일관성 (공식 결과 ↔ 원본 파일) | 전항목 | ✓ PASS |
| 수식·부호 규약 일관성 | 전항목 | ✓ PASS |
| v2 PASS / G2 FAIL 구분 | 전항목 | ✓ PASS |
| Abstract ↔ 본문 결론 일치 | 전항목 | ✓ PASS |
| 11편 citation key 존재 여부 | 전항목 | ✓ PASS |
| 선행 연구 차별성 과장 여부 | 전항목 | ✓ PASS |
| 미수행 실험 허위 주장 | 없음 | ✓ PASS |
| **복합 손익분기 예시 수치 오류 (§6.9, §7.4)** | **1건** | **✗ 수정 필요** |
| 그림 본문 참조 미작성 (Fig. 1–3) | 3건 | △ 미완성 (제출 전 필수) |
| Azure 2024 트레이스 미인용 (§5.1, §6.1) | 1건 | △ 경미 |
| BibTeX 오류 (related-work-matrix.md) | 외부 파일 | △ 제출 전 필요 |

**최종 판정**: **NEEDS REVISION** (1건 수치 수정 + 그림 참조 추가 후 READY FOR VENUE FORMATTING)

---

## 2. 절별 검토 결과

### Abstract

| 항목 | Abstract 기재 | 본문 (§9.3–§9.5) | 판정 |
|------|-------------|-----------------|------|
| H1v2 | −2.34 %p | −2.34 %p | ✓ |
| H2v2 | −9.02 %p | −9.02 %p | ✓ |
| G1\_SMODEL\_C | 1,560.860 ms | 1,560.860 ms | ✓ |
| mean ΔI | 2,405.932 ms | 2,405.932 ms | ✓ |
| mean G\_model | −845.072 ms | −845.072 ms | ✓ |
| G\_model > 0 판정 | 3회 모두 불충족 | 3회 모두 불충족 | ✓ |
| 단일변수 손익분기 (ΔI) | 35.1% 감소 필요 | 35.1% 감소 필요 | ✓ |
| 단일변수 손익분기 (T*) | 31.8% 감소 필요 | 31.8% 감소 필요 | ✓ |

**판정**: ✓ PASS — Abstract와 본문 결론 완전 일치.

---

### §1. Introduction

- 연구 질문 ("Can predictive memory reclamation pay for its own CPU cost?") § 9.1에서 정확히 반복됨. ✓
- 6가지 기여 목록이 모두 본문에서 뒷받침됨 (§4.5 G2W3, §4.6 parity, §6.8 G2 판정, §6.9 감도 분석). ✓
- LLAMA 400 ns / 48× 수치: 본문 §2.2에서 동일 반복. ✓
- RecMG 92 µs / 43% speedup: 본문 §2.4에서 동일 반복. ✓

**판정**: ✓ PASS

---

### §2. Background and Related Work

- 11편 선행 연구 모두 올바른 citation key로 인용. ✓
- Table 1의 RecMG 항목 "92 µs/call" 및 본문 §2.4의 "up to 43% end-to-end speedup" — `bibliography-verification.md`에서 확인된 수정 후 값(43%)이 반영되어 있음. ✓
- pagereclaim2025 항목: "SC Workshops (not main track)" 주의사항이 bibliography-verification.md에 기록됨. 본문 §2.3에서는 "A recent measurement study"로만 언급하여 주석을 붙이지 않음. 제출 venue에 따라 "SC Workshops '25" 명시를 추가할 수 있으나 현재 형태도 허용 가능. ✓
- §2.6 Table 1 memperf2023 항목: `\textless3\%` → MEMTIS 항목을 가리키나 열 순서에서 memperf2023 행에 위치. 확인하면 Table 1에서 memperf2023 행의 "CPU overhead" 셀이 비어있음 — 이것이 맞음(MemPerf는 직접 ML 비용을 측정하지 않음). `\textless3\%`은 MEMTIS 행에 있음. ✓

**판정**: ✓ PASS

---

### §3. Problem Formulation

- 정의 일관성: ΔR = R\_rule − R\_C3 (음수 = C3가 더 많이 반환), ΔH = H\_rule − H\_C3 (양수 = C3가 유휴 보유 적음), ΔI = CPU\_C3 − CPU\_rule (양수). ✓
- G\_model 수식: §3.2 정의 → §6.8 적용 → §7.2 분해 → §9.4 요약. 전 절에 걸쳐 일치. ✓
- G\_CPU "분해 보조만" 역할 명시: §3.2, §9.4. ✓
- §3.3 Table 2의 ΔH = 16,756.72 MiB·s는 g1/simulation/summary.json의 정밀 값 16,756.7231181967 MiB·s의 소수 2자리 반올림. 계산값(1,816.171 ms)은 정밀 값 기반 — 반올림 불일치가 아니라 표기 단순화. ✓

**판정**: ✓ PASS

---

### §4. Methodology

- §4.3 C3 경로 5개 증분 연산 나열: compute\_p1b\_features, compute\_Tk, online\_kappa\_T, run\_fallback3(AI fallback), run\_fallback3(C3 selection). §7.3, L11과 일치. ✓
- §4.5 G2W3 측정 경계: Table 3이 §4.3 C3 경로 설명과 일치. ✓
- §4.6 parity 검증: 420/420 PASS, 7,671,239 슬라이스 비교, T\_TOL = 0.0. `parity-check-log.txt` 출력과 일치. ✓
- §4.4 P1b vs P1 선택 근거: P1b = 1,560.860 ms > P1 = 1,483.856 ms. `g1/simulation/summary.json` 확인. ✓

**판정**: ✓ PASS

---

### §5. Experimental Setup

- §5.1 개발 40일 420창 (G1/G2) vs v2 봉인 41일 405창: 명확히 구분. ✓
- §5.1 "14 from an Azure 2024 trace dataset" — 이 데이터셋에 대한 citation이 없음. v1 sealed 범위 한정 설명이므로 중요도는 낮으나, 출처 불명 데이터 사용으로 보일 수 있음. △ 경미
- §5.2 의사결정 수 2,003,785회: `measure-summary-official.json`의 `n_decisions_rule`/`n_decisions_c3` 값(2,003,785)과 일치. ✓
- §5.3 Git commit 5c07328c…: 측정 시 커밋과 일치. ✓
- §5.4 체크섬 결정론성: Table 11의 `b2bb4067b5354578`/`ba240bd4f8369334`와 일치. ✓

**판정**: ✓ PASS (Azure 인용 누락 경미)

---

### §6. Results

#### §6.1–§6.3 v1, C3 선택, T*/c 보정

- v1 판정 수치 (H1: 37/12/4, p=1.0×10⁻⁷; H2: 9/14/30, p=0.0011): `sealed-results-v1.md` 참조. ✓
- c = 25.6 + 256.2 = 281.8 µs/MiB, T* = 2.6 s: `docs/additional-analysis-plan-v4.md` §1 명세와 일치. ✓

#### §6.4 v2 봉인 실험

- H1v2 −2.34 %p, H2v2 −9.02 %p, H3v2 (27/9/4, p=3.4×10⁻⁵): `sealed-results-v2.md` 참조. ✓
- 절대 비용 C3=767,024.9 MiB, rule=786,730.4 MiB: `sealed-results-v2.md` 참조. ✓

#### §6.5 Task A

- 범위 주석 "Python 3.12.3 / scikit-learn 1.9.1 단일 스레드 구현에 한정": 명확. ✓
- G2와의 혼동 방지를 위한 "별도 보고" 문장 존재. ✓

#### §6.6 Task B

- 사후 탐색적 분석 명시. ✓
- P\_shuffle 성분 99.4–99.8%: `b-placebo-results.md` 참조. ✓

#### §6.7 G1

- G1\_SMODEL\_C = 1,560.860 ms, ΔR = −906.0 MiB, ΔH = 16,756.72 MiB·s:
  `g1/simulation/summary.json` S\_total\_times\_c\_s = 1.5608601902... s = 1,560.860 ms ✓
- §6.7 단위 주석: `281.8 × 5,538.9 MiB` (MiB 단위 명시, v3에서 누락됐던 부분, v4에서 수정). ✓

#### §6.8 G2W3 공식 측정

| 항목 | 원고 | 원본 (`measure-summary-official.json`) | 판정 |
|------|------|---------------------------------------|------|
| Iter 1 ΔI | 2,329.017 ms | 2.329017150 s | ✓ |
| Iter 2 ΔI | 2,379.671 ms | 2.379671421 s | ✓ |
| Iter 3 ΔI | 2,509.109 ms | 2.509108815 s | ✓ |
| Mean ΔI | 2,405.932 ms | 2.405932462 s (3자리 반올림) | ✓ |
| Iter 1 G\_model | −768.157 ms | −0.7681569597 s | ✓ |
| Iter 2 G\_model | −818.811 ms | −0.8188112307 s | ✓ |
| Iter 3 G\_model | −948.249 ms | −0.9482486247 s | ✓ |
| Mean G\_model | −845.072 ms | 계산: (−768.157+−818.811+−948.249)/3 = −845.072 ms | ✓ |
| Mean G\_CPU | −2,661.243 ms | 계산: (−2584.328+−2634.982+−2764.420)/3 = −2,661.243 ms | ✓ |
| σ (pop.) | 75.831 ms | √(Σ(ΔI−mean)²/3) = √5750.22 = 75.831 ms | ✓ |
| ΔI 범위 | 180.092 ms | 2509.109−2329.017 = 180.092 ms | ✓ |
| checksum\_rule | b2bb4067b5354578 | raw.json 3회 모두 동일 | ✓ |
| P1b model SHA | 35338f0f… | sha256sum p1b\_model.bin = 35338f0fba9bfe27… | ✓ |
| Harness SHA | a8eec698… | sha256sum g2\_harness = a8eec698… | ✓ |

**판정**: ✓ PASS — 공식 측정 수치 전항목 원본 일치.

#### §6.9 손익분기 감도 분석

단일변수 경계값 검증:

| 변수 | 원고 경계 | 계산값 | 판정 |
|------|---------|-------|------|
| ΔI | < 1,560.860 ms | G\_model=0 경계 = G1\_SMODEL\_C = 1,560.860 ms | ✓ |
| T* | < 1.774 s | ΔH / (ΔI/c − ΔR) = 1.7744 s → 소수 3자리 1.774 s | ✓ |
| c | > 434.371 µs/MiB | ΔI/S\_units = 2,405,932/5538.894 = 434.371 µs/MiB | ✓ |
| ΔH | > 24,553.699 MiB·s | (ΔI/c − ΔR) × T* = 9443.730 × 2.6 = 24,553.699 MiB·s | ✓ |
| ΔR | > +2,092.837 MiB | ΔI/c − ΔH/T* = 8537.730 − 6444.893 = 2092.837 MiB | ✓ |
| −35.1% | (ΔI − G1\_SC)/ΔI = 845.072/2405.932 = 35.12% | ✓ |
| −31.8% | (2.6 − 1.774)/2.6 = 31.77% | ✓ |
| +54.1% | (434.371 − 281.8)/281.8 = 54.14% | ✓ |
| +46.5% | (24553.699 − 16756.72)/16756.72 = 46.53% | ✓ |

**⚠ 복합 조건 예시 — 수치 오류 발견**

원고 §6.9 마지막 단락 및 §7.4 마지막 단락:

> "a 20\% reduction in $\Delta I$ combined with a 15\% reduction in $T^*$ would bring
> $G_\mathrm{model}$ to approximately zero"

**검산 결과:**

```
ΔI_new = 2405.932 × 0.80 = 1924.746 ms
T*_new = 2.6 × 0.85    = 2.21 s
G1_new = 281.8 × (−906 + 16756.72/2.21) / 1000 = 1881.361 ms
G_model_new = 1881.361 − 1924.746 = −43.385 ms
```

**−43.385 ms는 "approximately zero"가 아니다.** 초기 결손(−845 ms)의 95%를 줄였지만
두 경로 모두 경계에 도달하지 못한 상태다.

정확한 복합 조건:
- 20% ΔI 감소 시 → T* 감소 필요량 = **16.7%** (not 15%)  
  `T*_exact = ΔH / (ΔI_new/c − ΔR) = 16756.72 / 9443.73 = 2.166 s`
- 15% T* 감소 시 → ΔI 감소 필요량 = **21.8%** (not 20%)  
  `ΔI_max = G1_new(T*=2.21) = 1881.361 ms → 감소율 = 21.8%`

---

### §7. Discussion

- §7.1: v2 PASS / G2 FAIL 대립 구조를 "policy quality is positive but insufficient to cover execution cost"로 명확히 정리. ✓
- §7.2 Table 13 분해:

| 항 | 계산식 | 원고값 | 검산값 | 판정 |
|----|--------|-------|-------|------|
| c·ΔR | 281.8 × (−906.0) µs | −255.311 ms | 281.8×906 = 255,310.8 µs = −255.311 ms | ✓ |
| c·ΔH/T* | 281.8 × 16,756.7231/2.6 µs | +1,816.171 ms | 281.8×6444.893 = 1,816,170.8 µs = +1,816.171 ms | ✓ |
| G1\_SMODEL\_C | −255.311 + 1,816.171 | +1,560.860 ms | 1,560.860 ms | ✓ |
| G\_model | 1,560.860 − 2,405.932 | −845.072 ms | −845.072 ms | ✓ |

- §7.3: 증분 연산 5가지 목록이 §4.3 Table 3과 일치. 개별 기여도 "현재 측정에서 알 수 없음" 명시. ✓
- §7.4: 단일변수 조건표 §6.9 Table 12와 일치. **단, 마지막 단락 복합 조건 예시가 §6.9와 동일한 오류를 포함함.** ← 수정 대상
- §7.5: "not support the conclusion that ML-based memory reclamation is generally unviable" 범위 제한 표현 존재. ✓

---

### §8. Limitations

12가지 한계(L1–L12) 모두 실제 미수행 사항을 올바르게 기술:

| 항목 | 내용 요약 | 판정 |
|------|---------|------|
| L1 | BurstGPT는 요청 수준 트레이스, 할당기 이벤트 없음 | ✓ |
| L2 | 유체 모델은 실제 메모리 연산 없음 | ✓ |
| L3 | 실제 jemalloc/TCMalloc 통합 실험 없음 | ✓ |
| L4 | 실제 OS 페이지 회수 측정 없음 | ✓ |
| L5 | 실제 RSS/클라우드 비용 감소 미측정 | ✓ |
| L6 | 단일 하드웨어 플랫폼 | ✓ |
| L7 | 단일 코어 고정 | ✓ |
| L8 | 3회 반복 | ✓ |
| L9 | n=3 표준편차는 기술 통계 | ✓ |
| L10 | P1b 기반 C3 정책에 한정 | ✓ |
| L11 | 세부 함수 CPU 분해 없음 | ✓ |
| L12 | 타 환경 일반화 없음 | ✓ |

**판정**: ✓ PASS — 미수행 실험 허위 주장 없음.

---

### §9. Conclusion

- §9.3–§9.4: v2 PASS 수치와 G2 FAIL 수치가 §6.4, §6.8과 일치. ✓
- §9.5: 결론이 범위 내로 한정 ("Under the conditions of this evaluation"). ✓
- §9.6: G2 FAIL의 범위 명시 (P1b 기반 C3, BurstGPT, T*=2.6 s, 특정 하드웨어). ✓
- §9.7: 세 미래 방향 모두 "would require a new pre-registered evaluation"으로 종결. ✓

**판정**: ✓ PASS

---

## 3. 수식·수치 전체 검증

### 3.1 핵심 수치 대조표

| 수치 | 원고 | 원본 근거 | 판정 |
|------|------|---------|------|
| G1\_SMODEL\_C | 1,560.860 ms | summary.json S\_total\_times\_c\_s = 1.5608601903 s | ✓ |
| ΔR (P1b) | −906.0 MiB | summary.json dR\_total\_MiB = -906.0 | ✓ |
| ΔH (P1b) | 16,756.72 MiB·s | summary.json dH\_total\_MiB = 16756.7231181967 (반올림) | ✓ |
| c | 281.8 µs/MiB | additional-analysis-plan-v4.md §1 | ✓ |
| T* | 2.6 s | prereg-v2 동결값 | ✓ |
| mean ΔI | 2,405.932 ms | measure-summary-official.json (3회 평균) | ✓ |
| mean G\_model | −845.072 ms | G1\_SC − mean ΔI = 1560.860 − 2405.932 | ✓ |
| mean G\_CPU | −2,661.243 ms | c·ΔR/1000 − mean ΔI = −255.311 − 2405.932 | ✓ |
| σ(ΔI) pop. | 75.831 ms | √(Σ(x−mean)²/3) = 75.831 ms | ✓ |
| ΔI 범위 | 180.092 ms | 2509.109 − 2329.017 | ✓ |
| H1v2 | −2.34 %p | sealed-results-v2.md | ✓ |
| H2v2 | −9.02 %p | sealed-results-v2.md | ✓ |
| H3v2 p값 | 3.4×10⁻⁵ | sealed-results-v2.md | ✓ |
| c·ΔR | −255.311 ms | 281.8 × 906 = 255,310.8 µs | ✓ |
| c·ΔH/T* | +1,816.171 ms | 281.8 × 6444.893 = 1,816,170.8 µs | ✓ |

### 3.2 손익분기 경계값 검증 (G\_model = 0)

| 변수 | 원고 경계 | 계산값 | 판정 |
|------|---------|-------|------|
| ΔI | < 1,560.860 ms | 1,560.860 ms | ✓ |
| T* | < 1.774 s | 1.7744 s (3자리 1.774) | ✓ |
| c | > 434.371 µs/MiB | 434.371 µs/MiB | ✓ |
| ΔH | > 24,553.699 MiB·s | 24,553.699 MiB·s | ✓ |
| ΔR | > +2,092.837 MiB | 2,092.837 MiB | ✓ |

---

## 4. 인용 및 참고문헌 검증

### 4.1 citation key 11개 본문 출현 확인

| 키 | §1 | §2 | §3 | §5 | §6 | §8 | §9 |
|----|----|----|----|----|----|----|-----|
| jemalloc | ✓ | ✓ | — | — | — | — | — |
| tcmalloc | ✓ | ✓ | — | — | — | — | — |
| llama2020 | ✓ | ✓ | — | — | — | — | — |
| tmo2022 | ✓ | ✓ | — | — | — | — | — |
| memtis2023 | — | ✓ | — | — | — | — | — |
| lecar2018 | — | ✓ | — | — | — | — | — |
| recmg2025 | ✓ | ✓ | — | — | — | — | — |
| kml2023 | — | ✓ | — | — | — | — | — |
| memperf2023 | — | ✓ | — | — | — | — | — |
| burstgpt2025 | ✓ | — | ✓ | ✓ | — | ✓ | — |
| pagereclaim2025 | — | ✓ | — | — | — | — | — |

11/11 모두 존재. 고아 키 없음. ✓

### 4.2 BibTeX 오류 현황 (related-work-matrix.md §8)

`bibliography-verification.md`에 7건의 오류가 기록되어 있으며, 그 중 6건이 치명적이다.
**BibTeX 수정은 manuscript-v4.md 본문 외 파일(related-work-matrix.md)에서 수행해야 한다.**

| # | 키 | 오류 유형 | 중요도 | 수정 여부 |
|---|----|---------|---------|---------| 
| 1 | recmg2025 | 저자 조작, arXiv 번호 오류, 제목 오류 | 치명 | bibliography-verification.md에 수정본 있음 |
| 2 | tmo2022 | 저자 성씨 오류 (Iyer → Agarwal), pages 누락 | 치명 | bibliography-verification.md에 수정본 있음 |
| 3 | memtis2023 | 제1저자 성씨 오류 (Kim → Lee) | 치명 | bibliography-verification.md에 수정본 있음 |
| 4 | lecar2018 | 저자 성명 오류, 2명 누락 | 치명 | bibliography-verification.md에 수정본 있음 |
| 5 | memperf2023 | 제목·저자·DOI 오류, @inproceedings→@article | 치명 | bibliography-verification.md에 수정본 있음 |
| 6 | burstgpt2025 | 1–3번째 저자 성명 오류 | 치명 | bibliography-verification.md에 수정본 있음 |
| 7 | llama2020 | 5번째 저자 오류 | 경미 | bibliography-verification.md에 수정본 있음 |
| 8 | kml2023 | @inproceedings → @article 필요 (TOCS 학술지) | 경미 | 미적용 |

**주의**: `bibliography-verification.md`는 수정 내용만 기록하고 `related-work-matrix.md` §8은 아직 직접 수정되지 않았을 수 있다. 제출 전 BibTeX 파일에 수정본을 반드시 적용해야 한다.

### 4.3 본문 선행 연구 설명 정확성

| 논문 | 원고 주요 주장 | 근거 | 판정 |
|------|-------------|------|------|
| LLAMA | 400 ns/call, 48× 느림, 캐싱으로 비용 분산 | bibliography-verification.md ✓ | ✓ |
| RecMG | LSTM+Attention, 92 µs/call, up to 43% speedup, 파이프라이닝 주장 | bibliography-verification.md ✓ | ✓ |
| TMO | 20–32% DRAM 절감, ML 없음, CPU 비용 미보고 | bibliography-verification.md ✓ | ✓ |
| MEMTIS | 히스토그램 기반, <3% 오버헤드 목표 | bibliography-verification.md ✓ | ✓ |
| KML | 커널 ML, <0.2% CPU 오버헤드 | related-work-audit.md ✓ | ✓ |
| LeCaR | 온라인 RL, 공간 오버헤드만 보고, CPU 미측정 | bibliography-verification.md ✓ | ✓ |
| MemPerf | 분석 프로파일러, ML 정책 없음 | bibliography-verification.md ✓ | ✓ |
| pagereclaim2025 | MGLRU/2QLRU 사이클 프로파일링, ML 없음 | bibliography-verification.md ✓ | ✓ |

**판정**: ✓ PASS — 선행 연구 차별성 과장 없음.

---

## 5. 문법 및 학술 문체 검토

### 5.1 발견된 경미한 사항

1. **§5.1 "14 from an Azure 2024 trace dataset"**  
   인용 없음. v1 sealed evaluation 범위 한정 설명이므로 연구 주장에 직접 영향은 없지만, 인용 가능한 출처가 있다면 추가할 것. 없으면 각주로 출처를 명시 권장. △ 경미

2. **§6.9 / §7.4 복합 손익분기 예시**  
   수치 오류 (3절 §6.9 항에 별도 기술). ✗ 수정 필요

3. **그림 참조 미작성 (Manuscript Status에 Fig. 1–3 "pending" 명시)**  
   본문에 "(Figure 1)", "(Figure 2)", "(Figure 3)" 형태의 참조가 없음.
   제출 버전에서는 그림 삽입 위치 및 본문 참조를 추가해야 한다. △ 미완성

4. **LaTeX 명령어 혼용 (Table 4, Table 1)**  
   `\textasciitilde`, `\textless3\%` 등이 Markdown 미리보기에서 LaTeX 명령으로 보임.
   LaTeX 소스 파일에서는 정상 렌더링됨. ✓ (LaTeX 원고로서 허용)

5. **§6.8 per-decision 수치 일관성**  
   "per-decision ΔI averages 1.201 µs/decision (2,405.932 ms / 2,003,785 decisions)":  
   2,405,932 / 2,003,785 = 1.2008 µs/decision → 소수 3자리 1.201 ✓  
   "per-decision simulation value is 0.779 µs/decision (1,560.860 ms / 2,003,785 decisions)":  
   1,560,860 / 2,003,785 = 0.7790 → 소수 3자리 0.779 ✓

6. **반복 표현**  
   - 연구 범위 한계 문장("P1b-based C3, BurstGPT, T*=2.6 s, specified hardware")이 §7.5, §9.5, §9.6에 각각 등장. 의도적 반복으로 적절함. ✓
   - §7.4 Break-Even Analysis 표가 §6.9 Table 12와 내용 동일 반복. 허용 가능 (Discussion 맥락에서 재제시). ✓

### 5.2 용어 일관성

| 용어 | 사용 패턴 | 판정 |
|------|---------|------|
| G1\_SMODEL\_C | 약자 정의 §3.2, 이후 일관 사용 | ✓ |
| G2 FAIL | 소문자 없이 대문자 고정 | ✓ |
| (sim) / (meas) / (calc) | §6 Notation에서 정의, 이후 일관 사용 | ✓ |
| fluid model | 소문자 고정 | ✓ |
| G2W3 | 대문자, 이탤릭 없음 | ✓ |

---

## 6. 수정이 필요한 항목

### 6.1 ✗ 수정 필수 — 복합 손익분기 예시 (§6.9, §7.4)

**위치**: §6.9 마지막 단락, §7.4 마지막 단락 (동일 문장이 두 곳에 존재)

**현재 문장** (§6.9):
> "For example, a 20\% reduction in $\Delta I$ combined with a 15\% reduction in $T^*$ would
> bring $G_\mathrm{model}$ to approximately zero; the exact threshold depends on the starting
> point for each variable."

**현재 문장** (§7.4):
> "a 20\% reduction in $\Delta I$ combined with a 15\% reduction in $T^*$ would, for instance,
> bring $G_\mathrm{model}$ approximately to zero."

**오류**: 20% ΔI 감소 + 15% T* 감소 시 실제 G\_model = −43.4 ms (zero와 거리가 있음).

**수정 권고 옵션 A** (두 조건 모두 명시):
> "For example, a 22\% reduction in $\Delta I$ combined with a 15\% reduction in $T^*$,
> or equivalently a 20\% reduction in $\Delta I$ combined with a 17\% reduction in $T^*$,
> would bring $G_\mathrm{model}$ approximately to zero."

**수정 권고 옵션 B** (20% ΔI 기준, 필요 T* 명시):
> "For example, a 20\% reduction in $\Delta I$ (to $\approx 1{,}924$ ms) would require
> an additional 17\% reduction in $T^*$ (to $\approx 2.17$ s) to bring $G_\mathrm{model}$
> approximately to zero."

**수정 범위**: §6.9 1개 문장 + §7.4 1개 문장 (총 2개소, 동일 오류).

---

### 6.2 △ 미완성 — 그림 본문 참조 (제출 전 필수)

**위치**: §3.4 (파이프라인 설명), §4.3 (C3 구조), §7.2 / §9.4 (G\_model 분해)

Manuscript Status에 Fig. 1–3이 "pending"으로 명시되어 있으나, 본문에 `(Figure 1)` 형태의 참조가 없음.

**권고**: 각 그림이 삽입되기 전이라도 "(see Figure 1 for the evaluation pipeline)" 형태의 선언적 참조를 본문에 추가한 뒤, 그림과 함께 최종 위치를 결정할 것.

---

### 6.3 △ 경미 — Azure 2024 트레이스 미인용

**위치**: §5.1, §6.1  
"14 from an Azure 2024 trace dataset"에 대한 인용이 없음.

**권고**: 인용 가능한 출처가 없다면 각주에 "Azure OpenAI 2024 internal trace; not publicly released." 등의 출처 명시를 고려할 것.

---

### 6.4 △ 외부 파일 — BibTeX 수정 미적용

**위치**: `docs/related-work-matrix.md` §8  
`bibliography-verification.md`에 수정본이 준비되어 있으나, 아직 BibTeX 파일에 적용되었는지 확인 필요.

**권고**: 제출 전 `bibliography-verification.md`의 수정본을 `related-work-matrix.md` §8에 반영하고, 특히 kml2023의 항목 유형을 `@inproceedings`에서 `@article`로 변경할 것.

---

## 7. 검증 체크리스트 (전체)

| 검토 항목 | 결과 |
|----------|------|
| 1. Abstract ↔ 본문 결론 수치 일치 | ✓ PASS |
| 2. 연구 질문 ↔ 6가지 기여 일관성 | ✓ PASS |
| 3. v2 PASS / G2 FAIL 구분 (§3.4, §7.1, §9.3–§9.4) | ✓ PASS |
| 4. ΔR, ΔH, ΔI 정의와 부호 규약 일관성 | ✓ PASS |
| 5. G\_model 수식·계산값 검증 | ✓ PASS |
| 6. G\_CPU 수식·계산값 검증 | ✓ PASS |
| 7. G2 공식 측정 3회 수치 (ΔI, G\_model, checksum) | ✓ PASS |
| 8. 표준편차 σ = 75.831 ms 검증 | ✓ PASS |
| 9. G1 수치 (G1\_SC, ΔR, ΔH) 원본 일치 | ✓ PASS |
| 10. 단일변수 손익분기 5개 경계값 | ✓ PASS |
| 11. Table 13 분해식 계산 | ✓ PASS |
| 12. 표 번호 (Table 1–14)와 본문 참조 일치 | ✓ PASS |
| 13. 11편 citation key 존재 및 연결 | ✓ PASS |
| 14. 선행 연구 차별성 과장 여부 | ✓ PASS — 과장 없음 |
| 15. 미수행 실험 허위 주장 | ✓ PASS — 없음 |
| 16. 적용 범위 한계와 결론 일치 (§8 L1–L12) | ✓ PASS |
| 17. 복합 손익분기 예시 (§6.9, §7.4) | ✗ 수치 오류 — **수정 필요** |
| 18. 그림 본문 참조 | △ 미작성 — 제출 전 추가 필요 |
| 19. Azure 2024 트레이스 인용 | △ 미인용 — 경미 |
| 20. BibTeX 오류 (외부 파일) | △ 수정 준비됨, 미적용 가능 |

---

## 8. 최종 판정

### **NEEDS REVISION**

**수정 없이 READY FOR VENUE FORMATTING 판정을 내릴 수 없는 이유:**

1. **§6.9 / §7.4 복합 손익분기 예시 수치 오류** (1건, 각 1문장)  
   "20% ΔI + 15% T*" 복합 조건이 G\_model ≈ 0을 만족하지 않음 (실제 −43 ms).  
   올바른 수치: "20% ΔI + 17% T*" 또는 "22% ΔI + 15% T*". 수정 범위 최소.

2. **그림 본문 참조 미작성** (Manuscript Status에 pending으로 명시)  
   Fig. 1–3 삽입과 본문 참조를 추가해야 완성본이 됨.

**상기 2개 항목 수정 완료 시: READY FOR VENUE FORMATTING**

**경미한 미결 항목 (판정에 영향 없음):**
- Azure 2024 트레이스 인용 (각주 추가 고려)
- BibTeX 오류 7건 (별도 파일 수정 필요, bibliography-verification.md에 준비됨)

---

*Git commit/push 미수행. 원고 본문, 실험 결과, G1/G2 산출물 미수정.*
