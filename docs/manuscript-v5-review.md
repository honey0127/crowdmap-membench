# Manuscript v5 최종 검토 보고서

> 작성: 2026-11-06  
> 대상: `docs/manuscript-v5.md`  
> 목적: `docs/manuscript-v4-review.md`에서 발견된 NEEDS REVISION 항목 해소 여부 확인  
> 참조: `docs/manuscript-v4-review.md`, `docs/bibliography-verification.md`,
>       `docs/related-work-matrix.md` §8, `DATA.md`

---

## 1. 검토 요약

| 범주 | v4 판정 | v5 적용 여부 | v5 판정 |
|------|---------|-------------|---------|
| §6.9 / §7.4 복합 손익분기 수치 오류 | ✗ 수정 필요 | ✓ 수정됨 | ✓ PASS |
| 그림 본문 참조 미작성 (Fig. 1–3) | △ 미완성 | ✓ 추가됨 | ✓ PASS |
| Azure 2024 트레이스 미인용 (§5.1, §6.1) | △ 경미 | ✓ 인용 추가됨 | ✓ PASS |
| BibTeX 오류 (related-work-matrix.md §8) | △ 미적용 | ✓ 확인됨 | ✓ PASS |
| 수치 일관성 (§7–§9 ↔ §3, §6) 계승 | ✓ PASS (v4) | 미변경 | ✓ PASS |
| 부호 규약 통일 | ✓ PASS (v4) | 미변경 | ✓ PASS |
| G2 공식 측정 수치 정확성 | ✓ PASS (v4) | 미변경 | ✓ PASS |

**최종 판정**: **READY FOR VENUE FORMATTING**

---

## 2. 수정 항목별 검증

### 2.1 §6.9 복합 손익분기 예시 수정

**v4 오류 문장** (수정 전):
> "a 20% reduction in ΔI combined with a 15% reduction in T* would bring G_model to
> approximately zero"

**v5 수정 문장**:
> "The exact two-variable thresholds are: a 20% reduction in ΔI (to ≈1,924 ms) requires
> an additional 17% reduction in T* (to ≈2.17 s) to reach G_model = 0; a 15% reduction in T*
> (to 2.21 s) requires a 22% reduction in ΔI. For instance, combining a 22% reduction in ΔI
> with a 15% reduction in T* yields G_model ≈ +5 ms, and combining a 20% reduction in ΔI
> with a 17% reduction in T* yields G_model ≈ +8 ms. These small positive values exceed the
> break-even threshold in expectation but fall within the population standard deviation
> σ_ΔI = 75.831 ms and do not represent stable economic viability."

**검산:**

| 복합 조건 | ΔI_new | T*_new | G1_new | G_model_new | 판정 |
|----------|--------|--------|--------|-------------|------|
| 20% ΔI + 15% T* (old) | 1924.746 ms | 2.21 s | 1881.361 ms | −43.385 ms | ✗ 오류 (not ≈ 0) |
| 20% ΔI + 17% T* (v5) | 1924.746 ms | 2.158 s | 1932.838 ms | +8.092 ms | ✓ ≈ 0 (within σ) |
| 22% ΔI + 15% T* (v5) | 1876.627 ms | 2.21 s | 1881.361 ms | +4.734 ms | ✓ ≈ 0 (within σ) |

정확 임계값 검산:
- 20% ΔI → 필요 T* = ΔH / (ΔI_new/c − ΔR) = 16756.72 / (1924.746/0.2818 − (−906.0)) = 16756.72 / (6830.4 + 906.0) = 16756.72 / 7736.4 = **2.166 s** (−16.7%)
- 15% T* → 필요 ΔI_max = G1_new(T*=2.21) = 281.8 × (−906 + 7582.0) / 1000 = **1881.4 ms** (−21.8% 감소 필요)

**판정**: ✓ PASS — 수정 후 수치가 올바름. σ=75.831 ms 대비 "not stable viability" 주석 포함.

---

### 2.2 §7.4 복합 손익분기 예시 수정

**v4 오류 문장**:
> "a 20% reduction in ΔI combined with a 15% reduction in T* would, for instance, bring
> G_model approximately to zero."

**v5 수정 문장**:
> "The exact paired thresholds are: a 20% reduction in ΔI (to ≈1,924 ms) requires a 17%
> reduction in T* (to ≈2.17 s) to reach G_model = 0; a 15% reduction in T* requires a 22%
> reduction in ΔI. For instance, combining a 20% reduction in ΔI with a 17% reduction in T*
> yields G_model ≈ +8 ms — a value that crosses zero but lies within σ_ΔI = 75.831 ms and
> does not represent stable economic viability."

**판정**: ✓ PASS — §6.9와 일관된 수정. σ 주석 일치.

---

### 2.3 Fig. 1 — 평가 파이프라인

**위치**: §3.4 말미 (Figure~1 참조 + 도식 삽입)

**내용 검증**:
- G1 (유체 모델, Python) → ΔR, ΔH → G1_SMODEL_C = +1,560.860 ms ✓
- G2 (C++ 측정, 3회) → ΔI = +2,405.932 ms ✓
- G_model = G1_SMODEL_C − ΔI = −845.072 ms ✓
- G2 verdict: FAIL ✓
- 본문 참조 "Figure~1 illustrates the two-phase evaluation pipeline." ✓

**판정**: ✓ PASS

---

### 2.4 Fig. 2 — C3 정책 실행 경로

**위치**: §4.3 말미 (Figure~2 참조 + 도식 삽입)

**내용 검증**:
- 증분 연산 5개 (compute_p1b_features, compute_Tk, online_kappa_T, run_fallback3×2) 모두 표시됨 ✓
- 공유 연산 (Ledger build, compute_grp_r, online_group_T, run_fallback3 rule) 구분됨 ✓
- §4.3 본문 및 Table 3과 일치 ✓
- 본문 참조 "Figure~2 shows the C3 execution path" ✓

**판정**: ✓ PASS

---

### 2.5 Fig. 3 — G_model 항별 분해

**위치**: §7.2 말미 (Table 13 직후, Figure~3 참조 + 도식 삽입)

**내용 검증**:
- c·ΔR = −255.311 ms ✓
- c·ΔH/T* = +1,816.171 ms ✓
- G1_SMODEL_C = +1,560.860 ms ✓
- ΔI = −2,405.932 ms ✓
- G_model = −845.072 ms ✓
- σ_ΔI = 75.831 ms 표시 ✓
- 본문 참조 "Figure~3 visualizes the term-by-term decomposition." ✓

**판정**: ✓ PASS

---

### 2.6 Azure 2024 트레이스 인용

**§5.1 수정 전**: "14 from an Azure 2024 trace dataset"  
**§5.1 수정 후**: "14 from an Azure 2024 LLM inference trace dataset \cite{azurellm2024}"

**§6.1 수정 전**: "14 Azure 2024"  
**§6.1 수정 후**: "14 Azure 2024 \cite{azurellm2024}"

**azurellm2024 BibTeX 항목** (related-work-matrix.md §8에 추가됨):
```bibtex
@misc{azurellm2024,
  author       = {{Microsoft Azure}},
  title        = {{AzureLLMInferenceTrace}: {Azure} {LLM} Inference Trace Dataset (2024)},
  howpublished = {\url{https://github.com/Azure/AzurePublicDataset}},
  year         = {2024},
  note         = {code 1-week (2024-05-10--16) and conv 1-week (2024-05-12--18) traces;
                  v1 sealed evaluation only (14 windows out of 53);
                  not used in v2 sealed, G1, or G2 evaluations}
}
```

**출처 검증** (`DATA.md`):
- code: `azure2024/AzureLLMInferenceTrace_code_1week.csv` → GitHub releases URL 확인 ✓
- conv: `azure2024/AzureLLMInferenceTrace_conv_1week.csv` → GitHub releases URL 확인 ✓
- v1 sealed에서만 14창 사용; v2/G1/G2는 BurstGPT만 사용 ✓

**판정**: ✓ PASS

---

### 2.7 BibTeX 오류 동기화

**related-work-matrix.md §8 현황 확인**:

| # | 키 | 오류 유형 | v5 상태 |
|---|----|---------|---------| 
| 1 | recmg2025 | 저자·arXiv·제목 오류 | ✓ 수정 적용됨 (Ren, Jie 등 7명, eprint=2511.08568) |
| 2 | tmo2022 | 저자 오류 (Iyer→Agarwal, 11명, pages 누락) | ✓ 수정 적용됨 |
| 3 | memtis2023 | 제1저자 오류 (Kim→Lee) | ✓ 수정 적용됨 |
| 4 | lecar2018 | 저자 성명·누락 2명 | ✓ 수정 적용됨 (8명 전원) |
| 5 | memperf2023 | 제목·저자·DOI 오류, @article 필요 | ✓ 수정 적용됨 (@article, Zhou 등 9명) |
| 6 | burstgpt2025 | 1–3번째 저자 오류 | ✓ 수정 적용됨 (Wang Yuxin 등 14명) |
| 7 | llama2020 | 5번째 저자 오류 | ✓ 수정 적용됨 (McKinley) |
| 8 | kml2023 | @inproceedings→@article 필요 | ✓ 이미 @article로 되어 있음 |
| 신규 | azurellm2024 | 신규 항목 | ✓ 추가됨 |

**판정**: ✓ PASS — 모든 7건 수정 확인됨, kml2023 이미 올바름, azurellm2024 추가됨.

---

## 3. v4에서 계승된 항목 (변경 없음)

v4-review에서 PASS 판정을 받은 항목들은 v5에서 해당 본문 내용이 변경되지 않았으므로 판정을 계승한다.

| 항목 | v4 판정 | v5 계승 |
|------|---------|---------|
| Abstract ↔ 본문 결론 수치 일치 | ✓ PASS | ✓ 계승 |
| ΔR, ΔH, ΔI 부호 규약 일관성 | ✓ PASS | ✓ 계승 |
| G_model 수식 계산값 | ✓ PASS | ✓ 계승 |
| G2 공식 측정 3회 수치 | ✓ PASS | ✓ 계승 |
| Table 13 항별 분해식 | ✓ PASS | ✓ 계승 |
| 단일변수 손익분기 5개 경계값 | ✓ PASS | ✓ 계승 |
| §8 한계 L1–L12 미수행 실험 허위 주장 없음 | ✓ PASS | ✓ 계승 |
| §9 결론 범위 한정 | ✓ PASS | ✓ 계승 |
| 선행 연구 차별성 과장 없음 | ✓ PASS | ✓ 계승 |

---

## 4. 수치 검증 (v5 신규 추가 항목)

### 4.1 §6.9 복합 조건 정확도

```
20% ΔI + 17% T*:
  ΔI_new = 2405.932 × 0.80 = 1924.746 ms
  T*_new = 2.6 × 0.83 = 2.158 s
  ΔH/T*_new = 16756.72 / 2.158 = 7764.9 MiB (approx)
  G1_new = 281.8 × (−906.0 + 7764.9) / 1000 = 281.8 × 6858.9 / 1000 = 1932.8 ms
  G_model_new = 1932.8 − 1924.7 = +8.1 ms  ✓ (원고: "≈ +8 ms")

22% ΔI + 15% T*:
  ΔI_new = 2405.932 × 0.78 = 1876.627 ms
  T*_new = 2.6 × 0.85 = 2.21 s
  ΔH/T*_new = 16756.72 / 2.21 = 7582.0 MiB (approx)
  G1_new = 281.8 × (−906.0 + 7582.0) / 1000 = 281.8 × 6676.0 / 1000 = 1881.3 ms
  G_model_new = 1881.3 − 1876.6 = +4.7 ms  ✓ (원고: "≈ +5 ms")
```

σ_ΔI = 75.831 ms (pop.) vs. +8 ms, +5 ms: 두 값 모두 σ 범위 내 → "not stable viability" ✓

### 4.2 20% ΔI 필요 T* 임계값

```
G_model = 0:
  c(ΔR + ΔH/T*_new) = ΔI_new
  281.8 × (−906 + 16756.72/T*_new) = 1924746 µs
  16756.72/T*_new = 1924746/281.8 + 906 = 6830.4 + 906.0 = 7736.4
  T*_new = 16756.72 / 7736.4 = 2.166 s
  감소율 = (2.6 − 2.166) / 2.6 = 16.7%  ✓ (원고: "17%", 반올림 ✓)
```

### 4.3 azurellm2024 citation key 일관성

- §5.1: `\cite{azurellm2024}` 추가 ✓
- §6.1: `\cite{azurellm2024}` 추가 ✓
- Verification Checklist 항목 9: `azurellm2024` 포함 (12개 키) ✓
- related-work-matrix.md §8: `@misc{azurellm2024, ...}` 추가 ✓

---

## 5. 검증 체크리스트 (전체)

| 검토 항목 | v4 판정 | v5 판정 |
|----------|---------|---------|
| 1. Abstract ↔ 본문 결론 수치 일치 | ✓ | ✓ |
| 2. 연구 질문 ↔ 6가지 기여 일관성 | ✓ | ✓ |
| 3. v2 PASS / G2 FAIL 구분 | ✓ | ✓ |
| 4. ΔR, ΔH, ΔI 부호 규약 일관성 | ✓ | ✓ |
| 5. G_model 수식·계산값 검증 | ✓ | ✓ |
| 6. G_CPU 수식·계산값 검증 | ✓ | ✓ |
| 7. G2 공식 측정 3회 수치 | ✓ | ✓ |
| 8. σ = 75.831 ms 검증 | ✓ | ✓ |
| 9. G1 수치 (G1_SC, ΔR, ΔH) 원본 일치 | ✓ | ✓ |
| 10. 단일변수 손익분기 5개 경계값 | ✓ | ✓ |
| 11. Table 13 분해식 계산 | ✓ | ✓ |
| 12. 표 번호와 본문 참조 일치 | ✓ | ✓ |
| 13. 11편 citation key 존재 및 연결 | ✓ | ✓ (12편, azurellm2024 추가) |
| 14. 선행 연구 차별성 과장 여부 | ✓ | ✓ |
| 15. 미수행 실험 허위 주장 | ✓ | ✓ |
| 16. 적용 범위 한계와 결론 일치 | ✓ | ✓ |
| **17. 복합 손익분기 예시 (§6.9, §7.4)** | ✗ 오류 | **✓ 수정됨** |
| **18. 그림 본문 참조 (Fig. 1–3)** | △ 미작성 | **✓ 추가됨** |
| **19. Azure 2024 트레이스 인용** | △ 미인용 | **✓ 인용됨** |
| **20. BibTeX 오류 동기화** | △ 미확인 | **✓ 확인 및 완료** |

---

## 6. 최종 판정

### **READY FOR VENUE FORMATTING**

manuscript-v4-review.md에서 NEEDS REVISION 판정의 근거였던 2개 항목이 모두 해소됨:

1. **§6.9 / §7.4 복합 손익분기 수치 오류 → 수정됨**  
   "20% + 15% ≈ 0" 오류를 "20% + 17% → +8 ms (σ 이내)" 및  
   "22% + 15% → +5 ms (σ 이내)"로 교체. 안정적 경제성 미달 주석 포함.

2. **그림 본문 참조 → 추가됨**  
   Fig. 1 (§3.4 평가 파이프라인), Fig. 2 (§4.3 C3 경로),  
   Fig. 3 (§7.2 G_model 분해) — 텍스트 기반 도식 + 본문 참조.

추가 해소 항목 (경미 → 완료):
- Azure 2024 트레이스 citation key `azurellm2024` 추가 (§5.1, §6.1)
- BibTeX 7건 수정 확인됨 (related-work-matrix.md §8 이미 수정 적용)
- kml2023 `@article` 타입 이미 올바름

**잔여 미결 항목**: 없음.

**제출 전 venue 형식화 시 권고 사항** (판정에 영향 없음):
- Fig. 1–3을 텍스트 도식에서 벡터 그래픽(TikZ 또는 PDF)으로 교체할 것.
- pagereclaim2025를 "SC Workshops '25" (본회의 아님)로 인용 시 본문에 명시 가능.
- §6.8의 `\textasciitilde`, `\textless` 등 LaTeX 명령어는 LaTeX 소스로 변환 시 정상 렌더링됨.

---

*Git commit/push 미수행. 실험 결과 파일 미수정.*
