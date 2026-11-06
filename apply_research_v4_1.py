#!/usr/bin/env python3
from pathlib import Path
import re
import sys

ROOT = Path.cwd()
plan_src = Path(__file__).with_name("additional-analysis-plan-v4.md")
q_src = Path(__file__).with_name("questions-for-advisor.md")
s1_src = Path(__file__).with_name("current-research-brief-section1-v4.1.md")
s5_src = Path(__file__).with_name("current-research-brief-section5-v4.1.md")
brief_path = ROOT / "docs/current-research-brief.md"
plan_path = ROOT / "docs/additional-analysis-plan-v4.md"
q_path = ROOT / "questions-for-advisor.md"

required = [plan_src, q_src, s1_src, s5_src, brief_path, plan_path, q_path]
missing = [str(p) for p in required if not p.is_file()]
if missing:
    print("필수 파일이 없어 중단합니다:")
    print("\n".join(missing))
    sys.exit(1)

brief = brief_path.read_text(encoding="utf-8")
# Locate section 1/2 and section 5/6 without guessing the body text.
sec1 = re.search(r"(?m)^## 1(?:\.|\s).*?$", brief)
sec2 = re.search(r"(?m)^## 2(?:\.|\s).*?$", brief)
sec5 = re.search(r"(?m)^## 5(?:\.|\s).*?$", brief)
sec6 = re.search(r"(?m)^## 6(?:\.|\s).*?$", brief)
if not all([sec1, sec2, sec5, sec6]) or not (sec1.start() < sec2.start() < sec5.start() < sec6.start()):
    print("브리프의 ## 1, ## 2, ## 5, ## 6 경계를 찾지 못해 아무 파일도 수정하지 않았습니다.")
    sys.exit(1)

s1 = s1_src.read_text(encoding="utf-8").rstrip() + "\n\n"
s5 = s5_src.read_text(encoding="utf-8").rstrip() + "\n\n"
# Do not duplicate the new subsection if this script is run twice.
section1_text = brief[sec1.start():sec2.start()]
if "### c의 실측 정의와 T*의 가격 환산" not in section1_text:
    brief = brief[:sec2.start()] + s1 + brief[sec2.start():]
    sec5 = re.search(r"(?m)^## 5(?:\.|\s).*?$", brief)
    sec6 = re.search(r"(?m)^## 6(?:\.|\s).*?$", brief)
brief = brief[:sec5.start()] + s5 + brief[sec6.start():]

# Read all replacement data before writing anything.
plan = plan_src.read_text(encoding="utf-8")
questions = q_src.read_text(encoding="utf-8")
# Final sanity checks.
for needle in ["주 판정은 `G_model`", "모든 조각의 표 조회", "공통 분위 격자"]:
    if needle not in plan:
        raise SystemExit(f"연구 계획 파일 검증 실패: {needle}")
if "G_CPU" not in questions or "비용 분해로만" not in questions:
    raise SystemExit("지도교수 질문 파일 검증 실패")

# Write only the three intended tracked documents.
plan_path.write_text(plan, encoding="utf-8")
brief_path.write_text(brief, encoding="utf-8")
q_path.write_text(questions, encoding="utf-8")
print("v4.1 문서 적용 완료: docs/current-research-brief.md, docs/additional-analysis-plan-v4.md, questions-for-advisor.md")
print("git diff --check 및 세 문서 diff를 검토한 뒤 커밋하세요. 자동 커밋은 하지 않았습니다.")
