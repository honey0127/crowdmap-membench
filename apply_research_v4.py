from pathlib import Path

ROOT = Path(__file__).resolve().parent
DOCS = ROOT / "docs"
brief_path = DOCS / "current-research-brief.md"
section_path = DOCS / "current-research-brief-section5-v4.md"
plan_path = DOCS / "additional-analysis-plan-v4.md"

for path in (brief_path, section_path, plan_path):
    if not path.is_file():
        raise SystemExit(f"필요한 파일이 없습니다: {path}")

brief = brief_path.read_text(encoding="utf-8")
section = section_path.read_text(encoding="utf-8").rstrip() + "\n\n"
start_marker = "## 5. 현재 범위 밖"
end_marker = "## 6. 문서 지도"
if brief.count(start_marker) != 1 or brief.count(end_marker) != 1:
    raise SystemExit("연구 지침의 §5/§6 제목이 예상과 다릅니다. 자동 변경을 중단합니다.")
start = brief.index(start_marker)
end = brief.index(end_marker)
if end <= start:
    raise SystemExit("§5와 §6의 순서가 올바르지 않습니다.")

brief = brief[:start] + section + brief[end:]
brief = brief.replace("# 현행 연구 지침 (2026-10-31)", "# 현행 연구 지침 (2026-11-04)", 1)

# §6 문서 지도에 신규 계획을 기록한다.
needle = "## 6. 문서 지도\n\n- 현행\n"
addition = "  - `additional-analysis-plan-v4.md`(2026-11-04 신규 저차원 조회표·순이득 실험 계획)\n"
if needle not in brief:
    raise SystemExit("§6 문서 지도 구조가 예상과 다릅니다. 자동 변경을 중단합니다.")
if "additional-analysis-plan-v4.md" not in brief:
    brief = brief.replace(needle, needle + addition, 1)

brief_path.write_text(brief, encoding="utf-8")
print("갱신 완료:", brief_path)
print("추가 계획:", plan_path)
print("변경 내용을 확인한 뒤 git diff --check 및 git diff를 실행하세요.")
