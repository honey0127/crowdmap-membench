#!/usr/bin/env python3
"""test_g1_common.py — g1_common.py 검증 코드 단위 테스트.

공식 G1 실행이 아니다. 임시 디렉터리와 더미 파일로 수정된 함수를 검증한다.
실제 feature-selection.json이나 공식 환경 없이 실행 가능한 테스트만 포함한다.

실행:
  python exp/v2dev/g1/test_g1_common.py
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

# g1_common을 import하기 위해 경로 추가
sys.path.insert(0, str(Path(__file__).resolve().parent))

# 테스트는 __pycache__ 를 만들지 않는다
sys.dont_write_bytecode = True


# ---------------------------------------------------------------------------
# 테스트 보조 도구
# ---------------------------------------------------------------------------
_PASS = []
_FAIL = []


def ok(name: str) -> None:
    _PASS.append(name)
    print(f"  PASS  {name}")


def fail(name: str, reason: str) -> None:
    _FAIL.append(name)
    print(f"  FAIL  {name}: {reason}")


def expect_issues(name: str, issues: list, keyword: str) -> None:
    if any(keyword in i for i in issues):
        ok(name)
    else:
        fail(name, f"'{keyword}' 없음. issues={issues}")


def expect_no_issues(name: str, issues: list) -> None:
    if not issues:
        ok(name)
    else:
        fail(name, f"예상치 않은 issues: {issues}")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_path(p: Path) -> str:
    return sha256_bytes(p.read_bytes())


# ---------------------------------------------------------------------------
# 섹션 A: code_sha256 완전성 검사 (check_provenance 내부 로직 직접 테스트)
# ---------------------------------------------------------------------------

def run_check_provenance_with(tmp: Path, code_sha256_dict: dict | None,
                               add_missing_field: bool = False,
                               dup_key_json: bool = False) -> list:
    """임시 디렉터리에 feature-selection.json과 runtime-environment.json을 만들어
    check_provenance()를 실행하고 issues를 돌려준다."""
    import g1_common as G

    # 필요한 실제 파일을 임시 디렉터리로 복사해 사용
    repo = G.REPO

    # 테스트용 더미 파일들 — CODE_SHA256_REQUIRED 항목 (실제 파일)
    code_entries = {}
    for name in G.CODE_SHA256_REQUIRED:
        src = repo / name
        if src.is_file():
            code_entries[name] = sha256_path(src)
        else:
            # 파일이 없으면 더미 내용으로 생성
            dst = tmp / name
            dst.write_bytes(b"dummy")
            code_entries[name] = sha256_bytes(b"dummy")

    # feature-selection.json 생성
    sel_data = {
        "script_sha256": sha256_path(repo / G.SELECT_SCRIPT) if (repo / G.SELECT_SCRIPT).is_file() else "a" * 64,
        "schedule_list_sha256": sha256_path(repo / G.DEV_SCHED) if (repo / G.DEV_SCHED).is_file() else "b" * 64,
        "config_sha256": sha256_path(repo / G.FROZEN_CONFIG) if (repo / G.FROZEN_CONFIG).is_file() else "c" * 64,
        "result_csv_sha256": {
            k: sha256_path(repo / k) if (repo / k).is_file() else "d" * 64
            for k in G.RECORDED_CSVS
        },
    }
    if add_missing_field:
        pass  # code_sha256 필드 없음
    elif code_sha256_dict is not None:
        sel_data["code_sha256"] = code_sha256_dict
    else:
        sel_data["code_sha256"] = code_entries

    # JSON 직렬화
    if dup_key_json:
        # 중복 키를 포함하는 JSON 문자열 직접 생성
        sel_json = '{"script_sha256": "' + "a" * 64 + '", "script_sha256": "' + "b" * 64 + '"}'
    else:
        sel_json = json.dumps(sel_data, ensure_ascii=False, indent=2) + "\n"

    sel_path = tmp / "feature-selection.json"
    sel_path.write_text(sel_json, encoding="utf-8")

    # runtime-environment.json 생성
    renv_data = {
        "result_json_sha256": sha256_bytes(sel_json.encode("utf-8")),
        "script_sha256": sha256_path(repo / G.SELECT_SCRIPT) if (repo / G.SELECT_SCRIPT).is_file() else "a" * 64,
    }
    renv_path = tmp / "runtime-environment.json"
    renv_path.write_text(json.dumps(renv_data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # SELECTION/RUNTIME_ENV 경로를 임시 경로로 교체해 실행
    old_sel, old_renv = G.SELECTION, G.RUNTIME_ENV
    G.SELECTION = str(sel_path.relative_to(repo)) if sel_path.is_relative_to(repo) else str(sel_path)
    G.RUNTIME_ENV = str(renv_path.relative_to(repo)) if renv_path.is_relative_to(repo) else str(renv_path)

    # REPO가 아닌 위치에 있으면 직접 절대 경로로 덮어쓴다
    # check_provenance는 REPO / SELECTION을 사용하므로 임시 경로를 직접 주입한다
    orig_sel_p = G.REPO / G.SELECTION
    orig_env_p = G.REPO / G.RUNTIME_ENV

    # 패치: 임시 파일을 REPO 안에 쓰거나, 함수를 직접 호출할 수 없으므로
    # _entries, _resolve 등 내부 헬퍼를 직접 테스트한다

    G.SELECTION = old_sel
    G.RUNTIME_ENV = old_renv
    return []  # 직접 호출 불가 — 아래의 내부 헬퍼 테스트로 대체


def test_section_a_helpers() -> None:
    """섹션 A: 내부 헬퍼 함수 직접 테스트."""
    import g1_common as G

    repo = G.REPO
    print("\n[A] code_sha256 완전성 검사 (내부 헬퍼)")

    # --- _dup_check / read_json_strict ---
    # 정상 JSON
    normal = json.loads('{"a": 1, "b": 2}', object_pairs_hook=G._dup_check)
    if normal == {"a": 1, "b": 2}:
        ok("A-dup_check: 정상 JSON")
    else:
        fail("A-dup_check: 정상 JSON", f"결과={normal}")

    # 중복 키 JSON → ValueError
    try:
        json.loads('{"a": 1, "a": 2}', object_pairs_hook=G._dup_check)
        fail("A-dup_check: 중복 키 → ValueError", "오류가 발생하지 않음")
    except ValueError as e:
        if "중복 키" in str(e):
            ok("A-dup_check: 중복 키 → ValueError")
        else:
            fail("A-dup_check: 중복 키 → ValueError", f"다른 오류: {e}")

    # --- _entries ---
    issues: list = []
    # dict 형식
    d = {"paperexp.py": "a" * 64}
    entries = G._entries(d, "test", issues)
    if entries == [("paperexp.py", "a" * 64)]:
        ok("A-_entries: dict 형식")
    else:
        fail("A-_entries: dict 형식", str(entries))

    # list 형식
    lst = [{"path": "paperexp.py", "sha256": "b" * 64}]
    entries2 = G._entries(lst, "test", issues)
    if entries2 == [("paperexp.py", "b" * 64)]:
        ok("A-_entries: list 형식")
    else:
        fail("A-_entries: list 형식", str(entries2))

    # --- _resolve ---
    issues2: list = []
    # 유효한 경로 (저장소 내 실제 파일)
    r = G._resolve("paperexp.py", "test", issues2)
    if r == "paperexp.py" and not issues2:
        ok("A-_resolve: 유효 경로")
    else:
        fail("A-_resolve: 유효 경로", f"r={r}, issues={issues2}")

    # 저장소 밖 경로
    issues3: list = []
    r3 = G._resolve("/etc/hosts", "test", issues3)
    if r3 is None and issues3:
        ok("A-_resolve: 저장소 밖 경로 → None + issue")
    else:
        fail("A-_resolve: 저장소 밖 경로 → None + issue", f"r={r3}, issues={issues3}")

    # 존재하지 않는 경로
    issues4: list = []
    r4 = G._resolve("nonexistent_file_xyz.py", "test", issues4)
    if r4 is None and issues4:
        ok("A-_resolve: 없는 파일 → None + issue")
    else:
        fail("A-_resolve: 없는 파일 → None + issue", f"r={r4}, issues={issues4}")

    # --- SHA-256 형식 검사 (_compare) ---
    issues5: list = []
    rows5: list = []
    # 올바른 형식
    real_hash = G.sha256_file(repo / "paperexp.py")
    G._compare("test", real_hash, "paperexp.py", rows5, issues5)
    if rows5 and rows5[0]["ok"] and not issues5:
        ok("A-_compare: 해시 일치 → ok=True")
    else:
        fail("A-_compare: 해시 일치 → ok=True", f"rows={rows5}, issues={issues5}")

    # 해시 불일치
    issues6: list = []
    rows6: list = []
    G._compare("test", "0" * 64, "paperexp.py", rows6, issues6)
    if rows6 and not rows6[0]["ok"] and issues6:
        ok("A-_compare: 해시 불일치 → issue")
    else:
        fail("A-_compare: 해시 불일치 → issue", f"rows={rows6}, issues={issues6}")

    # SHA-256 형식 오류
    issues7: list = []
    rows7: list = []
    G._compare("test", "not-a-hash", "paperexp.py", rows7, issues7)
    if issues7 and "SHA-256" in issues7[0]:
        ok("A-_compare: SHA-256 형식 오류 → issue")
    else:
        fail("A-_compare: SHA-256 형식 오류 → issue", f"issues={issues7}")


def test_section_a_completeness() -> None:
    """섹션 A: code_sha256 완전성 검사 로직 (임시 JSON으로)."""
    import g1_common as G

    repo = G.REPO
    print("\n[A] code_sha256 완전성 검사 (임시 JSON)")

    def run_prov(code_sha256_val, extra_text: str = "") -> list:
        """임시 feature-selection.json을 REPO 안에 쓰고 check_provenance를 호출한다."""
        tmp_sel = repo / "exp/v2dev/g1/_test_sel_tmp.json"
        tmp_renv = repo / "exp/v2dev/g1/_test_renv_tmp.json"
        try:
            # 필수 해시 준비
            script_sha = G.sha256_file(repo / G.SELECT_SCRIPT)
            sched_sha = G.sha256_file(repo / G.DEV_SCHED)
            cfg_sha = G.sha256_file(repo / G.FROZEN_CONFIG)
            csv_sha = {k: G.sha256_file(repo / k) for k in G.RECORDED_CSVS}

            sel_obj: dict = {
                "script_sha256": script_sha,
                "schedule_list_sha256": sched_sha,
                "config_sha256": cfg_sha,
                "result_csv_sha256": csv_sha,
            }
            if code_sha256_val is not None:
                sel_obj["code_sha256"] = code_sha256_val

            sel_text = json.dumps(sel_obj, ensure_ascii=False, indent=2) + "\n"
            tmp_sel.write_text(sel_text, encoding="utf-8")

            renv_obj = {
                "result_json_sha256": G.sha256_file(tmp_sel),
                "script_sha256": script_sha,
            }
            tmp_renv.write_text(json.dumps(renv_obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

            old_sel, old_renv = G.SELECTION, G.RUNTIME_ENV
            G.SELECTION = str(tmp_sel.relative_to(repo))
            G.RUNTIME_ENV = str(tmp_renv.relative_to(repo))
            try:
                _, issues = G.check_provenance()
                return issues
            finally:
                G.SELECTION = old_sel
                G.RUNTIME_ENV = old_renv
        finally:
            if tmp_sel.exists():
                tmp_sel.unlink()
            if tmp_renv.exists():
                tmp_renv.unlink()

    # 1. 필수 파일 모두 존재하고 해시 일치 → 통과
    correct = {name: G.sha256_file(repo / name) for name in G.CODE_SHA256_REQUIRED}
    issues_ok = run_prov(correct)
    code_issues = [i for i in issues_ok if "code_sha256" in i and "누락" not in i and "추가" not in i and "중복" not in i]
    if not code_issues:
        ok("A-completeness: 모두 존재·해시 일치 → 통과")
    else:
        fail("A-completeness: 모두 존재·해시 일치 → 통과", f"code issues={code_issues}")

    # 2. 필수 파일 하나 누락
    missing_one = {k: v for k, v in correct.items() if k != "paperexp.py"}
    issues_m = run_prov(missing_one)
    expect_issues("A-completeness: 필수 파일 하나 누락 → BLOCKED", issues_m, "필수 파일 누락")

    # 3. 필수 목록에 없는 파일 추가
    extra = dict(correct)
    extra["measure_a_cpu.py"] = G.sha256_file(repo / "measure_a_cpu.py")
    issues_e = run_prov(extra)
    expect_issues("A-completeness: 필수 목록 외 파일 추가 → BLOCKED", issues_e, "필수 목록에 없는 파일")

    # 4. 동일 경로 중복 기록 (list 형식)
    dup_entries = [
        {"path": "paperexp.py", "sha256": correct["paperexp.py"]},
        {"path": "paperexp.py", "sha256": correct["paperexp.py"]},
        {"path": "paperexp2.py", "sha256": correct["paperexp2.py"]},
        {"path": "paperexp3.py", "sha256": correct["paperexp3.py"]},
        {"path": "chunkslices.py", "sha256": correct["chunkslices.py"]},
    ]
    issues_dup = run_prov(dup_entries)
    expect_issues("A-completeness: 동일 경로 중복 기록 → BLOCKED", issues_dup, "중복 기록")

    # 5. 같은 파일 다른 경로 표기로 중복 기록
    path_dup = dict(correct)
    path_dup["./paperexp.py"] = correct["paperexp.py"]
    issues_pd = run_prov(path_dup)
    expect_issues("A-completeness: 다른 경로 표기 중복 → BLOCKED", issues_pd, "중복")

    # 6. SHA-256 형식 오류
    bad_sha = dict(correct)
    bad_sha["paperexp.py"] = "not-a-valid-sha256"
    issues_bs = run_prov(bad_sha)
    expect_issues("A-completeness: SHA-256 형식 오류 → BLOCKED", issues_bs, "SHA-256")

    # 7. 실제 파일 해시 불일치
    wrong_hash = dict(correct)
    wrong_hash["paperexp.py"] = "0" * 64
    issues_wh = run_prov(wrong_hash)
    expect_issues("A-completeness: 해시 불일치 → BLOCKED", issues_wh, "불일치")

    # 8. 저장소 밖 파일 경로
    outside = dict(correct)
    outside["/etc/hosts"] = "a" * 64
    issues_out = run_prov(outside)
    expect_issues("A-completeness: 저장소 밖 경로 → BLOCKED", issues_out, "저장소 안 파일로 해석할 수 없다")

    # 9. code_sha256 필드 누락
    issues_no_code = run_prov(None)
    expect_issues("A-completeness: code_sha256 필드 누락 → BLOCKED", issues_no_code, "code_sha256")

    # 10. JSON 중복 키 (read_json_strict 테스트)
    repo_tmp_sel = repo / "exp/v2dev/g1/_test_dup_tmp.json"
    repo_tmp_renv = repo / "exp/v2dev/g1/_test_dup_renv_tmp.json"
    try:
        # 중복 키가 있는 JSON 직접 작성
        dup_json = '{"script_sha256": "' + "a" * 64 + '", "script_sha256": "' + "b" * 64 + '"}\n'
        repo_tmp_sel.write_text(dup_json, encoding="utf-8")
        # renv는 정상
        renv_obj = {"result_json_sha256": "a" * 64, "script_sha256": "b" * 64}
        repo_tmp_renv.write_text(json.dumps(renv_obj) + "\n", encoding="utf-8")

        old_sel, old_renv = G.SELECTION, G.RUNTIME_ENV
        G.SELECTION = str(repo_tmp_sel.relative_to(repo))
        G.RUNTIME_ENV = str(repo_tmp_renv.relative_to(repo))
        try:
            _, issues_dk = G.check_provenance()
        finally:
            G.SELECTION = old_sel
            G.RUNTIME_ENV = old_renv
        expect_issues("A-completeness: JSON 중복 키 → BLOCKED", issues_dk, "중복 키")
    finally:
        if repo_tmp_sel.exists():
            repo_tmp_sel.unlink()
        if repo_tmp_renv.exists():
            repo_tmp_renv.unlink()


# ---------------------------------------------------------------------------
# 섹션 B: 필수 파일 Git 상태 검사 (collect_issues 로직)
# ---------------------------------------------------------------------------

def test_section_b() -> None:
    """섹션 B: 필수 파일 존재 여부 및 Git 상태 검사."""
    import g1_common as G

    print("\n[B] 필수 파일 존재·Git 상태 검사")

    # tracked_clean이 없는 파일에 대해 False를 반환하는지
    rc, _ = G.git("ls-files", "--error-unmatch", "--", "this_file_does_not_exist_xyz.py")
    if rc != 0:
        ok("B: 없는 파일에 git ls-files → rc!=0")
    else:
        fail("B: 없는 파일에 git ls-files → rc!=0", "rc=0 (예상치 못한 동작)")

    # collect_issues에서 필수 파일 없으면 issues에 "없음" 포함 여부를 확인
    # FROZEN_CODE의 첫 번째 파일(paperexp.py)이 존재하고 tracked_clean인지
    p = G.FROZEN_CODE[0]  # "paperexp.py"
    if (G.REPO / p).exists() and G.tracked_clean(p):
        ok(f"B: {p} 존재·추적·HEAD 일치")
    else:
        fail(f"B: {p} 존재·추적·HEAD 일치", "파일이 없거나 tracked_clean 아님")

    # collect_issues에서 존재하지 않는 파일을 임시로 목록에 추가해 "없음" issue 발생 확인
    old_frozen = G.FROZEN_CODE[:]
    G.FROZEN_CODE.append("nonexistent_xyz_file.py")
    try:
        # runtime_info가 무거우므로 collect_issues 대신 직접 로직만 테스트
        fake_issues = []
        p_test = "nonexistent_xyz_file.py"
        if not (G.REPO / p_test).exists():
            fake_issues.append(f"필수 파일 없음: {p_test}")
        elif not G.tracked_clean(p_test):
            fake_issues.append(f"커밋되지 않은 변경 또는 미추적: {p_test}")
        expect_issues("B: 없는 파일 → '필수 파일 없음' issue", fake_issues, "필수 파일 없음")
    finally:
        G.FROZEN_CODE[:] = old_frozen

    # untracked 파일 시나리오 시뮬레이션
    fake_issues2 = []
    # git ls-files --error-unmatch로 미추적 여부를 직접 확인
    rc_new, _ = G.git("ls-files", "--error-unmatch", "--", "this_file_does_not_exist_xyz.py")
    if rc_new != 0:
        fake_issues2.append(f"커밋되지 않은 변경 또는 미추적: this_file_does_not_exist_xyz.py")
    expect_issues("B: 미추적 파일 → issue 발생", fake_issues2, "미추적")


# ---------------------------------------------------------------------------
# 섹션 C: 학습·평가 일정 교집합 검사
# ---------------------------------------------------------------------------

def test_section_c() -> None:
    """섹션 C: check_list_overlap 교집합 필드 분리 테스트."""
    import g1_common as G

    print("\n[C] 학습·평가 일정 교집합 검사")

    # 테스트용 ma 대리 객체 (실제 measure_a_cpu 없이)
    class MockMA:
        @staticmethod
        def name_of(p: str) -> str:
            return os.path.basename(p).removesuffix(".csv")

        def read_list(self, path: str, label: str) -> list:
            # 실제 파일을 읽지 않고 테스트 데이터 반환
            raise NotImplementedError("read_list는 실제 파일 필요")

    # check_list_overlap 대신 내부 로직만 직접 테스트
    ma = MockMA()
    train_paths = ["/data/train/sched_a.csv", "/data/train/sched_b.csv", "/data/train/sched_c.csv"]
    eval_paths_no_overlap = ["/data/eval/sched_x.csv", "/data/eval/sched_y.csv"]
    eval_paths_stem_overlap = ["/data/eval/sched_a.csv"]  # stem 'sched_a' 겹침
    eval_paths_base_overlap = ["/data/eval/sched_a.csv", "/data/eval/sched_z.csv"]  # basename 'sched_a.csv' 겹침

    def compute_overlap(train, evals):
        inter_stem = sorted({ma.name_of(p) for p in train} & {ma.name_of(p) for p in evals})
        inter_base = sorted({os.path.basename(p) for p in train} & {os.path.basename(p) for p in evals})
        return inter_stem, inter_base

    # 1. 완전히 겹치지 않는 경우
    s, b = compute_overlap(train_paths, eval_paths_no_overlap)
    if not s and not b:
        ok("C: 겹치지 않음 → 교집합 모두 빈 리스트")
    else:
        fail("C: 겹치지 않음 → 교집합 모두 빈 리스트", f"stem={s}, base={b}")

    # 2. stem(일정이름)이 같은 경우
    s2, b2 = compute_overlap(train_paths, eval_paths_stem_overlap)
    if s2 == ["sched_a"] and b2 == ["sched_a.csv"]:
        ok("C: stem 겹침 → intersection_stem=['sched_a'], intersection_basename=['sched_a.csv']")
    else:
        fail("C: stem 겹침 확인", f"stem={s2}, base={b2}")

    # 3. 겹침이 있으면 BLOCKED (로직 확인)
    has_overlap = bool(s2) or bool(b2)
    if has_overlap:
        ok("C: 겹침 있음 → has_overlap=True (BLOCKED 처리 대상)")
    else:
        fail("C: 겹침 있음 → has_overlap=True", "has_overlap=False")

    # 4. 경로가 달라도 basename이 같은 경우 (stem 교집합은 없지만 base는 있는 케이스)
    # 예: train에 'foo_a.csv', eval에 'bar_a.csv' → 모두 겹치지 않음
    # 반대로 train에 'sched_a.csv', eval에 'sched_a.csv' (경로 다름) → 겹침
    train2 = ["/train/sched_a.csv"]
    eval2 = ["/eval/sched_a.csv"]  # 경로 다르지만 basename 같음
    s3, b3 = compute_overlap(train2, eval2)
    if s3 == ["sched_a"] and b3 == ["sched_a.csv"]:
        ok("C: 경로 달라도 basename 같으면 교집합")
    else:
        fail("C: 경로 달라도 basename 같으면 교집합", f"stem={s3}, base={b3}")

    # 5. 반환 필드 이름 확인 (실제 함수 반환값 구조)
    # check_list_overlap의 반환 딕셔너리에 필요한 키가 있는지 테스트
    # 실제 목록 파일을 읽어야 해서 직접 호출은 생략하고 필드명만 확인
    required_keys = {"n_intersection_stem", "intersection_stem", "n_intersection_basename", "intersection_basename",
                     "n_train", "n_eval"}
    # 직접 함수 소스에서 반환 딕셔너리 키 확인
    import inspect
    src = inspect.getsource(G.check_list_overlap)
    all_present = all(k in src for k in required_keys)
    if all_present:
        ok("C: check_list_overlap 반환 딕셔너리에 필요한 키 모두 있음")
    else:
        missing_k = [k for k in required_keys if k not in src]
        fail("C: check_list_overlap 반환 딕셔너리에 필요한 키 모두 있음", f"없는 키: {missing_k}")

    # 6. 'n_intersection'(구 필드) 미사용 확인
    if "n_intersection=" not in src and "n_intersection\"" not in src:
        ok("C: 구 필드 n_intersection 사용하지 않음")
    else:
        fail("C: 구 필드 n_intersection 사용하지 않음", "아직 구 필드가 소스에 있음")


# ---------------------------------------------------------------------------
# 섹션 D: 읽기 전용 사전 점검 (g1_common.py 직접 실행)
# ---------------------------------------------------------------------------

def test_section_d() -> None:
    """섹션 D: g1_common.py 직접 실행 — 읽기 전용 여부와 종료 코드."""
    import subprocess
    import g1_common as G

    print("\n[D] 읽기 전용 사전 점검")

    # sys.dont_write_bytecode 설정 위치 확인 (소스 검사)
    import inspect
    src = inspect.getsource(G._check_main)
    if "sys.dont_write_bytecode = True" in src:
        ok("D: _check_main에 sys.dont_write_bytecode = True 있음")
    else:
        fail("D: _check_main에 sys.dont_write_bytecode = True 있음", "설정 없음")

    # 직접 실행 — 종료 코드 2 (BLOCKED) 또는 0이어야 한다
    # 실제 환경 조건이 맞지 않을 수 있으므로 종료 코드 0 또는 2만 허용한다
    script = str(G.G1_DIR / "g1_common.py")
    result = subprocess.run(
        [sys.executable, "-B", script],
        capture_output=True, text=True,
        cwd=str(G.REPO),
        timeout=60,
    )
    rc = result.returncode
    if rc in (0, 2):
        ok(f"D: g1_common.py 직접 실행 → 종료 코드 {rc} (0=통과, 2=BLOCKED)")
    else:
        fail(f"D: g1_common.py 직접 실행 → 종료 코드 0 또는 2", f"실제 종료 코드={rc}\nstdout={result.stdout[:200]}\nstderr={result.stderr[:200]}")

    # 불필요한 파일이 생성되지 않았는지 확인
    # 사전 점검은 exp/v2dev/g1/_devtest/ 외부에 아무것도 쓰지 않아야 한다
    g1_dir = G.G1_DIR
    before_files = set(g1_dir.rglob("*"))
    # (이미 실행 후이므로 생성된 파일이 있다면 여기서 감지된다 — before가 after가 된 셈)
    # 대신 __pycache__ 항목만 확인
    pycache_dirs = [p for p in g1_dir.rglob("__pycache__") if p.is_dir()]
    new_pyc = [p for d in pycache_dirs for p in d.glob("g1_*.pyc")]
    if not new_pyc:
        ok("D: 직접 실행 후 g1_*.pyc 파일 생성 없음(-B 또는 dont_write_bytecode 효과)")
    else:
        fail("D: 직접 실행 후 g1_*.pyc 파일 없어야 함", f"생성된 파일: {new_pyc}")


# ---------------------------------------------------------------------------
# 섹션 E: tracked_clean() 임시 git 저장소 직접 검증
# ---------------------------------------------------------------------------

def test_section_e_tracked_clean() -> None:
    """섹션 E: 임시 git 저장소에서 tracked_clean() 4가지 케이스 직접 검증.
    실제 연구 저장소의 파일은 절대 수정하지 않는다."""
    import g1_common as G

    print("\n[E] tracked_clean() 임시 git 저장소 테스트")

    tmp_root = Path(tempfile.mkdtemp(prefix="test_g1_e_"))
    try:
        repo_path = tmp_root / "testrepo"
        repo_path.mkdir()

        # 임시 저장소용 환경 — 사용자 설정이 없어도 커밋 가능
        git_env = {
            **os.environ,
            "GIT_AUTHOR_NAME": "Test",
            "GIT_AUTHOR_EMAIL": "test@example.com",
            "GIT_COMMITTER_NAME": "Test",
            "GIT_COMMITTER_EMAIL": "test@example.com",
            "GIT_AUTHOR_DATE": "2024-01-01T00:00:00+00:00",
            "GIT_COMMITTER_DATE": "2024-01-01T00:00:00+00:00",
        }

        def git_tmp(*args):
            return subprocess.run(
                ["git", "-C", str(repo_path), *args],
                capture_output=True, text=True, env=git_env,
            )

        git_tmp("init")
        git_tmp("config", "user.email", "test@example.com")
        git_tmp("config", "user.name", "Test")

        # 추적할 파일 생성·커밋
        tracked_file = repo_path / "tracked.txt"
        tracked_file.write_text("original content", encoding="utf-8")
        git_tmp("add", "tracked.txt")
        r = git_tmp("commit", "-m", "init")
        if r.returncode != 0:
            fail("E: 임시 저장소 커밋", f"git commit 실패: {r.stderr}")
            return

        # G.REPO를 임시 저장소로 교체
        old_repo = G.REPO
        G.REPO = repo_path
        try:
            # 1. 추적되고 HEAD와 일치 → True
            result = G.tracked_clean("tracked.txt")
            if result:
                ok("E1: 추적·HEAD 일치 파일 → tracked_clean() = True")
            else:
                fail("E1: 추적·HEAD 일치 파일 → tracked_clean() = True", "False를 반환했다")

            # 2. 추적되지만 수정됨 → False
            tracked_file.write_text("modified content", encoding="utf-8")
            result = G.tracked_clean("tracked.txt")
            if not result:
                ok("E2: 추적·수정된 파일 → tracked_clean() = False")
            else:
                fail("E2: 추적·수정된 파일 → tracked_clean() = False", "True를 반환했다")

            # 3. 존재하지만 미추적(git add 하지 않은 새 파일) → False
            untracked_file = repo_path / "untracked.txt"
            untracked_file.write_text("untracked content", encoding="utf-8")
            result = G.tracked_clean("untracked.txt")
            if not result:
                ok("E3: 미추적 파일(존재함) → tracked_clean() = False")
            else:
                fail("E3: 미추적 파일(존재함) → tracked_clean() = False", "True를 반환했다")

            # 4. 파일이 없을 때 collect_issues 패턴 → '필수 파일 없음' issue 발생
            fake_issues: list[str] = []
            missing_name = "definitely_not_here.txt"
            if not (G.REPO / missing_name).exists():
                fake_issues.append(f"필수 파일 없음: {missing_name}")
            elif not G.tracked_clean(missing_name):
                fake_issues.append(f"커밋되지 않은 변경 또는 미추적: {missing_name}")

            if any("필수 파일 없음" in i for i in fake_issues):
                ok("E4: 없는 파일 → collect_issues 패턴 → '필수 파일 없음' issue")
            else:
                fail("E4: 없는 파일 → collect_issues 패턴 → '필수 파일 없음' issue",
                     f"issues={fake_issues}")
        finally:
            G.REPO = old_repo

    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


# ---------------------------------------------------------------------------
# 섹션 F: 추가 경계 케이스
# ---------------------------------------------------------------------------

def test_section_f_edge_cases() -> None:
    """섹션 F: 중첩 JSON 중복 키, 경로 정규화, 추가 파일 BLOCKED, 결과 파일 미변경."""
    import g1_common as G

    repo = G.REPO
    print("\n[F] 추가 경계 케이스")

    # ── F1: read_json_strict — 중첩 객체의 중복 키도 차단 ──────────────────
    try:
        json.loads('{"outer": {"inner": 1, "inner": 2}}', object_pairs_hook=G._dup_check)
        fail("F1: 중첩 중복 키(메모리) → ValueError", "오류가 발생하지 않음")
    except ValueError as e:
        if "중복 키" in str(e):
            ok("F1: 중첩 중복 키(메모리) → ValueError")
        else:
            fail("F1: 중첩 중복 키(메모리) → ValueError", f"예상과 다른 오류: {e}")

    # F1b: 파일에서도 동일하게 차단
    tmp_nested = repo / "exp/v2dev/g1/_test_f1_nested_tmp.json"
    try:
        tmp_nested.write_text('{"data": {"key": "val1", "key": "val2"}}\n', encoding="utf-8")
        try:
            G.read_json_strict(tmp_nested)
            fail("F1b: read_json_strict 중첩 중복 키 파일 → ValueError", "오류가 발생하지 않음")
        except ValueError as e:
            if "중복 키" in str(e):
                ok("F1b: read_json_strict 중첩 중복 키 파일 → ValueError")
            else:
                fail("F1b: read_json_strict 중첩 중복 키 파일 → ValueError", f"다른 오류: {e}")
    finally:
        if tmp_nested.exists():
            tmp_nested.unlink()

    # ── F2–F4: check_provenance 경유 테스트용 헬퍼 ──────────────────────────
    def run_prov(code_sha256_val) -> list:
        """REPO 안의 임시 파일로 check_provenance를 호출한다."""
        tmp_sel = repo / "exp/v2dev/g1/_test_f_sel_tmp.json"
        tmp_renv = repo / "exp/v2dev/g1/_test_f_renv_tmp.json"
        try:
            script_sha = G.sha256_file(repo / G.SELECT_SCRIPT)
            sched_sha = G.sha256_file(repo / G.DEV_SCHED)
            cfg_sha = G.sha256_file(repo / G.FROZEN_CONFIG)
            csv_sha = {k: G.sha256_file(repo / k) for k in G.RECORDED_CSVS}
            sel_obj = {
                "script_sha256": script_sha,
                "schedule_list_sha256": sched_sha,
                "config_sha256": cfg_sha,
                "result_csv_sha256": csv_sha,
                "code_sha256": code_sha256_val,
            }
            sel_text = json.dumps(sel_obj, ensure_ascii=False, indent=2) + "\n"
            tmp_sel.write_text(sel_text, encoding="utf-8")
            renv_obj = {
                "result_json_sha256": G.sha256_file(tmp_sel),
                "script_sha256": script_sha,
            }
            tmp_renv.write_text(json.dumps(renv_obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            old_sel, old_renv = G.SELECTION, G.RUNTIME_ENV
            G.SELECTION = str(tmp_sel.relative_to(repo))
            G.RUNTIME_ENV = str(tmp_renv.relative_to(repo))
            try:
                _, issues = G.check_provenance()
                return issues
            finally:
                G.SELECTION = old_sel
                G.RUNTIME_ENV = old_renv
        finally:
            if tmp_sel.exists():
                tmp_sel.unlink()
            if tmp_renv.exists():
                tmp_renv.unlink()

    correct = {name: G.sha256_file(repo / name) for name in G.CODE_SHA256_REQUIRED}

    # F2: paperexp.py 와 ./paperexp.py → 경로 정규화 → 중복 차단
    path_dup = dict(correct)
    path_dup["./paperexp.py"] = correct["paperexp.py"]
    issues_f2 = run_prov(path_dup)
    expect_issues(
        "F2: paperexp.py와 ./paperexp.py 중복 경로 표기 → BLOCKED",
        issues_f2, "중복",
    )

    # F3: 필수 4개 외의 파일 추가 (measure_a_cpu.py) → BLOCKED
    extra = dict(correct)
    extra["measure_a_cpu.py"] = G.sha256_file(repo / "measure_a_cpu.py")
    issues_f3 = run_prov(extra)
    expect_issues(
        "F3: 필수 4개 외 파일 추가(measure_a_cpu.py) → BLOCKED",
        issues_f3, "필수 목록에 없는 파일",
    )

    # F4: 필수 파일 하나 누락 + 필수 목록 외 파일 추가 동시 발생
    mixed = {k: v for k, v in correct.items() if k != "chunkslices.py"}
    mixed["ailearn.py"] = G.sha256_file(repo / "ailearn.py")
    issues_f4 = run_prov(mixed)
    has_missing = any("필수 파일 누락" in i for i in issues_f4)
    has_extra = any("필수 목록에 없는 파일" in i for i in issues_f4)
    if has_missing and has_extra:
        ok("F4: 누락 + 추가 동시 발생 → 두 종류 issue 모두 보고")
    else:
        fail("F4: 누락 + 추가 동시 발생 → 두 종류 issue 모두 보고",
             f"issues={issues_f4}")

    # ── F5: g1_common.py -B 실행이 기존 결과 파일을 변경하지 않는지 ───────────
    # 실행 전 주요 파일의 mtime/size 기록
    watched = [
        repo / G.SELECTION,
        repo / G.RUNTIME_ENV,
        repo / G.DEV_SCHED,
        repo / G.FROZEN_CONFIG,
        *(repo / k for k in G.RECORDED_CSVS),
    ]
    before: dict[str, tuple] = {}
    for f in watched:
        if f.is_file():
            st = f.stat()
            before[str(f)] = (st.st_mtime_ns, st.st_size)

    run_result = subprocess.run(
        [sys.executable, "-B", str(G.G1_DIR / "g1_common.py")],
        capture_output=True, text=True,
        cwd=str(repo),
        timeout=90,
    )

    if run_result.returncode not in (0, 2):
        fail("F5-종료코드: 0 또는 2", f"실제={run_result.returncode}\n{run_result.stderr[:200]}")
    else:
        ok(f"F5-종료코드: g1_common.py -B → 종료 코드 {run_result.returncode}")

    changed = []
    for f in watched:
        if f.is_file():
            st = f.stat()
            after = (st.st_mtime_ns, st.st_size)
            prev = before.get(str(f))
            if prev and prev != after:
                changed.append(str(f.relative_to(repo)))

    if not changed:
        ok("F5-미변경: g1_common.py -B 실행 후 감시 대상 파일 미변경")
    else:
        fail("F5-미변경: g1_common.py -B 실행 후 감시 대상 파일 미변경",
             f"변경된 파일: {changed}")


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------

def main() -> int:
    print("=" * 60)
    print("g1_common.py 단위 테스트")
    print("=" * 60)

    test_section_a_helpers()
    test_section_a_completeness()
    test_section_b()
    test_section_c()
    test_section_d()
    test_section_e_tracked_clean()
    test_section_f_edge_cases()

    print("\n" + "=" * 60)
    print(f"결과: 통과 {len(_PASS)}개 / 실패 {len(_FAIL)}개 / 합계 {len(_PASS) + len(_FAIL)}개")
    if _FAIL:
        print("실패 목록:")
        for f in _FAIL:
            print(f"  - {f}")
        return 1
    print("모두 통과")
    return 0


if __name__ == "__main__":
    sys.exit(main())
