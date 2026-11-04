"""g1_common.py — G1 저차원 생존 GBDT·조회표의 공통 코드.

계획: docs/additional-analysis-plan-v4.md §2.3·§2.4. 기존 연구 코드(paperexp*.py, chunkslices.py,
measure_a_cpu.py)는 고치지 않고 불러 쓴다.
  - 학습 행·가중치: measure_a_cpu.prepare_pre(= run_frozen.Pretrained)와 같은 순서·정규화
  - 생존 곡선: measure_a_cpu.predict_knots(단조 감소 보정 포함)를 그대로 부른다
  - 기본 반환 시점: paperexp.survival_timeouts(Sk, T*, 1.0)를 그대로 부르고, 반환한 시각을 매듭 배열
    KNOTS = [0, τ1..τ10]의 위치(0~10)로 바꾼다. 반환값이 KNOTS 원소와 정확히 같지 않으면 멈춘다.
진입 스크립트는 numpy·sklearn보다 이 모듈을 먼저 부른다. 이 모듈이 numpy를 부르기 전에 스레드 환경 변수를 1로 정한다.
공식 실행 사전 점검에는 특징 선택 결과의 출처 해시 대조(check_provenance)가 들어 있다.
이 파일을 직접 실행하면 점검만 하고 아무것도 쓰지 않는다: python exp/v2dev/g1/g1_common.py
"""
import hashlib
import io
import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
THREAD_ENV_BEFORE = {k: os.environ.get(k) for k in THREAD_VARS}
if "numpy" not in sys.modules:
    for _k in THREAD_VARS:
        os.environ[_k] = "1"
elif any(os.environ.get(k) != "1" for k in THREAD_VARS):
    raise RuntimeError("numpy가 이미 다른 스레드 설정으로 불렸다. g1_common을 numpy보다 먼저 부른다")

import numpy as np  # noqa: E402

G1_DIR = Path(__file__).resolve().parent
REPO = G1_DIR.parents[2]  # exp/v2dev/g1 → 저장소 루트

TSTAR = 2.6
FEATS = "f15"
SEED = 0
BRANCH = "exp/a-cpu"
HGB_PARAMS = dict(max_iter=120, learning_rate=0.1, max_leaf_nodes=31, l2_regularization=1.0, random_state=SEED)
CANDIDATES = {"P1": dict(n_feat=2, max_bins=255), "P1b": dict(n_feat=3, max_bins=64)}
EXPECTED_ENV = {"python": "3.12.3", "numpy": "2.5.4", "scipy": "1.18.1", "sklearn": "1.9.1"}
EXPECTED_AFFINITY = [1]
COMPILE_BATCH = 65_536  # 표 생성 묶음 크기(실행 전 고정)
VERIFY_BATCH = 4_099  # 전 조합 검증 묶음 크기(생성과 다르게 둔다: 묶음 크기와 무관함을 함께 확인)
N_DEV = 26
N_EVAL = 420
N_TRAIN_EXPECTED = 820_340  # 작업 A 사전 학습 조각 수(a-cpu-results §3.2)
N_TE_TOTAL = 2_003_785      # 평가 창 전체 TE 조각 수 (exp/v2dev/cpu_a/pre_sim_costs.csv 합계)

DEV_SCHED = "exp/dev_sched.txt"
EVAL_LIST = "exp/v2dev/daypool/all.txt"
CPU_A_ENV = "exp/v2dev/cpu_a/environment.json"
PRE_SIM_COSTS = "exp/v2dev/cpu_a/pre_sim_costs.csv"
PLAN = "docs/additional-analysis-plan-v4.md"
SELECTION = "exp/v2dev/g1/feature-selection.json"
RUNTIME_ENV = "exp/v2dev/g1/runtime-environment.json"
FROZEN_CODE = ["paperexp.py", "paperexp2.py", "paperexp3.py", "chunkslices.py", "policysim.py", "ailearn.py",
               "measure_a_cpu.py", "g1_select_features.py"]
SELECT_SCRIPT = "g1_select_features.py"  # 특징 선택 스크립트(저장소 루트)
# g1_select_features.py가 code_sha256에 기록하는 파일 목록(feature-selection.json의 필수 집합)
CODE_SHA256_REQUIRED = ["paperexp.py", "paperexp2.py", "paperexp3.py", "chunkslices.py"]
FROZEN_CONFIG = "exp/frozen.json"
RECORDED_CSVS = ["exp/v2dev/chunk/res_q1.csv", "exp/v2dev/chunk/res_q2.csv"]  # 특징 선택 검증 B의 기록 파일
G1_CODE = ["exp/v2dev/g1/g1_common.py", "exp/v2dev/g1/g1_build_models.py", "exp/v2dev/g1/g1_compile_tables.py",
           "exp/v2dev/g1/g1_verify_tables.py", "exp/v2dev/g1/g1_simulate.py"]


# ---------------------------------------------------------------- 경로·해시·원자적 저장
def rel(p):
    p = Path(p).resolve()
    try:
        return str(p.relative_to(REPO))
    except ValueError:
        return str(p)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def sha256_array(a):
    """dtype·shape·C 순서 바이트의 해시"""
    a = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(f"{a.dtype.str}|{a.shape}|".encode())
    h.update(a.tobytes())
    return h.hexdigest()


def _guard(path, force):
    path = Path(path)
    if path.exists() and not force:
        raise SystemExit(f"이미 있다(조용히 덮어쓰지 않는다): {rel(path)} sha256={sha256_file(path)}. "
                         "확인한 뒤 다시 만들 때만 --force를 준다.")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _atomic_replace(path, writer):
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with open(tmp, "wb") as f:
            writer(f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def write_json(path, obj, force=False):
    path = _guard(path, force)
    data = (json.dumps(obj, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    _atomic_replace(path, lambda f: f.write(data))
    return sha256_file(path)


def write_text(path, text, force=False):
    path = _guard(path, force)
    data = text.encode("utf-8")
    _atomic_replace(path, lambda f: f.write(data))
    return sha256_file(path)


def write_npy(path, arr, force=False):
    path = _guard(path, force)
    buf = io.BytesIO()
    np.save(buf, arr, allow_pickle=False)
    _atomic_replace(path, lambda f: f.write(buf.getvalue()))
    return sha256_file(path)


def write_joblib(path, obj, force=False):
    import joblib
    path = _guard(path, force)
    buf = io.BytesIO()
    joblib.dump(obj, buf)
    _atomic_replace(path, lambda f: f.write(buf.getvalue()))
    return sha256_file(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _dup_check(pairs):
    d = {}
    for k, v in pairs:
        if k in d:
            raise ValueError(f"JSON 중복 키: {k!r}")
        d[k] = v
    return d


def read_json_strict(path):
    """JSON을 읽되 같은 수준의 중복 키는 오류로 처리한다(출처 해시 검사에 사용)."""
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=_dup_check)


MANIFEST_NAME = "model-table-manifest.json"


def load_manifest(out):
    p = Path(out) / MANIFEST_NAME
    return read_json(p) if p.exists() else dict(schema="g1-model-table-manifest/1")


def save_manifest(out, man):
    """매니페스트는 단계마다 자기 절만 더해 원자적으로 다시 쓴다(단계 절의 덮어쓰기는 각 단계가 막는다)"""
    p = Path(out) / MANIFEST_NAME
    p.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(man, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    _atomic_replace(p, lambda f: f.write(data))
    return sha256_file(p)


def check_sha(path, expected, what):
    got = sha256_file(path)
    if got != expected:
        raise SystemExit(f"{what} 해시가 매니페스트와 다르다: {rel(path)} {got} ≠ {expected}")
    return got


def jsonable(d):
    out = {}
    for k, v in d.items():
        out[k] = v if isinstance(v, (int, float, str, bool, type(None))) else repr(v)
    return out


# ---------------------------------------------------------------- 실행 환경·Git 점검
def git(*args):
    r = subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True)
    return r.returncode, r.stdout.strip()


def git_state():
    _, head = git("rev-parse", "HEAD")
    _, branch = git("rev-parse", "--abbrev-ref", "HEAD")
    _, dirty = git("status", "--porcelain")
    return dict(head=head, branch=branch, dirty=dirty.splitlines() if dirty else [])


def tracked_clean(path):
    """Git이 추적하고 HEAD와 같으면 True"""
    rc1, _ = git("ls-files", "--error-unmatch", "--", path)
    rc2, _ = git("diff", "--quiet", "HEAD", "--", path)
    return rc1 == 0 and rc2 == 0


def runtime_info():
    import scipy
    import sklearn
    import joblib
    from threadpoolctl import threadpool_info
    cpu = ""
    try:
        for line in open("/proc/cpuinfo", encoding="utf-8"):
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    pools = [dict(user_api=p.get("user_api"), internal_api=p.get("internal_api"), num_threads=p.get("num_threads"),
                  filepath=p.get("filepath")) for p in threadpool_info()]
    return dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                sklearn=sklearn.__version__, joblib=joblib.__version__, executable=sys.executable,
                platform=platform.platform(), cpu=cpu, affinity=sorted(os.sched_getaffinity(0)),
                thread_env={k: os.environ.get(k) for k in THREAD_VARS}, thread_env_before=THREAD_ENV_BEFORE,
                threadpools=pools)


HEX64 = re.compile(r"^[0-9a-f]{64}$")
_MISSING = object()  # 필드가 없거나 모호해 이미 어긋남으로 적은 경우


def _find_key(obj, key, path="$"):
    """JSON 안의 모든 위치에서 이름이 key인 필드를 찾는다(중첩 위치를 가정하지 않는다)"""
    hits = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{path}.{k}"
            if k == key:
                hits.append((p, v))
            hits += _find_key(v, key, p)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            hits += _find_key(v, key, f"{path}[{i}]")
    return hits


def _field(obj, key, where, issues):
    """필드 하나: 없거나, 같은 이름이 서로 다른 값으로 여러 곳에 있으면 어긋남으로 기록한다"""
    hits = _find_key(obj, key)
    if not hits:
        issues.append(f"provenance: {where}에 '{key}' 필드가 없다")
        return _MISSING, None
    if len({json.dumps(v, sort_keys=True, ensure_ascii=False) for _, v in hits}) > 1:
        issues.append(f"provenance: {where}의 '{key}'가 서로 다른 값으로 {len(hits)}곳에 있다: {[p for p, _ in hits]}")
        return _MISSING, None
    return hits[0][1], hits[0][0]


def _entries(val, label, issues):
    """{경로: sha256} 또는 [{path|file: 경로, sha256: 값}] → [(경로, sha256)]. 다른 형식은 추측하지 않는다."""
    if isinstance(val, dict) and val and all(isinstance(v, str) for v in val.values()):
        return list(val.items())
    if isinstance(val, list) and val and all(isinstance(e, dict) for e in val):
        out = []
        for e in val:
            p, s = e.get("path", e.get("file")), e.get("sha256")
            if not isinstance(p, str) or not isinstance(s, str):
                issues.append(f"provenance: {label} 항목 형식을 알 수 없다(키 {list(e)})")
                return []
            out.append((p, s))
        return out
    issues.append(f"provenance: {label} 형식을 알 수 없거나 비어 있다({type(val).__name__})")
    return []


def _resolve(key, label, issues):
    """기록된 경로를 저장소 안의 실제 파일로 해석한다. 다른 이름으로 바꿔 맞추지 않는다."""
    p = Path(key)
    cand = (p if p.is_absolute() else REPO / p).resolve()
    if not cand.is_file() or REPO not in cand.parents:
        issues.append(f"provenance: {label}의 경로 {key!r}를 저장소 안 파일로 해석할 수 없다")
        return None
    return rel(cand)


def _compare(label, recorded, path, rows, issues):
    if path is None or recorded is _MISSING:
        return
    if not isinstance(recorded, str) or not HEX64.match(recorded):
        issues.append(f"provenance: {label} 기록값이 SHA-256 16진 문자열이 아니다: {recorded!r}")
        return
    if not (REPO / path).is_file():
        issues.append(f"provenance: {label} 대상 파일이 없다: {path}")
        return
    cur = sha256_file(REPO / path)
    rows.append(dict(check=label, path=path, recorded=recorded, current=cur, ok=cur == recorded))
    if cur != recorded:
        issues.append(f"provenance: {label} 해시 불일치 {path}: 기록 {recorded[:12]}… ≠ 현재 {cur[:12]}…")


def check_provenance():
    """특징 선택 결과의 출처 해시를 현재 파일과 대조한다(읽기만 한다).
    feature-selection.json: script_sha256, schedule_list_sha256, config_sha256, code_sha256, result_csv_sha256
    runtime-environment.json: result_json_sha256(= feature-selection.json), script_sha256(= g1_select_features.py)
    필드는 이름으로 찾고 위치(중첩)는 가정하지 않는다. 없거나 모호하거나 다르면 어긋남이다.
    code_sha256에는 CODE_SHA256_REQUIRED의 파일이 정확히 모두 있어야 한다(누락·추가·중복 모두 차단)."""
    issues, rows = [], []
    sel_p, env_p = REPO / SELECTION, REPO / RUNTIME_ENV
    not_found = [rel(p) for p in (sel_p, env_p) if not p.is_file()]
    if not_found:
        issues.append(f"provenance 검사 불가: 파일 없음 {not_found}")
        return dict(checks=rows, fields={}), issues
    try:
        sel = read_json_strict(sel_p)
    except ValueError as e:
        issues.append(f"provenance: {SELECTION} JSON 중복 키 감지: {e}")
        return dict(checks=rows, fields={}), issues
    try:
        renv = read_json_strict(env_p)
    except ValueError as e:
        issues.append(f"provenance: {RUNTIME_ENV} JSON 중복 키 감지: {e}")
        return dict(checks=rows, fields={}), issues
    fields = {}

    def one(obj, key, where):
        val, at = _field(obj, key, where, issues)
        if at is not None:
            fields[f"{where}:{key}"] = at
        return val

    _compare("selection.script_sha256", one(sel, "script_sha256", "feature-selection.json"), SELECT_SCRIPT, rows, issues)
    _compare("selection.schedule_list_sha256", one(sel, "schedule_list_sha256", "feature-selection.json"), DEV_SCHED,
             rows, issues)
    _compare("selection.config_sha256", one(sel, "config_sha256", "feature-selection.json"), FROZEN_CONFIG, rows, issues)
    code = one(sel, "code_sha256", "feature-selection.json")
    if code is not _MISSING:
        entries = _entries(code, "code_sha256", issues)
        if entries:
            # 경로 정규화 후 중복 탐지
            seen_paths = {}  # canonical_path → (original_key, sha)
            for key, sha in entries:
                r = _resolve(key, "code_sha256", issues)
                if r is None:
                    continue
                if r in seen_paths:
                    orig_key = seen_paths[r][0]
                    issues.append(
                        f"provenance: code_sha256에 같은 파일이 중복 기록됐다: {key!r}와 {orig_key!r} → {r}"
                    )
                else:
                    seen_paths[r] = (key, sha)
            # 필수 파일 집합과 기록된 집합의 완전성 검사
            required = set(CODE_SHA256_REQUIRED)
            recorded = set(seen_paths.keys())
            missing_code = sorted(required - recorded)
            extra_code = sorted(recorded - required)
            if missing_code:
                issues.append(f"provenance: code_sha256에서 필수 파일 누락: {missing_code}")
            if extra_code:
                issues.append(f"provenance: code_sha256에 필수 목록에 없는 파일 추가: {extra_code}")
            for path in sorted(seen_paths.keys()):
                orig_key, sha = seen_paths[path]
                _compare(f"selection.code_sha256[{orig_key}]", sha, path, rows, issues)
    res = one(sel, "result_csv_sha256", "feature-selection.json")
    if res is not _MISSING:
        seen = []
        for key, sha in _entries(res, "result_csv_sha256", issues):
            path = _resolve(key, "result_csv_sha256", issues)
            seen.append(path)
            _compare(f"selection.result_csv_sha256[{key}]", sha, path, rows, issues)
        if sorted(p for p in seen if p) != sorted(RECORDED_CSVS):
            issues.append(f"provenance: result_csv_sha256가 {RECORDED_CSVS} 두 파일을 정확히 가리키지 않는다: {seen}")
    _compare("runtime_env.result_json_sha256", one(renv, "result_json_sha256", "runtime-environment.json"), SELECTION,
             rows, issues)
    _compare("runtime_env.script_sha256", one(renv, "script_sha256", "runtime-environment.json"), SELECT_SCRIPT, rows,
             issues)
    return dict(checks=rows, fields=fields, n_checks=len(rows), n_ok=sum(r["ok"] for r in rows)), issues


def check_list_overlap(ma, eval_list=EVAL_LIST):
    """학습 26일정과 평가 창의 중복 검사. measure_a_cpu와 같은 규칙: read_list(파일 존재, 목록 안 이름 중복 금지),
    name_of(파일 이름에서 .csv를 뗀 것)의 교집합 금지. 확장자를 포함한 파일 이름 교집합도 별도로 기록한다.
    두 종류의 교집합 중 하나라도 있으면 공식 검증은 BLOCKED 처리한다."""
    train = ma.read_list(DEV_SCHED, "사전 학습원")
    evals = ma.read_list(eval_list, "평가 창")
    inter_stem = sorted({ma.name_of(p) for p in train} & {ma.name_of(p) for p in evals})
    inter_base = sorted({os.path.basename(p) for p in train} & {os.path.basename(p) for p in evals})
    return dict(rule="measure_a_cpu.read_list + name_of 교집합(사전 학습원과 대상 창은 겹치지 않는다)",
                train_list=DEV_SCHED, train_list_sha256=sha256_file(REPO / DEV_SCHED), n_train=len(train),
                eval_list=eval_list, eval_list_sha256=sha256_file(REPO / eval_list), n_eval=len(evals),
                n_intersection_stem=len(inter_stem), intersection_stem=inter_stem[:20],
                n_intersection_basename=len(inter_base), intersection_basename=inter_base[:20])


def collect_issues():
    """공식 실행 조건 점검(읽기만 한다). 스레드 풀이 1이 아니면 바로 멈춘다. 반환: (환경 기록, 어긋남 목록)"""
    info = runtime_info()
    issues = []
    for k, v in EXPECTED_ENV.items():
        if info[k] != v:
            issues.append(f"{k} {info[k]} ≠ {v}")
    if info["affinity"] != EXPECTED_AFFINITY:
        issues.append(f"CPU affinity {info['affinity']} ≠ {EXPECTED_AFFINITY}(taskset -c 1)")
    bad_pool = [p for p in info["threadpools"] if p["num_threads"] != 1]
    if bad_pool:  # 스레드 수는 개발 시험에서도 1이어야 한다
        raise SystemExit(f"스레드 풀이 1이 아니다: {bad_pool}")
    gs = git_state()
    if gs["branch"] != BRANCH:
        issues.append(f"브랜치 {gs['branch']} ≠ {BRANCH}")
    for p in [SELECTION, RUNTIME_ENV]:
        if not (REPO / p).is_file():
            issues.append(f"없음: {p}")
        elif not tracked_clean(p):
            issues.append(f"커밋 필요(추적되지 않거나 HEAD와 다르다): {p}")
    for p in FROZEN_CODE + [PLAN, DEV_SCHED, EVAL_LIST, FROZEN_CONFIG] + RECORDED_CSVS:
        if not (REPO / p).exists():
            issues.append(f"필수 파일 없음: {p}")
        elif not tracked_clean(p):
            issues.append(f"커밋되지 않은 변경 또는 미추적: {p}")
    for p in G1_CODE:
        if not tracked_clean(p):
            issues.append(f"G1 코드 커밋 필요(계획 v4: 실행 전 코드 지문 커밋): {p}")
    prov, prov_issues = check_provenance()
    issues += prov_issues
    return dict(runtime=info, git=gs, provenance=prov), issues


def preflight(args, stage):
    """공식 실행의 사전 점검. --dev-test면 어긋남을 기록만 하고 계속한다(결과는 _devtest/로 간다).
    반환: (환경 기록, 어긋남 목록)"""
    env, issues = collect_issues()
    if issues and not args.dev_test:
        msg = "\n  - ".join(issues)
        raise SystemExit(f"[{stage}] 공식 실행 조건을 채우지 못했다(BLOCKED):\n  - {msg}\n"
                         "개발 시험만 하려면 --dev-test(결과는 exp/v2dev/g1/_devtest/에 쓴다).")
    return env, issues


def out_dir(args):
    """공식 결과는 exp/v2dev/g1/, 개발 시험 결과는 exp/v2dev/g1/_devtest/"""
    return G1_DIR / "_devtest" if args.dev_test else G1_DIR


def add_common_args(ap):
    ap.add_argument("--candidates", default="P1,P1b", help="쉼표로 구분한 후보(P1, P1b)")
    ap.add_argument("--force", action="store_true", help="이미 있는 결과를 다시 만든다(확인한 뒤에만)")
    ap.add_argument("--dev-test", action="store_true",
                    help="개발 시험: 환경·Git 어긋남을 기록만 하고 계속한다. 결과는 _devtest/에 쓰며 공식 결과가 아니다")
    ap.add_argument("--selection", default=SELECTION, help="특징 선택 파일(개발 시험에서만 바꾼다)")


def parse_candidates(s):
    cands = [c.strip() for c in s.split(",") if c.strip()]
    bad = [c for c in cands if c not in CANDIDATES]
    if bad or not cands:
        raise SystemExit(f"알 수 없는 후보: {bad or s} (가능: {list(CANDIDATES)})")
    return cands


# ---------------------------------------------------------------- 연구 코드 불러오기
_CHUNK_ON = False


def research_modules():
    """paperexp·paperexp2·paperexp3·measure_a_cpu를 부르고 1 MiB 청크 모델을 한 번만 켠다"""
    global _CHUNK_ON
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    import paperexp as pe
    import paperexp2 as p2
    import paperexp3 as p3
    import measure_a_cpu as ma
    if not _CHUNK_ON:
        p3.use_chunk_model()
        _CHUNK_ON = True
    return pe, p2, p3, ma


def knots(pe):
    return np.concatenate([[0.0], pe.TAU])


# ---------------------------------------------------------------- 특징 선택
def load_selection(path, feat_names):
    """feature-selection.json의 P1·P1b 특징을 읽고 Trace2.feat_names와 대조한다. 순서는 파일 순서를 따른다."""
    obj = read_json(REPO / path)
    out = {}
    for cand, spec in CANDIDATES.items():
        ent = obj.get(cand)
        if not isinstance(ent, dict):
            raise SystemExit(f"{path}에 {cand} 항목이 없다(최상위 키: {list(obj)})")
        idx = next((ent[k] for k in ("feature_indices", "index", "indices") if k in ent), None)
        names = ent.get("features")
        if idx is None or names is None:
            raise SystemExit(f"{path}의 {cand}에 특징 이름·인덱스가 없다(키: {list(ent)})")
        idx = [int(i) for i in idx]
        if len(idx) != spec["n_feat"] or len(set(idx)) != len(idx):
            raise SystemExit(f"{cand} 특징 수가 계획({spec['n_feat']})과 다르거나 중복이다: {idx}")
        actual = [feat_names[i] for i in idx]
        if actual != list(names):
            raise SystemExit(f"{cand} 특징 이름·인덱스 불일치: 파일 {names} vs Trace2 {actual}")
        out[cand] = dict(features=list(names), indices=idx)
    return out, sha256_file(REPO / path)


# ---------------------------------------------------------------- 학습 행(사전 학습 C3와 같다)
def load_pretrain_rows(pe, p2, ma):
    """measure_a_cpu.prepare_pre와 같은 학습원·순서·가중치: 목록 순서대로 각 일정의 TR 행(원래 순서)을 붙이고,
    가중치는 일정마다 h / h.sum() × 1000이다. 원시 15특징 전부를 돌려준다(격자·학습은 선택 열만 쓴다)."""
    paths = ma.read_list(DEV_SCHED, "사전 학습원")
    if len(paths) != N_DEV:
        raise SystemExit(f"개발 일정이 {N_DEV}개가 아니다: {len(paths)}")
    Xs, gs, ws, ids, per = [], [], [], [], []
    feat_names = None
    for si, p in enumerate(paths):
        t = p2.Trace2(p, FEATS)
        if feat_names is None:
            feat_names = list(t.feat_names)
        elif list(t.feat_names) != feat_names:
            raise SystemExit(f"특징 이름·순서가 일정마다 다르다: {p}")
        m = t.TR
        n = int(m.sum())
        per.append(dict(schedule=rel(p), sha256=sha256_file(p), n_tr=n, n_all=int(len(m)),
                        weight_sum=float((t.h[m] / t.h[m].sum() * 1000).sum()) if n else 0.0))
        if n == 0:
            continue
        Xs.append(t.X[m])
        gs.append(t.gtr[m])
        ws.append(t.h[m] / t.h[m].sum() * 1000)
        ids.append(np.column_stack([np.full(n, si, dtype=np.int64), np.where(m)[0].astype(np.int64)]))
    X, g, w, row_ids = np.vstack(Xs), np.concatenate(gs), np.concatenate(ws), np.vstack(ids)
    if len(g) != N_TRAIN_EXPECTED:
        raise SystemExit(f"학습 조각 수 {len(g)} ≠ 작업 A 사전 학습 {N_TRAIN_EXPECTED}")
    return dict(X=X, g=g, w=w, row_ids=row_ids, feat_names=feat_names, per_schedule=per,
                list_sha256=sha256_file(REPO / DEV_SCHED))


# ---------------------------------------------------------------- 공통 분위 격자와 정수 코드
def build_grid(values, max_bins):
    """계획 v4 §2.4: TR 유한값 전체, q = k/max_bins (k = 1..max_bins−1), numpy.quantile(method="linear"),
    정렬 후 중복 제거. ±Inf가 있으면 멈춘다. NaN은 분위 계산에서 뺀다(결측 코드로 간다)."""
    v = np.asarray(values, dtype=np.float64)
    if np.isinf(v).any():
        raise SystemExit(f"학습 값에 ±Inf가 {int(np.isinf(v).sum())}개 있다(격자 생성 중단)")
    fin = v[np.isfinite(v)]
    q = np.arange(1, max_bins, dtype=np.float64) / max_bins
    raw = np.quantile(fin, q, method="linear")
    cut = np.unique(raw)  # 정렬·중복 제거
    if len(cut) + 1 > max_bins:
        raise SystemExit("코드 수가 max_bins를 넘는다")
    return cut, dict(n_values=int(len(v)), n_nan=int(np.isnan(v).sum()), n_finite=int(len(fin)),
                     n_cut_raw=int(len(raw)), n_cut=int(len(cut)), n_dup_removed=int(len(raw) - len(cut)),
                     n_codes=int(len(cut) + 1), train_min=float(fin.min()), train_max=float(fin.max()))


def encode(x, cut, sentinel):
    """원시 특징 → 정수 코드. searchsorted(side="right"): cutpoint와 같은 값은 위쪽 칸. NaN → sentinel. ±Inf는 오류."""
    x = np.asarray(x, dtype=np.float64)
    if np.isinf(x).any():
        raise ValueError(f"입력에 ±Inf가 {int(np.isinf(x).sum())}개 있다")
    c = np.searchsorted(cut, x, side="right").astype(np.int16)
    c[np.isnan(x)] = sentinel
    return c


def encode_matrix(Xraw, grid):
    """선택 열(grid['indices'] 순서)을 코드 행렬(n × k, int16)로"""
    cols = [encode(Xraw[:, fi], np.asarray(cut), grid["sentinel"])
            for fi, cut in zip(grid["indices"], grid["cutpoints"])]
    return np.column_stack(cols)


def model_input(codes, sentinel):
    """sklearn 입력: 유효 코드는 그 정수 값, 결측 sentinel은 실제 NaN(sentinel 정수를 넣지 않는다)"""
    Xm = codes.astype(np.float64)
    Xm[codes == sentinel] = np.nan
    return Xm


def grid_from_json(obj):
    g = dict(obj)
    g["cutpoints"] = [np.asarray(c, dtype=np.float64) for c in obj["cutpoints"]]
    return g


# ---------------------------------------------------------------- 예측 → 기본 반환 시점 색인
def predict_index(pe, ma, models, Xm):
    """measure_a_cpu.predict_knots(10개 매듭, 단조 감소 보정) → paperexp.survival_timeouts(T*, κ = 1)
    → KNOTS 위치. 반환값이 KNOTS 원소와 정확히 같지 않으면 멈춘다."""
    Sk = ma.predict_knots(models, Xm)
    if not (np.diff(Sk, axis=1) <= 0).all():
        raise SystemExit("생존 확률이 단조 감소가 아니다")
    rec = pe.survival_timeouts(Sk, TSTAR, 1.0)
    K = knots(pe)
    idx = np.searchsorted(K, rec)
    ok = (idx < len(K)) & (K[np.minimum(idx, len(K) - 1)] == rec)
    if not ok.all():
        raise SystemExit(f"survival_timeouts 반환값이 11개 선택지(KNOTS)와 다르다: {np.unique(rec[~ok])[:5]}")
    return idx.astype(np.uint8)


def table_shape(grid):
    return tuple([grid["sentinel"] + 1] * len(grid["indices"]))


def all_codes(shape, start, stop):
    """C 순서 평탄 위치 [start, stop)의 코드 조합(n × k)"""
    flat = np.arange(start, stop, dtype=np.int64)
    return np.column_stack(np.unravel_index(flat, shape)).astype(np.int16)


class Timer:
    def __enter__(self):
        self.c0, self.w0 = time.process_time_ns(), time.perf_counter_ns()
        return self

    def __exit__(self, *exc):
        self.cpu_s = (time.process_time_ns() - self.c0) / 1e9
        self.wall_s = (time.perf_counter_ns() - self.w0) / 1e9


# ---------------------------------------------------------------- 점검만 하는 진입점
def _check_main():
    import argparse
    sys.dont_write_bytecode = True  # 점검만 하므로 연구 모듈 import 시 __pycache__ 작성을 막는다
    ap = argparse.ArgumentParser(
        description="공식 실행 전 점검만 한다: 실행 환경, Git, 특징 선택 provenance 해시, 학습·평가 목록 중복. "
                    "학습·표 생성·검증을 하지 않고 아무 파일도 쓰지 않는다. 통과하면 0, BLOCKED면 2로 끝난다.")
    ap.parse_args()
    env, issues = collect_issues()
    ov = None
    try:
        _, _, _, ma = research_modules()
        ov = check_list_overlap(ma)
        if ov["n_train"] != N_DEV or ov["n_eval"] != N_EVAL:
            issues.append(f"목록 크기: 학습 {ov['n_train']}개(기대 {N_DEV}), 평가 {ov['n_eval']}개(기대 {N_EVAL})")
        has_overlap = ov["n_intersection_stem"] > 0 or ov["n_intersection_basename"] > 0
        if has_overlap:
            parts = []
            if ov["n_intersection_stem"]:
                parts.append(f"일정이름 {ov['n_intersection_stem']}개(예: {ov['intersection_stem'][:3]})")
            if ov["n_intersection_basename"]:
                parts.append(f"파일이름 {ov['n_intersection_basename']}개(예: {ov['intersection_basename'][:3]})")
            issues.append("학습·평가 일정이 겹친다: " + ", ".join(parts))
    except SystemExit as e:
        issues.append(f"목록 검사 불가: {e}")
    for r in env["provenance"]["checks"]:
        print(f"  {'OK' if r['ok'] else 'X '} {r['check']} ({r['path']})")
    if ov is not None:
        print(f"  학습 {ov['n_train']}개, 평가 {ov['n_eval']}개, "
              f"교집합(일정이름) {ov['n_intersection_stem']}개 / (파일이름) {ov['n_intersection_basename']}개")
    if issues:
        print("BLOCKED:\n  - " + "\n  - ".join(issues))
        return 2
    print("점검 통과: 공식 실행 조건을 채웠다")
    return 0


if __name__ == "__main__":
    sys.exit(_check_main())
