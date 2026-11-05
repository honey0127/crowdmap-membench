"""g2_common.py — G2 공통 상수·이진 형식 (G2W2 / G2MB).

G2: P1b 조회표 기반 C3 경로의 C++ 대응 구현 및 CPU 비용 대칭·측정.

형식 변경 이력
  G2W1 → G2W2 (Part E 경계 수정):
    제거: T_base, T_rb(입력), Tk(입력) — C++가 직접 계산
    추가: X_p1b[n,3] — C++가 encode + lookup + κ-scale
    추가: T_base_py 기준선 — parity 검증용
    모델 아티팩트(cuts, table, KNOTS)를 별도 G2MB 파일로 분리
"""
import struct
import numpy as np

# ---------------------------------------------------------------- 경로·후보
BINARY_CAND = "P1b"
MiB = 1024.0 * 1024.0

# ---------------------------------------------------------------- 알고리즘 상수
TSTAR = 2.6
CAP   = 52.0           # min(300.0, 20.0 * TSTAR)
W_CAL = 900.0
U_PERIOD = 30.0
W_FB  = 300.0
K_SIZE = 9
KGRID = [0.25, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0, 2.8, 4.0]
KNOTS = [0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0]
MIN_FRAC = 0.02
MIN_N = 20
INITIAL_K = 4          # KGRID[4] = 1.0
TABLE_DIM = 65         # 64 bins + 1 sentinel
TABLE_SIZE = TABLE_DIM ** 3   # 274,625

# ---------------------------------------------------------------- G2W2 창 형식
MAGIC_WIN = b"G2W2"
FMT_WIN_HDR = "<4sIIIddddd"
# magic(4s), n(I), G_r(I), n_grid(I), b(d), cap(d), W_cal(d), U(d), W_fb(d)
WIN_HDR_SIZE = struct.calcsize(FMT_WIN_HDR)  # 52 bytes

# 배열 순서 (헤더 뒤):
#   float64[n]: g, h
#   uint8[n]:   r
#   float64[n]: ts
#   uint8[n]:   dec_mask, te_mask
#   int32[n]:   grp_r
#   float64[n_grid]: grid
#   float64[n*3]: X_p1b (row-major)
# Python 기준선:
#   float64[n]: T_base_py, T_rule_py, T_rb_py, T_ai_py, T_ai_fb_py, T_c3_py
#   double:     R_py, H_py

def write_window(path, n, G_r, n_grid, b, cap, W_cal, U, W_fb,
                 g, h, r, ts, dec_mask, te_mask, grp_r,
                 grid, X_p1b,
                 T_base, T_rule, T_rb, T_ai, T_ai_fb, T_c3, R, H):
    """G2W2 창 이진 파일을 쓴다.

    X_p1b: shape (n, 3) float64 — P1b 원시 특징값 (encode + lookup은 C++에서)
    T_base, T_rule, T_rb, T_ai, T_ai_fb, T_c3: Python 기준선 (parity 검증용)
    """
    def asf64(a): return np.asarray(a, dtype=np.float64)
    def asui8(a): return np.asarray(a, dtype=np.uint8)
    def asi32(a): return np.asarray(a, dtype=np.int32)

    hdr = struct.pack(FMT_WIN_HDR, MAGIC_WIN, n, G_r, n_grid, b, cap, W_cal, U, W_fb)
    xp = np.asarray(X_p1b, dtype=np.float64)  # (n, 3), C 순서로 저장
    if xp.shape != (n, 3):
        raise ValueError(f"X_p1b shape 오류: {xp.shape}, 기대 ({n}, 3)")

    with open(path, "wb") as f:
        f.write(hdr)
        for arr in (asf64(g), asf64(h), asui8(r), asf64(ts),
                    asui8(dec_mask), asui8(te_mask), asi32(grp_r)):
            f.write(arr.tobytes())
        f.write(asf64(grid).tobytes())    # n_grid
        f.write(xp.tobytes())             # n * 3 float64, row-major
        # Python 기준선
        for arr in (asf64(T_base), asf64(T_rule), asf64(T_rb),
                    asf64(T_ai), asf64(T_ai_fb), asf64(T_c3)):
            f.write(arr.tobytes())
        f.write(struct.pack("<dd", float(R), float(H)))


def read_window(path):
    """G2W2 창 이진 파일을 읽어 dict로 반환한다."""
    with open(path, "rb") as f:
        raw = f.read()
    off = 0

    def take_hdr():
        nonlocal off
        v = struct.unpack_from(FMT_WIN_HDR, raw, off)
        off += WIN_HDR_SIZE
        return v

    def take_arr(dtype, count):
        nonlocal off
        a = np.frombuffer(raw, dtype=dtype, count=count, offset=off).copy()
        off += a.nbytes
        return a

    magic, n, G_r, n_grid, b, cap, W_cal, U, W_fb = take_hdr()
    if magic != MAGIC_WIN:
        raise ValueError(f"매직 불일치: {magic!r} (기대 {MAGIC_WIN!r})")

    g         = take_arr(np.float64, n)
    h         = take_arr(np.float64, n)
    r         = take_arr(np.uint8,   n).astype(bool)
    ts        = take_arr(np.float64, n)
    dec_mask  = take_arr(np.uint8,   n).astype(bool)
    te_mask   = take_arr(np.uint8,   n).astype(bool)
    grp_r     = take_arr(np.int32,   n)
    grid      = take_arr(np.float64, n_grid)
    X_p1b     = take_arr(np.float64, n * 3).reshape(n, 3)
    T_base_py    = take_arr(np.float64, n)
    T_rule_py    = take_arr(np.float64, n)
    T_rb_py      = take_arr(np.float64, n)
    T_ai_py      = take_arr(np.float64, n)
    T_ai_fb_py   = take_arr(np.float64, n)
    T_c3_py      = take_arr(np.float64, n)
    R_py, H_py   = struct.unpack_from("<dd", raw, off)

    return dict(
        n=n, G_r=G_r, n_grid=n_grid, b=b, cap=cap,
        W_cal=W_cal, U=U, W_fb=W_fb,
        g=g, h=h, r=r, ts=ts,
        dec_mask=dec_mask, te_mask=te_mask, grp_r=grp_r,
        grid=grid, X_p1b=X_p1b,
        T_base_py=T_base_py, T_rule_py=T_rule_py, T_rb_py=T_rb_py,
        T_ai_py=T_ai_py, T_ai_fb_py=T_ai_fb_py, T_c3_py=T_c3_py,
        R_py=R_py, H_py=H_py,
    )


# ---------------------------------------------------------------- G2MB 모델 형식
# P1b 아티팩트: cuts (3개 배열), KNOTS, table
MAGIC_MOD = b"G2MB"
FMT_MOD_HDR = "<4sIIIII"
# magic(4s), n_cuts_0(I), n_cuts_1(I), n_cuts_2(I), n_knots(I), sentinel(I)
MOD_HDR_SIZE = struct.calcsize(FMT_MOD_HDR)  # 24 bytes

def write_model(path, cuts_list, knots, table):
    """G2MB 모델 파일을 쓴다.

    cuts_list: list of 3 np.float64 arrays (cutpoints per feature)
    knots:     np.float64 array, length 11
    table:     np.uint8 array, shape (65, 65, 65)
    """
    cuts = [np.asarray(c, dtype=np.float64) for c in cuts_list]
    kn   = np.asarray(knots, dtype=np.float64)
    tbl  = np.asarray(table, dtype=np.uint8)
    if tbl.shape != (TABLE_DIM, TABLE_DIM, TABLE_DIM):
        raise ValueError(f"table shape 오류: {tbl.shape}")
    sentinel = TABLE_DIM - 1  # 64

    hdr = struct.pack(FMT_MOD_HDR, MAGIC_MOD,
                      len(cuts[0]), len(cuts[1]), len(cuts[2]),
                      len(kn), sentinel)
    with open(path, "wb") as f:
        f.write(hdr)
        for c in cuts:
            f.write(c.tobytes())
        f.write(kn.tobytes())
        f.write(tbl.tobytes())


def read_model(path):
    """G2MB 모델 파일을 읽어 dict로 반환한다."""
    with open(path, "rb") as f:
        raw = f.read()
    off = 0

    magic, nc0, nc1, nc2, nk, sentinel = struct.unpack_from(FMT_MOD_HDR, raw, off)
    off += MOD_HDR_SIZE
    if magic != MAGIC_MOD:
        raise ValueError(f"모델 매직 불일치: {magic!r}")

    def take(dtype, count):
        nonlocal off
        a = np.frombuffer(raw, dtype=dtype, count=count, offset=off).copy()
        off += a.nbytes
        return a

    cuts_0 = take(np.float64, nc0)
    cuts_1 = take(np.float64, nc1)
    cuts_2 = take(np.float64, nc2)
    knots  = take(np.float64, nk)
    table  = take(np.uint8, TABLE_DIM**3).reshape(TABLE_DIM, TABLE_DIM, TABLE_DIM)

    return dict(
        cuts=[cuts_0, cuts_1, cuts_2],
        knots=knots,
        table=table,
        sentinel=sentinel,
    )


# ---------------------------------------------------------------- 결과 이진 형식 (G2W2용)
# T_base 추가됨 (parity 검증용)
FMT_RESULT_HDR = "<II"   # n, reserved

def write_results(path, n, T_base, T_rule, T_rb, T_ai, T_ai_fb, T_c3, R, H):
    def asf64(a): return np.asarray(a, dtype=np.float64)
    with open(path, "wb") as f:
        f.write(struct.pack(FMT_RESULT_HDR, n, 0))
        for arr in (asf64(T_base), asf64(T_rule), asf64(T_rb),
                    asf64(T_ai), asf64(T_ai_fb), asf64(T_c3)):
            f.write(arr.tobytes())
        f.write(struct.pack("<dd", float(R), float(H)))


def read_results(path, n):
    with open(path, "rb") as f:
        raw = f.read()
    hdr_size = struct.calcsize(FMT_RESULT_HDR)
    n_check, _ = struct.unpack_from(FMT_RESULT_HDR, raw, 0)
    if n_check != n:
        raise ValueError(f"n 불일치: {n_check} != {n}")
    off = hdr_size

    def take(count):
        nonlocal off
        a = np.frombuffer(raw, dtype=np.float64, count=count, offset=off).copy()
        off += a.nbytes
        return a

    T_base  = take(n)
    T_rule  = take(n)
    T_rb    = take(n)
    T_ai    = take(n)
    T_ai_fb = take(n)
    T_c3    = take(n)
    R, H    = struct.unpack_from("<dd", raw, off)
    return dict(T_base=T_base, T_rule=T_rule, T_rb=T_rb,
                T_ai=T_ai, T_ai_fb=T_ai_fb, T_c3=T_c3, R=R, H=H)
