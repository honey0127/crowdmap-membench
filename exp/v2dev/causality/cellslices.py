"""cellslices.py — policysim.slices와 같은 LIFO 분해에 '띠 칸' 분할을 더한 변형(인과성 점검용).

원래 분해: 띠 [D0, top)가 ts에 놀기 시작하면, 뒤의 부분 재사용이 띠를 조각으로 가른다.
  조각의 pos = 조각 중점 − D0, 이전놀이 칸 = 조각 중점 높이 → 조각 경계가 미래 재사용으로 정해지므로 ts에 알 수 없다.
변형
  mode="cellmid": 조각을 띠 기준 1 MiB 칸으로 더 나누되 특징은 원래 조각 값 그대로(분해 세분화만의 효과, 대조군)
  mode="causal" : 같은 칸으로 나누고 pos·이전놀이 칸을 칸 중점으로 정한다(칸 경계 = D0 + k·cell, ts에 안다)
비용은 조각 단위 합이라 세 분해에서 같은 T를 주면 같다(고정 정책으로 확인).
"""
import bisect
import math

import numpy as np

MiB = 1024.0 * 1024.0


def slices_cells(ta, tf, sz, mode="causal", cell=1.0 * MiB):
    assert mode in ("cellmid", "causal")
    t = np.concatenate([ta, tf])
    d = np.concatenate([sz, -sz])
    order = np.lexsort((d, t))
    t, d = t[order], d[order]
    t0 = float(t[0])
    D = 0.0
    idle = []  # [lo, hi, ts, D0]
    out = {k: [] for k in ("g", "h", "reused", "ts", "D0", "pos", "prevgap")}
    cold = 0.0
    area = 0.0
    prev = t0
    BUCKET = 1.0 * MiB
    hist = {}

    def lookup(b, ts):
        ends, gaps = hist.get(b, ((), ()))
        i = bisect.bisect_right(ends, ts) - 1
        return gaps[i] if i >= 0 else math.nan

    def record(b, t_end, g):
        e = hist.setdefault(b, ([], []))
        e[0].append(t_end)
        e[1].append(g)

    def emit(g, lo, hi, ts, D0, top, reused, t_end_gap):
        if mode == "cellmid":
            b = int(((lo + hi) / 2) // BUCKET)
            pg = lookup(b, ts)
            pos = (lo + hi) / 2 - D0
        k = int((lo - D0) // cell)
        while True:
            c_lo = D0 + k * cell
            if c_lo >= hi:
                break
            c_hi = min(D0 + (k + 1) * cell, top)
            p_lo, p_hi = max(lo, c_lo), min(hi, c_hi)
            if p_hi > p_lo:
                if mode == "causal":
                    mid = (c_lo + c_hi) / 2
                    bc = int(mid // BUCKET)
                    out["pos"].append(mid - D0)
                    out["prevgap"].append(lookup(bc, ts))
                    if reused:
                        record(bc, t_end_gap, g)
                else:
                    out["pos"].append(pos)
                    out["prevgap"].append(pg)
                out["g"].append(g)
                out["h"].append(p_hi - p_lo)
                out["reused"].append(reused)
                out["ts"].append(ts)
                out["D0"].append(D0)
            k += 1
        if mode == "cellmid" and reused:
            record(b, t_end_gap, g)

    for ti, di in zip(t.tolist(), d.tolist()):
        area += D * (ti - prev)
        prev = ti
        if di < 0:
            lo = D + di
            if idle and idle[-1][2] == ti and idle[-1][0] == D:
                idle[-1][0] = lo
                idle[-1][3] = lo
            else:
                idle.append([lo, D, ti, lo])
            D = lo
        else:
            target = D + di
            level = D
            while level < target and idle:
                lo, hi, ts, D0 = idle[-1]
                top = min(hi, target)
                emit(ti - ts, lo, top, ts, D0, hi, True, ti)
                if hi <= target:
                    idle.pop()
                else:
                    idle[-1][0] = target
                level = top
            if level < target:
                cold += target - level
            D = target
    t_end = float(t[-1])
    for lo, hi, ts, D0 in idle:
        emit(t_end - ts, lo, hi, ts, D0, hi, False, t_end)
    S = {k: np.array(v) for k, v in out.items()}
    S["reused"] = S["reused"].astype(bool)
    S.update(cold=cold, dur=t_end - t0, t0=t0, t_end=t_end, dmean=area / (t_end - t0))
    return S
