"""chunkslices.py — v2 평가 모델(청크 풀 유체 모델): 요청 본문은 ceil(크기 / 청크)개의 청크를 쓰고, 조각 = 청크 하나의 놀이 구간이다.

v1 모델(policysim.slices)과의 차이: v1은 띠가 뒤의 부분 재사용으로 갈라진 '조각'마다 특징 pos(조각 중점 − D0)와 이전놀이 칸
(조각 중점 높이)을 매겼다. 조각 경계는 놀기 시작한 뒤의 도착 크기로 정해지므로, 이 특징에는 결정 시각 ts 뒤의 정보가 섞였다
(docs/causality-check.md). 3단계 청크 풀(chunkpool.cpp, export_decisions.py)과 같은 올림을 쓴다.

policysim.slices와 같은 LIFO 재사용(가장 최근에 놀기 시작한 띠의 바닥부터 다시 쓴다)을 쓴다.
크기를 청크 배수로 올리면 사용량 D(t)·띠 경계·재사용 경계가 모두 청크 경계에 놓여, 청크마다 놀이 시간이 하나다.
특징은 청크 위치로 정한다(놀기 시작할 때 안다).
  pos      = (띠 안 청크 번호 k + 0.5) × 청크          ← 원래 분해의 '조각 중점'은 뒤의 재사용 크기로 정해져 미래 정보가 섞였다
  D0       = 띠 바닥(놀기 시작한 직후 사용량)
  이전놀이 = 같은 절대 높이 청크에서 ts 이전에 끝난 마지막 놀이 시간
행 수(청크 수)도 ts에 정해진다. 그래서 시각 t의 모든 결정은 t 이전 사건만으로 정해진다(causality_check.py로 확인).
"""
import bisect
import math

import numpy as np

MiB = 1024.0 * 1024.0


def quantize(sz, chunk=MiB):
    return np.ceil(np.asarray(sz, dtype=float) / chunk) * chunk


def slices_chunk(ta, tf, sz, chunk=MiB):
    sz = quantize(sz, chunk)
    t = np.concatenate([ta, tf])
    d = np.concatenate([sz, -sz])
    order = np.lexsort((d, t))  # 같은 시각이면 반납(음수) 먼저
    t, d = t[order], d[order]
    t0 = float(t[0])
    D = 0.0
    idle = []  # [lo, hi, ts, D0]
    out = {k: [] for k in ("g", "h", "reused", "ts", "D0", "pos", "prevgap")}
    cold = 0.0
    area = 0.0
    prev = t0
    hist = {}  # 절대 청크 번호 → ([끝난 시각...], [놀이 시간...])

    def emit(g, lo, hi, ts, D0, reused, t_end_gap):
        k0 = int(round((lo - D0) / chunk))
        k1 = int(round((hi - D0) / chunk))
        for k in range(k0, k1):
            mid = D0 + (k + 0.5) * chunk
            ab = int(mid // chunk)
            ends, gaps = hist.get(ab, ((), ()))
            i = bisect.bisect_right(ends, ts) - 1
            out["g"].append(g)
            out["h"].append(chunk)
            out["reused"].append(reused)
            out["ts"].append(ts)
            out["D0"].append(D0)
            out["pos"].append(mid - D0)
            out["prevgap"].append(gaps[i] if i >= 0 else math.nan)
            if reused:
                e = hist.setdefault(ab, ([], []))
                e[0].append(t_end_gap)
                e[1].append(g)

    for ti, di in zip(t.tolist(), d.tolist()):
        area += D * (ti - prev)
        prev = ti
        if di < 0:
            lo = D + di
            if idle and idle[-1][2] == ti and idle[-1][0] == D:  # 같은 시각의 연속 감소는 한 띠로 합친다
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
                emit(ti - ts, lo, top, ts, D0, True, ti)
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
        emit(t_end - ts, lo, hi, ts, D0, False, t_end)
    S = {k: np.array(v) for k, v in out.items()}
    # 행 순서를 (놀기 시작한 시각, 절대 높이)로 정한다. 만든 순서(다시 쓰인 시각)는 미래에 따라 바뀌고,
    # 생존 모델(HistGradientBoosting)의 조기 종료 검증 분할이 행 순서에 따라 달라지기 때문이다(docs/causality-check.md §2b).
    o = np.lexsort((S["D0"] + S["pos"], S["ts"]))
    S = {k: v[o] for k, v in S.items()}
    S["reused"] = S["reused"].astype(bool)
    S.update(cold=cold, dur=t_end - t0, t0=t0, t_end=t_end, dmean=area / (t_end - t0))
    return S
