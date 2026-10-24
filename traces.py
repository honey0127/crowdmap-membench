#!/usr/bin/env python3
"""traces.py — 무거운 요청의 도착·크기 시계열을 만들고, 할당/반납 일정으로 바꾸고, 예측할 거리가 있는지 검사한다.

공통 트레이스 형식(trace.csv):  t_s,size_bytes,work
  t_s        도착 시각(초, 0부터)        size_bytes  본문 크기       work  처리량 단위(LLM이면 출력 토큰, 없으면 0)

하위 명령
  synth     합성 트레이스 (periodic: 정해진 주기로 몰림 / random: 같은 양이 무작위 시각에 몰림 / poisson: 몰림 없음)
  azure     Azure LLM 추론 트레이스(TIMESTAMP,ContextTokens,GeneratedTokens) 변환
  burstgpt  BurstGPT 트레이스(Timestamp,Model,Request tokens,Response tokens,...) 변환
  check     예측할 거리가 있는지 1차 검사(몰림 정도, 주기, 단순 예측 비교)
  schedule  트레이스 → 할당/반납 일정(schedule.csv: t_alloc_s,t_free_s,size_bytes). 워커 c개 FIFO로 완료 시각 계산

예
  python3 traces.py synth --pattern periodic --out syn_periodic.csv
  python3 traces.py synth --pattern random   --out syn_random.csv
  python3 traces.py azure --in AzureLLMInferenceTrace_conv.csv --out azure_conv.csv --duration 600
  python3 traces.py check --in syn_periodic.csv --png check_periodic.png
  python3 traces.py schedule --in syn_periodic.csv --out sched_periodic.csv --workers 2 --rho 0.6
필요: numpy (그림은 matplotlib 선택)
"""
import argparse
import calendar
import csv
import heapq
import math
import sys
from datetime import datetime

import numpy as np

KiB, MiB = 1024, 1024 * 1024


# ---------------------------------------------------------------- 공통 입출력
def write_trace(path, t, size, work):
    order = np.argsort(t, kind="stable")
    t, size, work = t[order], size[order], work[order]
    t = t - (t[0] if len(t) else 0.0)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "size_bytes", "work"])
        for a, b, c in zip(t, size, work):
            w.writerow([f"{a:.6f}", int(b), f"{c:g}"])
    summarize(t, size, path)


def read_trace(path):
    t, s, w = [], [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            t.append(float(row["t_s"]))
            s.append(float(row["size_bytes"]))
            w.append(float(row.get("work") or 0))
    t, s, w = np.array(t), np.array(s), np.array(w)
    order = np.argsort(t, kind="stable")
    return t[order], s[order], w[order]


def summarize(t, size, label):
    if len(t) == 0:
        print(f"[{label}] 요청 0건")
        return
    dur = max(t[-1], 1e-9)
    print(f"[{label}] 요청 {len(t)}건, 길이 {dur:.1f}s, 평균 도착 {len(t) / dur:.2f}건/s, "
          f"크기 중앙값 {np.median(size) / MiB:.2f}MiB / p90 {np.percentile(size, 90) / MiB:.2f} / 최대 {size.max() / MiB:.2f}, "
          f"평균 유입 {size.sum() / dur / MiB:.2f}MiB/s")


def window(t, *cols, start=0.0, duration=0.0):
    t = t - t.min()
    m = t >= start
    if duration > 0:
        m &= t < start + duration
    return (t[m] - start,) + tuple(c[m] for c in cols)


def scale_sizes(proxy, median_mb, min_kb, max_mb):
    """실제 트레이스의 토큰 수 → 바이트. 분포 모양은 유지하고 중앙값만 median_mb에 맞춘다."""
    proxy = np.maximum(proxy, 1.0)
    k = median_mb * MiB / np.median(proxy)
    print(f"크기 환산: 1 단위 = {k / KiB:.1f} KiB (중앙값 {median_mb} MiB에 맞춤)")
    return np.clip(proxy * k, min_kb * KiB, max_mb * MiB)


# ---------------------------------------------------------------- synth
def cmd_synth(a):
    rng = np.random.default_rng(a.seed)
    T, lam = a.duration, a.rate
    f = 0.0 if a.pattern == "poisson" else a.burst_frac
    base = (1.0 - f) * lam  # 몰림 밖 평소 도착(초당)
    if a.day_period > 0:  # 하루 주기 같은 느린 파도(압축한 주기)
        lmax = base * (1 + a.day_amp)
        cand = np.sort(rng.uniform(0, T, rng.poisson(lmax * T)))
        keep = rng.uniform(0, 1, len(cand)) < (1 + a.day_amp * np.sin(2 * np.pi * cand / a.day_period)) / (1 + a.day_amp)
        tb = cand[keep]
    else:
        tb = rng.uniform(0, T, rng.poisson(base * T))
    parts = [tb]
    if f > 0:
        starts = np.arange(a.period / 2, T, a.period)  # periodic: 매 period초마다 같은 위상 → 예측 가능
        if a.pattern == "random":  # 몰림 횟수·양은 같고 시각만 무작위 → 예측 불가(대조군)
            starts = np.sort(rng.uniform(0, max(T - a.burst_width, 0), len(starts)))
        per_burst = f * lam * a.period  # 몰림 1회당 평균 요청 수
        for s in starts:
            m = 1 + a.day_amp * np.sin(2 * np.pi * s / a.day_period) if (a.wave_all and a.day_period > 0) else 1.0
            parts.append(s + rng.uniform(0, a.burst_width, rng.poisson(per_burst * m)))
    t = np.concatenate(parts)
    t = np.sort(t[(t >= 0) & (t < T)])
    size = np.exp(rng.normal(math.log(a.median_mb * MiB), a.sigma, len(t)))
    size = np.clip(size, a.min_kb * KiB, a.max_mb * MiB)
    write_trace(a.out, t, size, np.zeros(len(t)))


# ---------------------------------------------------------------- 실제 트레이스 변환
def parse_azure_ts(s):
    s = s.strip()
    off = 0
    if len(s) > 19 and s[-6] in "+-" and s[-3] == ":":  # 2024판의 '+00:00' 같은 시간대 꼬리
        off = (int(s[-5:-3]) * 3600 + int(s[-2:]) * 60) * (1 if s[-6] == "+" else -1)
        s = s[:-6]
    base = calendar.timegm(datetime.strptime(s[:19], "%Y-%m-%d %H:%M:%S").timetuple())
    return base + (float("0" + s[19:]) if len(s) > 19 else 0.0) - off  # 소수점 7자리까지 처리


def cmd_azure(a):
    t, ctx, gen = [], [], []
    with open(a.inp, newline="") as f:
        for row in csv.DictReader(f):
            t.append(parse_azure_ts(row["TIMESTAMP"]))
            ctx.append(float(row["ContextTokens"]))
            gen.append(float(row["GeneratedTokens"]))
    t, ctx, gen = window(np.array(t), np.array(ctx), np.array(gen), start=a.start, duration=a.duration)
    write_trace(a.out, t, scale_sizes(ctx, a.median_mb, a.min_kb, a.max_mb), gen)


def cmd_burstgpt(a):
    rng = np.random.default_rng(a.seed)
    t, req, resp = [], [], []
    with open(a.inp, newline="") as f:
        for row in csv.DictReader(f):
            if a.model and row["Model"] != a.model:
                continue
            r = float(row["Response tokens"])
            if r <= 0 and not a.keep_failed:
                continue
            t.append(float(row["Timestamp"]))
            req.append(float(row["Request tokens"]))
            resp.append(r)
    t, req, resp = np.array(t), np.array(req), np.array(resp)
    if len(t) and np.all(t == np.round(t)):  # 초 단위 정수 시각 → 1초 안에 고르게 흩는다
        t = t + rng.uniform(0, 1, len(t))
    t, req, resp = window(t, req, resp, start=a.start, duration=a.duration)
    write_trace(a.out, t, scale_sizes(req, a.median_mb, a.min_kb, a.max_mb), resp)


# ---------------------------------------------------------------- check
def acf(x, maxlag):
    x = x - x.mean()
    n = len(x)
    fx = np.fft.rfft(x, 2 * n)
    r = np.fft.irfft(fx * np.conj(fx))[: maxlag + 1]
    return r / r[0] if r[0] > 0 else np.zeros(maxlag + 1)


def cmd_check(a):
    t, size, _ = read_trace(a.inp)
    summarize(t, size, a.inp)
    T = t[-1]
    nb = lambda b: int(T // b) + 1
    print("\n1) 몰림 정도: 구간별 도착 수의 분산/평균 (포아송이면 약 1, 클수록 몰림)")
    for b in (0.1, 1.0, 10.0):
        if nb(b) > 20_000_000:  # 너무 긴 트레이스는 가는 구간을 건너뛴다
            continue
        c = np.bincount((t // b).astype(np.int64), minlength=nb(b))
        print(f"   {b:>5}s 구간: {c.var() / max(c.mean(), 1e-12):8.2f}")
    x = np.bincount((t // a.bin).astype(np.int64), weights=size, minlength=nb(a.bin)) / MiB  # 구간별 도착 MiB
    edges = np.arange(len(x)) * a.bin
    maxlag = min(int(a.max_lag / a.bin), len(x) // 3)
    if maxlag < 4:
        print("트레이스가 너무 짧다")
        return
    r = acf(x, maxlag)
    # 처음 내려가는 구간(직전 값과 비슷한 효과)을 지난 뒤의 봉우리 중 가장 높은 것을 주기로 본다
    m0 = next((k for k in range(1, maxlag) if r[k] < r[k - 1] and r[k] <= r[k + 1]), 1)
    peaks = [k for k in range(max(2, m0 + 1), maxlag) if r[k] >= r[k - 1] and r[k] >= r[k + 1]]
    P = max(peaks, key=lambda k: r[k]) if peaks else 0
    print(f"\n2) 주기: 자기상관이 가장 높은 지연 = {P * a.bin:g}s (상관 {r[P] if P else 0:.2f}; 0.3 이상이면 뚜렷한 주기)")
    for lag_s in [float(v) for v in a.lags.split(",") if v.strip()]:
        k = int(round(lag_s / a.bin))
        if 0 < k <= maxlag:
            print(f"   지연 {lag_s:g}s의 자기상관: {r[k]:.2f}")
    h = max(1, int(round(a.horizon / a.bin)))
    cs = np.concatenate([[0.0], np.cumsum(x)])
    y = cs[1:] - cs[np.maximum(np.arange(1, len(x) + 1) - h, 0)]  # y[i] = x[i-h+1..i] 합 = 직전 h구간 도착량
    start = max(P, 10 * h)
    idx = np.arange(start, len(x) - h)
    truth = y[idx + h]  # 다음 h구간 도착량(예측 대상)
    pers = y[idx]  # 직전 h구간 그대로
    ma = (cs[idx + 1] - cs[idx + 1 - 10 * h]) / 10.0  # 최근 10h구간 평균 × h
    mae = lambda p: np.mean(np.abs(truth - p))
    print(f"\n3) 다음 {h * a.bin:g}s 동안 들어올 바이트 예측, 평균 절대오차(MiB) — 낮을수록 좋다")
    print(f"   직전 값 그대로      : {mae(pers):8.3f}")
    print(f"   최근 평균           : {mae(ma):8.3f}")
    if P:
        seas = y[idx + h - P]  # 한 주기 전 같은 위치
        print(f"   한 주기 전 값({P * a.bin:g}s): {mae(seas):8.3f}   → 직전 값 대비 {mae(seas) / max(mae(pers), 1e-12):.2f}배")
        ratio = mae(seas) / max(min(mae(pers), mae(ma)), 1e-12)
        if r[P] >= 0.3 and ratio < 0.8:
            print("   판정: 주기가 있고 '한 주기 전'이 더 잘 맞는다 → 예측기로 이길 여지가 있다")
        else:
            print("   판정: 주기 신호가 약하다 → 예측 이득이 작을 가능성이 크다(이 트레이스는 대조군으로 쓸 만하다)")
    if a.png:
        plot_check(a.png, edges, x, r, a.bin, P, a.inp)


def plot_check(path, edges, x, r, binw, P, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid, blue = "#0b0b0b", "#898781", "#e1e0d9", "#2a78d6"
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 6), facecolor="#fcfcfb")
    n = min(len(x), int(600 / binw))  # 앞 10분만
    a1.plot(edges[:n], x[:n], color=blue, lw=1.5)
    a1.set_title(f"{title}: arriving MiB per {binw:g}s (first {n * binw:g}s)", color=ink, fontsize=10, loc="left")
    a1.set_xlabel("time (s)", color=muted)
    a1.set_ylabel("MiB", color=muted)
    a2.plot(np.arange(len(r)) * binw, r, color=blue, lw=1.5)
    if P:
        a2.axvline(P * binw, color=muted, lw=1, ls="--")
        a2.annotate(f"peak {P * binw:g}s", (P * binw, r[P]), xytext=(6, 6), textcoords="offset points", color=ink, fontsize=9)
    a2.set_title("autocorrelation of arriving bytes", color=ink, fontsize=10, loc="left")
    a2.set_xlabel("lag (s)", color=muted)
    for ax in (a1, a2):
        ax.set_facecolor("#fcfcfb")
        ax.grid(color=grid, lw=0.8)
        ax.tick_params(colors=muted)
        for s in ax.spines.values():
            s.set_color("#c3c2b7")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"그림: {path}")


# ---------------------------------------------------------------- schedule
def cmd_schedule(a):
    t, size, work = read_trace(a.inp)
    S = (a.svc_base_ms + a.svc_ms_per_mb * size / MiB + a.svc_ms_per_work * work) / 1000.0
    dur0 = max(t[-1], 1e-9)
    rho0 = len(t) / dur0 * S.mean() / a.workers
    speed = a.rho / rho0 if a.rho else a.speed
    t = t / speed  # speed > 1이면 시간을 압축(도착이 잦아짐)
    if a.max_duration:
        m = t <= a.max_duration
        t, size, S = t[m], size[m], S[m]
    n = len(t)
    free_at = [0.0] * a.workers  # 워커가 비는 시각(최소 힙)
    heapq.heapify(free_at)
    tf, wait = np.empty(n), np.empty(n)
    late = 0
    for i in range(n):
        s0 = heapq.heappop(free_at)
        start = max(t[i], s0)
        wait[i] = start - t[i]
        if a.deadline_s and wait[i] > a.deadline_s:  # 꺼낼 때 마감: 처리하지 않고 버퍼 반납
            tf[i] = start
            late += 1
            heapq.heappush(free_at, start)
        else:
            tf[i] = start + S[i]
            heapq.heappush(free_at, tf[i])
    # 살아 있는 바이트 D(t)
    ev_t = np.concatenate([t, tf])
    ev_d = np.concatenate([size, -size])
    o = np.lexsort((ev_d, ev_t))  # 같은 시각이면 반납 먼저
    ev_t, ev_d = ev_t[o], ev_d[o]
    live = np.cumsum(ev_d)
    span = ev_t[-1] - ev_t[0]
    live_mean = np.sum(live[:-1] * np.diff(ev_t)) / span if span > 0 else 0.0
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_alloc_s", "t_free_s", "size_bytes"])
        for x, y, z in zip(t, tf, size):
            w.writerow([f"{x:.6f}", f"{y:.6f}", int(z)])
    rho = n / max(t[-1], 1e-9) * S.mean() / a.workers
    print(f"[{a.out}] 요청 {n}건, 재생 길이 {tf.max():.1f}s, 시간 배율 {speed:.3g}, 워커 {a.workers}, "
          f"평균 처리시간 {S.mean() * 1e3:.1f}ms, 평균 이용률 ρ {rho:.2f}")
    print(f"  큐 대기 Wq p50 {np.percentile(wait, 50) * 1e3:.1f}ms / p99 {np.percentile(wait, 99) * 1e3:.1f}ms / "
          f"최대 {wait.max() * 1e3:.1f}ms, 마감 초과 {late}건")
    print(f"  큐 메모리(살아 있는 본문): 평균 {live_mean / MiB:.1f}MiB, 최대 {live.max() / MiB:.1f}MiB")
    if live.max() / MiB > a.mem_limit_mb:
        print(f"  [경고] 최대 {live.max() / MiB:.0f}MiB > --mem-limit-mb {a.mem_limit_mb}. 보드 RAM을 넘지 않게 "
              f"--median-mb를 줄이거나 --rho를 낮춘다.")


# ---------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def size_opts(q, median):
        q.add_argument("--median-mb", type=float, default=median, help="본문 크기 중앙값(MiB)")
        q.add_argument("--min-kb", type=float, default=64)
        q.add_argument("--max-mb", type=float, default=32)

    q = sub.add_parser("synth")
    q.add_argument("--out", required=True)
    q.add_argument("--pattern", choices=["periodic", "random", "poisson"], default="periodic")
    q.add_argument("--duration", type=float, default=600, help="길이(s)")
    q.add_argument("--rate", type=float, default=8, help="평균 도착(건/s)")
    q.add_argument("--burst-frac", type=float, default=0.4, help="몰림으로 오는 요청 비율 f")
    q.add_argument("--period", type=float, default=30, help="몰림 주기 T(s). random이면 평균 간격")
    q.add_argument("--burst-width", type=float, default=3, help="몰림 폭 w(s)")
    q.add_argument("--day-period", type=float, default=0, help="느린 파도 주기(s), 0이면 끔")
    q.add_argument("--day-amp", type=float, default=0.5)
    q.add_argument("--wave-all", action="store_true", help="느린 파도를 몰림 요청 수에도 건다(기본은 평소 도착에만)")
    q.add_argument("--sigma", type=float, default=0.7, help="크기 로그정규 σ")
    q.add_argument("--seed", type=int, default=1)
    size_opts(q, 2.0)
    q.set_defaults(fn=cmd_synth)

    for name, fn in (("azure", cmd_azure), ("burstgpt", cmd_burstgpt)):
        q = sub.add_parser(name)
        q.add_argument("--in", dest="inp", required=True)
        q.add_argument("--out", required=True)
        q.add_argument("--start", type=float, default=0, help="잘라 쓸 시작 시각(트레이스 시작부터 s)")
        q.add_argument("--duration", type=float, default=0, help="잘라 쓸 길이(s), 0이면 전체")
        size_opts(q, 2.0)
        q.add_argument("--seed", type=int, default=1)
        if name == "burstgpt":
            q.add_argument("--model", default="", help="ChatGPT 또는 GPT-4만 고르기(빈 값이면 전부)")
            q.add_argument("--keep-failed", action="store_true", help="Response tokens=0(실패) 행도 쓰기")
        q.set_defaults(fn=fn)

    q = sub.add_parser("check")
    q.add_argument("--in", dest="inp", required=True)
    q.add_argument("--bin", type=float, default=1.0, help="시계열 구간(s)")
    q.add_argument("--horizon", type=float, default=1.0, help="예측 지평(s)")
    q.add_argument("--max-lag", type=float, default=600, help="주기를 찾을 최대 지연(s)")
    q.add_argument("--lags", default="", help="자기상관을 따로 볼 지연(s), 쉼표로. 예: 86400,604800")
    q.add_argument("--png", default="")
    q.set_defaults(fn=cmd_check)

    q = sub.add_parser("schedule")
    q.add_argument("--in", dest="inp", required=True)
    q.add_argument("--out", required=True)
    q.add_argument("--workers", type=int, default=2, help="무거운 요청 워커 수 c")
    q.add_argument("--svc-base-ms", type=float, default=10, help="처리시간 = base + per_mb×크기 + per_work×work")
    q.add_argument("--svc-ms-per-mb", type=float, default=20)
    q.add_argument("--svc-ms-per-work", type=float, default=0, help="LLM 트레이스면 출력 토큰당 ms")
    q.add_argument("--rho", type=float, default=0, help="평균 이용률을 이 값에 맞추도록 시간 배율 자동 계산")
    q.add_argument("--speed", type=float, default=1, help="시간 배율(--rho가 없을 때). 2면 2배 빠르게 재생")
    q.add_argument("--deadline-s", type=float, default=2.0, help="꺼낼 때 마감(v5 기본 2s), 0이면 끔")
    q.add_argument("--max-duration", type=float, default=0, help="재생 길이 상한(s)")
    q.add_argument("--mem-limit-mb", type=float, default=1024)
    q.set_defaults(fn=cmd_schedule)

    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
