#!/usr/bin/env python3
"""plot.py — 측정 결과를 표와 그림으로 정리한다. (그림 글자는 한글 글꼴 문제를 피하려고 영어로 쓴다)

  python3 plot.py L      out/L.csv        [out/L.png]       크기별 확보 지연 L 표·그림
  python3 plot.py Lsum   out/L_run1/L.csv out/L_run2/L.csv ... [--out out/L_thp.png]
                                                          여러 회차·THP 조건을 표 하나로(회차 중앙값·범위)
  python3 plot.py curve  out/results.csv  [out/curve.png]   낭비–지연 곡선(같은 일정, 설정만 바꾼 점들)
  python3 plot.py series out/series/je_0.csv out/series/je_-1.csv ... [--out out/series.png]
                                                          시간에 따른 B(확보)와 D(사용) — 둘 사이 틈이 낭비
필요: matplotlib
"""
import csv
import math
import statistics as st
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogFormatter

PLAIN = FuncFormatter(lambda v, _: f"{v:g}")  # 로그 축 눈금을 10^0 대신 1, 2, 5처럼

# 차트 색: 기본 검증 팔레트의 앞 슬롯(파랑, 주황, 청록, 노랑), 글자는 잉크색
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"


def style(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURF)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(which="both", colors=MUTED, labelsize=9)
    for s in ax.spines.values():
        s.set_color(AXIS)
    ax.set_title(title, color=INK, fontsize=10, loc="left")
    ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)


def read_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(line for line in f if not line.startswith("#")))


# ---------------------------------------------------------------- L
def cmd_L(path, png):
    rows = read_rows(path)
    by = defaultdict(dict)  # (thp) -> {(mode, size, allocfamily): row}
    for r in rows:
        fam = "je" if r["alloc"].startswith("jemalloc") else "glibc"
        by[r["thp"]][(r["mode"], int(r["size_bytes"]), fam)] = r
    for thp, d in by.items():
        sizes = sorted({k[1] for k in d if k[0] == "os_fault"})
        print(f"\n[THP {thp}]  단위: ms (L = os_fault − warm_touch)")
        print(f"{'size':>8} | {'L p50':>8} {'L p99':>8} | {'populate':>8} | {'memset':>8} | "
              f"{'glibc malloc':>12} {'je malloc':>10} | {'L/memset':>8}")
        for s in sizes:
            g = lambda m, fam="glibc", c="p50_us": float(d[(m, s, fam)][c]) / 1000 if (m, s, fam) in d else float("nan")
            Lp50 = g("os_fault") - g("warm_touch")
            Lp99 = g("os_fault", c="p99_us") - g("warm_touch")
            print(f"{s / 2**20:>7.3g}M | {Lp50:8.3f} {Lp99:8.3f} | {g('os_populate'):8.3f} | {g('memset_warm'):8.3f} | "
                  f"{g('malloc_touch'):12.3f} {g('malloc_touch', 'je'):10.3f} | {Lp50 / g('memset_warm'):8.1f}x")
    if not png:
        return
    thp = "default" if "default" in by else next(iter(by))
    d = by[thp]
    sizes = sorted({k[1] for k in d if k[0] == "os_fault"})
    x = [s / 2**20 for s in sizes]
    fig, ax = plt.subplots(figsize=(7.5, 4.6), facecolor=SURF)
    series = [("os_fault", "new memory: page fault + zero fill (= L)", "L"),
              ("os_populate", "new memory: MAP_POPULATE", "populate"),
              ("memset_warm", "write body into a ready buffer", "body write"),
              ("warm_touch", "touch a ready buffer", "ready")]
    for (mode, name, short), c in zip(series, SLOTS):
        y = [float(d[(mode, s, "glibc")]["p50_us"]) / 1000 for s in sizes]
        ax.plot(x, y, color=c, lw=2, marker="o", ms=6, label=name)
        ax.annotate(short, (x[-1], y[-1]), xytext=(6, 0), textcoords="offset points", color=INK2, fontsize=8,
                    va="center")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.xaxis.set_major_formatter(PLAIN)
    ax.xaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    ax.yaxis.set_major_formatter(PLAIN)
    style(ax, f"Time to get a body buffer ready, p50 (THP {thp})", "body size (MiB)", "time (ms)")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper left")
    fig.tight_layout()
    fig.savefig(png, dpi=140)
    print(f"\n그림: {png}")


# ---------------------------------------------------------------- Lsum
THP_ORDER = {"default": 0, "off": 1, "on": 2}


def cmd_Lsum(paths, png):
    """여러 회차의 L.csv를 THP 조건별로 묶는다: 회차 중앙값과 범위, MiB당 시간, 1Gbps 수신 시간 대비 비율."""
    MiB = 2**20
    agg = defaultdict(list)  # (thp, size) -> [(L p50, L p99), ...] 회차별
    for p in paths:
        d = {}
        for r in read_rows(p):
            if r["alloc"].startswith("glibc"):  # os_* 행은 할당기와 무관(glibc 실행에서만 기록)
                d[(r["thp"], int(r["size_bytes"]), r["mode"])] = r
        for (thp, size, mode), r in d.items():
            w = d.get((thp, size, "warm_touch"))
            if mode == "os_fault" and w:
                l50 = (float(r["p50_us"]) - float(w["p50_us"])) / 1000
                l99 = (float(r["p99_us"]) - float(w["p50_us"])) / 1000
                agg[(thp, size)].append((l50, l99))
    if not agg:
        print("L.csv에서 os_fault 행을 찾지 못했다")
        return
    thps = sorted({k[0] for k in agg}, key=lambda t: THP_ORDER.get(t, 9))
    sizes = sorted({k[1] for k in agg})
    print(f"회차 {len(paths)}개, 단위 ms. L = 새 메모리(페이지 폴트 + 0 채우기) − 이미 쓴 버퍼. 값은 회차 중앙값.")
    print(f"{'크기':>7} {'THP':>8} {'회차':>4} | {'L p50':>8} {'(회차 범위)':>17} | {'L p99':>8} | "
          f"{'ms/MiB':>7} | {'1Gbps 수신 대비':>14}")
    med = {}
    for s in sizes:
        net_ms = s * 8 / 1e9 * 1000  # 같은 크기를 1Gbps로 받는 시간
        for t in thps:
            v = agg.get((t, s))
            if not v:
                continue
            l50s, l99s = [a for a, _ in v], [b for _, b in v]
            m50, m99 = st.median(l50s), st.median(l99s)
            med[(t, s)] = (m50, min(l50s), max(l50s))
            rng = f"({min(l50s):.3f}–{max(l50s):.3f})"
            print(f"{s / MiB:>6.3g}M {t:>8} {len(v):>4} | {m50:8.3f} {rng:>17} | {m99:8.3f} | "
                  f"{m50 / (s / MiB):7.3f} | {m50 / net_ms * 100:13.1f}%")
    if "default" in thps and "on" in thps:
        r = [med[("on", s)][0] / med[("default", s)][0] for s in sizes
             if ("on", s) in med and ("default", s) in med and s >= 2 * MiB and med[("default", s)][0] > 0]
        if r:
            print(f"\nTHP on ÷ default (2MiB 이상, p50): {min(r):.2f}–{max(r):.2f}배")
    if not png:
        return
    x = [s / MiB for s in sizes]
    fig, ax = plt.subplots(figsize=(7.5, 4.6), facecolor=SURF)
    for i, t in enumerate(thps):
        xs = [s / MiB for s in sizes if (t, s) in med]
        ys = [med[(t, s)][0] for s in sizes if (t, s) in med]
        lo = [y - med[(t, s)][1] for y, s in zip(ys, [s for s in sizes if (t, s) in med])]
        hi = [med[(t, s)][2] - y for y, s in zip(ys, [s for s in sizes if (t, s) in med])]
        ax.errorbar(xs, ys, yerr=[lo, hi], color=SLOTS[i % 4], lw=2, marker="o", ms=6, capsize=3,
                    label=f"THP {t}: L p50 (median, run range)")
    ax.plot(x, [s * 8 / 1e9 * 1000 for s in sizes], color=MUTED, lw=1.2, ls="--", label="receive at 1 Gbps (reference)")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.xaxis.set_major_formatter(PLAIN)
    ax.xaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    ax.yaxis.set_major_formatter(PLAIN)
    style(ax, f"New-memory latency L by THP mode ({len(paths)} runs)", "body size (MiB)", "time (ms)")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper left")
    fig.tight_layout()
    fig.savefig(png, dpi=140)
    print(f"그림: {png}")


# ---------------------------------------------------------------- curve
def decay_key(label):
    v = label.split("_", 1)[1]
    return math.inf if v == "-1" else float(v)


def cmd_curve(path, png):
    rows = read_rows(path)
    g = defaultdict(list)
    for r in rows:
        if float(r["lag_max_ms"]) > 50:
            print(f"[제외] {r['label']} rep {r['rep']}: 재생 지연 {r['lag_max_ms']}ms")
            continue
        g[r["label"]].append(r)
    agg = {}
    for lab, rs in g.items():
        m = lambda c: st.mean(float(r[c]) for r in rs)
        lo_hi = lambda c: (min(float(r[c]) for r in rs), max(float(r[c]) for r in rs))
        agg[lab] = dict(n=len(rs), waste=m("waste_mean_MB"), waste_rng=lo_hi("waste_mean_MB"),
                        a99=m("A_p99_us") / 1000, a99_rng=tuple(v / 1000 for v in lo_hi("A_p99_us")),
                        asum=m("A_sum_ms"), f99=m("F_p99_us") / 1000, flt=m("minflt"), cpu=m("cpu_ms"),
                        bmax=m("B_max_MB"), dmax=m("D_max_MB"))
    je = sorted([k for k in agg if k.startswith("je_")], key=decay_key)
    others = sorted(k for k in agg if not k.startswith("je_"))
    print(f"{'label':>10} {'reps':>4} | {'낭비 평균MB':>10} | {'A p99 ms':>9} {'A 합 ms':>9} | {'free p99 ms':>11} | "
          f"{'페이지폴트':>10} {'CPU ms':>8} | {'B최대':>7} {'D최대':>7}")
    for k in others + je:
        a = agg[k]
        print(f"{k:>10} {a['n']:>4} | {a['waste']:10.2f} | {a['a99']:9.3f} {a['asum']:9.1f} | {a['f99']:11.3f} | "
              f"{a['flt']:10.0f} {a['cpu']:8.0f} | {a['bmax']:7.1f} {a['dmax']:7.1f}")
    if not png:
        return
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.4), facecolor=SURF)
    for ax, key, rng, ylab, title in (
        (axs[0], "a99", "a99_rng", "allocation latency p99 per request (ms)", "Waste vs per-request delay"),
        (axs[1], "asum", None, "total allocation latency (ms)", "Waste vs total delay"),
    ):
        if je:
            xs = [agg[k]["waste"] for k in je]
            ys = [max(agg[k][key], 1e-3) for k in je]
            ax.plot(xs, ys, color=SLOTS[0], lw=2, marker="o", ms=7, label="jemalloc decay sweep (line to beat)")
            for k, x, y in zip(je, xs, ys):
                d = k.split("_", 1)[1]
                ax.annotate("never" if d == "-1" else f"{d}ms", (x, y), xytext=(5, 5), textcoords="offset points",
                            color=INK2, fontsize=8)
            if rng:
                for k, x in zip(je, xs):
                    lo, hi = agg[k][rng]
                    ax.plot([x, x], [max(lo, 1e-3), max(hi, 1e-3)], color=SLOTS[0], lw=1, alpha=0.5)
        for i, k in enumerate(others):
            c = SLOTS[1 + i % 3]
            ax.plot([agg[k]["waste"]], [max(agg[k][key], 1e-3)], color=c, marker="s", ms=8, ls="none", label=k)
            ax.annotate(k, (agg[k]["waste"], max(agg[k][key], 1e-3)), xytext=(5, -11), textcoords="offset points",
                        color=INK2, fontsize=8)
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(PLAIN)
        ax.yaxis.set_minor_formatter(LogFormatter(labelOnlyBase=False, minor_thresholds=(2, 0.5)))
        style(ax, title + "  (lower-left is better)", "mean idle memory  W / time  (MB)", ylab)
        ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="best")
    fig.tight_layout()
    fig.savefig(png, dpi=140)
    print(f"그림: {png}")


# ---------------------------------------------------------------- series
def cmd_series(paths, png):
    fig, ax = plt.subplots(figsize=(10, 4.2), facecolor=SURF)
    drew_d = False
    for i, p in enumerate(paths):
        rows = read_rows(p)
        t = [float(r["t_s"]) for r in rows]
        if not drew_d:
            ax.fill_between(t, [float(r["D_MB"]) for r in rows], color=MUTED, alpha=0.35, lw=0, label="D: in use")
            drew_d = True
        name = p.rsplit("/", 1)[-1].removesuffix(".csv")
        ax.plot(t, [float(r["B_MB"]) for r in rows], color=SLOTS[i % 4], lw=1.5, label=f"B: held ({name})")
    style(ax, "Held memory B(t) vs used memory D(t) — the gap is waste", "time (s)", "MB")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper right")
    fig.tight_layout()
    fig.savefig(png, dpi=140)
    print(f"그림: {png}")


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    cmd = sys.argv[1]
    if cmd == "L":
        cmd_L(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "")
    elif cmd == "curve":
        cmd_curve(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "")
    elif cmd in ("series", "Lsum"):
        args = sys.argv[2:]
        png = "series.png" if cmd == "series" else ""
        if "--out" in args:
            i = args.index("--out")
            png = args[i + 1]
            args = args[:i] + args[i + 2 :]
        (cmd_series if cmd == "series" else cmd_Lsum)(args, png)
    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
