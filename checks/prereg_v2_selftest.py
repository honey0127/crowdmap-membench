#!/usr/bin/env python3
"""prereg_v2_selftest.py — prereg_v2_analysis.py(사전 등록 v2 개정 6)의 계산 규칙과 판정 중단 규칙을 답을 아는 가짜 입력으로 확인한다.

  python3 checks/prereg_v2_selftest.py      # 저장소 루트에서. 모두 맞으면 'selftest OK'를 출력하고 0으로 끝난다

확인하는 계산 규칙(경우 A·B, 판정까지 가야 한다)
  1. 창 비용 → 날 합 → 주 블록 합(windows_day.csv의 block 열, blocks.csv와 일치 확인). 블록 d를 손 계산과 비교한다.
  2. H1v2 = 41일 총비용 d_total(관측 판정, ±0.5%p). 주 블록 부호 검정은 참고로만 나온다.
     경우 B: 주 블록 대부분에서 조금씩 이기고 한 블록에서 크게 지면, 블록 방향은 주 후보 쪽이어도 H1v2는 '반대 방향'이다.
  3. 기준 비용이 0인 날·블록·합성 일정은 비교에서 빠지고 따로 적힌다(±∞나 무승부로 세지 않는다).
  4. ai = none인 창(AI를 쓸 수 없어 대체 정책으로 계산한 정상 처리)은 누락이 아니다. 빠지지 않고 비용 합에 들어가며 목록에 적힌다.
  5. H3v2는 합성 40개(사전 등록 순서: 조건 8 × 반복 5, 시드 = 201 + 8 × (반복 − 1) + 조건 번호) 부호 검정(보정 없음)이고,
     합성 조건별 승/무/패가 적힌다.
판정 중단 규칙(경우 C 이하, 개정 6): 아래 입력에서는 종료 코드 2, 표준 오류에 '판정 중단: <이유>', 표준 출력에 판정 줄이 없어야 한다.
  C 블록 목록 불일치, C2 block 열이 정의와 다름, D 창 하나의 결과가 통째로 없음, E 한 창에서 정책 하나의 결과 한 줄이 없음,
  E2 합성 일정 하나에서 주 후보 결과 한 줄이 없음, F 판정 T*가 아닌 T*의 결과가 한 창에서 없음, G 합성 시드가 다름(901~940),
  G2 합성 39개, G3 시드와 조건의 짝이 다름, H 비용 NaN, H2 비용 음수, I 중복 행, J 목록에 없는 창, K ai 열 없음, K2 ai 값 이상,
  L --tlist에 없는 T*, M 판정 T*가 --tlist에 없음, N 평가 모델이 chunk가 아님.
  G4: 시드가 다른 합성도 --test-mode(시험 실행)에서는 끝까지 계산하되 머리에 '시험 실행 … 판정이 아니다'를 적는다.
"""
import math
import os
import re
import subprocess
import sys
import tempfile

import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
P, RULE, BASE, V1 = "C3 AI→규칙대체", "규칙-온라인+대체", "고정-온라인", "AI생존κ-온라인+대체"
POLS = [BASE, RULE, V1, "C1 AI세분규칙-온라인+대체", "C2 AI생존κ칸별-온라인+대체", "C4 AI생존세밀κ-온라인+대체", P, "오라클"]
CONDS = ["spn", "spw", "sbn", "sbw", "Spn", "Spw", "Sbn", "Sbw"]  # run_sealed_v2.sh의 조건 순서(조건 번호 0~7)
B = 2.6
TL = (B, 1.0)  # 실행한 T* 목록(판정 T*와 다른 T* 하나). --tlist 2.6,1


def rows(trace, cost, ai="ok"):
    """cost: 정책 → 비용(MiB). 나머지 정책은 고정-온라인과 같게 둔다. T* 목록의 T*마다 같은 비용을 쓴다"""
    out = []
    for t in TL:
        for pol in POLS:
            c = cost.get(pol, cost[BASE])
            out.append(dict(src="test", model="chunk", trace=trace, Tstar=t, policy=pol, cost_MiB=c,
                            rel_fixon=(c / cost[BASE] - 1) * 100 if cost[BASE] > 0 else 0.0, refault_MiB=0.0,
                            held_mean_MiB=0.0, n_te=1, rule="높이", base_frac=0.5 if pol == P else "", ai=ai))
    return out


def fx_rows(base=201):
    """합성 40개: 반복 5 × 조건 8, 시드 = base + 8 × 반복 + 조건 번호. 조건 Sbw만 주 후보가 지고(5패) 나머지는 이긴다.
    첫 반복의 spn(시드 base)은 기준 비용 0이다"""
    fx = []
    for rep in range(5):
        for k, cnd in enumerate(CONDS):
            if cnd == "Sbw":
                cst = {BASE: 100.0, RULE: 90.0, P: 95.0}
            elif cnd == "spn" and rep == 0:
                cst = {BASE: 0.0, RULE: 0.0, P: 0.0}
            else:
                cst = {BASE: 100.0, RULE: 90.0, P: 80.0}
            fx += rows(f"fx_{cnd}_{base + 8 * rep + k}", cst)
    return fx


def run(win, cost, ai_none=(), blocks_override=None, real_mut=None, fx_mut=None, w_mut=None, fx_base=201,
        tlist="2.6,1", tstar=B, extra=()):
    real = []
    for n, _ in win:
        real += rows(n, cost[n], ai="none" if n in ai_none else "ok")
    W = pd.DataFrame([dict(name=n, cluster=c, picked=1) for n, c in win])
    W["day_index"] = [int(c.split("_d")[1]) for c in W.cluster]
    W["week"] = W.day_index // 7
    W["block"] = [f"{c.split('_d')[0]}_w{int(c.split('_d')[1]) // 7:02d}" for c in W.cluster]
    Bk = pd.DataFrame([dict(block=b, source=b.split("_")[0], week=int(b.split("_w")[1]), n_days=g.cluster.nunique(),
                            n_windows=len(g), days=" ".join(sorted(g.cluster.unique()))) for b, g in W.groupby("block")])
    R, F = pd.DataFrame(real), pd.DataFrame(fx_rows(fx_base))
    if blocks_override is not None:
        Bk = blocks_override(Bk)
    if real_mut is not None:
        R = real_mut(R)
    if fx_mut is not None:
        F = fx_mut(F)
    if w_mut is not None:
        W = w_mut(W)
    with tempfile.TemporaryDirectory() as d:
        R.to_csv(f"{d}/real.csv", index=False)
        F.to_csv(f"{d}/fx.csv", index=False)
        W.to_csv(f"{d}/w.csv", index=False)
        Bk.to_csv(f"{d}/b.csv", index=False)
        return subprocess.run([sys.executable, os.path.join(REPO, "prereg_v2_analysis.py"), "--res", f"{d}/real.csv",
                               "--fx", f"{d}/fx.csv", "--windows", f"{d}/w.csv", "--blocks", f"{d}/b.csv",
                               "--primary", P, "--tstar", str(tstar), "--tlist", tlist, *extra], capture_output=True, text=True)


def drop(df, **kw):
    m = pd.Series(True, index=df.index)
    for k, v in kw.items():
        m &= df[k] == v
    assert m.any(), kw
    return df[~m]


def setv(df, col, val, **kw):
    df = df.copy()
    m = pd.Series(True, index=df.index)
    for k, v in kw.items():
        m &= df[k] == v
    assert m.sum() == 1, kw
    df.loc[m, col] = val
    return df


def main():
    fails = []

    def need(out, pat, what):
        if not re.search(pat, out):
            fails.append(what)

    def must_stop(name, r, why):
        """판정 중단: 종료 코드 2, 표준 오류에 이유, 표준 출력에 판정 줄 없음"""
        if r.returncode != 2:
            fails.append(f"경우 {name}: 종료 코드 {r.returncode}(기대 2)")
        if "판정 중단:" not in r.stderr or why not in r.stderr:
            fails.append(f"경우 {name}: 표준 오류에 '판정 중단: …{why}…'가 없다 ({r.stderr.strip()[-200:]})")
        if "■ H1v2" in r.stdout or "→ 판정:" in r.stdout:
            fails.append(f"경우 {name}: 판정 줄을 썼다")

    # 경우 A: 날 bg2_d000·d001(블록 bg2_w00), bg2_d008(bg2_w01), bg3_d000(bg3_w00, 기준 비용 0인 날 하나뿐인 블록)
    winA = [("bg2_h0001", "bg2_d000"), ("bg2_h0002", "bg2_d000"), ("bg2_h0030", "bg2_d001"),
            ("bg2_h0200", "bg2_d008"), ("bg3_h0005", "bg3_d000")]
    costA = {
        "bg2_h0001": {BASE: 100.0, RULE: 90.0, P: 85.0},      # 날 d000: 고정 300, 규칙 270, 주 후보 250
        "bg2_h0002": {BASE: 200.0, RULE: 180.0, P: 165.0},
        "bg2_h0030": {BASE: 50.0, RULE: 40.0, P: 41.0},       # 날 d001: 주 후보가 규칙보다 1 높다
        "bg2_h0200": {BASE: 80.0, RULE: 70.0, P: 70.0},       # 블록 w01: 무승부(d = 0)
        "bg3_h0005": {BASE: 0.0, RULE: 0.001, P: 0.0},        # 기준 비용 0: 블록·날 비교에서 빠져야 한다
    }
    A = dict(ai_none={"bg2_h0002"})  # bg2_h0002는 AI를 쓸 수 없어 대체 정책으로 계산한 창(정상 처리, 누락 아님)
    r = run(winA, costA, **A)
    out = r.stdout
    if r.returncode != 0:
        fails.append("경우 A 실행 실패: " + r.stderr[-300:])
    if out.startswith("시험 실행"):
        fails.append("경우 A: 사전 등록 입력인데 시험 실행 머리말을 썼다")
    need(out, r"완전성 확인 통과: 실제 창 5개 × T\* 2개 × 정책 8개, 합성 일정 40개\(시드 201~240, 사전 등록과 같음\)", "완전성 확인 통과 줄")
    need(out, r"AI를 쓸 수 없어 AI 정책을 대체 정책으로 계산한 창 1개: bg2_h0002", "ai = none 창 목록")
    need(out, r"창 5개 → 날 4개 → 주 블록 3개\(blocks.csv와 같음", "창 → 날 → 주 블록")
    need(out, r"기준 비용 0인 주 블록 1개\(비교에서 뺀다\): bg3_w00", "기준 비용 0 블록 제외")
    need(out, r"기준 비용 0인 날 1개\(비교에서 뺀다\): bg3_d000", "기준 비용 0 날 제외")
    need(out, r"기준 비용 0인 합성 일정 1개\(비교에서 뺀다\): fx_spn_201 ", "기준 비용 0 합성 제외")
    # 총비용: 창 5개 합(기준 0 창 포함): 주 후보 361, 규칙 380.001, 고정 430 → d_total = −4.42%p(충족)
    need(out, r"주 후보 361\.0, 규칙 380\.0, 고정-온라인 430\.0", "총비용")
    need(out, re.escape(f"d_total {100 * (361 - 380.001) / 430:+.2f}%p → 판정: 충족(총비용이 낮았다)"), "H1v2 총비용 판정")
    # 방향(참고): 블록 bg2_w00 = 날 d000 + d001: 고정 350, 규칙 310, 주 후보 291 → d = −5.43(승), bg2_w01 = 0(무)
    need(out, r"방향\(참고, 주 블록 독립을 가정한 근사 p\): n=2 승/무/패 1/1/0", "H1v2 블록 방향(참고)")
    need(out, re.escape(f"d 중앙 {(100 * (291 - 310) / 350 + 0) / 2:+.2f}%p"), "블록 d 중앙값")
    need(out, re.escape(f"d_total {100 * (361 - 430) / 430:+.2f}%p, 주 블록 후퇴 비율 0.0% → 판정: 충족"), "H2v2 판정")
    need(out, r"날 단위\(날 독립을 가정한 요약\): 주 후보 vs 규칙 n=3 승/무/패 1/1/1", "날 단위 승/무/패")
    w3, l3 = 34, 5   # 기준 0 하나 빠져 39개
    p3 = min(1.0, 2 * sum(math.comb(w3 + l3, i) for i in range(l3 + 1)) / 2 ** (w3 + l3))
    need(out, re.escape(f"n=39 승/무/패 34/0/5, p={p3:.4g}"), "H3v2 승/무/패·p(보정 없음)")
    need(out, r"H3v2 \(2차, 통계 검정\).*\n.*→ 판정: 충족", "H3v2 판정")
    need(out, r"Sbw 0/0/5", "합성 조건별(Sbw)")
    need(out, r"spn 4/0/0", "합성 조건별(기준 0 하나 뺀 spn)")
    need(out, r"쓸 문장 틀\(§2\.3\): H1v2 = 충족\(총비용이 낮았다\)", "문장 틀")
    need(out, r"T\*=1: 주 후보 vs 규칙", "탐색: T*별(판정 T*가 아닌 T*)")

    # 경우 B: 주 블록 6개 중 5개에서 주 후보가 1%씩 이기고, 큰 블록 하나에서 크게 진다 → 총비용은 주 후보가 높다
    winB, costB = [], {}
    for k in range(5):
        n = f"bg3_h{k * 7 * 24:04d}"
        winB.append((n, f"bg3_d{k * 7:03d}"))
        costB[n] = {BASE: 100.0, RULE: 90.0, P: 89.0}
    winB.append(("bg3_h0840", "bg3_d035"))
    costB["bg3_h0840"] = {BASE: 1000.0, RULE: 800.0, P: 900.0}
    r = run(winB, costB)
    out = r.stdout
    if r.returncode != 0:
        fails.append("경우 B 실행 실패: " + r.stderr[-300:])
    tB = 100 * ((5 * 89 + 900) - (5 * 90 + 800)) / (5 * 100 + 1000)
    need(out, re.escape(f"d_total {tB:+.2f}%p → 판정: 미충족(반대 방향: 총비용이 높았다)"), "경우 B: 총비용 반대 방향")
    need(out, r"방향\(참고, 주 블록 독립을 가정한 근사 p\): n=6 승/무/패 5/0/1", "경우 B: 블록 방향은 주 후보 쪽")

    # 판정 중단 경우(개정 6). 모두 경우 A의 입력을 한 군데만 바꾼다
    def swap(Bk):
        Bk = Bk.copy()
        Bk.loc[0, "days"] = "bg2_d999"
        return Bk

    def wrong_block(W):
        W = W.copy()
        W.loc[W.cluster == "bg2_d008", "block"] = "bg2_w00"
        return W

    def dup(R):
        return pd.concat([R, R.iloc[[3]]], ignore_index=True)

    def extra_window(R):
        return pd.concat([R, pd.DataFrame(rows("bg2_h0099", {BASE: 10.0}))], ignore_index=True)

    def extra_t(R):
        x = R[R.trace == "bg2_h0200"].copy()
        x = x[x.Tstar == B].assign(Tstar=3.0)
        return pd.concat([R, x], ignore_index=True)

    def cond_swap(F):
        return F.assign(trace=F.trace.replace({"fx_spn_201": "fx_spw_201", "fx_spw_202": "fx_spn_202"}))

    neg = [
        ("C", dict(blocks_override=swap), "blocks.csv와 windows_day.csv의 블록 구성이 다르다"),
        ("C2", dict(w_mut=wrong_block), "block 열이 정의"),
        ("D", dict(real_mut=lambda R: R[R.trace != "bg2_h0030"]), "실제 결과가 없는 일정 1개: bg2_h0030"),
        ("E", dict(real_mut=lambda R: drop(R, trace="bg2_h0001", Tstar=B, policy=RULE)),
         "실제 빠진 (일정, T*, 정책) 1개: bg2_h0001 T*=2.6 규칙-온라인+대체"),
        ("E2", dict(fx_mut=lambda F: drop(F, trace="fx_sbn_203", Tstar=B, policy=P)),
         "합성 빠진 (일정, T*, 정책) 1개: fx_sbn_203 T*=2.6 C3 AI→규칙대체"),
        ("F", dict(real_mut=lambda R: drop(R, trace="bg2_h0200", Tstar=1.0)), "실제 빠진 (일정, T*, 정책) 8개"),
        ("G", dict(fx_base=901), "합성: 일정 집합이 사전 등록(조건 8 × 반복 5, 시드 201~240)과 다르다"),
        ("G2", dict(fx_mut=lambda F: F[F.trace != "fx_Sbw_240"]), "합성: 일정 집합이 사전 등록"),
        ("G3", dict(fx_mut=cond_swap), "합성: 일정 집합이 사전 등록"),
        ("H", dict(real_mut=lambda R: setv(R, "cost_MiB", float("nan"), trace="bg2_h0001", Tstar=B, policy=P)),
         "실제 비용이 숫자가 아니거나 음수인 행 1개"),
        ("H2", dict(real_mut=lambda R: setv(R, "cost_MiB", -1.0, trace="bg2_h0002", Tstar=1.0, policy=RULE)),
         "실제 비용이 숫자가 아니거나 음수인 행 1개"),
        ("I", dict(real_mut=dup), "실제: 중복 행 1개"),
        ("J", dict(real_mut=extra_window), "실제 목록에 없는 일정 1개: bg2_h0099"),
        ("K", dict(real_mut=lambda R: R.drop(columns="ai")), "실제: ai 열이 없다"),
        ("K2", dict(real_mut=lambda R: R.assign(ai=R.ai.where(R.trace != "bg2_h0030", "fail"))), "ai 열 값이 ok·none이 아니다"),
        ("L", dict(real_mut=extra_t), "실제 --tlist에 없는 T*: [3.0]"),
        ("M", dict(tlist="1"), "판정 T* 2.6가 --tlist 1에 없다"),
        ("N", dict(real_mut=lambda R: R.assign(model=R.model.where(R.trace != "bg2_h0001", "v1"))), "평가 모델이 chunk가 아니다"),
    ]
    for name, kw, why in neg:
        must_stop(name, run(winA, costA, **{**A, **kw}), why)

    # 경우 G4: 시드가 다른 합성(시험 실행)은 --test-mode에서만 끝까지 계산하고, 머리에 '판정이 아니다'를 적는다
    r = run(winA, costA, fx_base=901, extra=("--test-mode",), **A)
    if r.returncode != 0:
        fails.append("경우 G4 실행 실패: " + r.stderr[-300:])
    if not r.stdout.startswith("시험 실행(--test-mode): 사전 등록 입력이 아닐 수 있어 아래 판정 줄은 판정이 아니다"):
        fails.append("경우 G4: 시험 실행 머리말이 첫 줄에 없다")
    need(r.stdout, r"합성 일정 40개\(시드 901~940\)\n", "경우 G4: 시드 901~940을 적고 '사전 등록과 같음'을 쓰지 않는다")
    # 시험 실행에서도 실제 창의 완전성 검사는 그대로다
    must_stop("G5", run(winA, costA, fx_base=901, extra=("--test-mode",), real_mut=lambda R: R[R.trace != "bg2_h0030"], **A),
              "실제 결과가 없는 일정 1개: bg2_h0030")

    if fails:
        print("selftest 실패:", "; ".join(fails))
        sys.exit(1)
    print(f"selftest OK (판정 경우 A·B·G4, 판정 중단 경우 {len(neg) + 1}개)")


if __name__ == "__main__":
    main()
