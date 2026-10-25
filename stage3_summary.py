#!/usr/bin/env python3
"""stage3_summary.py — 3단계 측정 요약

  python3 stage3_summary.py tstar exp/stage3/tstar.csv            # 조건별 실효 T*와 판정용 T*(규칙: 아래 TSTAR_RULE)
  python3 stage3_summary.py replay exp/stage3/replay.csv exp/stage3/dec   # 정책별 실제 비용 vs 유체 모델 예측

비용 단위: 'MiB·초 환산'. CPU 1초의 값 = 가격 비율 × 1024 MiB·초(가격 비율 = vCPU 1시간 가격 ÷ GB 1시간 가격, 기본 9.107).
  실제 비용 = (반납 + 재확보 CPU 초) × 가격 비율 × 1024 + 붙잡은 MiB·초(장부, RSS 대조로 확인)
  유체 예측 = 재확보 MiB × T* + 낭비 MiB·초 (내보낼 때 쓴 T*)
"""
import csv
import glob
import json
import os
import statistics as st
import sys

RATIO = 9.107
TSTAR_RULE = dict(chunk_bytes="1048576", method="dontneed", thp="default", batch="16")  # 판정용 T* 조건(v2 사전 등록 초안과 같다)


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(line for line in f if not line.startswith("#")))


def tstar(path):
    rows = read_csv(path)
    print("chunk method thp batch | 반납 µs/MiB  재확보 µs/MiB  폴트/MiB  RSS감소 | 실효 T*(초)")
    for r in rows:
        ok = float(r["rss_drop_frac"]) >= 0.5
        print(f"{int(r['chunk_bytes']) >> 20}M {r['method']:8s} {r['thp']:7s} {r['batch']:>3s} | {float(r['release_cpu_us_per_MiB']):9.1f}"
              f"  {float(r['refault_cpu_us_per_MiB']):12.1f}  {float(r['faults_per_MiB']):8.1f}  {float(r['rss_drop_frac']):6.2f} | "
              f"{float(r['tstar_eff_s']):6.2f}{'' if ok else '  (반납해도 RSS가 안 줄어 비용 비교에 쓰지 않는다)'}")
    sel = [r for r in rows if all(r[k] == v for k, v in TSTAR_RULE.items())]
    if sel:
        print(f"\n판정용 T* 조건 {TSTAR_RULE}: 실효 T* = {st.median(float(r['tstar_eff_s']) for r in sel):.2f}초 "
              f"(행 {len(sel)}개의 중앙값; 여러 번 돌렸으면 모두 합친 중앙값을 쓴다)")


def replay(path, decdir):
    rows = read_csv(path)
    by = {}
    for r in rows:
        by.setdefault((r["trace"], r["chunk"], r["tstar"], r["method"], r["thp"]), {})[r["policy"]] = r
    for key, pols in sorted(by.items()):
        trace, chunk, b, method, thp = key
        fl = json.load(open(os.path.join(decdir, f"{trace}_c{chunk}_t{b}_fluid.json")))["policies"]
        b = float(b)
        print(f"\n■ {trace} 청크 {chunk} T*(내보내기) {b:g}초, {method}/{thp}")
        print("정책   | 재확보 MiB(실측/유체) 반납 MiB | CPU 초  RSS오차max | 실제 비용  유체 예측 (MiB·초 환산) | 고정 대비 실제·유체(%) | 함축 T*")
        base = None
        out = []
        for pol in ("fixed", "rule", "ai", "never", "now"):
            if pol not in pols:
                continue
            r, f = pols[pol], fl[pol]
            cpu = float(r["rel_cpu_s"]) + float(r["ref_cpu_s"])
            real = cpu * RATIO * 1024 + f["waste_MiB_s"]
            pred = f["refault_MiB"] * b + f["waste_MiB_s"]
            imp = cpu / f["refault_MiB"] * RATIO * 1024 if f["refault_MiB"] > 0 else float("nan")
            out.append((pol, float(r["ref_MiB"]), f["refault_MiB"], float(r["rel_MiB"]), cpu, float(r["rss_dev_max_MiB"]), real, pred, imp))
            if pol == "fixed":
                base = (real, pred)
        for pol, rm, fm, rel, cpu, dev, real, pred, imp in out:
            rr = (real / base[0] - 1) * 100 if base else float("nan")
            pp = (pred / base[1] - 1) * 100 if base else float("nan")
            print(f"{pol:6s} | {rm:9.0f}/{fm:<9.0f} {rel:9.0f} | {cpu:7.3f} {dev:8.1f} | {real:10.0f} {pred:10.0f} | {rr:+7.1f} {pp:+7.1f} | {imp:6.2f}")


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "tstar":
        tstar(sys.argv[2])
    elif len(sys.argv) >= 4 and sys.argv[1] == "replay":
        replay(sys.argv[2], sys.argv[3])
    else:
        sys.exit(__doc__)
