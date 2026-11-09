#!/usr/bin/env python3
"""g2_breakdown_summary.py — breakdown.json 요약: 연산별 CPU(3회 평균·범위), ΔI 대비 비율, 체크섬 확인, 손익분기 참고값"""
import json, sys
d = json.load(open(sys.argv[1]))
its = d["iterations"]; ops = d["ops"]
S_C = 1.5608601903  # V_pool = S_model·c (초), g2_measure.py의 G1 확정값
ms = lambda ns: ns / 1e6
ok = all(i["checksum_c3"] == i["checksum_split"] for i in its) and len({i["checksum_c3"] for i in its}) == 1 and len({i["checksum_rule"] for i in its}) == 1
print(f"창 {d['n_windows']}개, 체크섬(c3 = split, 회차 간 동일): {'통과' if ok else '실패'}")
dI = [i["dI_ns"] for i in its]
print(f"ΔI(원본 방식) 평균 {ms(sum(dI)/3):.3f} ms, 범위 {ms(min(dI)):.3f}~{ms(max(dI)):.3f}")
tot = [sum(i["op_ns"]) for i in its]
for k, name in enumerate(ops):
    v = [i["op_ns"][k] for i in its]
    print(f"  op{k+1} {name:22s} {ms(sum(v)/3):10.3f} ms ({ms(min(v)):.3f}~{ms(max(v)):.3f})  Σop 대비 {sum(v)/sum(tot)*100:5.1f}%")
print(f"  Σop 평균 {ms(sum(tot)/3):.3f} ms, ΔI − Σop(잔여) 평균 {ms((sum(dI)-sum(tot))/3):.3f} ms")
sc = [i["cpu_split_ns"] - i["cpu_rule_ns"] - sum(i["op_ns"]) for i in its]
print(f"  분해 실행의 잔여(split − rule − Σop) 평균 {ms(sum(sc)/3):.3f} ms (공통 작업 차이 + 타이머 비용)")
m = [sum(i['op_ns'][k] for i in its) / 3 / 1e9 for k in range(5)]
dI_s = sum(dI) / 3 / 1e9
I_AI = m[1]; I_fix = dI_s - I_AI   # I_fixed = ΔI − I_AI (잔여 포함, ΔI 전체를 보존)
print("\n[손익분기 참고, 계산] 가속 가능 I_AI = op2(부호화+표 조회+κ 배율), I_fixed = ΔI − I_AI")
if d["n_windows"] != 420:
    print("  420창이 아니므로 V_pool과 비교하지 않는다(스모크 시험)."); sys.exit(0)
print(f"  I_AI = {I_AI*1e3:.3f} ms, I_fixed = {I_fix*1e3:.3f} ms, V_pool = {S_C*1e3:.3f} ms")
print(f"  NPU가 무한히 빨라도(k→∞) G = V_pool − I_fixed = {(S_C - I_fix)*1e3:.3f} ms")
den = S_C - I_fix
print("  " + (f"k* = I_AI / (V_pool − I_fixed) = {I_AI/den:.3f} (C_HW = 0일 때)" if den > 0 else "분모 ≤ 0: 어떤 k로도 G > 0이 되지 않는다"))
