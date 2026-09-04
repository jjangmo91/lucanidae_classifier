"""테스트셋 크기별 검정력·신뢰구간 계산.

DESIGN.md §12.1 의 수치를 산출한 스크립트.
실행: python scripts/analysis/power_analysis.py
출력: experiments/analysis_v7/
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "experiments" / "analysis_v7"
OUTDIR.mkdir(parents=True, exist_ok=True)

import math
import numpy as np
from scipy import stats

OUT = []
SCR = str(OUTDIR) + "/"

p = 0.73
OUT.append("=== accuracy 95% CI half-width at p=0.73 ===")
for n in [199, 200, 400, 600, 840, 1000, 1900]:
    hw = 1.96 * math.sqrt(p * (1 - p) / n)
    OUT.append(f"  n={n:5d}  +/- {hw*100:4.1f} points")

OUT.append("")
OUT.append("=== n required for target precision (p=0.73) ===")
for target in [0.06, 0.05, 0.04, 0.03, 0.02]:
    n = p * (1 - p) * (1.96 / target) ** 2
    OUT.append(f"  +/- {target*100:3.0f} points -> n = {math.ceil(n):5d}")

# McNemar power: two models on same test set, differing on b/c discordant pairs
OUT.append("")
OUT.append("=== paired comparison (McNemar) power to detect a true difference ===")
OUT.append("assumes discordance rate 20% of test items")
for n in [199, 400, 800, 1600]:
    for delta in [0.03, 0.05, 0.10]:
        disc = 0.20 * n
        # among discordant, prob favoring better model
        pi = 0.5 + (delta * n) / (2 * disc) if disc > 0 else 0.5
        if pi >= 1:
            OUT.append(f"  n={n:5d} delta={delta*100:4.1f}pt -> not representable")
            continue
        se = math.sqrt(0.25 / disc)
        z = (pi - 0.5) / se
        power = 1 - stats.norm.cdf(1.96 - z)
        OUT.append(f"  n={n:5d} delta={delta*100:4.1f}pt -> power = {power*100:4.1f}%")

# per-class precision for rare species
OUT.append("")
OUT.append("=== per-species recall CI half-width (p=0.6) ===")
for n in [5, 10, 20, 30, 50, 100]:
    hw = 1.96 * math.sqrt(0.6 * 0.4 / n)
    OUT.append(f"  n_test={n:4d} per species -> +/- {hw*100:4.1f} points")

open(SCR + "power_out.txt", "w", encoding="utf-8").write("\n".join(OUT))
