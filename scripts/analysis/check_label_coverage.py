"""성별·형태 라벨의 실제 매칭률 점검 (split 키 유실 진단).

DESIGN.md §4.3 의 수치를 산출한 스크립트.
실행: python scripts/analysis/check_label_coverage.py
출력: experiments/analysis_v7/
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "experiments" / "analysis_v7"
OUTDIR.mkdir(parents=True, exist_ok=True)

import glob, sys
sys.path.insert(0, str(ROOT))
from src.training.dataset import _load_sex_labels, _label_key

R = str(ROOT) + '/'
lab = _load_sex_labels(R + "data/labels/sex_labels.csv")
print("csv keys:", len(lab))
for mode, d in [("full", "data/final"), ("bbox", "data/final_bbox")]:
    tot = ks = kf = miss = 0
    for f in glob.glob(R + d + "/*/*/*"):
        tot += 1
        i = lab.get(_label_key(f))
        if i is None:
            miss += 1
            continue
        if i["sex"] != "unknown":
            ks += 1
        if i["male_form"] != "unknown":
            kf += 1
    print(f"{mode:5s} n={tot:5d} miss={miss:4d} sex={ks:5d} ({ks/tot*100:.1f}%) form={kf:5d} ({kf/tot*100:.1f}%)")
