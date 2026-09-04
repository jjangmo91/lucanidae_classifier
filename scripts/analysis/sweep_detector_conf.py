"""detector confidence 임계값별 다중검출 감소 효과 측정.

DESIGN.md §4.8.1 의 수치를 산출한 스크립트.
실행: python scripts/analysis/sweep_detector_conf.py
출력: experiments/analysis_v7/
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "experiments" / "analysis_v7"
OUTDIR.mkdir(parents=True, exist_ok=True)

"""confidence / 최대박스 정책별로 다중검출이 얼마나 줄어드는지 측정."""
import json, random
from pathlib import Path
from PIL import Image
from ultralytics import YOLO

ROOT = Path("C:/Users/doruc/project/lucanidae_classifier")
SCR = Path(__file__).parent
multi = list(json.load(open(SCR / "multi.json")).keys())

# 다중검출 205장 + 정상 단일검출 표본 200장
allimgs = []
for key in multi:
    split, sp, stem = key.split("/")
    c = list((ROOT / f"data/final/{split}/{sp}").glob(stem + ".*"))
    if c:
        allimgs.append(("multi", c[0]))

random.seed(0)
singles = []
for p in (ROOT / "data/final").glob("*/*/*"):
    key = "/".join([p.parts[-3], p.parts[-2], p.stem])
    if key not in multi:
        singles.append(p)
for p in random.sample(singles, 200):
    allimgs.append(("single", p))

model = YOLO(str(ROOT / "models/weights/best_detector.pt"))
out = {}
for conf in [0.25, 0.40, 0.50, 0.60, 0.70]:
    stat = {"multi_still_multi": 0, "multi_now_single": 0, "multi_now_zero": 0,
            "single_stays_single": 0, "single_now_zero": 0, "single_became_multi": 0}
    for kind, p in allimgs:
        img = Image.open(p).convert("RGB")
        r = model.predict(img, conf=conf, iou=0.45, verbose=False)[0]
        n = 0 if r.boxes is None else len(r.boxes)
        if kind == "multi":
            if n >= 2:
                stat["multi_still_multi"] += 1
            elif n == 1:
                stat["multi_now_single"] += 1
            else:
                stat["multi_now_zero"] += 1
        else:
            if n == 1:
                stat["single_stays_single"] += 1
            elif n == 0:
                stat["single_now_zero"] += 1
            else:
                stat["single_became_multi"] += 1
    out[conf] = stat
    print(conf, stat, flush=True)

json.dump(out, open(SCR / "confsweep.json", "w"))
