"""다중검출 원본의 박스 겹침 구조 측정 (한 개체 과분할 vs 실제 다개체).

DESIGN.md §4.8.1 의 수치를 산출한 스크립트.
실행: python scripts/analysis/probe_multi_detect.py
출력: experiments/analysis_v7/
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "experiments" / "analysis_v7"
OUTDIR.mkdir(parents=True, exist_ok=True)

"""205개 다중검출 원본에 detector를 재실행해 박스 겹침 구조를 측정한다.
목적: '한 개체의 부위 분할' vs '실제 여러 개체'를 구분.
"""
import json, sys
from pathlib import Path
from PIL import Image
from ultralytics import YOLO

ROOT = Path("C:/Users/doruc/project/lucanidae_classifier")
SCR = Path(__file__).parent
multi = json.load(open(SCR / "multi.json"))

model = YOLO(str(ROOT / "models/weights/best_detector.pt"))


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0, 0.0
    inter = (x2 - x1) * (y2 - y1)
    aa = (a[2] - a[0]) * (a[3] - a[1])
    bb = (b[2] - b[0]) * (b[3] - b[1])
    smaller = min(aa, bb)
    return inter / (aa + bb - inter), inter / smaller  # IoU, containment of smaller


rows = []
for key in multi:
    split, sp, stem = key.split("/")
    cand = list((ROOT / f"data/final/{split}/{sp}").glob(stem + ".*"))
    if not cand:
        continue
    img = Image.open(cand[0]).convert("RGB")
    W, H = img.size
    r = model.predict(img, conf=0.25, iou=0.45, verbose=False)[0]
    if r.boxes is None or len(r.boxes) == 0:
        continue
    boxes = [[float(v) for v in b] for b in r.boxes.xyxy.cpu().numpy()]
    confs = [float(c) for c in r.boxes.conf.cpu().numpy()]
    areas = [((b[2] - b[0]) * (b[3] - b[1])) / (W * H) for b in boxes]

    n = len(boxes)
    # 가장 큰 박스에 대한 각 박스의 포함률
    big = max(range(n), key=lambda i: areas[i])
    contained = 0
    for i in range(n):
        if i == big:
            continue
        _, c = iou(boxes[i], boxes[big])
        if c >= 0.80:
            contained += 1
    # 서로 거의 겹치지 않는 큰 박스 쌍 = 진짜 다개체 신호
    disjoint_big = 0
    idx_big = [i for i in range(n) if areas[i] >= 0.10]
    for ii in range(len(idx_big)):
        for jj in range(ii + 1, len(idx_big)):
            i, j = idx_big[ii], idx_big[jj]
            _, c = iou(boxes[i], boxes[j])
            if c < 0.30:
                disjoint_big += 1
    rows.append({
        "key": key, "n": n, "areas": areas, "confs": confs,
        "contained_in_largest": contained, "disjoint_big_pairs": disjoint_big,
        "largest_area": areas[big],
    })

json.dump(rows, open(SCR / "boxprobe.json", "w"))
print("analyzed", len(rows))
