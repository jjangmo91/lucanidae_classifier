"""다중검출 이미지 검수 큐를 생성한다.

검출기가 한 장에서 박스를 2개 이상 잡은 원본을 찾아, 번호를 매긴 박스를 그린
검수용 이미지와 판정 입력용 CSV를 만든다. 사람이 CSV의 verdict 칸만 채우면 된다.

배경: v6 데이터에서 205장(10.3%)이 2개 이상 검출됐고, 육안 확인 결과 대부분
여러 마리가 아니라 한 마리를 다리·더듬이 조각으로 쪼갠 것이었다 (DESIGN.md 4.8).
모든 crop이 원본의 종 라벨을 그대로 상속하므로 조각이 종 라벨을 달고 학습에 들어간다.

verdict 값:
    single    한 마리인데 과분할됨   -> 박스 병합 후 1장으로 저장
    multiple  실제로 여러 마리       -> 개체별 종 라벨링 필요
    false     오검출(배경·손 등 포함) -> 해당 박스 제거
    unclear   판단 불가              -> 학습에서 제외

사용법:
    python scripts/review_multi_detect.py
    python scripts/review_multi_detect.py --split train --limit 50
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import yaml
from PIL import Image, ImageDraw
from tqdm import tqdm
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
PALETTE = ["#e84a3f", "#2f7fd1", "#3aa06a", "#f2a03d", "#8e5bd0",
           "#00a0a0", "#d94f8a", "#7a7a2a", "#4a6fd0", "#c04040"]


def _iou_contain(a, b) -> tuple[float, float]:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0, 0.0
    inter = (x2 - x1) * (y2 - y1)
    aa = (a[2] - a[0]) * (a[3] - a[1])
    bb = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (aa + bb - inter), inter / min(aa, bb)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=str(ROOT / "data" / "final"))
    ap.add_argument("--out", default=str(ROOT / "experiments" / "multi_detect_review"))
    ap.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    ap.add_argument("--split", default=None, help="train|val|test (기본: 전체)")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    cfg = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    det = cfg["detection"]
    model = YOLO(str(ROOT / det["weights_path"].lstrip("./")))
    conf = det.get("confidence", 0.25)
    iou = det.get("iou_threshold", 0.45)

    src = Path(a.source)
    out = Path(a.out)
    (out / "images").mkdir(parents=True, exist_ok=True)

    paths = [p for p in src.glob("*/*/*") if p.suffix in EXTS]
    if a.split:
        paths = [p for p in paths if p.parts[-3] == a.split]
    paths.sort()
    if a.limit:
        paths = paths[: a.limit]

    rows = []
    n_multi = 0
    for p in tqdm(paths, desc="scanning"):
        img = Image.open(p).convert("RGB")
        W, H = img.size
        r = model.predict(img, conf=conf, iou=iou, verbose=False)[0]
        if r.boxes is None or len(r.boxes) < 2:
            continue
        n_multi += 1

        boxes = [[float(v) for v in b] for b in r.boxes.xyxy.cpu().numpy()]
        confs = [float(c) for c in r.boxes.conf.cpu().numpy()]
        areas = [((b[2] - b[0]) * (b[3] - b[1])) / (W * H) for b in boxes]

        # 최대 박스에 대한 포함 관계 — 과분할 판단 힌트
        big = max(range(len(boxes)), key=lambda i: areas[i])
        contained = sum(
            1 for i in range(len(boxes))
            if i != big and _iou_contain(boxes[i], boxes[big])[1] >= 0.80
        )

        vis = img.copy()
        d = ImageDraw.Draw(vis)
        for i, b in enumerate(boxes):
            c = PALETTE[i % len(PALETTE)]
            d.rectangle(b, outline=c, width=max(2, int(min(W, H) * 0.004)))
            d.text((b[0] + 4, b[1] + 4), f"{i}", fill=c)
        name = f"{p.parts[-3]}__{p.parts[-2]}__{p.stem}.jpg"
        vis.save(out / "images" / name, quality=88)

        rows.append({
            "review_image": name,
            "source_path": str(p.relative_to(ROOT)).replace("\\", "/"),
            "split": p.parts[-3],
            "species": p.parts[-2],
            "n_boxes": len(boxes),
            "areas": ";".join(f"{x:.3f}" for x in areas),
            "confs": ";".join(f"{x:.2f}" for x in confs),
            "n_contained_in_largest": contained,
            "hint": "over-seg?" if contained == len(boxes) - 1 else "",
            "verdict": "",
            "keep_box_indices": "",
            "note": "",
        })

    csv_path = out / "review_queue.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["review_image"])
        w.writeheader()
        w.writerows(rows)

    (out / "README.md").write_text(
        "\n".join([
            "# 다중검출 검수 큐",
            "",
            f"검사 대상 {len(paths)}장 중 **{n_multi}장**이 박스 2개 이상.",
            "",
            "## 작업 방법",
            "1. `images/` 의 이미지를 연다. 박스마다 번호가 붙어 있다.",
            "2. `review_queue.csv` 의 `verdict` 칸을 채운다.",
            "",
            "| verdict | 뜻 | 후속 처리 |",
            "|---|---|---|",
            "| `single` | 한 마리인데 과분할됨 | 박스 병합 후 1장 저장 |",
            "| `multiple` | 실제로 여러 마리 | 개체별 종 라벨링 |",
            "| `false` | 오검출 포함 | 해당 박스 제거 |",
            "| `unclear` | 판단 불가 | 학습 제외 |",
            "",
            "3. `multiple` 이나 `false` 인 경우 `keep_box_indices` 에 남길 번호를",
            "   쉼표로 적는다 (예: `0,2`).",
            "",
            "`hint` 열이 `over-seg?` 이면 부수 박스가 전부 최대 박스 안에 들어간 경우로,",
            "`single` 일 가능성이 높다. 다만 확인 없이 신뢰하지 말 것 — 다리처럼",
            "최대 박스 밖으로 뻗은 조각은 이 힌트에 걸리지 않는다.",
            "",
            "근거: DESIGN.md 4.8",
        ]),
        encoding="utf-8", newline="",
    )

    print(f"\n검사 {len(paths)}장 | 다중검출 {n_multi}장")
    print(f"검수 큐: {csv_path}")
    print(f"검수 이미지: {out / 'images'}")


if __name__ == "__main__":
    main()
