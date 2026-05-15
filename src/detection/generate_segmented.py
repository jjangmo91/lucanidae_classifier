"""
학습된 Detector를 사용해 data/final/ 이미지를 전처리하고
data/final_{mode}/ 에 ImageFolder 구조로 저장한다.

지원 모드:
    bbox      - BBox 직사각형 crop (배경 일부 포함)
    seg_hard  - Binary 마스크 + 회색 배경
    seg_soft  - Alpha blending 마스크 + 회색 배경

사용법:
    python -m src.detection.generate_segmented              # default.yaml mode 사용
    python -m src.detection.generate_segmented --mode bbox
    python -m src.detection.generate_segmented --mode seg_soft
"""

import argparse
import shutil
from pathlib import Path

import yaml
from PIL import Image
from tqdm import tqdm

from src.ml.detector import BeetleSegmenter

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def generate(config_path: str = "configs/default.yaml", mode: str | None = None) -> None:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    det_cfg    = config["detection"]
    source_dir = Path("./data/final")

    if mode is None:
        inf_mode = config.get("inference", {}).get("preprocessing_mode", "seg_hard")
        mode     = inf_mode if inf_mode != "full" else "seg_hard"

    if mode not in ("bbox", "seg_hard", "seg_soft"):
        raise ValueError(f"mode must be bbox|seg_hard|seg_soft, got: {mode}")

    output_dir = Path(f"./data/final_{mode}")

    segmenter = BeetleSegmenter(
        weights_path  = det_cfg["weights_path"],
        confidence    = det_cfg.get("confidence", 0.25),
        iou_threshold = det_cfg.get("iou_threshold", 0.45),
        mode          = mode,
    )

    if output_dir.exists():
        shutil.rmtree(output_dir)

    stats = {"total": 0, "fallback": 0, "crops": 0}

    for split_dir in sorted(source_dir.iterdir()):
        if not split_dir.is_dir():
            continue
        split = split_dir.name

        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            class_name = class_dir.name

            out_class_dir = output_dir / split / class_name
            out_class_dir.mkdir(parents=True, exist_ok=True)

            images = [p for p in class_dir.iterdir() if p.suffix in IMAGE_EXTENSIONS]

            for img_path in tqdm(images, desc=f"{split}/{class_name}", leave=False):
                stats["total"] += 1
                try:
                    image           = Image.open(img_path).convert("RGB")
                    crops, detected = segmenter.segment(image)

                    if not detected:
                        stats["fallback"] += 1

                    for i, crop in enumerate(crops):
                        suffix = img_path.suffix.lower() or ".jpg"
                        # sex_labels.csv 매칭을 위해 단일 crop은 원본 파일명 유지
                        out_name = img_path.name if (i == 0 and len(crops) == 1) else f"{img_path.stem}_{i}{suffix}"
                        crop.save(out_class_dir / out_name)
                        stats["crops"] += 1

                except Exception as e:
                    shutil.copy2(img_path, out_class_dir / img_path.name)
                    stats["fallback"] += 1
                    stats["crops"]    += 1

    print(f"\n[{mode}] 세그먼테이션 완료")
    print(f"  처리 이미지: {stats['total']}장")
    print(f"  Fallback:   {stats['fallback']}장  ({stats['fallback'] / stats['total'] * 100:.1f}%)")
    print(f"  저장 crops: {stats['crops']}장")
    print(f"  출력 경로:  {output_dir}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default=None,
                        help="bbox | seg_hard | seg_soft (기본값: default.yaml inference.preprocessing_mode)")
    args = parser.parse_args()
    generate(mode=args.mode)
