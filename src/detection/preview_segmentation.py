"""
세그먼테이션 마스킹 결과 미리보기

학습된 detector로 랜덤 샘플 이미지를 처리하고
원본 | 마스킹 결과 를 나란히 저장하여 품질을 시각적으로 확인한다.

사용법:
    python -m src.detection.preview_segmentation           # 클래스당 2장 (기본)
    python -m src.detection.preview_segmentation --n 5     # 클래스당 5장
    python -m src.detection.preview_segmentation --n 1 --split val
"""

import argparse
import random
from pathlib import Path

import yaml
from PIL import Image, ImageDraw, ImageFont

from src.ml.detector import BeetleSegmenter

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def make_comparison(original: Image.Image, crops: list[Image.Image], detected: bool, label: str) -> Image.Image:
    """원본과 마스킹 결과를 가로로 이어붙인 비교 이미지 생성."""
    thumb_size = (400, 400)
    original_thumb = original.copy()
    original_thumb.thumbnail(thumb_size, Image.LANCZOS)

    crop_thumbs = []
    for crop in crops:
        t = crop.copy()
        t.thumbnail(thumb_size, Image.LANCZOS)
        crop_thumbs.append(t)

    n_cols  = 1 + len(crop_thumbs)
    w       = thumb_size[0] * n_cols + 10 * (n_cols + 1)
    h       = thumb_size[1] + 60
    canvas  = Image.new("RGB", (w, h), (240, 240, 240))

    # 원본
    x = 10
    canvas.paste(original_thumb, (x, 50))
    draw = ImageDraw.Draw(canvas)
    draw.text((x, 10), f"원본  [{label}]", fill=(0, 0, 0))

    # 마스킹 결과
    for i, ct in enumerate(crop_thumbs):
        x = 10 + (i + 1) * (thumb_size[0] + 10)
        canvas.paste(ct, (x, 50))
        tag = f"Crop #{i}" if detected else "Fallback (미검출)"
        draw.text((x, 10), tag, fill=(200, 0, 0) if not detected else (0, 120, 0))

    return canvas


def preview(config_path: str = "configs/default.yaml", n_per_class: int = 2, split: str = "train") -> None:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    source_dir  = Path(config["training"]["data_dir"]) / split
    output_dir  = Path(config["detection"]["segmented_dir"]).parent / "preview_segmentation"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not Path(config["detection"]["weights_path"]).exists():
        print(f"[오류] detector 가중치 없음: {config['detection']['weights_path']}")
        print("먼저 python train_detector.py 를 실행하세요.")
        return

    segmenter = BeetleSegmenter.from_config(config_path)

    stats = {"total": 0, "detected": 0, "fallback": 0}

    for class_dir in sorted(source_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name
        images = [p for p in class_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]

        random.seed(42)
        samples = random.sample(images, min(n_per_class, len(images)))

        for img_path in samples:
            stats["total"] += 1
            image          = Image.open(img_path).convert("RGB")
            crops, detected = segmenter.segment(image)

            if detected:
                stats["detected"] += 1
            else:
                stats["fallback"] += 1

            comparison = make_comparison(image, crops, detected, class_name)
            out_name   = f"{class_name}__{img_path.stem}.jpg"
            comparison.save(output_dir / out_name, quality=90)

    print(f"미리보기 저장: {output_dir}  ({stats['total']}장)")
    print(f"  검출 성공: {stats['detected']}장  |  Fallback(미검출): {stats['fallback']}장")
    print()
    print("Windows 탐색기에서 해당 폴더를 열어 이미지를 확인하세요.")
    print(f"  explorer {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="세그먼테이션 마스킹 결과 미리보기")
    parser.add_argument("--n",     type=int, default=2,       help="클래스당 샘플 수 (기본: 2)")
    parser.add_argument("--split", type=str, default="train", help="데이터 split (train/val/test)")
    parser.add_argument("--config",type=str, default="configs/default.yaml")
    args = parser.parse_args()

    preview(config_path=args.config, n_per_class=args.n, split=args.split)


if __name__ == "__main__":
    main()
