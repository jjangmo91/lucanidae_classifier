"""
YOLOv8-seg 기반 사슴벌레 세그먼테이션 추론 wrapper

- best_detector.pt 를 로드하여 이미지에서 사슴벌레를 검출·마스킹
- 세그멘테이션 마스크로 배경을 중립색(회색)으로 치환 후 bbox crop 반환
  → 분류기가 배경 노이즈 없이 벌레 형태에만 집중할 수 있음
- Fallback 정책: 검출 없음 → 원본 이미지 반환 (분류기에 그대로 전달)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

# 배경을 채울 중립색 (ImageNet 평균에 근접한 회색)
_BG_COLOR = (128, 128, 128)


class BeetleSegmenter:
    def __init__(
        self,
        weights_path: str,
        confidence: float = 0.25,
        iou_threshold: float = 0.45,
    ):
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError(
                "ultralytics 가 설치되지 않았습니다.\n"
                "pip install ultralytics"
            ) from e

        self.model         = YOLO(weights_path)
        self.confidence    = confidence
        self.iou_threshold = iou_threshold

    def segment(self, image: Image.Image) -> tuple[list[Image.Image], bool]:
        """
        단일 PIL 이미지에서 사슴벌레를 검출하고 crop 목록을 반환한다.

        Returns:
            (crops, detected)
            - detected=True:  세그먼테이션 마스크로 배경 제거 후 bbox crop 목록
            - detected=False: [원본 이미지] (Fallback)
        """
        results = self.model.predict(
            source=image,
            conf=self.confidence,
            iou=self.iou_threshold,
            verbose=False,
        )

        crops = self._extract_crops(image, results)
        if not crops:
            return [image], False
        return crops, True

    def segment_path(self, image_path: str | Path) -> tuple[list[Image.Image], bool]:
        """파일 경로를 받아 segment() 호출."""
        image = Image.open(image_path).convert("RGB")
        return self.segment(image)

    def _extract_crops(self, image: Image.Image, results) -> list[Image.Image]:
        crops = []
        w, h  = image.size
        img_array = np.array(image)  # (H, W, 3)

        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                continue

            has_masks = result.masks is not None and len(result.masks.data) > 0

            boxes = result.boxes.xyxy.cpu().numpy()
            masks = result.masks.data.cpu().numpy() if has_masks else None

            for i, box in enumerate(boxes):
                x1 = max(0, int(box[0]))
                y1 = max(0, int(box[1]))
                x2 = min(w, int(box[2]))
                y2 = min(h, int(box[3]))

                if x2 <= x1 or y2 <= y1:
                    continue

                if has_masks:
                    # 마스크를 원본 이미지 크기로 리사이즈
                    mask_raw = masks[i]  # (mask_h, mask_w), float [0,1]
                    mask_pil = Image.fromarray((mask_raw * 255).astype(np.uint8))
                    mask_full = np.array(mask_pil.resize((w, h), Image.NEAREST)) > 127

                    # 배경 → 중립색 치환
                    masked = img_array.copy()
                    masked[~mask_full] = _BG_COLOR

                    crop = Image.fromarray(masked[y1:y2, x1:x2])
                else:
                    # seg 마스크 없을 경우 bbox crop으로 fallback
                    crop = image.crop((x1, y1, x2, y2))

                crops.append(crop)

        return crops

    @classmethod
    def from_config(cls, config_path: str = "configs/default.yaml") -> "BeetleSegmenter":
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        cfg = config["detection"]
        return cls(
            weights_path  = cfg["weights_path"],
            confidence    = cfg.get("confidence", 0.25),
            iou_threshold = cfg.get("iou_threshold", 0.45),
        )
