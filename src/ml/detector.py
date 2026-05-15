"""
YOLOv8-seg 기반 사슴벌레 세그먼테이션 추론 wrapper

지원 모드:
    full      - Detector 미사용, 원본 이미지 그대로 (서비스 레이어에서 처리)
    bbox      - BBox 직사각형 crop (마스크 없음, 배경 일부 포함)
    seg_hard  - Binary 마스크 + 회색 배경 (sharp 경계)
    seg_soft  - Alpha blending 마스크 + 회색 배경 (부드러운 경계)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

_BG_COLOR   = (128, 128, 128)
_SOFT_SIGMA = 5   # GaussianBlur radius for seg_soft


class BeetleSegmenter:
    def __init__(
        self,
        weights_path: str,
        confidence: float = 0.25,
        iou_threshold: float = 0.45,
        mode: str = "seg_hard",
    ):
        if mode not in ("bbox", "seg_hard", "seg_soft"):
            raise ValueError(f"mode must be bbox|seg_hard|seg_soft, got: {mode}")

        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError("pip install ultralytics") from e

        self.model         = YOLO(weights_path)
        self.confidence    = confidence
        self.iou_threshold = iou_threshold
        self.mode          = mode

    def segment(self, image: Image.Image) -> tuple[list[Image.Image], bool]:
        """
        Returns:
            (crops, detected)
            detected=False → [원본 이미지] fallback
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
        return self.segment(Image.open(image_path).convert("RGB"))

    def _extract_crops(self, image: Image.Image, results) -> list[Image.Image]:
        crops     = []
        w, h      = image.size
        img_array = np.array(image)

        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                continue

            has_masks = result.masks is not None and len(result.masks.data) > 0
            boxes     = result.boxes.xyxy.cpu().numpy()
            masks     = result.masks.data.cpu().numpy() if has_masks else None

            for i, box in enumerate(boxes):
                x1, y1 = max(0, int(box[0])), max(0, int(box[1]))
                x2, y2 = min(w, int(box[2])), min(h, int(box[3]))
                if x2 <= x1 or y2 <= y1:
                    continue

                if self.mode == "bbox":
                    crops.append(image.crop((x1, y1, x2, y2)))

                elif self.mode == "seg_hard":
                    if has_masks:
                        mask_pil  = Image.fromarray((masks[i] * 255).astype(np.uint8))
                        mask_full = np.array(mask_pil.resize((w, h), Image.NEAREST)) > 127
                        masked    = img_array.copy()
                        masked[~mask_full] = _BG_COLOR
                        crops.append(Image.fromarray(masked[y1:y2, x1:x2]))
                    else:
                        crops.append(image.crop((x1, y1, x2, y2)))

                elif self.mode == "seg_soft":
                    if has_masks:
                        mask_pil     = Image.fromarray((masks[i] * 255).astype(np.uint8))
                        mask_resized = mask_pil.resize((w, h), Image.BILINEAR)
                        mask_soft    = mask_resized.filter(ImageFilter.GaussianBlur(radius=_SOFT_SIGMA))
                        alpha        = np.array(mask_soft, dtype=np.float32) / 255.0
                        bg           = np.full_like(img_array, _BG_COLOR, dtype=np.float32)
                        blended      = (img_array.astype(np.float32) * alpha[..., None]
                                        + bg * (1.0 - alpha[..., None])).astype(np.uint8)
                        crops.append(Image.fromarray(blended[y1:y2, x1:x2]))
                    else:
                        crops.append(image.crop((x1, y1, x2, y2)))

        return crops

    @classmethod
    def from_config(cls, config_path: str = "configs/default.yaml") -> "BeetleSegmenter":
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        det_cfg  = config["detection"]
        inf_mode = config.get("inference", {}).get("preprocessing_mode", "seg_hard")
        mode     = inf_mode if inf_mode != "full" else "seg_hard"
        return cls(
            weights_path  = det_cfg["weights_path"],
            confidence    = det_cfg.get("confidence", 0.25),
            iou_threshold = det_cfg.get("iou_threshold", 0.45),
            mode          = mode,
        )
