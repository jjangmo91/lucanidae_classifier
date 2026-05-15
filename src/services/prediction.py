"""
추론 서비스 — API 레이어와 ML 레이어 사이의 비즈니스 로직.

흐름:
    이미지 → Segmenter(crop) → Classifier(종 예측) + OOD(병렬) → result_type 결정 → 결과 반환
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from PIL import Image

from src.ml.manager import ModelManager
from src.training.dataset import EVAL_TRANSFORM

_KO_NAMES: dict[str, str] = {
    "Prosopocoilus_astacoides_blanchardi":   "두점박이사슴벌레",
    "Dorcus_titanus_castanicolor":           "넓적사슴벌레",
    "Lucanus_maculifemoratus_dybowskyi":     "사슴벌레",
    "Dorcus_hopei_binodulosus":              "왕사슴벌레",
    "Prosopocoilus_inclinatus_inclinatus":   "톱사슴벌레",
    "Dorcus_rectus_rectus":                  "애사슴벌레",
    "Dorcus_rubrofemoratus_rubrofemoratus":  "홍다리사슴벌레",
    "Prismognathus_dauricus":                "다우리아사슴벌레",
    "Dorcus_carinulatus_koreanus":           "털보왕사슴벌레",
    "Platycerus_hongwonpyoi_hongwonpyoi":    "원표애보라사슴벌레",
    "Dorcus_consentaneus_consentaneus":      "참넓적사슴벌레",
    "Aegus_laevicollis_subnitidus":          "꼬마넓적사슴벌레",
    "Nigidius_miwai":                        "뿔꼬마사슴벌레",
    "Dorcus_tenuihirsutus":                  "엷은털왕사슴벌레",
    "Figulus_punctatus":                     "길쭉꼬마사슴벌레",
    "Figulus_binodulus":                     "큰꼬마사슴벌레",
}

# result_type 결정 기준
_CONFIDENCE_THRESHOLD = 0.40   # 이하면 low_confidence
_OOD_THRESHOLD        = 0.65   # 이상이면 uncertain


@dataclass
class TopPrediction:
    species: str
    confidence: float


@dataclass
class PredictionResult:
    prediction_id:     Optional[str]
    result_type:       str           # identified | low_confidence | uncertain | no_beetle
    species:           str
    species_ko:        Optional[str]
    confidence:        float
    is_ood:            bool
    ood_score:         float
    top3:              list[TopPrediction]
    is_fallback:       bool
    crop_index:        int            # -1 이면 fallback
    bbox:              Optional[list[int]]
    preprocessing_mode: str
    latency_ms:        float


def _determine_result_type(
    detected: bool,
    ood_score: float,
    confidence: float,
) -> str:
    if not detected:
        return "no_beetle"
    if ood_score >= _OOD_THRESHOLD:
        return "uncertain"
    if confidence < _CONFIDENCE_THRESHOLD:
        return "low_confidence"
    return "identified"


class PredictionService:

    def __init__(self, manager: ModelManager, preprocessing_mode: str = "seg_hard"):
        self._manager            = manager
        self._preprocessing_mode = preprocessing_mode

    async def predict_image(self, image: Image.Image) -> list[PredictionResult]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._predict_sync, image)

    async def predict_path(self, image_path: str | Path) -> list[PredictionResult]:
        image = Image.open(image_path).convert("RGB")
        return await self.predict_image(image)

    # ------------------------------------------------------------------ #
    # Internal (sync, thread-safe)                                         #
    # ------------------------------------------------------------------ #

    def _predict_sync(self, image: Image.Image) -> list[PredictionResult]:
        import time
        t0 = time.monotonic()

        # full 모드는 원본 그대로 사용 (세그멘터 미사용)
        if self._preprocessing_mode == "full":
            crops, detected, is_fallback = [image], True, False
        else:
            crops, detected = self._manager.segmenter.segment(image)
            is_fallback     = not detected
            if not detected:
                crops = [image]

        results = []
        for i, crop in enumerate(crops):
            tensor  = EVAL_TRANSFORM(crop).unsqueeze(0).to(self._manager.device)
            cls_out = self._classify(tensor)
            ood_out = self._manager.ood.predict(tensor)

            top_species    = cls_out[0].species
            top_confidence = cls_out[0].confidence
            result_type    = _determine_result_type(detected, ood_out.score, top_confidence)

            results.append(PredictionResult(
                prediction_id      = None,
                result_type        = result_type,
                species            = top_species,
                species_ko         = _KO_NAMES.get(top_species),
                confidence         = top_confidence,
                is_ood             = ood_out.is_ood,
                ood_score          = ood_out.score,
                top3               = cls_out,
                is_fallback        = is_fallback,
                crop_index         = -1 if is_fallback else i,
                bbox               = None,
                preprocessing_mode = self._preprocessing_mode,
                latency_ms         = (time.monotonic() - t0) * 1000,
            ))
        return results

    @torch.no_grad()
    def _classify(self, tensor: torch.Tensor) -> list[TopPrediction]:
        outputs = self._manager.classifier(tensor)
        # multi-task 모델은 (species, sex, ...) tuple 반환
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        probs   = F.softmax(outputs, dim=1)[0]
        k       = min(3, len(self._manager.class_names))
        top_probs, top_idx = torch.topk(probs, k=k)
        return [
            TopPrediction(
                species    = self._manager.class_names[idx],
                confidence = float(prob),
            )
            for prob, idx in zip(top_probs.cpu(), top_idx.cpu())
        ]
