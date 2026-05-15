"""
통합 추론 파이프라인: Segmenter + Classifier

1. BeetleSegmenter 로 이미지에서 사슴벌레 crop 추출
2. LucanidaePredictor 로 각 crop 분류
3. Fallback: 검출 없음 → 원본 이미지 그대로 분류

사용법:
    python pipeline.py                          # data/test_images/ 일괄 추론
    python pipeline.py --image path/to/img.jpg  # 단일 이미지
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from predict import LucanidaePredictor
from src.ml.detector import BeetleSegmenter
from src.ml.ood_detector import OODDetector
from src.training.dataset import EVAL_TRANSFORM


class LucanidaePipeline:
    def __init__(
        self,
        classifier_weights: str,
        detector_weights: str,
        confidence: float = 0.25,
        iou_threshold: float = 0.45,
        ood_centroid_path: str | None = None,
        ood_threshold: float = 0.65,
    ):
        self.classifier  = LucanidaePredictor(classifier_weights)
        self.segmenter   = BeetleSegmenter(
            weights_path  = detector_weights,
            confidence    = confidence,
            iou_threshold = iou_threshold,
        )
        self.ood_threshold = ood_threshold
        self.ood_detector  = None
        if ood_centroid_path and Path(ood_centroid_path).exists():
            self.ood_detector = OODDetector(
                backbone       = self.classifier.model.backbone,
                device         = self.classifier.device,
                centroid_path  = ood_centroid_path,
            )
            self.ood_detector.load_centroids()
            print(f"OOD 탐지기 로드 완료: {ood_centroid_path}")

    @classmethod
    def from_config(cls, config_path: str = "configs/default.yaml") -> "LucanidaePipeline":
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return cls(
            classifier_weights = config["training"]["weights_path"],
            detector_weights   = config["detection"]["weights_path"],
            confidence         = config["detection"].get("confidence", 0.25),
            iou_threshold      = config["detection"].get("iou_threshold", 0.45),
            ood_centroid_path  = config["inference"].get("ood_centroid_path", None),
            ood_threshold      = config["inference"].get("ood_threshold", 0.65),
        )

    def predict(self, image_path: str | Path) -> list[dict]:
        """
        단일 이미지에 대해 검출 → 분류를 수행한다.

        반환: [
            {
                "crop_index": int,           # 0부터 시작 (fallback이면 -1)
                "is_fallback": bool,
                "top_predictions": [
                    {"label": str, "score": float},
                    ...
                ]
            },
            ...
        ]
        """
        image           = Image.open(image_path).convert("RGB")
        crops, detected = self.segmenter.segment(image)
        is_fallback     = not detected

        results = []
        for i, crop in enumerate(crops):
            probs, indices = self._classify_pil(crop)
            top_preds = [
                {"label": self.classifier.class_names[idx], "score": float(prob)}
                for prob, idx in zip(probs, indices)
            ]

            ood_score = None
            is_uncertain = False
            if self.ood_detector is not None:
                tensor = EVAL_TRANSFORM(crop).unsqueeze(0)
                ood_score    = self.ood_detector.score(tensor)
                is_uncertain = ood_score > self.ood_threshold

            results.append({
                "crop_index":      -1 if is_fallback else i,
                "is_fallback":     is_fallback,
                "top_predictions": top_preds,
                "ood_score":       ood_score,
                "is_uncertain":    is_uncertain,
            })

        return results

    @torch.no_grad()
    def _classify_pil(self, image: Image.Image):
        device = self.classifier.device
        tensor = EVAL_TRANSFORM(image).unsqueeze(0).to(device)
        outputs = self.classifier.model(tensor)
        sp_out  = outputs[0] if isinstance(outputs, tuple) else outputs  # multi_task 대응
        probs   = F.softmax(sp_out, dim=1)[0]
        k       = min(3, len(self.classifier.class_names))
        top_probs, top_indices = torch.topk(probs, k=k)
        return top_probs.cpu().numpy(), top_indices.cpu().numpy()


def _print_result(img_name: str, results: list[dict]) -> None:
    print("=" * 65)
    print(f" Image: {img_name}")
    print("=" * 65)
    for res in results:
        label = "Fallback (원본)" if res["is_fallback"] else f"Crop #{res['crop_index']}"
        ood_tag = ""
        if res.get("ood_score") is not None:
            ood_tag = f"  [OOD={res['ood_score']:.3f}{' UNCERTAIN' if res['is_uncertain'] else ''}]"
        print(f" [{label}]{ood_tag}")
        for i, pred in enumerate(res["top_predictions"]):
            score  = pred["score"] * 100
            bar    = "#" * int(score / 5)
            marker = "[BEST]" if i == 0 else "      "
            print(f"  {marker} {pred['label']:45s} | {score:6.2f}% | {bar}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Lucanidae 통합 추론 파이프라인")
    parser.add_argument("--image",  type=str, default=None, help="단일 이미지 경로")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    pipeline = LucanidaePipeline.from_config(args.config)

    if args.image:
        image_files = [Path(args.image)]
    else:
        test_dir = Path("data/test_images")
        if not test_dir.exists():
            test_dir.mkdir(parents=True)
            print(f"폴더 생성: {test_dir}")
            print("이미지를 넣고 다시 실행하세요.")
            return
        image_files = [
            f for f in test_dir.iterdir()
            if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]

    if not image_files:
        print("이미지 파일이 없습니다.")
        return

    for img_path in image_files:
        results = pipeline.predict(img_path)
        _print_result(img_path.name, results)


if __name__ == "__main__":
    main()
