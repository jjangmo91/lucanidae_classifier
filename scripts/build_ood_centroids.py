"""
DINOv2 backbone으로 train 데이터 종별 centroid 계산.

사용법:
    python scripts/build_ood_centroids.py
    python scripts/build_ood_centroids.py --weights models/weights/best_model.pth
"""

import argparse
import torch
import yaml

from src.ml.classifier import build_model
from src.ml.ood_detector import OODDetector


def main(weights_path: str, data_dir: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint   = torch.load(weights_path, map_location=device, weights_only=False)
    num_classes  = checkpoint["num_classes"]
    architecture = checkpoint.get("architecture", "convnext_tiny")
    mode         = checkpoint.get("mode", "species_only")

    model = build_model(num_classes=num_classes, architecture=architecture, mode=mode)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    # backbone만 추출 (MultiTaskClassifier.backbone)
    backbone = model.backbone

    detector = OODDetector(backbone=backbone, device=device)
    detector.compute_centroids(data_dir=data_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights",  default=None)
    parser.add_argument("--data_dir", default=None)
    args = parser.parse_args()

    with open("configs/default.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["training"]

    main(
        weights_path=args.weights  or cfg["weights_path"],
        data_dir    =args.data_dir or cfg["data_dir"],
    )
