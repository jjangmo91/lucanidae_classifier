"""
DINOv2 특징 벡터 기반 OOD 탐지기.

학습 데이터의 종별 centroid를 계산해 저장하고,
추론 시 새 이미지의 특징 벡터와 centroid 거리로 OOD를 판별한다.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torchvision import datasets
from torch.utils.data import DataLoader

from src.training.dataset import EVAL_TRANSFORM, NUM_WORKERS


@dataclass
class OODResult:
    is_ood: bool
    score: float


class OODDetector:
    def __init__(self, backbone: torch.nn.Module, device: torch.device,
                 centroid_path: str = "models/weights/ood_centroids.pt"):
        self.backbone      = backbone
        self.device        = device
        self.centroid_path = Path(centroid_path)
        self.centroids     = None  # {class_idx: 384-dim tensor}
        self.class_names   = None

    def compute_centroids(self, data_dir: str):
        """train 데이터에서 종별 centroid 계산 후 저장."""
        dataset    = datasets.ImageFolder(Path(data_dir) / "train", EVAL_TRANSFORM)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False,
                                num_workers=NUM_WORKERS,
                                pin_memory=torch.cuda.is_available())

        self.class_names = dataset.classes
        accum  = {i: [] for i in range(len(dataset.classes))}

        self.backbone.eval()
        with torch.no_grad():
            for inputs, labels in tqdm(dataloader, desc="Centroid 계산"):
                inputs   = inputs.to(self.device)
                features = self._extract_features(inputs)
                for feat, label in zip(features, labels):
                    accum[label.item()].append(feat.cpu())

        centroids = {}
        for cls_idx, feats in accum.items():
            stacked = torch.stack(feats)
            centroids[cls_idx] = F.normalize(stacked.mean(dim=0), dim=0)

        self.centroids = centroids
        self.centroid_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"centroids": centroids, "class_names": self.class_names},
                   self.centroid_path)
        print(f"Centroid 저장 완료: {self.centroid_path}")

    def load_centroids(self):
        data             = torch.load(self.centroid_path, map_location="cpu", weights_only=False)
        self.centroids   = data["centroids"]
        self.class_names = data["class_names"]

    def predict(self, image_tensor: torch.Tensor, threshold: float = 0.65) -> OODResult:
        """centroids가 없으면 OOD 아님으로 처리 (graceful fallback)."""
        if self.centroids is None:
            return OODResult(is_ood=False, score=0.0)
        s = self.score(image_tensor)
        return OODResult(is_ood=s >= threshold, score=s)

    def score(self, image_tensor: torch.Tensor) -> float:
        """
        OOD score 반환 (0~1). 높을수록 OOD 가능성 높음.
        가장 가까운 centroid와의 코사인 거리 기반.
        """
        assert self.centroids is not None, "load_centroids() 먼저 호출하세요."

        self.backbone.eval()
        with torch.no_grad():
            feat = self._extract_features(image_tensor.to(self.device))
            feat = F.normalize(feat[0], dim=0).cpu()

        # 코사인 유사도 → 가장 높은 값 = 가장 가까운 종
        sims = torch.stack([
            F.cosine_similarity(feat.unsqueeze(0),
                                self.centroids[i].unsqueeze(0))
            for i in range(len(self.centroids))
        ])
        max_sim  = sims.max().item()
        ood_score = 1.0 - max_sim  # 유사도 낮을수록 OOD score 높음
        return float(ood_score)

    def _extract_features(self, inputs: torch.Tensor) -> torch.Tensor:
        out = self.backbone(inputs)
        # DINOv2는 dict 또는 tensor 반환
        if isinstance(out, dict):
            return out["x_norm_clstoken"]
        return out
