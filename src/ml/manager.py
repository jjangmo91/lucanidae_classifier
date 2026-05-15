"""
ModelManager — 앱 시작 시 모델을 1회 로드하고 핫스왑을 지원하는 싱글턴.

사용법:
    manager = ModelManager.get_instance()
    classifier = manager.classifier
    segmenter  = manager.segmenter
    ood        = manager.ood
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional

import torch
import yaml

from src.ml.classifier import build_model
from src.ml.detector import BeetleSegmenter
from src.ml.ood_detector import OODDetector


class ModelManager:
    _instance: Optional["ModelManager"] = None
    _lock = threading.Lock()

    def __init__(self, config_path: str = "configs/default.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self._config = yaml.safe_load(f)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_all()

    # ------------------------------------------------------------------ #
    # Singleton                                                            #
    # ------------------------------------------------------------------ #

    @classmethod
    def get_instance(cls, config_path: str = "configs/default.yaml") -> "ModelManager":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(config_path)
        return cls._instance

    # ------------------------------------------------------------------ #
    # Load                                                                 #
    # ------------------------------------------------------------------ #

    def _load_all(self) -> None:
        self.classifier = self._load_classifier()
        self.segmenter  = self._load_segmenter()
        self.ood        = self._load_ood()

    def _load_classifier(self):
        cfg  = self._config["training"]
        ckpt = torch.load(cfg["weights_path"], map_location=self.device)

        model = build_model(
            num_classes  = ckpt["num_classes"],
            architecture = ckpt.get("architecture", "convnext_tiny"),
            mode         = ckpt.get("mode", "species_only"),
        )
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(self.device).eval()

        self.class_names = ckpt["class_names"]
        return model

    def _load_segmenter(self) -> BeetleSegmenter:
        cfg = self._config["detection"]
        return BeetleSegmenter(
            weights_path  = cfg["weights_path"],
            confidence    = cfg.get("confidence", 0.25),
            iou_threshold = cfg.get("iou_threshold", 0.45),
        )

    def _load_ood(self) -> OODDetector:
        cfg           = self._config["inference"]
        centroid_path = cfg.get("ood_centroid_path", None)
        detector      = OODDetector(
            backbone      = self.classifier.backbone,
            device        = self.device,
            centroid_path = centroid_path or "models/weights/ood_centroids.pt",
        )
        if centroid_path and Path(centroid_path).exists():
            detector.load_centroids()
        return detector

    # ------------------------------------------------------------------ #
    # Hot-swap (무중단 모델 교체)                                          #
    # ------------------------------------------------------------------ #

    def hot_swap_classifier(self, new_weights_path: str) -> None:
        """새 체크포인트로 분류기를 무중단 교체한다."""
        ckpt      = torch.load(new_weights_path, map_location=self.device)
        new_model = build_model(
            num_classes  = ckpt["num_classes"],
            architecture = ckpt.get("architecture", "convnext_tiny"),
            mode         = ckpt.get("mode", "species_only"),
        )
        new_model.load_state_dict(ckpt["model_state_dict"])
        new_model.to(self.device).eval()

        with self._lock:
            self.classifier  = new_model
            self.class_names = ckpt["class_names"]
