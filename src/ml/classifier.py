import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ViTWrapper(nn.Module):
    """ViT는 224×224 고정 입력이므로 448 입력을 resize해서 전달."""
    def __init__(self, vit: nn.Module):
        super().__init__()
        self.vit = vit

    def forward(self, x):
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        return self.vit(x)


class MultiTaskClassifier(nn.Module):
    """
    단일 backbone + 최대 3개 head (species / sex / male_form).

    mode:
      "species_only" — species head만 사용
      "sex_only"     — species + sex head (form head 없음)
      "form_only"    — species + form head (sex head 없음)
      "multi_task"   — 세 head 모두 사용
    """

    def __init__(self, backbone: nn.Module, in_features: int,
                 num_species: int, mode: str = "species_only"):
        super().__init__()
        self.backbone     = backbone
        self.mode         = mode
        self.species_head = nn.Linear(in_features, num_species)

        if mode in ("multi_task", "sex_only"):
            self.sex_head = nn.Linear(in_features, 2)   # male / female (unknown은 masked loss로 제외)
        if mode in ("multi_task", "form_only"):
            self.male_form_head = nn.Linear(in_features, 3)  # major / minor / intermediate (unknown 제외)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """t-SNE / UMAP 시각화용 backbone feature 벡터 반환."""
        return self.backbone(x)

    def forward(self, x):
        features = self.backbone(x)
        species  = self.species_head(features)

        if self.mode == "multi_task":
            return species, self.sex_head(features), self.male_form_head(features)
        if self.mode == "sex_only":
            return species, self.sex_head(features)
        if self.mode == "form_only":
            return species, self.male_form_head(features)
        return species


def build_model(num_classes: int, architecture: str = "convnext_tiny",
                mode: str = "species_only") -> nn.Module:
    arch = architecture.lower()

    if arch == "convnext_tiny":
        from torchvision.models import ConvNeXt_Tiny_Weights
        base        = models.convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
        in_features = base.classifier[2].in_features
        base.classifier[2] = nn.Identity()
        backbone    = base

    elif arch == "efficientnet_b3":
        from torchvision.models import EfficientNet_B3_Weights
        base        = models.efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT)
        in_features = base.classifier[1].in_features
        base.classifier = nn.Identity()  # Dropout(0.3) + Linear 전체 제거
        backbone    = base

    elif arch == "swin_tiny":
        from torchvision.models import Swin_T_Weights
        base        = models.swin_t(weights=Swin_T_Weights.DEFAULT)
        in_features = base.head.in_features
        base.head   = nn.Identity()
        backbone    = base

    elif arch == "vit_small":
        from torchvision.models import ViT_B_16_Weights
        base        = models.vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        in_features = base.heads.head.in_features
        base.heads.head = nn.Identity()
        backbone    = ViTWrapper(base)

    elif arch == "dinov2_vits14":
        backbone    = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        in_features = backbone.embed_dim  # 384

    else:
        raise ValueError(
            f"Unknown architecture: {architecture}. "
            "Choose: convnext_tiny | efficientnet_b3 | swin_tiny | vit_small | dinov2_vits14"
        )

    return MultiTaskClassifier(backbone, in_features, num_classes, mode=mode)
