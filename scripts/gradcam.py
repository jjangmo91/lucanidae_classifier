"""
Grad-CAM 시각화 — 모델이 어떤 부위를 보고 분류하는지 확인.

사용법:
    python scripts/gradcam.py --weights models/weights/best_model.pth
    python scripts/gradcam.py --weights models/weights/best_model.pth --n_samples 5
"""

import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
from PIL import Image
from torchvision import datasets

from src.ml.classifier import build_model
from src.training.dataset import EVAL_TRANSFORM


# ── Grad-CAM 구현 ─────────────────────────────────────────────────────────

class GradCAM:
    def __init__(self, model, target_layer):
        self.model        = model
        self.target_layer = target_layer
        self.gradients    = None
        self.activations  = None
        self._hooks       = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(_, __, output):
            self.activations = output.detach()

        def backward_hook(_, __, grad_output):
            self.gradients = grad_output[0].detach()

        self._hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self._hooks.append(self.target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()

    def __call__(self, x, class_idx=None):
        self.model.eval()
        output = self.model(x)
        sp_out = output[0] if isinstance(output, tuple) else output

        if class_idx is None:
            class_idx = sp_out.argmax(dim=1).item()

        self.model.zero_grad()
        sp_out[0, class_idx].backward()

        grad = self.gradients
        act  = self.activations

        if grad.dim() == 4:
            # channels-last (B, H, W, C) → Swin 등 → channels-first로 변환
            if grad.shape[-1] > grad.shape[1]:
                grad = grad.permute(0, 3, 1, 2).contiguous()
                act  = act.permute(0, 3, 1, 2).contiguous()
            # CNN: (B, C, H, W) — ConvNeXt, EfficientNet, Swin(변환 후)
            weights = grad.mean(dim=(2, 3), keepdim=True)
            cam     = (weights * act).sum(dim=1, keepdim=True)
        elif grad.dim() == 3:
            # Transformer: (B, N_tokens, C) — ViT, Swin, DINOv2
            B, N, C = grad.shape
            # CLS 토큰 제거: N-1이 완전제곱수이면 CLS 있음
            n = N
            if int((n - 1) ** 0.5) ** 2 == n - 1:
                grad = grad[:, 1:, :]
                act  = act[:, 1:, :]
                n    = n - 1
            h = w = int(n ** 0.5)
            # embed_dim 방향으로 gradient 가중 평균 → 공간 지도
            weights  = grad.mean(dim=2)                              # (B, n_patches)
            cam_flat = (weights.unsqueeze(-1) * act).sum(dim=2)     # (B, n_patches)
            cam      = cam_flat.reshape(B, 1, h, w)
        else:
            raise ValueError(f"지원하지 않는 activation shape: {grad.shape}")

        cam = F.relu(cam)
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, class_idx


def _get_target_layer(model, architecture: str):
    """아키텍처별 마지막 feature map 레이어 반환."""
    arch = architecture.lower()
    if "convnext" in arch:
        return model.backbone.stages[-1][-1].conv_dw
    elif "efficientnet" in arch:
        return model.backbone.features[-1][0]
    elif "swin" in arch:
        return model.backbone.features[-1][-1].norm2
    elif "vit" in arch or "dinov2" in arch:
        blocks = list(model.backbone.blocks)
        return blocks[-1].norm1
    raise ValueError(f"지원하지 않는 아키텍처: {architecture}")


# ── 이미지 역정규화 ────────────────────────────────────────────────────────

MEAN = np.array([0.485, 0.456, 0.406])
STD  = np.array([0.229, 0.224, 0.225])

def denormalize(tensor):
    img = tensor.cpu().numpy().transpose(1, 2, 0)
    img = img * STD + MEAN
    return np.clip(img, 0, 1)


# ── 시각화 ────────────────────────────────────────────────────────────────

def visualize_gradcam(model, gradcam, dataset, class_names, device,
                      n_samples: int, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    # 클래스별 1장씩 샘플링 (최대 n_samples개 클래스)
    class_to_idx = {v: k for k, v in enumerate(class_names)}
    shown_classes = set()
    samples_done  = 0

    for img_tensor, label in dataset:
        sp_name = class_names[label]
        if sp_name in shown_classes:
            continue
        shown_classes.add(sp_name)

        x   = img_tensor.unsqueeze(0).to(device)
        cam, pred_idx = gradcam(x)
        pred_name = class_names[pred_idx]

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        # 원본
        orig = denormalize(img_tensor)
        axes[0].imshow(orig)
        axes[0].set_title(f"True: {sp_name.replace('_', ' ')}", fontsize=8)
        axes[0].axis("off")

        # Grad-CAM heatmap
        heatmap = cm.jet(cam)[:, :, :3]
        axes[1].imshow(heatmap)
        axes[1].set_title("Grad-CAM", fontsize=9)
        axes[1].axis("off")

        # Overlay
        overlay = 0.5 * orig + 0.5 * heatmap
        overlay = np.clip(overlay, 0, 1)
        correct = "O" if pred_idx == label else "X"
        axes[2].imshow(overlay)
        axes[2].set_title(f"Pred: {pred_name.replace('_', ' ')} {correct}", fontsize=8)
        axes[2].axis("off")

        plt.tight_layout()
        fname = output_dir / f"gradcam_{sp_name}.png"
        fig.savefig(fname, dpi=120, bbox_inches="tight")
        plt.close(fig)

        samples_done += 1
        if samples_done >= n_samples:
            break

    print(f"Grad-CAM 저장: {output_dir} ({samples_done}장)")


# ── 메인 ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights",    required=True)
    parser.add_argument("--data_dir",   default=None)
    parser.add_argument("--split",      default="val", choices=["val", "test"])
    parser.add_argument("--n_samples",  type=int, default=16)
    parser.add_argument("--output_dir", default="experiments/gradcam")
    args = parser.parse_args()

    import yaml
    with open("configs/default.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["training"]
    data_dir = args.data_dir or cfg["data_dir"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt         = torch.load(args.weights, map_location=device, weights_only=False)
    class_names  = ckpt["class_names"]
    architecture = ckpt.get("architecture", "convnext_tiny")
    mode         = ckpt.get("mode", "species_only")

    model = build_model(num_classes=len(class_names), architecture=architecture, mode=mode)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    try:
        target_layer = _get_target_layer(model, architecture)
    except Exception as e:
        print(f"[경고] 타겟 레이어 자동 감지 실패: {e}")
        print("model 구조를 확인하고 --target_layer를 수동 지정하세요.")
        return

    gradcam = GradCAM(model, target_layer)
    dataset = datasets.ImageFolder(
        Path(data_dir) / args.split, transform=EVAL_TRANSFORM
    )

    visualize_gradcam(model, gradcam, dataset, class_names, device,
                      n_samples=args.n_samples,
                      output_dir=Path(args.output_dir))
    gradcam.remove_hooks()


if __name__ == "__main__":
    main()
