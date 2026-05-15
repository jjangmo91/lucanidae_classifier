"""
Temperature Scaling — val set으로 최적 온도 T 탐색 후 test set ECE 비교.

보정 전/후 ECE와 reliability diagram을 저장.

사용법:
    python scripts/calibrate.py --weights models/weights/best_model.pth
"""

import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path
from torch import nn, optim
from torchvision import datasets
from torch.utils.data import DataLoader

from src.ml.classifier import build_model
from src.training.dataset import EVAL_TRANSFORM, NUM_WORKERS


N_ECE_BINS = 15


# ── Temperature Scaling 모델 래퍼 ─────────────────────────────────────────

class TemperatureScaler(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model       = model
        self.log_t       = nn.Parameter(torch.zeros(1))  # T = exp(log_t) > 0 보장

    @property
    def temperature(self):
        return self.log_t.exp()

    def forward(self, x):
        out    = self.model(x)
        logits = out[0] if isinstance(out, tuple) else out
        return logits / self.temperature

    def fit(self, loader, device, max_iter=100, lr=0.01):
        """val set NLL 최소화로 T 학습."""
        self.model.eval()
        nll = nn.CrossEntropyLoss()
        opt = optim.LBFGS([self.log_t], lr=lr, max_iter=max_iter)

        all_logits, all_labels = [], []
        with torch.no_grad():
            for batch in loader:
                x      = batch[0].to(device)
                labels = batch[1]
                out    = self.model(x)
                sp_out = out[0] if isinstance(out, tuple) else out
                all_logits.append(sp_out.cpu())
                all_labels.append(labels)

        logits_t = torch.cat(all_logits).to(device)
        labels_t = torch.cat(all_labels).to(device)

        def eval_step():
            opt.zero_grad()
            loss = nll(logits_t / self.temperature, labels_t)
            loss.backward()
            return loss

        opt.step(eval_step)
        return self.temperature.item()


# ── ECE 계산 ──────────────────────────────────────────────────────────────

def compute_ece(probs, labels, n_bins=N_ECE_BINS):
    confs      = probs.max(axis=1)
    preds      = probs.argmax(axis=1)
    accs       = (preds == labels).astype(float)
    bin_edges  = np.linspace(0, 1, n_bins + 1)
    ece        = 0.0
    bin_data   = []

    for i in range(n_bins):
        mask = (confs >= bin_edges[i]) & (confs < bin_edges[i + 1])
        if mask.sum() > 0:
            b_acc  = accs[mask].mean()
            b_conf = confs[mask].mean()
            ece   += mask.sum() / len(confs) * abs(b_acc - b_conf)
            bin_data.append((b_conf, b_acc, mask.sum()))

    return float(ece), bin_data


def _draw_reliability_ax(ax, probs, labels, ece, title, n_bins=N_ECE_BINS):
    """Guo et al. (2017) 스타일 reliability diagram."""
    confs_all = probs.max(axis=1)
    preds_all = probs.argmax(axis=1)
    accs_all  = (preds_all == labels).astype(float)

    bin_edges  = np.linspace(0, 1, n_bins + 1)
    bar_w      = bin_edges[1] - bin_edges[0]
    bin_centers= (bin_edges[:-1] + bin_edges[1:]) / 2

    bar_accs   = np.zeros(n_bins)
    bar_confs  = np.zeros(n_bins)
    bar_counts = np.zeros(n_bins, dtype=int)

    for i in range(n_bins):
        mask = (confs_all >= bin_edges[i]) & (confs_all < bin_edges[i + 1])
        if mask.sum() > 0:
            bar_accs[i]   = accs_all[mask].mean()
            bar_confs[i]  = confs_all[mask].mean()
            bar_counts[i] = mask.sum()

    # 정확도 바 (파란색)
    ax.bar(bin_centers, bar_accs, width=bar_w * 0.9,
           color="#4477AA", alpha=0.85, label="Accuracy", zorder=2)

    # 과잉확신 gap (빨간색 — 대각선보다 낮은 구간)
    gap = bin_centers - bar_accs
    over_mask = gap > 0
    ax.bar(bin_centers[over_mask],
           gap[over_mask],
           bottom=bar_accs[over_mask],
           width=bar_w * 0.9,
           color="#EE6677", alpha=0.6, label="Gap (overconfidence)", zorder=2)

    # 과소확신 gap (청록색 — 대각선보다 높은 구간)
    under_mask = gap < 0
    ax.bar(bin_centers[under_mask],
           -gap[under_mask],
           bottom=bin_centers[under_mask],
           width=bar_w * 0.9,
           color="#44AA99", alpha=0.6, label="Gap (underconfidence)", zorder=2)

    # 완벽 보정선
    ax.plot([0, 1], [0, 1], "--", color="black",
            linewidth=1.2, zorder=3, label="Perfect calibration")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence", fontsize=9)
    ax.set_ylabel("Accuracy", fontsize=9)
    ax.set_title(title, fontsize=10, pad=7)
    ax.set_aspect("equal")
    ax.text(0.03, 0.93, f"ECE = {ece:.4f}", transform=ax.transAxes,
            fontsize=8.5, va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    ax.legend(fontsize=7.5, framealpha=0.9, loc="lower right")
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.5)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def plot_reliability(bin_data_before, bin_data_after,
                     ece_before, ece_after, out_path, temperature=None,
                     probs_before=None, labels_before=None,
                     probs_after=None,  labels_after=None):
    plt.rcParams.update({
        "font.family": "serif", "font.size": 9,
    })
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))

    t_str = f"  ($T={temperature:.3f}$)" if temperature else ""
    _draw_reliability_ax(axes[0], probs_before, labels_before,
                         ece_before, "(a) Before calibration")
    _draw_reliability_ax(axes[1], probs_after,  labels_after,
                         ece_after,  f"(b) After calibration{t_str}")

    plt.tight_layout(pad=1.8)
    out_pdf = out_path.with_suffix(".pdf")
    fig.savefig(out_pdf,  bbox_inches="tight")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"저장: {out_path.name}, {out_pdf.name}")


# ── 추론 ─────────────────────────────────────────────────────────────────

def get_probs_and_labels(model, loader, device, apply_temperature=None):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            x      = batch[0].to(device)
            labels = batch[1]
            out    = model(x)
            logits = out[0] if isinstance(out, tuple) else out
            if apply_temperature:
                logits = logits / apply_temperature
            probs  = F.softmax(logits, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels)


# ── 메인 ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights",    required=True)
    parser.add_argument("--data_dir",   default=None)
    parser.add_argument("--output_dir", default="experiments/calibration")
    args = parser.parse_args()

    import yaml
    with open("configs/default.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["training"]
    data_dir = args.data_dir or cfg["data_dir"]

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt         = torch.load(args.weights, map_location=device, weights_only=False)
    class_names  = ckpt["class_names"]
    architecture = ckpt.get("architecture", "convnext_tiny")
    mode         = ckpt.get("mode", "species_only")

    model = build_model(num_classes=len(class_names), architecture=architecture, mode=mode)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    def make_loader(split):
        ds = datasets.ImageFolder(Path(data_dir) / split, EVAL_TRANSFORM)
        return DataLoader(ds, batch_size=32, shuffle=False,
                          num_workers=NUM_WORKERS,
                          pin_memory=torch.cuda.is_available())

    val_loader  = make_loader("val")
    test_loader = make_loader("test")

    # Temperature 학습 (val set)
    scaler = TemperatureScaler(model).to(device)
    T      = scaler.fit(val_loader, device)
    print(f"최적 Temperature T = {T:.4f}")

    # Val ECE
    probs_val, labels_val = get_probs_and_labels(model, val_loader, device)
    ece_val_before, _     = compute_ece(probs_val, labels_val)
    probs_val_t, _        = get_probs_and_labels(model, val_loader, device, apply_temperature=T)
    ece_val_after, _      = compute_ece(probs_val_t, labels_val)

    # Test ECE
    probs_test, labels_test  = get_probs_and_labels(model, test_loader, device)
    ece_test_before, bd_b    = compute_ece(probs_test, labels_test)
    probs_test_t, _          = get_probs_and_labels(model, test_loader, device, apply_temperature=T)
    ece_test_after,  bd_a    = compute_ece(probs_test_t, labels_test)

    print(f"\nVal  ECE: {ece_val_before:.4f} → {ece_val_after:.4f}")
    print(f"Test ECE: {ece_test_before:.4f} → {ece_test_after:.4f}")

    stem = Path(args.weights).stem
    plot_reliability(bd_b, bd_a, ece_test_before, ece_test_after,
                     out / f"reliability_{stem}.png", temperature=T,
                     probs_before=probs_test,   labels_before=labels_test,
                     probs_after=probs_test_t,  labels_after=labels_test)

    result = {
        "architecture":   architecture,
        "weights":        args.weights,
        "temperature":    round(T, 4),
        "val_ece_before": round(ece_val_before,  4),
        "val_ece_after":  round(ece_val_after,   4),
        "test_ece_before":round(ece_test_before, 4),
        "test_ece_after": round(ece_test_after,  4),
    }
    with open(out / f"calibration_{stem}.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"저장: calibration_{stem}.json")


if __name__ == "__main__":
    main()
