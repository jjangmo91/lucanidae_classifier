"""
N4 Data Flywheel 시뮬레이션.

학습 데이터의 N%만 사용 → pseudo-labeling → 재학습으로 accuracy 회복 정도 측정.
논문 기여: 웹 서비스를 통한 데이터 축적이 모델 성능을 얼마나 향상시키는지 정량화.

단계:
  1. Full labeled data로 oracle 모델 학습  (baseline)
  2. 레이블 X%만 사용하여 seed 모델 학습   (degraded)
  3. seed 모델로 hold-out X%를 pseudo-label
  4. pseudo-label 포함하여 재학습            (recovered)
  5. val_acc: oracle ≥ recovered > degraded 관계 시각화

사용법:
    python scripts/simulate_flywheel.py --weights models/weights/best_model.pth
    python scripts/simulate_flywheel.py --weights models/weights/best_model.pth --holdout_fracs 0.2 0.4 0.6
"""

import argparse
import json
import random
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from sklearn.utils.class_weight import compute_class_weight

from src.ml.classifier import build_model
from src.training.dataset import EVAL_TRANSFORM, TRAIN_TRANSFORM, NUM_WORKERS


# ── Seed helper ──────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── Fast mini-trainer (no MLflow, no checkpoint overhead) ────────────────────

def quick_train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    num_epochs: int,
    class_weights: torch.Tensor,
) -> float:
    """Returns best val_acc seen during training."""
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    best_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            logits = out[0] if isinstance(out, tuple) else out
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                logits = out[0] if isinstance(out, tuple) else out
                correct += (logits.argmax(1) == y).sum().item()
                total   += len(y)
        val_acc = correct / total if total > 0 else 0.0
        scheduler.step(val_acc)
        if val_acc > best_acc:
            best_acc = val_acc

    return best_acc


# ── Pseudo-label generation ───────────────────────────────────────────────────

def pseudo_label(
    model: nn.Module,
    holdout_dataset: datasets.ImageFolder,
    holdout_indices: list[int],
    threshold: float,
    device: torch.device,
) -> list[tuple[str, int]]:
    """
    Returns list of (image_path, pseudo_label_int) for samples
    where model confidence >= threshold.
    """
    model.eval()
    loader = DataLoader(
        Subset(holdout_dataset, holdout_indices),
        batch_size=32, shuffle=False, num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    accepted = []
    sample_idx = 0
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            out = model(x)
            logits = out[0] if isinstance(out, tuple) else out
            probs = F.softmax(logits, dim=1)
            confs, preds = probs.max(dim=1)
            for conf, pred in zip(confs.cpu(), preds.cpu()):
                if conf.item() >= threshold:
                    global_idx = holdout_indices[sample_idx]
                    path = holdout_dataset.samples[global_idx][0]
                    accepted.append((path, pred.item()))
                sample_idx += 1

    return accepted


# ── Temp dataset builder (ImageFolder from subset) ────────────────────────────

def build_pseudo_dataset(
    labeled_root: Path,
    pseudo_items: list[tuple[str, int]],
    class_names: list[str],
    tmp_dir: Path,
) -> datasets.ImageFolder:
    """
    Copies labeled + pseudo-labeled images into a temp ImageFolder directory.
    """
    for cls in class_names:
        (tmp_dir / cls).mkdir(parents=True, exist_ok=True)

    # Copy labeled images
    for cls_dir in labeled_root.iterdir():
        if not cls_dir.is_dir():
            continue
        for img in cls_dir.iterdir():
            shutil.copy(img, tmp_dir / cls_dir.name / img.name)

    # Copy pseudo-labeled images (rename to avoid collision)
    for i, (path, label_idx) in enumerate(pseudo_items):
        cls = class_names[label_idx]
        ext = Path(path).suffix
        dst = tmp_dir / cls / f"pseudo_{i:05d}{ext}"
        shutil.copy(path, dst)

    return datasets.ImageFolder(str(tmp_dir), transform=TRAIN_TRANSFORM)


# ── Main simulation ───────────────────────────────────────────────────────────

def simulate(
    weights_path: str,
    data_dir: str,
    holdout_fracs: list[float],
    num_epochs: int,
    pseudo_threshold: float,
    seed: int,
    output_dir: Path,
) -> None:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt         = torch.load(weights_path, map_location=device, weights_only=False)
    class_names  = ckpt["class_names"]
    architecture = ckpt.get("architecture", "convnext_tiny")
    num_classes  = len(class_names)

    data_dir = Path(data_dir)
    train_root = data_dir / "train"
    val_root   = data_dir / "val"

    full_train_ds = datasets.ImageFolder(str(train_root), transform=TRAIN_TRANSFORM)
    val_ds        = datasets.ImageFolder(str(val_root),   transform=EVAL_TRANSFORM)
    val_loader    = DataLoader(val_ds, batch_size=32, shuffle=False,
                               num_workers=NUM_WORKERS,
                               pin_memory=torch.cuda.is_available())

    all_indices = list(range(len(full_train_ds)))
    all_labels  = [full_train_ds.targets[i] for i in all_indices]

    # ── Oracle (full data) ──────────────────────────────────────────────────
    print("\n[1/3] Oracle 학습 (전체 레이블 데이터)")
    oracle_model = build_model(num_classes=num_classes, architecture=architecture, mode="species_only")
    oracle_model.to(device)

    cw = compute_class_weight("balanced", classes=np.unique(all_labels), y=all_labels)
    cw_tensor = torch.tensor(cw, dtype=torch.float).to(device)

    full_loader = DataLoader(full_train_ds, batch_size=32, shuffle=True,
                             num_workers=NUM_WORKERS,
                             pin_memory=torch.cuda.is_available())
    oracle_acc = quick_train(oracle_model, full_loader, val_loader, device, num_epochs, cw_tensor)
    print(f"  Oracle val_acc = {oracle_acc:.4f}")

    results = []

    for frac in holdout_fracs:
        print(f"\n[Holdout fraction = {frac:.0%}]")
        rng = np.random.default_rng(seed)
        n_holdout = int(len(all_indices) * frac)

        holdout_indices = rng.choice(all_indices, n_holdout, replace=False).tolist()
        labeled_indices = [i for i in all_indices if i not in set(holdout_indices)]
        labeled_labels  = [full_train_ds.targets[i] for i in labeled_indices]

        # ── Degraded (labeled fraction only) ───────────────────────────────
        print(f"  [Degraded] {len(labeled_indices)} labeled 이미지")
        degraded_model = build_model(num_classes=num_classes, architecture=architecture, mode="species_only")
        degraded_model.to(device)

        deg_cw     = compute_class_weight("balanced", classes=np.unique(labeled_labels), y=labeled_labels)
        deg_cw_t   = torch.tensor(deg_cw, dtype=torch.float).to(device)
        deg_loader = DataLoader(
            Subset(full_train_ds, labeled_indices),
            batch_size=32, shuffle=True, num_workers=NUM_WORKERS,
            pin_memory=torch.cuda.is_available(),
        )
        degraded_acc = quick_train(degraded_model, deg_loader, val_loader, device, num_epochs, deg_cw_t)
        print(f"    Degraded val_acc = {degraded_acc:.4f}")

        # ── Pseudo-labeling ─────────────────────────────────────────────────
        pseudo_items = pseudo_label(
            degraded_model, full_train_ds, holdout_indices, pseudo_threshold, device
        )
        pseudo_accept_rate = len(pseudo_items) / max(len(holdout_indices), 1)
        print(f"    Pseudo-label 채택: {len(pseudo_items)}/{len(holdout_indices)} ({pseudo_accept_rate:.1%})")

        # ── Recovered (labeled + pseudo-labeled) ────────────────────────────
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp) / "pseudo_train"

            # Build labeled-only temp dataset (subset of full_train_ds)
            labeled_tmp = Path(tmp) / "labeled_only"
            for idx in labeled_indices:
                path, lbl = full_train_ds.samples[idx]
                cls = class_names[lbl]
                dst_dir = labeled_tmp / cls
                dst_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy(path, dst_dir / Path(path).name)

            pseudo_ds = build_pseudo_dataset(labeled_tmp, pseudo_items, class_names, tmp_path)
            all_pseudo_labels = pseudo_ds.targets
            ps_cw = compute_class_weight("balanced", classes=np.unique(all_pseudo_labels), y=all_pseudo_labels)
            ps_cw_t = torch.tensor(ps_cw, dtype=torch.float).to(device)

            recovered_model = build_model(num_classes=num_classes, architecture=architecture, mode="species_only")
            recovered_model.to(device)
            ps_loader = DataLoader(pseudo_ds, batch_size=32, shuffle=True,
                                   num_workers=NUM_WORKERS,
                                   pin_memory=torch.cuda.is_available())
            recovered_acc = quick_train(recovered_model, ps_loader, val_loader, device, num_epochs, ps_cw_t)

        print(f"    Recovered val_acc = {recovered_acc:.4f}")
        results.append({
            "holdout_frac":       frac,
            "labeled_count":      len(labeled_indices),
            "pseudo_count":       len(pseudo_items),
            "pseudo_accept_rate": round(pseudo_accept_rate, 4),
            "oracle_acc":         round(oracle_acc, 4),
            "degraded_acc":       round(degraded_acc, 4),
            "recovered_acc":      round(recovered_acc, 4),
            "gap_closed":         round((recovered_acc - degraded_acc) /
                                        max(oracle_acc - degraded_acc, 1e-8), 4),
        })

    # ── Save results ─────────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    out_json = output_dir / "flywheel_simulation.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n저장: {out_json}")

    # ── Plot ─────────────────────────────────────────────────────────────────
    fracs   = [r["holdout_frac"] for r in results]
    deg_acc = [r["degraded_acc"]  * 100 for r in results]
    rec_acc = [r["recovered_acc"] * 100 for r in results]
    orc_acc = results[0]["oracle_acc"] * 100 if results else 0

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(fracs, deg_acc, "o--", color="tomato",    label="Degraded (labeled only)")
    ax.plot(fracs, rec_acc, "s-",  color="steelblue", label="Recovered (+ pseudo-labels)")
    ax.axhline(orc_acc, color="gray", linestyle=":", label=f"Oracle (100% labeled) {orc_acc:.1f}%")
    ax.fill_between(fracs, deg_acc, rec_acc, alpha=0.15, color="steelblue", label="Flywheel gain")
    ax.set_xlabel("Holdout Fraction (simulated unlabeled data)")
    ax.set_ylabel("Val Accuracy (%)")
    ax.set_title("Data Flywheel Simulation — Pseudo-labeling Recovery")
    ax.legend(fontsize=9)
    ax.set_xlim(0, max(fracs) + 0.05)
    ax.set_ylim(0, 100)
    plt.tight_layout()
    fig.savefig(output_dir / "fig_flywheel_simulation.png", dpi=150)
    plt.close(fig)
    print(f"저장: fig_flywheel_simulation.png")

    print("\n── 결과 요약 ──────────────────────────────────────────")
    print(f"{'Holdout':>10}  {'Degraded':>10}  {'Recovered':>10}  {'Gap closed':>12}")
    for r in results:
        print(f"{r['holdout_frac']:>10.0%}  "
              f"{r['degraded_acc']*100:>9.1f}%  "
              f"{r['recovered_acc']*100:>9.1f}%  "
              f"{r['gap_closed']*100:>11.1f}%")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights",         required=True)
    parser.add_argument("--data_dir",        default=None)
    parser.add_argument("--holdout_fracs",   nargs="+", type=float, default=[0.2, 0.4, 0.6])
    parser.add_argument("--num_epochs",      type=int,   default=20)
    parser.add_argument("--pseudo_threshold",type=float, default=0.85)
    parser.add_argument("--seed",            type=int,   default=42)
    parser.add_argument("--output_dir",      default="experiments/flywheel")
    args = parser.parse_args()

    import yaml
    with open("configs/default.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["training"]
    data_dir = args.data_dir or cfg["data_dir"]

    simulate(
        weights_path      = args.weights,
        data_dir          = data_dir,
        holdout_fracs     = args.holdout_fracs,
        num_epochs        = args.num_epochs,
        pseudo_threshold  = args.pseudo_threshold,
        seed              = args.seed,
        output_dir        = Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
