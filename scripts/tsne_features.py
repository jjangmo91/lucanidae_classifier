"""
t-SNE / UMAP feature space 시각화.

학습된 모델의 backbone feature를 추출하여 2D 공간에 투영.
종별 / 성별 컬러링 두 버전 저장.

사용법:
    python scripts/tsne_features.py --weights models/weights/best_model.pth
    python scripts/tsne_features.py --weights models/weights/best_model.pth --method umap
"""

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from torchvision import datasets
from torch.utils.data import DataLoader

from src.ml.classifier import build_model
from src.training.dataset import EVAL_TRANSFORM, NUM_WORKERS


# ── Feature extraction ────────────────────────────────────────────────────────

def extract_features(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    feats, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            f = model.extract_features(x)
            feats.append(f.cpu().numpy())
            labels.append(y.numpy())
    return np.concatenate(feats), np.concatenate(labels)


def extract_features_with_sex(model, loader, device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MultiTaskDataset 로더에서 sex label도 함께 수집."""
    model.eval()
    feats, sp_labels, sex_labels = [], [], []
    with torch.no_grad():
        for batch in loader:
            x      = batch[0].to(device)
            sp_lbl = batch[1].numpy()
            sex_lbl = batch[2].numpy() if len(batch) > 2 else np.full(len(sp_lbl), -1)
            f = model.extract_features(x)
            feats.append(f.cpu().numpy())
            sp_labels.append(sp_lbl)
            sex_labels.append(sex_lbl)
    return np.concatenate(feats), np.concatenate(sp_labels), np.concatenate(sex_labels)


# ── 2D projection ─────────────────────────────────────────────────────────────

def reduce(feats: np.ndarray, method: str) -> np.ndarray:
    if method == "tsne":
        from sklearn.manifold import TSNE
        return TSNE(n_components=2, random_state=42, perplexity=30,
                    max_iter=1000, init="pca").fit_transform(feats)
    elif method == "umap":
        try:
            import umap
        except ImportError:
            raise ImportError("pip install umap-learn 후 재실행")
        return umap.UMAP(n_components=2, random_state=42).fit_transform(feats)
    elif method == "both":
        return None  # handled separately in main
    else:
        raise ValueError(f"method는 tsne|umap|both 중 하나: {method}")


# ── Plot helpers ──────────────────────────────────────────────────────────────

# Paul Tol high-contrast 팔레트 — 16종 전부 고유색 (반복 없음)
OKABE_ITO = [
    "#4477AA",  # blue
    "#EE6677",  # rose
    "#228833",  # green
    "#CCBB44",  # yellow
    "#66CCEE",  # cyan
    "#AA3377",  # purple
    "#BBBBBB",  # grey
    "#332288",  # indigo
    "#88CCEE",  # sky
    "#44AA99",  # teal
    "#117733",  # dark green
    "#999933",  # olive
    "#DDCC77",  # sand
    "#CC6677",  # rose-pink
    "#882255",  # wine
    "#E69F00",  # orange
]

plt.rcParams.update({
    "font.family":  "serif",
    "font.size":    8,
    "axes.titlesize": 9,
})


def _italic(cls: str) -> str:
    parts = cls.split("_")
    if len(parts) >= 2:
        return f"$\\it{{{parts[0]}}}$ $\\it{{{parts[1]}}}$"
    return f"$\\it{{{parts[0]}}}$"


SEX_MARKER = {-1: "o", 0: "o", 1: "*"}   # unknown / male / female
SEX_LABEL  = {-1: "Unknown", 0: "Male", 1: "Female"}
SEX_SIZE   = {-1: 10, 0: 20, 1: 30}


def plot_combined(emb, sp_labels, sex_labels, class_names, title, out_path):
    """종(색) + 성별(마커 모양)을 하나의 그림에 표현. 범례 통합."""
    n = len(emb)

    fig, ax = plt.subplots(figsize=(11, 8))

    # unknown 먼저(배경), 그 다음 male/female
    for sex_val in [-1, 0, 1]:
        sex_mask = sex_labels == sex_val
        if sex_mask.sum() == 0:
            continue
        for i, cls in enumerate(class_names):
            mask = sex_mask & (sp_labels == i)
            if mask.sum() == 0:
                continue
            ax.scatter(emb[mask, 0], emb[mask, 1],
                       c=OKABE_ITO[i % len(OKABE_ITO)],
                       marker=SEX_MARKER[sex_val],
                       s=SEX_SIZE[sex_val],
                       alpha=0.30 if sex_val == -1 else 0.82,
                       linewidths=0.3 if sex_val == 1 else 0)

    # ── 통합 범례 ──────────────────────────────────────────────
    handles = []

    # 종 섹션 헤더
    handles.append(plt.Line2D([0], [0], linestyle="none", label="— Species —",
                               marker="none", color="gray"))
    for i, cls in enumerate(class_names):
        if (sp_labels == i).sum() == 0:
            continue
        handles.append(plt.Line2D(
            [0], [0], marker="o", color="w",
            markerfacecolor=OKABE_ITO[i % len(OKABE_ITO)],
            markersize=6, label=_italic(cls)))

    # 성별 섹션 헤더
    handles.append(plt.Line2D([0], [0], linestyle="none", label="— Sex —",
                               marker="none", color="gray"))
    handles += [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor="#444444", markersize=6,  label="Male (●)"),
        plt.Line2D([0], [0], marker="*", color="w",
                   markerfacecolor="#444444", markersize=8,  label="Female (★)"),
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor="#AAAAAA", markersize=5,  label="Unknown (·)"),
    ]

    ax.legend(handles=handles, fontsize=7, ncol=2,
              framealpha=0.9, loc="upper right", bbox_to_anchor=(1.32, 1),
              handletextpad=0.3, borderpad=0.6, labelspacing=0.4)

    ax.set_title(f"{title} (n={n})", fontsize=9, pad=8)
    ax.axis("off")
    plt.tight_layout()
    out_pdf = out_path.with_suffix(".pdf")
    fig.savefig(out_pdf,  bbox_inches="tight")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"저장: {out_path.name}, {out_pdf.name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights",    required=True)
    parser.add_argument("--data_dir",   default=None)
    parser.add_argument("--split",      default="test",
                        help="train | val | test | val+test (기본값: test)")
    parser.add_argument("--method",     default="tsne", choices=["tsne", "umap", "both"])
    parser.add_argument("--max_samples",type=int, default=3000,
                        help="feature 추출 최대 샘플 수 (t-SNE 속도)")
    parser.add_argument("--output_dir", default="experiments/tsne")
    args = parser.parse_args()

    import yaml
    with open("configs/default.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["training"]
    data_dir = Path(args.data_dir or cfg["data_dir"])
    out      = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt         = torch.load(args.weights, map_location=device, weights_only=False)
    class_names  = ckpt["class_names"]
    architecture = ckpt.get("architecture", "convnext_tiny")
    mode         = ckpt.get("mode", "species_only")

    model = build_model(num_classes=len(class_names), architecture=architecture, mode=mode)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    splits = args.split.split("+")
    sex_labels_csv = Path("data/labels/sex_labels.csv")

    # 모델 mode 관계없이 sex CSV가 있으면 항상 로드
    if sex_labels_csv.exists():
        from src.training.dataset import MultiTaskDataset, _load_sex_labels
        sex_labels_dict = _load_sex_labels(str(sex_labels_csv))
        use_sex = True
    else:
        use_sex = False

    all_feats, all_sp, all_sex = [], [], []
    for sp in splits:
        if use_sex:
            ds = MultiTaskDataset(data_dir / sp, EVAL_TRANSFORM, sex_labels_dict)
            loader = DataLoader(ds, batch_size=32, shuffle=False,
                                num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())
            f, s, sx = extract_features_with_sex(model, loader, device)
        else:
            ds = datasets.ImageFolder(str(data_dir / sp), EVAL_TRANSFORM)
            loader = DataLoader(ds, batch_size=32, shuffle=False,
                                num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())
            f, s = extract_features(model, loader, device)
            sx = np.full(len(s), -1)
        all_feats.append(f); all_sp.append(s); all_sex.append(sx)

    feats      = np.concatenate(all_feats)
    sp_labels  = np.concatenate(all_sp)
    sex_labels = np.concatenate(all_sex)

    # 샘플 수 제한 (t-SNE 속도)
    if len(feats) > args.max_samples:
        idx = np.random.default_rng(42).choice(len(feats), args.max_samples, replace=False)
        feats, sp_labels, sex_labels = feats[idx], sp_labels[idx], sex_labels[idx]

    print(f"Feature shape: {feats.shape}  |  샘플 수: {len(feats)}")

    stem = Path(args.weights).stem
    methods = ["tsne", "umap"] if args.method == "both" else [args.method]

    for m in methods:
        print(f"\n[{m.upper()}] 차원 축소 중...")
        try:
            emb = reduce(feats, m)
        except ImportError as e:
            print(f"  스킵: {e}")
            continue

        plot_combined(emb, sp_labels, sex_labels, class_names,
                      f"{m.upper()} — Species & Sex ({args.split}, {stem})",
                      out / f"{m}_combined_{stem}.png")


if __name__ == "__main__":
    main()
