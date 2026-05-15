"""
Test set 최종 평가 스크립트.
Bootstrap CI, ECE, sex-stratified 평가, 전체 지표 포함.

사용법:
    python scripts/evaluate_test.py
    python scripts/evaluate_test.py --weights models/weights/sweep/convnext_full_SO_s42.pth
"""

import argparse
import json
import time
import torch
import torch.nn.functional as F
import yaml
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, cohen_kappa_score,
)

from src.ml.classifier import build_model
from src.training.dataset import (
    EVAL_TRANSFORM, NUM_WORKERS,
    _load_sex_labels, MultiTaskDataset,
)

N_BOOTSTRAP = 1000
N_ECE_BINS  = 15


# ── 데이터 로드 ──────────────────────────────────────────────────────────

def load_test_data(data_dir: str, mode: str,
                   sex_labels_csv: str = "data/labels/sex_labels.csv"):
    """test split 로드. mode 무관하게 항상 sex 라벨도 병행 로드."""
    from pathlib import Path as P
    sex_labels = _load_sex_labels(sex_labels_csv) if P(sex_labels_csv).exists() else {}
    ds = MultiTaskDataset(P(data_dir) / "test", EVAL_TRANSFORM, sex_labels)
    loader = DataLoader(ds, batch_size=32, shuffle=False,
                        num_workers=NUM_WORKERS,
                        pin_memory=torch.cuda.is_available())
    return loader, ds.classes


# ── 추론 ─────────────────────────────────────────────────────────────────

def run_inference(model, loader, device, mode):
    model.eval()
    all_preds, all_labels = [], []
    all_confs             = []
    all_top3_correct      = []
    all_sex_gt, all_form_gt = [], []

    with torch.no_grad():
        for batch in loader:
            inputs       = batch[0].to(device)
            labels       = batch[1]
            sex_labels_b = batch[2]
            form_labels_b = batch[3]

            out    = model(inputs)
            sp_out = out[0] if isinstance(out, tuple) else out

            probs = F.softmax(sp_out, dim=1)
            confs, preds = probs.max(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_confs.extend(confs.cpu().numpy())
            all_sex_gt.extend(sex_labels_b.numpy())
            all_form_gt.extend(form_labels_b.numpy())

            k = min(3, sp_out.size(1))
            top3    = torch.topk(sp_out, k=k, dim=1).indices
            correct = top3.cpu().eq(labels.unsqueeze(1)).any(dim=1)
            all_top3_correct.extend(correct.numpy())

    return {
        "preds":        np.array(all_preds),
        "labels":       np.array(all_labels),
        "confs":        np.array(all_confs),
        "top3_correct": np.array(all_top3_correct),
        "sex_gt":       np.array(all_sex_gt),
        "form_gt":      np.array(all_form_gt),
    }


# ── 지표 계산 ────────────────────────────────────────────────────────────

def compute_metrics(preds, labels, confs, top3_correct,
                    class_names, sex_gt=None, form_gt=None):
    report = classification_report(
        labels, preds, target_names=class_names,
        output_dict=True, zero_division=0,
    )

    m = {
        "accuracy":          float(accuracy_score(labels, preds)),
        "top3_acc":          float(top3_correct.mean()),
        "cohen_kappa":       float(cohen_kappa_score(labels, preds)),
        "macro_f1":          report["macro avg"]["f1-score"],
        "macro_precision":   report["macro avg"]["precision"],
        "macro_recall":      report["macro avg"]["recall"],
        "weighted_f1":       report["weighted avg"]["f1-score"],
        "weighted_precision":report["weighted avg"]["precision"],
        "weighted_recall":   report["weighted avg"]["recall"],
    }

    # ECE
    accs_np  = (preds == labels).astype(float)
    bin_edges = np.linspace(0, 1, N_ECE_BINS + 1)
    ece = 0.0
    for i in range(N_ECE_BINS):
        mask = (confs >= bin_edges[i]) & (confs < bin_edges[i + 1])
        if mask.sum() > 0:
            ece += mask.sum() / len(confs) * abs(accs_np[mask].mean() - confs[mask].mean())
    m["ece"] = float(ece)

    # Per-class
    for name in class_names:
        if name in report:
            m[f"f1_{name}"]        = report[name]["f1-score"]
            m[f"precision_{name}"] = report[name]["precision"]
            m[f"recall_{name}"]    = report[name]["recall"]

    # Sex-stratified (sex_gt 있으면 모드 무관하게 계산)
    if sex_gt is not None:
        male_mask   = sex_gt == 0
        female_mask = sex_gt == 1
        if male_mask.sum() > 0:
            m["male_species_acc"]   = float(accuracy_score(labels[male_mask], preds[male_mask]))
        if female_mask.sum() > 0:
            m["female_species_acc"] = float(accuracy_score(labels[female_mask], preds[female_mask]))

        # 종별 성별 정확도
        for sp_idx, sp_name in enumerate(class_names):
            sp_mask = labels == sp_idx
            mf = sp_mask & male_mask
            ff = sp_mask & female_mask
            if mf.sum() >= 3:
                m[f"male_acc_{sp_name}"]   = float(accuracy_score(labels[mf], preds[mf]))
            if ff.sum() >= 3:
                m[f"female_acc_{sp_name}"] = float(accuracy_score(labels[ff], preds[ff]))

    return m


def bootstrap_ci(preds, labels, top3_correct, n=N_BOOTSTRAP, ci=95):
    """accuracy / top3_acc / macro_f1 에 대한 bootstrap CI."""
    rng  = np.random.default_rng(42)
    accs, top3s, f1s = [], [], []
    n_samples = len(labels)
    for _ in range(n):
        idx  = rng.integers(0, n_samples, n_samples)
        accs.append(accuracy_score(labels[idx], preds[idx]))
        top3s.append(top3_correct[idx].mean())
        f1s.append(f1_score(labels[idx], preds[idx], average="macro", zero_division=0))
    lo = (100 - ci) / 2
    hi = 100 - lo
    return {
        "acc_ci":    (float(np.percentile(accs,  lo)), float(np.percentile(accs,  hi))),
        "top3_ci":   (float(np.percentile(top3s, lo)), float(np.percentile(top3s, hi))),
        "f1_ci":     (float(np.percentile(f1s,   lo)), float(np.percentile(f1s,   hi))),
    }


# ── Figure 저장 ──────────────────────────────────────────────────────────

def save_confusion_matrix(labels, preds, class_names, out_path):
    cm = confusion_matrix(labels, preds)
    short = [c.split("_")[0] for c in class_names]
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(short)))
    ax.set_yticks(range(len(short)))
    ax.set_xticklabels(short, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(short, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Test Confusion Matrix")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"저장: {out_path}")


def save_reliability_diagram(labels, preds, confs, out_path):
    accs_np   = (preds == labels).astype(float)
    bin_edges = np.linspace(0, 1, N_ECE_BINS + 1)
    ece = 0.0
    bin_accs, bin_confs = [], []
    bar_heights = []
    for i in range(N_ECE_BINS):
        mask = (confs >= bin_edges[i]) & (confs < bin_edges[i + 1])
        if mask.sum() > 0:
            b_acc  = accs_np[mask].mean()
            b_conf = confs[mask].mean()
            ece   += mask.sum() / len(confs) * abs(b_acc - b_conf)
            bin_accs.append(b_acc)
            bin_confs.append(b_conf)
            bar_heights.append(b_acc)
        else:
            bar_heights.append(0)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.bar(bin_edges[:-1], bar_heights, width=1/N_ECE_BINS,
           align="edge", alpha=0.4, color="steelblue", label="Accuracy/bin")
    ax.plot(bin_confs, bin_accs, "ro-", markersize=4, label=f"ECE={ece:.3f}")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title("Reliability Diagram (Test)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"저장: {out_path}")


# ── 메인 ─────────────────────────────────────────────────────────────────

def main(weights_path: str, data_dir: str, output_dir: str):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt         = torch.load(weights_path, map_location=device, weights_only=False)
    class_names  = ckpt["class_names"]
    architecture = ckpt.get("architecture", "convnext_tiny")
    mode         = ckpt.get("mode", "species_only")

    model = build_model(num_classes=len(class_names), architecture=architecture, mode=mode)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    loader, _ = load_test_data(data_dir, mode)

    print(f"Architecture : {architecture}  |  Mode : {mode}")
    print(f"Weights      : {weights_path}")
    print(f"Test samples : {len(loader.dataset)}\n")

    data = run_inference(model, loader, device, mode)

    # 추론 속도
    dummy = torch.randn(1, 3, 448, 448).to(device)
    with torch.no_grad():
        for _ in range(3):
            _ = model(dummy)
        t0 = time.perf_counter()
        for _ in range(50):
            _ = model(dummy)
    inference_ms = (time.perf_counter() - t0) / 50 * 1000

    metrics = compute_metrics(
        data["preds"], data["labels"], data["confs"], data["top3_correct"],
        class_names,
        sex_gt  = data["sex_gt"]  if (data["sex_gt"]  >= 0).any() else None,
        form_gt = data["form_gt"] if (data["form_gt"] >= 0).any() else None,
    )
    metrics["inference_ms"] = round(inference_ms, 2)

    ci = bootstrap_ci(data["preds"], data["labels"], data["top3_correct"])
    metrics["acc_ci_lo"],  metrics["acc_ci_hi"]  = ci["acc_ci"]
    metrics["top3_ci_lo"], metrics["top3_ci_hi"] = ci["top3_ci"]
    metrics["f1_ci_lo"],   metrics["f1_ci_hi"]   = ci["f1_ci"]

    # 출력
    print(f"{'='*60}")
    print(f"  Accuracy     : {metrics['accuracy']:.4f}  "
          f"95% CI [{ci['acc_ci'][0]:.4f}, {ci['acc_ci'][1]:.4f}]")
    print(f"  Top-3 Acc    : {metrics['top3_acc']:.4f}  "
          f"95% CI [{ci['top3_ci'][0]:.4f}, {ci['top3_ci'][1]:.4f}]")
    print(f"  Macro F1     : {metrics['macro_f1']:.4f}  "
          f"95% CI [{ci['f1_ci'][0]:.4f}, {ci['f1_ci'][1]:.4f}]")
    print(f"  Cohen's Kappa: {metrics['cohen_kappa']:.4f}")
    print(f"  ECE          : {metrics['ece']:.4f}")
    print(f"  Inference    : {inference_ms:.1f} ms/img")
    if "male_species_acc" in metrics:
        print(f"  Male Acc     : {metrics['male_species_acc']:.4f}")
        print(f"  Female Acc   : {metrics['female_species_acc']:.4f}")
    print(f"{'='*60}\n")
    print(classification_report(data["labels"], data["preds"],
                                  target_names=class_names, digits=4, zero_division=0))

    # 저장
    stem = Path(weights_path).stem
    with open(out / f"test_metrics_{stem}.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"저장: test_metrics_{stem}.json")

    save_confusion_matrix(data["labels"], data["preds"], class_names,
                          out / f"test_confusion_{stem}.png")
    save_reliability_diagram(data["labels"], data["preds"], data["confs"],
                             out / f"test_reliability_{stem}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights",    default=None)
    parser.add_argument("--data_dir",   default=None)
    parser.add_argument("--output_dir", default="experiments/test_results")
    args = parser.parse_args()

    with open("configs/default.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["training"]

    main(
        weights_path = args.weights  or cfg["weights_path"],
        data_dir     = args.data_dir or cfg["data_dir"],
        output_dir   = args.output_dir,
    )
