"""
MLflow에서 전체 실험 결과를 가져와 논문용 figure/table을 자동 생성한다.

사용법:
    python scripts/analyze_results.py
    python scripts/analyze_results.py --output_dir experiments/analysis
"""

import argparse
import re
import mlflow
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from pathlib import Path

MLFLOW_URI      = "http://localhost:5001"
EXPERIMENT_NAME = "lucanidae_classifier"

ARCH_ORDER = ["convnext_tiny", "efficientnet_b3", "swin_tiny", "vit_small", "dinov2_vits14"]
PREP_ORDER = ["full", "seg_soft", "seg_hard", "bbox"]
MODE_ORDER = ["species_only", "multi_task"]

ARCH_LABEL = {
    "convnext_tiny":   "ConvNeXt-T",
    "efficientnet_b3": "EffNet-B3",
    "swin_tiny":       "Swin-T",
    "vit_small":       "ViT-B/16",
    "dinov2_vits14":   "DINOv2-S",
}
PREP_LABEL = {
    "full":     "Full",
    "seg_soft": "Seg-Soft",
    "seg_hard": "Seg-Hard",
    "bbox":     "BBox",
}


# ── MLflow에서 run 파싱 ──────────────────────────────────────────────────

def parse_run_name(name: str) -> dict | None:
    """
    run_name에서 arch/prep/mode/seed 추출.
    두 가지 형식 지원:
      - 신규: convnext_full_SO_s42
      - 구형: convnext-tiny_full_species-only_v1
    """
    # 신규 형식
    m = re.match(
        r"^(convnext|efficientnet|swin|vit|dinov2)_(full|bbox|seg_hard|seg_soft)_(SO|MT|SX|FO)_s(\d+)$",
        name
    )
    if m:
        arch_map = {
            "convnext": "convnext_tiny", "efficientnet": "efficientnet_b3",
            "swin": "swin_tiny", "vit": "vit_small", "dinov2": "dinov2_vits14",
        }
        mode_map = {"SO": "species_only", "MT": "multi_task",
                    "SX": "sex_only",     "FO": "form_only"}
        return {
            "arch": arch_map[m.group(1)],
            "prep": m.group(2),
            "mode": mode_map[m.group(3)],
            "seed": int(m.group(4)),
        }

    # 구형 형식 (v1 실험들)
    arch_map = {
        "convnext-tiny": "convnext_tiny", "efficientnet-b3": "efficientnet_b3",
        "swin-tiny": "swin_tiny", "vit-small": "vit_small",
        "dinov2-vits14": "dinov2_vits14",
    }
    prep_map = {
        "full": "full", "bbox": "bbox", "seg-hard": "seg_hard",
        "seg-soft": "seg_soft", "seg-gray": "seg_hard",
    }
    mode_map = {"species-only": "species_only", "multi-task": "multi_task"}

    parts = name.split("_")
    if len(parts) >= 3:
        arch = arch_map.get(parts[0])
        prep = prep_map.get(parts[1])
        mode_raw = parts[2] if len(parts) > 2 else None
        mode = mode_map.get(mode_raw)
        if arch and prep and mode:
            return {"arch": arch, "prep": prep, "mode": mode, "seed": 0}

    return None


def fetch_runs() -> pd.DataFrame:
    mlflow.set_tracking_uri(MLFLOW_URI)
    client = mlflow.tracking.MlflowClient()
    exp    = client.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        raise RuntimeError(f"MLflow experiment '{EXPERIMENT_NAME}' 없음")

    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        max_results=2000,
    )

    records = []
    for run in runs:
        parsed = parse_run_name(run.info.run_name)
        if parsed is None:
            continue

        metrics = run.data.metrics
        params  = run.data.params
        rec = {
            **parsed,
            "run_name":           run.info.run_name,
            "val_acc":            metrics.get("best_val_acc", metrics.get("val_acc", None)),
            "val_loss":           metrics.get("val_loss", None),
            "train_acc":          metrics.get("train_acc", None),
            "macro_f1":           metrics.get("macro_f1", None),
            "macro_precision":    metrics.get("macro_precision", None),
            "macro_recall":       metrics.get("macro_recall", None),
            "weighted_f1":        metrics.get("weighted_f1", None),
            "weighted_precision": metrics.get("weighted_precision", None),
            "weighted_recall":    metrics.get("weighted_recall", None),
            "top3_acc":           metrics.get("top3_acc", None),
            "cohen_kappa":        metrics.get("cohen_kappa", None),
            "sex_acc":            metrics.get("sex_acc", None),
            "sex_macro_f1":       metrics.get("sex_macro_f1", None),
            "form_acc":           metrics.get("form_acc", None),
            "form_macro_f1":      metrics.get("form_macro_f1", None),
            "male_species_acc":   metrics.get("male_species_acc", None),
            "female_species_acc": metrics.get("female_species_acc", None),
            "num_params_M":       float(params["num_params_M"]) if "num_params_M" in params else None,
            "inference_ms":       float(params["inference_ms"])  if "inference_ms"  in params else None,
            "run_id":             run.info.run_id,
        }
        for k, v in metrics.items():
            if any(k.startswith(p) for p in
                   ("f1_", "precision_", "recall_", "male_acc_", "female_acc_")):
                rec[k] = v
        records.append(rec)

    return pd.DataFrame(records)


# ── Figure 생성 ──────────────────────────────────────────────────────────

def _bar_comparison(df, metric, ylabel, title, output_dir, fname):
    """아키텍처별 species_only vs multi_task 바 차트 (공용)"""
    data = (
        df[df["prep"] == "full"]
        .groupby(["arch", "mode"])[metric]
        .mean()
        .reset_index()
    )
    if data[metric].isna().all():
        print(f"  (데이터 없음, 스킵: {fname})")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    x     = np.arange(len(ARCH_ORDER))
    width = 0.35

    for i, mode in enumerate(MODE_ORDER):
        vals = [
            data[(data["arch"] == a) & (data["mode"] == mode)][metric].mean()
            if not data[(data["arch"] == a) & (data["mode"] == mode)].empty else 0
            for a in ARCH_ORDER
        ]
        scale = 100 if metric != "macro_f1" else 100
        label = "Species-Only" if mode == "species_only" else "Multi-Task"
        ax.bar(x + i * width, [v * scale for v in vals], width, label=label)

    ax.set_xticks(x + width / 2)
    ax.set_xticklabels([ARCH_LABEL[a] for a in ARCH_ORDER])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.set_ylim(0, 100)
    plt.tight_layout()
    fig.savefig(output_dir / fname, dpi=150)
    plt.close(fig)
    print(f"저장: {fname}")


def fig_architecture_comparison(df: pd.DataFrame, output_dir: Path):
    """아키텍처별 val_acc + macro_f1 바 차트"""
    _bar_comparison(df, "val_acc",   "Val Accuracy (%)",
                    "Architecture Comparison — Val Accuracy (Full Preprocessing)",
                    output_dir, "fig_architecture_comparison_acc.png")
    _bar_comparison(df, "macro_f1",  "Macro F1 (%)",
                    "Architecture Comparison — Macro F1 (Full Preprocessing)",
                    output_dir, "fig_architecture_comparison_f1.png")


def fig_preprocessing_ablation(df: pd.DataFrame, output_dir: Path):
    """전처리별 성능 비교 (convnext, species_only 기준)"""
    data = (
        df[(df["arch"] == "convnext_tiny") & (df["mode"] == "species_only")]
        .groupby("prep")["val_acc"]
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    vals    = [
        data[data["prep"] == p]["val_acc"].mean() * 100
        if not data[data["prep"] == p].empty else 0
        for p in PREP_ORDER
    ]
    bars = ax.bar([PREP_LABEL[p] for p in PREP_ORDER], vals, color="steelblue")
    ax.bar_label(bars, fmt="%.1f%%", padding=3)
    ax.set_ylabel("Val Accuracy (%)")
    ax.set_title("Preprocessing Ablation (ConvNeXt-T, Species-Only)")
    ax.set_ylim(0, 100)
    plt.tight_layout()
    fig.savefig(output_dir / "fig_preprocessing_ablation.png", dpi=150)
    plt.close(fig)
    print("저장: fig_preprocessing_ablation.png")


def _heatmap(df, metric, mode, title, fname, output_dir, vmin, vmax):
    pivot = pd.DataFrame(index=ARCH_ORDER, columns=PREP_ORDER, dtype=float)
    sub   = df[df["mode"] == mode].groupby(["arch", "prep"])[metric].mean()
    for arch in ARCH_ORDER:
        for prep in PREP_ORDER:
            if (arch, prep) in sub.index:
                pivot.loc[arch, prep] = sub[(arch, prep)] * 100
    if pivot.isna().all().all():
        print(f"  (데이터 없음, 스킵: {fname})")
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        pivot.astype(float),
        annot=True, fmt=".1f", cmap="YlOrRd",
        xticklabels=[PREP_LABEL[p] for p in PREP_ORDER],
        yticklabels=[ARCH_LABEL[a] for a in ARCH_ORDER],
        ax=ax, vmin=vmin, vmax=vmax,
    )
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(output_dir / fname, dpi=150)
    plt.close(fig)
    print(f"저장: {fname}")


def fig_full_heatmap(df: pd.DataFrame, output_dir: Path):
    """5 arch × 4 prep heatmap — val_acc & macro_f1"""
    for mode in MODE_ORDER:
        mode_label = "Species-Only" if mode == "species_only" else "Multi-Task"
        _heatmap(df, "val_acc",  mode,
                 f"Val Accuracy Heatmap — {mode_label}",
                 f"fig_heatmap_acc_{mode}.png", output_dir, vmin=50, vmax=90)
        _heatmap(df, "macro_f1", mode,
                 f"Macro F1 Heatmap — {mode_label}",
                 f"fig_heatmap_f1_{mode}.png",  output_dir, vmin=40, vmax=90)


def fig_multitask_effect(df: pd.DataFrame, output_dir: Path):
    """multi_task 효과 (SO 대비 MT 차이)"""
    data = (
        df[df["prep"] == "full"]
        .groupby(["arch", "mode"])["val_acc"]
        .mean()
        .unstack("mode")
        .reset_index()
    )
    if "multi_task" not in data.columns or "species_only" not in data.columns:
        return

    data["diff"] = (data["multi_task"] - data["species_only"]) * 100
    data = data.sort_values("diff", ascending=True)

    colors = ["tomato" if d < 0 else "steelblue" for d in data["diff"]]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh([ARCH_LABEL[a] for a in data["arch"]], data["diff"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Val Accuracy Change (%)")
    ax.set_title("Multi-Task Effect vs Species-Only (Full Preprocessing)")
    plt.tight_layout()
    fig.savefig(output_dir / "fig_multitask_effect.png", dpi=150)
    plt.close(fig)
    print("저장: fig_multitask_effect.png")


# ── Table 생성 ───────────────────────────────────────────────────────────

def _make_result_str(mean, std):
    if pd.isna(mean):
        return "-"
    if pd.isna(std) or std == 0:
        return f"{mean:.1f}"
    return f"{mean:.1f}±{std:.1f}"


def table_full_results(df: pd.DataFrame, output_dir: Path):
    """전체 결과 CSV + LaTeX 테이블 (val_acc, top3_acc, macro_f1, macro_precision, macro_recall)"""
    rows = []
    for (arch, prep, mode), g in df.groupby(["arch", "prep", "mode"]):
        def ms(col):
            m = g[col].mean() * 100 if col in g and g[col].notna().any() else float("nan")
            s = g[col].std()  * 100 if len(g) > 1 and col in g and g[col].notna().any() else float("nan")
            return _make_result_str(m, s)

        rows.append({
            "arch": arch, "prep": prep, "mode": mode,
            "acc":               ms("val_acc"),
            "top3_acc":          ms("top3_acc"),
            "macro_f1":          ms("macro_f1"),
            "macro_precision":   ms("macro_precision"),
            "macro_recall":      ms("macro_recall"),
        })

    agg = pd.DataFrame(rows)
    agg["arch"] = pd.Categorical(agg["arch"], ARCH_ORDER)
    agg["prep"] = pd.Categorical(agg["prep"], PREP_ORDER)
    agg = agg.sort_values(["arch", "prep", "mode"])

    agg.to_csv(output_dir / "table_full_results.csv", index=False, encoding="utf-8-sig")
    print("저장: table_full_results.csv")

    latex = agg.to_latex(index=False, escape=False)
    (output_dir / "table_full_results.tex").write_text(latex, encoding="utf-8")
    print("저장: table_full_results.tex")


def table_multitask_results(df: pd.DataFrame, output_dir: Path):
    """multi_task 전용 — sex/form 분류 성능 테이블"""
    sub = df[df["mode"] == "multi_task"].copy()
    if sub.empty or sub["sex_acc"].isna().all():
        print("  (multi_task 데이터 없음, 스킵: table_multitask_results)")
        return

    rows = []
    for (arch, prep), g in sub.groupby(["arch", "prep"]):
        def ms(col):
            m = g[col].mean() * 100 if col in g and g[col].notna().any() else float("nan")
            s = g[col].std()  * 100 if len(g) > 1 and col in g and g[col].notna().any() else float("nan")
            return _make_result_str(m, s)

        rows.append({
            "arch": arch, "prep": prep,
            "sex_acc":      ms("sex_acc"),
            "sex_macro_f1": ms("sex_macro_f1"),
            "form_acc":     ms("form_acc"),
            "form_macro_f1":ms("form_macro_f1"),
        })

    agg = pd.DataFrame(rows)
    agg["arch"] = pd.Categorical(agg["arch"], ARCH_ORDER)
    agg["prep"] = pd.Categorical(agg["prep"], PREP_ORDER)
    agg = agg.sort_values(["arch", "prep"])

    agg.to_csv(output_dir / "table_multitask_results.csv", index=False, encoding="utf-8-sig")
    print("저장: table_multitask_results.csv")

    latex = agg.to_latex(index=False, escape=False)
    (output_dir / "table_multitask_results.tex").write_text(latex, encoding="utf-8")
    print("저장: table_multitask_results.tex")


def fig_training_curves(df: pd.DataFrame, output_dir: Path):
    """ConvNeXt-T / full 기준 모든 seed·mode 학습 곡선 (val_loss, val_acc)"""
    mlflow.set_tracking_uri(MLFLOW_URI)
    client = mlflow.tracking.MlflowClient()
    exp    = client.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        return

    sub = df[(df["arch"] == "convnext_tiny") & (df["prep"] == "full")]
    if sub.empty:
        print("  (ConvNeXt-T/full 데이터 없음, 스킵: fig_training_curves)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    colors    = plt.cm.tab10.colors

    for idx, (_, row) in enumerate(sub.iterrows()):
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string=f"attributes.run_name = '{row['run_name']}'",
            max_results=1,
        )
        if not runs:
            continue
        run_id = runs[0].info.run_id
        color  = colors[idx % len(colors)]
        label  = f"{'SO' if row['mode'] == 'species_only' else 'MT'}_s{row['seed']}"

        for metric, ax in [("val_loss", axes[0]), ("val_acc", axes[1])]:
            hist   = client.get_metric_history(run_id, metric)
            steps  = [h.step for h in hist]
            vals   = [h.value for h in hist]
            ax.plot(steps, vals, label=label, color=color, alpha=0.8)

    axes[0].set(title="Val Loss (ConvNeXt-T, Full)", xlabel="Epoch", ylabel="Loss")
    axes[1].set(title="Val Accuracy (ConvNeXt-T, Full)", xlabel="Epoch", ylabel="Accuracy")
    for ax in axes:
        ax.legend(fontsize=7)
    plt.tight_layout()
    fig.savefig(output_dir / "fig_training_curves.png", dpi=150)
    plt.close(fig)
    print("저장: fig_training_curves.png")


def _bootstrap_ci(a: list, b: list, n_boot: int = 10000, ci: float = 0.95) -> tuple:
    """
    두 조건 간 평균 차이(b - a)에 대한 bootstrap 95% CI.
    n=3처럼 작은 표본에서도 Wilcoxon 대비 적합.
    CI가 0을 포함하지 않으면 유의미한 차이로 간주.
    """
    diffs = np.array(b) - np.array(a)
    rng   = np.random.default_rng(42)
    boot  = [rng.choice(diffs, len(diffs), replace=True).mean()
             for _ in range(n_boot)]
    lo = np.percentile(boot, (1 - ci) / 2 * 100)
    hi = np.percentile(boot, (1 + ci) / 2 * 100)
    return float(diffs.mean()), float(lo), float(hi), bool(lo > 0 or hi < 0)


def table_significance_test(df: pd.DataFrame, output_dir: Path):
    """아키텍처 간 val_acc Bootstrap 95% CI (ConvNeXt-T 대비, full/species_only).
    n=3에서 Wilcoxon은 p<0.05 달성 불가 → bootstrap CI 사용."""
    baseline  = "convnext_tiny"
    base_data = df[(df["arch"] == baseline) & (df["prep"] == "full") & (df["mode"] == "species_only")]
    rows = []

    for arch in ARCH_ORDER:
        if arch == baseline:
            continue
        cmp_data = df[(df["arch"] == arch) & (df["prep"] == "full") & (df["mode"] == "species_only")]
        common   = sorted(set(base_data["seed"]) & set(cmp_data["seed"]))
        if len(common) < 2:
            continue

        x = [base_data[base_data["seed"] == s]["val_acc"].mean() for s in common]
        y = [cmp_data[cmp_data["seed"]  == s]["val_acc"].mean() for s in common]
        diff, lo, hi, sig = _bootstrap_ci(x, y)

        rows.append({
            "baseline":        ARCH_LABEL[baseline],
            "comparison":      ARCH_LABEL[arch],
            "baseline_acc":    f"{np.mean(x)*100:.1f}",
            "comparison_acc":  f"{np.mean(y)*100:.1f}",
            "mean_diff":       f"{diff*100:+.1f}",
            "95%_CI":          f"[{lo*100:+.1f}, {hi*100:+.1f}]",
            "significant":     "Yes" if sig else "No",
        })

    if not rows:
        print("  (유의성 검정 데이터 부족)")
        return

    out_df = pd.DataFrame(rows)
    out_df.to_csv(output_dir / "table_significance_test.csv", index=False, encoding="utf-8-sig")
    latex = out_df.to_latex(index=False, escape=False)
    (output_dir / "table_significance_test.tex").write_text(latex, encoding="utf-8")
    print("저장: table_significance_test.csv / .tex")


def table_dataset_stats(output_dir: Path, data_dir: str = "data/final"):
    """종별 train/val/test 샘플 수 테이블"""
    base = Path(data_dir)
    if not base.exists():
        print(f"  (데이터 디렉토리 없음, 스킵: {data_dir})")
        return

    exts    = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    splits  = ["train", "val", "test"]
    records = {}

    for split in splits:
        split_dir = base / split
        if not split_dir.exists():
            continue
        for cls_dir in sorted(split_dir.iterdir()):
            if not cls_dir.is_dir():
                continue
            count = sum(1 for f in cls_dir.iterdir() if f.suffix in exts)
            records.setdefault(cls_dir.name, {})[split] = count

    df_stats = pd.DataFrame(records).T.reindex(columns=splits, fill_value=0).fillna(0).astype(int)
    df_stats["total"] = df_stats.sum(axis=1)
    df_stats = df_stats.sort_index()

    df_stats.to_csv(output_dir / "table_dataset_stats.csv", encoding="utf-8-sig")
    latex = df_stats.to_latex()
    (output_dir / "table_dataset_stats.tex").write_text(latex, encoding="utf-8")
    print("저장: table_dataset_stats.csv / .tex")


def table_model_efficiency(df: pd.DataFrame, output_dir: Path):
    """아키텍처별 파라미터 수 + 추론 시간 비교 테이블"""
    sub = df[df["prep"] == "full"].groupby("arch")[["num_params_M", "inference_ms"]].mean()
    if sub["num_params_M"].isna().all():
        print("  (num_params_M 없음, 스킵: table_model_efficiency)")
        return

    sub.index = [ARCH_LABEL.get(a, a) for a in sub.index]
    sub = sub.round(2)
    sub.columns = ["Params (M)", "Inference (ms/img)"]

    sub.to_csv(output_dir / "table_model_efficiency.csv", encoding="utf-8-sig")
    latex = sub.to_latex()
    (output_dir / "table_model_efficiency.tex").write_text(latex, encoding="utf-8")
    print("저장: table_model_efficiency.csv / .tex")


def fig_female_vs_male_accuracy(df: pd.DataFrame, output_dir: Path):
    """N1 핵심 figure: 종별 수컷 vs 암컷 종 분류 정확도 (MT 모드)"""
    sub = df[(df["mode"] == "multi_task") & (df["prep"] == "full")]
    if sub.empty:
        print("  (multi_task/full 데이터 없음, 스킵: fig_female_vs_male_accuracy)")
        return

    # 종 이름 추출 (female_acc_ prefix)
    female_cols = [c for c in sub.columns if c.startswith("female_acc_")]
    male_cols   = [c.replace("female_acc_", "male_acc_") for c in female_cols]

    paired = [(fc, mc) for fc, mc in zip(female_cols, male_cols) if mc in sub.columns]
    if not paired:
        print("  (per-species sex acc 없음, 스킵)")
        return

    species_labels = [fc.replace("female_acc_", "").split("_")[0] for fc, _ in paired]
    fem_means = [sub[fc].mean() * 100 for fc, _ in paired]
    mal_means = [sub[mc].mean() * 100 for _, mc in paired]

    x     = np.arange(len(species_labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width/2, mal_means, width, label="Male",   color="steelblue")
    ax.bar(x + width/2, fem_means, width, label="Female", color="tomato")
    ax.set_xticks(x)
    ax.set_xticklabels(species_labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Species Accuracy (%)")
    ax.set_title("Per-Species Accuracy: Male vs Female (Multi-Task, Full Prep)")
    ax.legend()
    ax.set_ylim(0, 100)
    plt.tight_layout()
    fig.savefig(output_dir / "fig_female_vs_male_accuracy.png", dpi=150)
    plt.close(fig)
    print("저장: fig_female_vs_male_accuracy.png")


def table_mt_ablation(df: pd.DataFrame, output_dir: Path):
    """
    SO / SX / FO / MT 4개 모드 비교 테이블.
    convnext_tiny / full 기준, 3 seeds 평균±표준편차.
    """
    ablation_modes = ["species_only", "sex_only", "form_only", "multi_task"]
    mode_label     = {"species_only": "SO", "sex_only": "SX (sex)",
                      "form_only": "FO (form)", "multi_task": "MT (full)"}

    # SX/FO가 실제로 존재하는 arch+prep 자동 탐지 (ablation이 best arch에서 돌아가므로)
    sx_fo = df[df["mode"].isin(["sex_only", "form_only"]) & (df["prep"] == "full")]
    if sx_fo.empty:
        print("  (MT ablation 데이터 없음, 스킵)")
        return
    ablation_arch = sx_fo["arch"].value_counts().idxmax()
    ablation_prep = "full"

    sub = df[(df["arch"] == ablation_arch) & (df["prep"] == ablation_prep) &
             (df["mode"].isin(ablation_modes))]
    if sub.empty:
        print("  (MT ablation 데이터 없음, 스킵)")
        return
    print(f"  MT ablation 기준: arch={ablation_arch}, prep={ablation_prep}")

    rows = []
    for mode in ablation_modes:
        g = sub[sub["mode"] == mode]
        if g.empty:
            continue

        def ms(col):
            m = g[col].mean() * 100 if col in g and g[col].notna().any() else float("nan")
            s = g[col].std()  * 100 if len(g) > 1 and col in g and g[col].notna().any() else float("nan")
            return _make_result_str(m, s)

        rows.append({
            "Mode":            mode_label[mode],
            "Val Acc":         ms("val_acc"),
            "Macro F1":        ms("macro_f1"),
            "Top-3 Acc":       ms("top3_acc"),
            "Sex Acc":         ms("sex_acc"),
            "Form Acc":        ms("form_acc"),
            "Male Sp. Acc":    ms("male_species_acc"),
            "Female Sp. Acc":  ms("female_species_acc"),
        })

    agg = pd.DataFrame(rows)
    agg.to_csv(output_dir / "table_mt_ablation.csv", index=False, encoding="utf-8-sig")
    latex = agg.to_latex(index=False, escape=False)
    (output_dir / "table_mt_ablation.tex").write_text(latex, encoding="utf-8")
    print("저장: table_mt_ablation.csv / .tex")


def fig_so_vs_mt_female_accuracy(df: pd.DataFrame, output_dir: Path):
    """
    N1 핵심 증거: 종별 암컷 분류 정확도 SO vs MT 직접 비교.
    Multi-task auxiliary loss가 암컷 정확도를 향상시키는지 검증.
    """
    sub_full = df[df["prep"] == "full"]
    female_cols = sorted([c for c in df.columns if c.startswith("female_acc_")])
    if not female_cols:
        print("  (per-species female_acc 없음, 스킵: fig_so_vs_mt_female_accuracy)")
        return

    so_df = sub_full[sub_full["mode"] == "species_only"]
    mt_df = sub_full[sub_full["mode"] == "multi_task"]
    if so_df.empty or mt_df.empty:
        print("  (SO 또는 MT 데이터 없음, 스킵)")
        return

    species = [c.replace("female_acc_", "") for c in female_cols]
    so_vals  = [so_df[c].mean() * 100 if c in so_df.columns and so_df[c].notna().any() else float("nan") for c in female_cols]
    mt_vals  = [mt_df[c].mean() * 100 if c in mt_df.columns and mt_df[c].notna().any() else float("nan") for c in female_cols]

    # Filter species with data in both modes
    valid = [(sp, s, m) for sp, s, m in zip(species, so_vals, mt_vals)
             if not (pd.isna(s) or pd.isna(m))]
    if not valid:
        print("  (SO/MT 공통 종 데이터 없음, 스킵)")
        return

    species_v, so_v, mt_v = zip(*valid)
    diffs = [m - s for s, m in zip(so_v, mt_v)]
    order = sorted(range(len(diffs)), key=lambda i: diffs[i])
    species_v = [species_v[i] for i in order]
    so_v      = [so_v[i]      for i in order]
    mt_v      = [mt_v[i]      for i in order]
    diffs     = [diffs[i]     for i in order]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: grouped bar SO vs MT per species
    x     = np.arange(len(species_v))
    width = 0.38
    axes[0].bar(x - width/2, so_v, width, label="Species-Only", color="steelblue", alpha=0.85)
    axes[0].bar(x + width/2, mt_v, width, label="Multi-Task",   color="tomato",    alpha=0.85)
    axes[0].set_xticks(x)
    short_sp = [s.split("_")[0][:10] for s in species_v]
    axes[0].set_xticklabels(short_sp, rotation=45, ha="right", fontsize=8)
    axes[0].set_ylabel("Female Accuracy (%)")
    axes[0].set_title("Female Classification: SO vs MT (Full Prep)")
    axes[0].legend(fontsize=9)
    axes[0].set_ylim(0, 110)

    # Right: delta bar (MT - SO)
    colors = ["tomato" if d < 0 else "steelblue" for d in diffs]
    axes[1].barh(short_sp, diffs, color=colors)
    axes[1].axvline(0, color="black", linewidth=0.8)
    axes[1].set_xlabel("ΔFemale Accuracy (MT − SO, %)")
    axes[1].set_title("Multi-Task Gain on Female Images")
    mean_diff = np.mean(diffs)
    axes[1].axvline(mean_diff, color="navy", linestyle="--", linewidth=1,
                    label=f"Mean Δ={mean_diff:+.1f}%")
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    fig.savefig(output_dir / "fig_so_vs_mt_female_accuracy.png", dpi=150)
    plt.close(fig)
    print("저장: fig_so_vs_mt_female_accuracy.png")


def fig_fewshot_scatter(df: pd.DataFrame, output_dir: Path,
                         data_dir: str = "data/final"):
    """N4: 클래스별 학습 샘플 수 vs macro F1 scatter"""
    train_dir = Path(data_dir) / "train"
    if not train_dir.exists():
        print("  (data/final/train 없음, 스킵: fig_fewshot_scatter)")
        return

    exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    sample_counts = {
        d.name: sum(1 for f in d.iterdir() if f.suffix in exts)
        for d in train_dir.iterdir() if d.is_dir()
    }

    sub = df[(df["prep"] == "full") & (df["mode"] == "species_only")]
    if sub.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for arch in ARCH_ORDER:
        arch_df = sub[sub["arch"] == arch]
        if arch_df.empty:
            continue
        f1_cols = [c for c in arch_df.columns if c.startswith("f1_")]
        for col in f1_cols:
            sp_name = col[3:]
            if sp_name in sample_counts:
                ax.scatter(sample_counts[sp_name], arch_df[col].mean() * 100,
                           alpha=0.6, label=ARCH_LABEL[arch] if col == f1_cols[0] else "")

    ax.set_xlabel("Training Samples per Class")
    ax.set_ylabel("Per-class F1 (%)")
    ax.set_title("Few-shot Performance: Samples vs F1 (Full Prep, Species-Only)")
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=8)
    plt.tight_layout()
    fig.savefig(output_dir / "fig_fewshot_scatter.png", dpi=150)
    plt.close(fig)
    print("저장: fig_fewshot_scatter.png")


def table_prep_significance(df: pd.DataFrame, output_dir: Path):
    """N3: 전처리 간 val_acc Bootstrap 95% CI (ConvNeXt-T, species_only)."""
    baseline  = "full"
    base_data = df[(df["arch"] == "convnext_tiny") & (df["prep"] == baseline)
                   & (df["mode"] == "species_only")]
    rows = []

    for prep in PREP_ORDER:
        if prep == baseline:
            continue
        cmp = df[(df["arch"] == "convnext_tiny") & (df["prep"] == prep)
                 & (df["mode"] == "species_only")]
        common = sorted(set(base_data["seed"]) & set(cmp["seed"]))
        if len(common) < 2:
            continue
        x = [base_data[base_data["seed"] == s]["val_acc"].mean() for s in common]
        y = [cmp[cmp["seed"] == s]["val_acc"].mean() for s in common]
        diff, lo, hi, sig = _bootstrap_ci(x, y)
        rows.append({
            "baseline":       PREP_LABEL[baseline],
            "comparison":     PREP_LABEL[prep],
            "baseline_acc":   f"{np.mean(x)*100:.1f}",
            "comparison_acc": f"{np.mean(y)*100:.1f}",
            "mean_diff":      f"{diff*100:+.1f}",
            "95%_CI":         f"[{lo*100:+.1f}, {hi*100:+.1f}]",
            "significant":    "Yes" if sig else "No",
        })

    if not rows:
        return
    out_df = pd.DataFrame(rows)
    out_df.to_csv(output_dir / "table_prep_significance.csv", index=False, encoding="utf-8-sig")
    (output_dir / "table_prep_significance.tex").write_text(
        out_df.to_latex(index=False, escape=False), encoding="utf-8")
    print("저장: table_prep_significance.csv / .tex")


def fig_lambda_ablation(output_dir: Path):
    """
    λ_sex / λ_form 민감도 분석 — 두 subplots.
    run_name 패턴: convnext_full_MT_lsex{v}_lform{v}_s42
    SO 기준선(convnext_full_SO_s42)도 함께 표시.
    """
    import re as _re
    mlflow.set_tracking_uri(MLFLOW_URI)
    client = mlflow.tracking.MlflowClient()
    exp    = client.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        return

    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        max_results=2000,
    )

    # λ_sex sweep (λ_form=0.2 고정)
    sex_pat  = _re.compile(r"convnext_full_MT_lsex([\d.]+)_lform0\.2_s42$")
    # λ_form sweep (λ_sex=0.3 고정)
    form_pat = _re.compile(r"convnext_full_MT_lsex0\.3_lform([\d.]+)_s42$")

    sex_rows, form_rows = [], []
    so_acc = None

    for run in runs:
        name = run.info.run_name
        acc  = run.data.metrics.get("best_val_acc", run.data.metrics.get("val_acc"))
        if acc is None:
            continue
        m = sex_pat.match(name)
        if m:
            sex_rows.append((float(m.group(1)), acc * 100))
        m = form_pat.match(name)
        if m:
            form_rows.append((float(m.group(1)), acc * 100))
        if name == "convnext_full_SO_s42":
            so_acc = acc * 100

    if not sex_rows and not form_rows:
        print("  (λ ablation 데이터 없음, 스킵: fig_lambda_ablation)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, rows, fixed_label, xlabel, title in [
        (axes[0], sex_rows,  "λ_form=0.2 fixed", "λ_sex",  "λ_sex Sensitivity"),
        (axes[1], form_rows, "λ_sex=0.3 fixed",  "λ_form", "λ_form Sensitivity"),
    ]:
        if not rows:
            ax.set_title(f"{title} (데이터 없음)")
            continue
        rows_sorted = sorted(rows, key=lambda x: x[0])
        xs   = [r[0] for r in rows_sorted]
        ys   = [r[1] for r in rows_sorted]
        ax.plot(xs, ys, "o-", color="steelblue", label=f"MT ({fixed_label})")
        if so_acc is not None:
            ax.axhline(so_acc, color="tomato", linestyle="--",
                       label=f"SO baseline ({so_acc:.1f}%)")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Val Accuracy (%)")
        ax.set_title(title)
        ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(output_dir / "fig_lambda_ablation.png", dpi=150)
    plt.close(fig)
    print("저장: fig_lambda_ablation.png")


def table_per_class_f1(df: pd.DataFrame, output_dir: Path):
    """종별 F1 비교 테이블 (full + 아키텍처별)"""
    f1_cols = [c for c in df.columns if c.startswith("f1_")]
    if not f1_cols:
        return

    sub = df[(df["prep"] == "full") & (df["mode"] == "species_only")]
    agg = sub.groupby("arch")[f1_cols].mean()
    agg.index = [ARCH_LABEL.get(a, a) for a in agg.index]
    agg = (agg * 100).round(1)

    agg.to_csv(output_dir / "table_per_class_f1.csv", encoding="utf-8-sig")
    print("저장: table_per_class_f1.csv")


# ── 메인 ─────────────────────────────────────────────────────────────────

def main(output_dir: str):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("MLflow에서 결과 가져오는 중...")
    df = fetch_runs()
    print(f"총 {len(df)}개 run 파싱 완료\n")

    if df.empty:
        print("파싱된 run 없음. run_name 형식을 확인하세요.")
        return

    df.to_csv(out / "all_runs.csv", index=False, encoding="utf-8-sig")
    print("저장: all_runs.csv\n")

    print("Figure 생성 중...")
    fig_architecture_comparison(df, out)
    fig_preprocessing_ablation(df, out)
    fig_full_heatmap(df, out)
    fig_multitask_effect(df, out)

    print("\nFigure 추가 생성 중...")
    fig_training_curves(df, out)
    fig_female_vs_male_accuracy(df, out)
    fig_so_vs_mt_female_accuracy(df, out)
    fig_fewshot_scatter(df, out)
    fig_lambda_ablation(out)

    print("\nTable 생성 중...")
    table_full_results(df, out)
    table_per_class_f1(df, out)
    table_multitask_results(df, out)
    table_mt_ablation(df, out)
    table_significance_test(df, out)
    table_prep_significance(df, out)
    table_dataset_stats(out)
    table_model_efficiency(df, out)

    print(f"\n완료. 결과 위치: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="experiments/analysis")
    args = parser.parse_args()
    main(args.output_dir)
