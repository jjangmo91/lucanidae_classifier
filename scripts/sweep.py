"""
Full factorial sweep: 5 architectures × 4 preprocessing × 2 modes × 3 seeds = 120 experiments.

완료된 run은 MLflow에서 확인해 자동 스킵한다.

사용법:
    python scripts/sweep.py                        # 전체 실행
    python scripts/sweep.py --seeds 42             # seed 42만
    python scripts/sweep.py --archs convnext_tiny  # 특정 아키텍처만
    python scripts/sweep.py --dry_run             # 실행 목록만 출력
"""

import argparse
import subprocess
import sys
from pathlib import Path
import mlflow

# ── 실험 설정 ──────────────────────────────────────────────────────────────
ARCHITECTURES = [
    "convnext_tiny",
    "efficientnet_b3",
    "swin_tiny",
    "vit_small",
    "dinov2_vits14",
]

PREPROCESSING = {
    "full":     "./data/final",
    "bbox":     "./data/final_bbox",
    "seg_hard": "./data/final_seg_hard",
    "seg_soft": "./data/final_seg_soft",
}

MODES = ["species_only", "multi_task"]

SEEDS = [42, 123, 456]

# DINOv2는 낮은 LR 필요
LR_MAP = {
    "dinov2_vits14": 0.00001,
}
LR_DEFAULT = 0.0001

ARCH_SHORT = {
    "convnext_tiny":   "convnext",
    "efficientnet_b3": "efficientnet",
    "swin_tiny":       "swin",
    "vit_small":       "vit",
    "dinov2_vits14":   "dinov2",
}
# ───────────────────────────────────────────────────────────────────────────


def get_completed_runs(tracking_uri: str, experiment_name: str) -> set:
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()
    exp    = client.get_experiment_by_name(experiment_name)
    if exp is None:
        return set()
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        max_results=1000,
    )
    return {r.info.run_name for r in runs}


def make_run_name(arch: str, prep: str, mode: str, seed: int) -> str:
    mode_short = "SO" if mode == "species_only" else "MT"
    return f"{ARCH_SHORT[arch]}_{prep}_{mode_short}_s{seed}"


def make_weights_path(arch: str, prep: str, mode: str, seed: int) -> str:
    return f"./models/weights/sweep/{make_run_name(arch, prep, mode, seed)}.pth"


def build_experiments(archs, preps, modes, seeds):
    exps = []
    for arch in archs:
        for prep in preps:
            data_dir = PREPROCESSING[prep]
            if not Path(data_dir.lstrip("./")).exists():
                continue  # 전처리 데이터 없으면 스킵
            for mode in modes:
                for seed in seeds:
                    exps.append({
                        "arch":         arch,
                        "prep":         prep,
                        "mode":         mode,
                        "seed":         seed,
                        "data_dir":     data_dir,
                        "lr":           LR_MAP.get(arch, LR_DEFAULT),
                        "run_name":     make_run_name(arch, prep, mode, seed),
                        "weights_path": make_weights_path(arch, prep, mode, seed),
                    })
    return exps


def run_experiment(exp: dict):
    cmd = [
        sys.executable, "train.py",
        "--architecture",  exp["arch"],
        "--mode",          exp["mode"],
        "--data_dir",      exp["data_dir"],
        "--preprocessing", exp["prep"],
        "--learning_rate", str(exp["lr"]),
        "--run_name",      exp["run_name"],
        "--weights_path",  exp["weights_path"],
        "--seed",          str(exp["seed"]),
    ]
    print(f"\n{'='*60}")
    print(f"  실행: {exp['run_name']}")
    print(f"{'='*60}")
    result = subprocess.run(cmd)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archs",   nargs="+", default=ARCHITECTURES)
    parser.add_argument("--preps",   nargs="+", default=list(PREPROCESSING.keys()))
    parser.add_argument("--modes",   nargs="+", default=MODES)
    parser.add_argument("--seeds",   nargs="+", type=int, default=SEEDS)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--mlflow_uri", default="http://localhost:5001")
    parser.add_argument("--experiment", default="lucanidae_classifier")
    args = parser.parse_args()

    experiments   = build_experiments(args.archs, args.preps, args.modes, args.seeds)
    completed     = get_completed_runs(args.mlflow_uri, args.experiment)

    pending   = [e for e in experiments if e["run_name"] not in completed]
    skipped   = [e for e in experiments if e["run_name"] in completed]

    print(f"전체: {len(experiments)}개 | 완료(스킵): {len(skipped)}개 | 실행 예정: {len(pending)}개\n")

    if skipped:
        print("[ 스킵 ]")
        for e in skipped:
            print(f"  [done] {e['run_name']}")
        print()

    print("[ 실행 예정 ]")
    for e in pending:
        print(f"  [pending] {e['run_name']}")

    if args.dry_run or not pending:
        return

    print("\n실행 시작...\n")
    failed = []
    for i, exp in enumerate(pending, 1):
        print(f"[{i}/{len(pending)}] {exp['run_name']}")
        success = run_experiment(exp)
        if not success:
            failed.append(exp["run_name"])
            print(f"  [FAIL] {exp['run_name']}")

    print(f"\n{'='*60}")
    print(f"완료: {len(pending) - len(failed)}개 / 실패: {len(failed)}개")
    if failed:
        print("실패 목록:")
        for name in failed:
            print(f"  - {name}")


if __name__ == "__main__":
    main()
