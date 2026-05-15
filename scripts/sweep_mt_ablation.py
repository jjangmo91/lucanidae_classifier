"""
Multi-Task 구성요소 ablation sweep.

SO vs sex_only vs form_only vs MT(full) 비교.
SO/MT는 메인 sweep에 이미 존재하므로 SX/FO만 추가 실행.

기준 (arch, prep) 선정 방식:
  1. MLflow에서 메인 sweep의 SO 결과 중 val_acc 최고 (arch, prep) 조합 자동 선택
  2. 또는 --arch / --prep 로 수동 지정

사용법:
    python scripts/sweep_mt_ablation.py                              # 자동 선정
    python scripts/sweep_mt_ablation.py --arch convnext_tiny --prep full  # 수동 지정
    python scripts/sweep_mt_ablation.py --dry_run
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path
import mlflow

ALL_MODES = ["sex_only", "form_only"]  # SO/MT는 메인 sweep에서 커버됨
SEEDS   = [42, 123, 456]              # 메인 sweep과 동일한 시드
DATA    = "./data/final"
LR      = 0.0001

MODE_SHORT = {"sex_only": "SX", "form_only": "FO"}

ARCH_ORDER = ["convnext_tiny", "efficientnet_b3", "swin_tiny", "vit_small", "dinov2_vits14"]
PREP_ORDER = ["full", "seg_soft", "seg_hard", "bbox"]


def get_completed(mlflow_uri: str) -> set[str]:
    mlflow.set_tracking_uri(mlflow_uri)
    client = mlflow.tracking.MlflowClient()
    exp    = client.get_experiment_by_name("lucanidae_classifier")
    if exp is None:
        return set()
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        max_results=2000,
    )
    return {r.info.run_name for r in runs}


def find_best_arch_prep(mlflow_uri: str) -> tuple[str, str] | None:
    """
    메인 sweep SO 결과에서 val_acc 기준 베스트 (arch, prep) 반환.
    run_name 패턴: convnext_full_SO_s42
    """
    mlflow.set_tracking_uri(mlflow_uri)
    client = mlflow.tracking.MlflowClient()
    exp    = client.get_experiment_by_name("lucanidae_classifier")
    if exp is None:
        return None

    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        max_results=2000,
    )

    arch_map = {
        "convnext": "convnext_tiny", "efficientnet": "efficientnet_b3",
        "swin": "swin_tiny", "vit": "vit_small", "dinov2": "dinov2_vits14",
    }
    pattern = re.compile(
        r"^(convnext|efficientnet|swin|vit|dinov2)_(full|bbox|seg_hard|seg_soft)_SO_s\d+$"
    )

    # (arch, prep) → [val_acc, ...]
    from collections import defaultdict
    scores: dict[tuple, list] = defaultdict(list)
    for run in runs:
        m = pattern.match(run.info.run_name)
        if not m:
            continue
        arch = arch_map[m.group(1)]
        prep = m.group(2)
        acc  = run.data.metrics.get("best_val_acc", run.data.metrics.get("val_acc"))
        if acc is not None:
            scores[(arch, prep)].append(acc)

    if not scores:
        return None

    best = max(scores, key=lambda k: sum(scores[k]) / len(scores[k]))
    mean_acc = sum(scores[best]) / len(scores[best])
    print(f"  베스트 조합: arch={best[0]}, prep={best[1]}  (SO val_acc 평균 {mean_acc*100:.1f}%)")
    return best


def build_experiments(arch: str, prep: str, modes: list[str]) -> list[dict]:
    short_arch = arch.split("_")[0]  # convnext_tiny → convnext
    exps = []
    for mode in modes:
        short_mode = MODE_SHORT[mode]
        for seed in SEEDS:
            name = f"{short_arch}_{prep}_{short_mode}_s{seed}"
            exps.append({
                "run_name":     name,
                "mode":         mode,
                "seed":         seed,
                "arch":         arch,
                "weights_path": f"./models/weights/mt_ablation/{name}.pth",
            })
    return exps


def run_experiment(exp: dict, prep: str) -> bool:
    cmd = [
        sys.executable, "train.py",
        "--architecture",  exp["arch"],
        "--mode",          exp["mode"],
        "--data_dir",      DATA,
        "--preprocessing", prep,
        "--learning_rate", str(LR),
        "--run_name",      exp["run_name"],
        "--weights_path",  exp["weights_path"],
        "--seed",          str(exp["seed"]),
    ]
    return subprocess.run(cmd).returncode == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch",       default=None, help="고정 아키텍처 (미지정 시 MLflow에서 자동 선택)")
    parser.add_argument("--prep",       default=None, help="고정 전처리 (미지정 시 MLflow에서 자동 선택)")
    parser.add_argument("--modes",      default=None, nargs="+", choices=["sex_only", "form_only"],
                        help="실행할 모드 (기본값: sex_only form_only)")
    parser.add_argument("--dry_run",    action="store_true")
    parser.add_argument("--mlflow_uri", default="http://localhost:5001")
    args = parser.parse_args()

    modes = args.modes if args.modes else ALL_MODES

    # (arch, prep) 결정
    if args.arch and args.prep:
        arch, prep = args.arch, args.prep
        print(f"수동 지정: arch={arch}, prep={prep}")
    else:
        print("MLflow에서 베스트 (arch, prep) 탐색 중...")
        result = find_best_arch_prep(args.mlflow_uri)
        if result is None:
            print("  메인 sweep 결과 없음. --arch / --prep 를 수동 지정하세요.")
            return
        arch, prep = result

    completed = get_completed(args.mlflow_uri)
    exps      = build_experiments(arch, prep, modes)
    pending   = [e for e in exps if e["run_name"] not in completed]

    print(f"\nMT ablation: 전체 {len(exps)}개 | 완료: {len(exps)-len(pending)}개 | 예정: {len(pending)}개")
    print(f"기준 조합: arch={arch}, prep={prep}  |  모드: {modes}  |  시드: {SEEDS}\n")
    for e in pending:
        print(f"  [pending] {e['run_name']}  (mode={e['mode']}, seed={e['seed']})")

    if args.dry_run or not pending:
        return

    Path("models/weights/mt_ablation").mkdir(parents=True, exist_ok=True)

    failed = []
    for i, exp in enumerate(pending, 1):
        print(f"\n[{i}/{len(pending)}] {exp['run_name']}")
        if not run_experiment(exp, prep):
            failed.append(exp["run_name"])

    print(f"\n완료: {len(pending)-len(failed)}개 / 실패: {len(failed)}개")
    if failed:
        for name in failed:
            print(f"  [FAIL] {name}")


if __name__ == "__main__":
    main()
