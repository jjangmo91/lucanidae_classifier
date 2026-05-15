"""
λ_sex / λ_form 민감도 분석 sweep.
main sweep 완료 후 실행. convnext_tiny / full / multi_task / seed=42 고정.

사용법:
    python scripts/sweep_lambda.py
    python scripts/sweep_lambda.py --dry_run
"""

import argparse
import subprocess
import sys
from pathlib import Path
import mlflow

# λ_sex 변화 (λ_form=0.2 고정)
LAMBDA_SEX_VALUES  = [0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
# λ_form 변화 (λ_sex=0.3 고정)
LAMBDA_FORM_VALUES = [0.05, 0.1, 0.2, 0.3, 0.5, 1.0]

FIXED_ARCH    = "convnext_tiny"
FIXED_DATA    = "./data/final"
FIXED_PREP    = "full"
FIXED_SEED    = 42
FIXED_LR      = 0.0001
EXPERIMENT    = "lucanidae_lambda_ablation"


def get_completed(tracking_uri, experiment_name):
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()
    exp    = client.get_experiment_by_name(experiment_name)
    if exp is None:
        return set()
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        max_results=500,
    )
    return {r.info.run_name for r in runs}


def build_experiments():
    exps = []
    # λ_sex 변화
    for ls in LAMBDA_SEX_VALUES:
        name = f"convnext_full_MT_lsex{ls}_lform0.2_s{FIXED_SEED}"
        exps.append({
            "run_name":    name,
            "lambda_sex":  ls,
            "lambda_form": 0.2,
            "weights_path": f"./models/weights/lambda_ablation/{name}.pth",
        })
    # λ_form 변화 (λ_sex=0.3 default 제외, 이미 위에서 포함)
    for lf in LAMBDA_FORM_VALUES:
        if lf == 0.2:
            continue  # 이미 위에서 포함
        name = f"convnext_full_MT_lsex0.3_lform{lf}_s{FIXED_SEED}"
        exps.append({
            "run_name":    name,
            "lambda_sex":  0.3,
            "lambda_form": lf,
            "weights_path": f"./models/weights/lambda_ablation/{name}.pth",
        })
    return exps


def run_experiment(exp, mlflow_uri, experiment_name):
    cmd = [
        sys.executable, "train.py",
        "--architecture",  FIXED_ARCH,
        "--mode",          "multi_task",
        "--data_dir",      FIXED_DATA,
        "--preprocessing", FIXED_PREP,
        "--learning_rate", str(FIXED_LR),
        "--run_name",      exp["run_name"],
        "--weights_path",  exp["weights_path"],
        "--seed",          str(FIXED_SEED),
        "--lambda_sex",    str(exp["lambda_sex"]),
        "--lambda_form",   str(exp["lambda_form"]),
    ]
    # 별도 MLflow experiment로 기록
    import os
    env = {**__import__("os").environ,
           "MLFLOW_EXPERIMENT_NAME": experiment_name}

    # train.py가 experiment 이름을 고정하므로 여기선 run_name으로 구분
    result = subprocess.run(cmd)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry_run",     action="store_true")
    parser.add_argument("--mlflow_uri",  default="http://localhost:5001")
    args = parser.parse_args()

    # λ ablation은 main experiment에 기록 (분리하면 분석이 복잡해짐)
    completed = get_completed(args.mlflow_uri, "lucanidae_classifier")
    exps      = build_experiments()
    pending   = [e for e in exps if e["run_name"] not in completed]

    print(f"λ ablation: 전체 {len(exps)}개 | 완료: {len(exps)-len(pending)}개 | 예정: {len(pending)}개\n")
    for e in pending:
        print(f"  [pending] {e['run_name']}  (λ_sex={e['lambda_sex']}, λ_form={e['lambda_form']})")

    if args.dry_run or not pending:
        return

    Path("models/weights/lambda_ablation").mkdir(parents=True, exist_ok=True)

    failed = []
    for i, exp in enumerate(pending, 1):
        print(f"\n[{i}/{len(pending)}] {exp['run_name']}")
        if not run_experiment(exp, args.mlflow_uri, "lucanidae_classifier"):
            failed.append(exp["run_name"])

    print(f"\n완료: {len(pending)-len(failed)}개 / 실패: {len(failed)}개")


if __name__ == "__main__":
    main()
