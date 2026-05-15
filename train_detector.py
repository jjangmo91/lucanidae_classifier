"""
YOLOv8n-seg Detector 학습 스크립트

data/detector_train/dataset.yaml 을 사용해
class-agnostic 사슴벌레 인스턴스 세그먼테이션 모델을 학습한다.

사전 조건:
    python -m src.detection.grounded_sam_annotate
    python -m src.detection.annotation_utils

사용법:
    python train_detector.py
"""

import shutil
import yaml
import mlflow
from pathlib import Path


def main():
    with open("configs/default.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    cfg          = config["detection"]
    detector_dir = Path(cfg["detector_train"])
    dataset_yaml = detector_dir / "dataset.yaml"
    weights_path = cfg["weights_path"]
    architecture = cfg.get("architecture", "yolov8n-seg")
    num_epochs   = cfg.get("num_epochs", 100)

    if not dataset_yaml.exists():
        raise FileNotFoundError(
            f"dataset.yaml 이 없습니다: {dataset_yaml}\n"
            "먼저 python -m src.detection.annotation_utils 를 실행하세요."
        )

    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise ImportError("pip install ultralytics") from e

    Path(weights_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"[Detector 학습]")
    print(f"  아키텍처: {architecture}  |  Epochs: {num_epochs}")
    print(f"  Dataset:  {dataset_yaml}")
    print()

    mlflow.set_tracking_uri("http://localhost:5001")
    mlflow.set_experiment("lucanidae_detector")

    with mlflow.start_run(run_name="yolov8n-seg_detector_v1"):
        mlflow.log_params({
            "architecture": architecture,
            "num_epochs":   num_epochs,
            "imgsz":        640,
            "dataset":      str(dataset_yaml),
        })

        model   = YOLO(f"models/weights/{architecture}.pt")
        results = model.train(
            data      = str(dataset_yaml),
            epochs    = num_epochs,
            imgsz     = 640,
            batch     = -1,
            patience  = 20,
            project   = "lucanidae-detector",
            name      = "yolov8n-seg_detector_v1",
            exist_ok  = True,
            device    = 0,
        )

        best_pt = Path(results.save_dir) / "weights" / "best.pt"
        if best_pt.exists():
            shutil.copy2(best_pt, weights_path)
            mlflow.log_artifact(str(best_pt))
            print(f"\nBest model 저장: {weights_path}")
        else:
            print(f"\n[경고] best.pt 를 찾을 수 없습니다: {best_pt}")

        # YOLO 결과 CSV에서 best metrics 로깅
        results_csv = Path(results.save_dir) / "results.csv"
        if results_csv.exists():
            import pandas as pd
            df = pd.read_csv(results_csv)
            df.columns = df.columns.str.strip()
            last = df.iloc[-1]
            mlflow.log_metrics({
                "best_box_map50":  float(last.get("metrics/mAP50(B)", 0)),
                "best_mask_map50": float(last.get("metrics/mAP50(M)", 0)),
            })

    print("\n다음 단계: python -m src.detection.generate_segmented")


if __name__ == "__main__":
    main()
