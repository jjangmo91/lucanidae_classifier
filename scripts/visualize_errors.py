"""
val 데이터셋에서 오분류 이미지를 experiments/error_analysis/ 에 저장한다.

사용법:
    python visualize_errors.py                        # default.yaml 설정 사용
    python visualize_errors.py --data_dir data/final/val
"""

import argparse
import shutil
import yaml
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm

from src.ml.classifier import build_model
from src.training.dataset import EVAL_TRANSFORM


def main(data_dir: str | None = None, weights_path: str | None = None):
    with open("configs/default.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["training"]

    device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights_path = weights_path or cfg["weights_path"]
    architecture = cfg["architecture"]

    if data_dir is None:
        # data_dir이 final_* 형태면 val 서브디렉토리 사용
        base = Path(cfg["data_dir"])
        data_dir = str(base / "val") if (base / "val").exists() else str(base)

    output_dir = Path("experiments/error_analysis")

    checkpoint  = torch.load(weights_path, map_location=device, weights_only=False)
    class_names = checkpoint["class_names"]
    num_classes = checkpoint["num_classes"]
    arch        = checkpoint.get("architecture", architecture)
    mode        = checkpoint.get("mode", "species_only")

    model = build_model(num_classes=num_classes, architecture=arch, mode=mode)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    dataset    = datasets.ImageFolder(data_dir, EVAL_TRANSFORM)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    correct, total = 0, 0

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(tqdm(dataloader, desc="Error Analysis")):
            inputs, labels = inputs.to(device), labels.to(device)
            out    = model(inputs)
            sp_out = out[0] if isinstance(out, tuple) else out
            _, preds = torch.max(sp_out, 1)

            total += 1
            if preds == labels:
                correct += 1
            else:
                img_path, _ = dataset.samples[i]
                true_label  = class_names[labels.item()]
                pred_label  = class_names[preds.item()]

                target_folder = output_dir / f"{true_label}__pred_{pred_label}"
                target_folder.mkdir(parents=True, exist_ok=True)
                shutil.copy(img_path, target_folder / Path(img_path).name)

    acc = correct / total * 100
    print(f"\nVal Accuracy: {correct}/{total} = {acc:.2f}%")
    print(f"오류 이미지 저장 위치: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=None,
                        help="val 데이터 경로 (기본값: default.yaml data_dir/val)")
    parser.add_argument("--weights", type=str, default=None,
                        help="모델 가중치 경로 (기본값: default.yaml weights_path)")
    args = parser.parse_args()
    main(data_dir=args.data_dir, weights_path=args.weights)
