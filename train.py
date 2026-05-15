import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import yaml
import mlflow
import argparse
import random
import numpy as np
from dotenv import load_dotenv
from sklearn.utils.class_weight import compute_class_weight

from src.training.dataset import get_dataloaders
from src.ml.classifier import build_model
from src.training.trainer import ModelTrainer

load_dotenv()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture",  type=str,   default=None)
    parser.add_argument("--mode",          type=str,   default=None)
    parser.add_argument("--data_dir",      type=str,   default=None)
    parser.add_argument("--preprocessing", type=str,   default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--run_name",      type=str,   default=None)
    parser.add_argument("--weights_path",  type=str,   default=None)
    parser.add_argument("--seed",          type=int,   default=42)
    parser.add_argument("--lambda_sex",    type=float, default=None)
    parser.add_argument("--lambda_form",   type=float, default=None)
    args = parser.parse_args()

    with open("configs/default.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    cfg = config["training"]

    # CLI args override yaml
    if args.architecture:  cfg["architecture"]  = args.architecture
    if args.mode:          cfg["mode"]          = args.mode
    if args.data_dir:      cfg["data_dir"]      = args.data_dir
    if args.preprocessing: cfg["preprocessing"] = args.preprocessing
    if args.learning_rate: cfg["learning_rate"] = args.learning_rate
    if args.run_name:      cfg["run_name"]      = args.run_name
    if args.weights_path:  cfg["weights_path"]  = args.weights_path
    if args.lambda_sex  is not None: cfg["lambda_sex"]  = args.lambda_sex
    if args.lambda_form is not None: cfg["lambda_form"] = args.lambda_form

    set_seed(args.seed)

    device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    training_mode = cfg.get("mode", "species_only")

    dataloaders, class_names = get_dataloaders(
        cfg["data_dir"], batch_size=cfg["batch_size"], mode=training_mode
    )
    num_classes = len(class_names)

    train_labels    = dataloaders["train"].dataset.targets
    unique_classes  = np.unique(train_labels)
    weights_present = compute_class_weight(
        class_weight="balanced",
        classes=unique_classes,
        y=train_labels,
    )
    # 훈련셋에 없는 클래스가 있으면 weight=1 (중립) 부여해 shape mismatch 방지
    class_weights = np.ones(num_classes, dtype=np.float64)
    for cls_idx, w in zip(unique_classes, weights_present):
        class_weights[cls_idx] = w
    weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)

    mlflow.set_tracking_uri("http://localhost:5001")
    mlflow.set_experiment("lucanidae_classifier")

    with mlflow.start_run(run_name=cfg["run_name"]):
        lambda_sex  = cfg.get("lambda_sex",  0.3)
        lambda_form = cfg.get("lambda_form", 0.2)

        mlflow.log_params({
            "architecture":  cfg["architecture"],
            "learning_rate": cfg["learning_rate"],
            "batch_size":    cfg["batch_size"],
            "num_epochs":    cfg["num_epochs"],
            "num_classes":   num_classes,
            "preprocessing": cfg.get("preprocessing", "full"),
            "training_mode": training_mode,
            "seed":          args.seed,
            "lambda_sex":    lambda_sex,
            "lambda_form":   lambda_form,
        })

        model = build_model(num_classes=num_classes, architecture=cfg["architecture"], mode=training_mode)
        model = model.to(device)

        # 모델 크기 및 추론 속도
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        mlflow.log_param("num_params_M", round(num_params / 1e6, 2))

        model.eval()
        dummy = torch.randn(1, 3, 448, 448).to(device)
        with torch.no_grad():
            for _ in range(3):   # warmup
                _ = model(dummy)
            t0 = time.perf_counter()
            for _ in range(50):
                _ = model(dummy)
        inference_ms = (time.perf_counter() - t0) / 50 * 1000
        mlflow.log_param("inference_ms", round(inference_ms, 2))

        criterion = nn.CrossEntropyLoss(weight=weights_tensor, label_smoothing=0.1)
        optimizer = optim.Adam(model.parameters(), lr=cfg["learning_rate"])
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=3)

        trainer = ModelTrainer(
            model        = model,
            dataloaders  = dataloaders,
            criterion    = criterion,
            optimizer    = optimizer,
            device       = device,
            class_names  = class_names,
            weights_path = cfg["weights_path"],
            architecture = cfg["architecture"],
            mode         = training_mode,
            scheduler    = scheduler,
            patience     = cfg["patience"],
            lambda_sex   = lambda_sex,
            lambda_form  = lambda_form,
        )

        trainer.fit(num_epochs=cfg["num_epochs"])

        mlflow.log_metric("best_val_acc", trainer.best_val_acc)
        mlflow.log_artifact(cfg["weights_path"])


if __name__ == "__main__":
    main()
