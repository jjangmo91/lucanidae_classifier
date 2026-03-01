import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import os
import wandb
from dotenv import load_dotenv
from pathlib import Path

from src.training.dataset import get_dataloaders
from src.models.classifier import build_model
from src.training.trainer import ModelTrainer

from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# .env 파일 보안 로드
load_dotenv()

def main():
    # 실험 환경 및 하이퍼파라미터 설정
    data_dir = "data/final"
    batch_size = 16        # 해상도를 높일 예정이므로 메모리 확보를 위해 16으로 조정
    num_epochs = 100 
    learning_rate = 1e-4 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 데이터 로더 초기화
    dataloaders, class_names = get_dataloaders(data_dir, batch_size=batch_size)
    num_classes = len(class_names)

    # 클래스 가중치 계산 로직
    # 학습 데이터셋의 라벨(targets)을 추출하여 불균형 정도를 계산합니다.
    train_labels = dataloaders['train'].dataset.targets
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_labels),
        y=train_labels
    )
    # 계산된 가중치를 텐서화하여 GPU(device)로 보냅니다.
    weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)

    # WandB 초기화
    wandb.init(
        project="lucanidae_classifier",
        name="convnext-tiny",
        config={
            "architecture": "ResNet-50",
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "num_classes": num_classes,
            "class_weights": "balanced"  # 설정 기록
        }
    )

    # 모델 및 손실함수 설정
    model = build_model(num_classes=num_classes)
    
    # 손실 함수에 weight 파라미터를 추가하여 데이터가 적은 종에 가산점을 줍니다.
    criterion = nn.CrossEntropyLoss(weight=weights_tensor, label_smoothing=0.1)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 스케줄러 설정
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

    # 트레이너 실행
    trainer = ModelTrainer(
        model=model,
        dataloaders=dataloaders,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler,
        patience=10
    )
    
    trainer.fit(num_epochs=num_epochs)
    wandb.finish()

if __name__ == "__main__":
    main()