import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from pathlib import Path

def get_dataloaders(data_dir: str, batch_size: int = 16):
    base_path = Path(data_dir)
    
    # Train에는 변형을 많이, Val에는 정석적인 변환만 적용
    data_transforms = {
        'train': transforms.Compose([
            # 무작위 크롭 및 크기 조정 (사슴벌레가 사진 어디에 있든 인식하게 함)
            transforms.RandomResizedCrop(448, scale=(0.8, 1.0)), 
            # 좌우 반전 (곤충의 대칭성 활용)
            transforms.RandomHorizontalFlip(p=0.5),
            # 무작위 회전 (다양한 각도 대응)
            transforms.RandomRotation(degrees=20),
            # 밝기, 대비, 채도 조절 (다양한 촬영 환경 대응)
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            # ImageNet 데이터셋의 평균과 표준편차로 정규화 (Pre-trained 모델 사용 시 필수)
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(512),
            transforms.CenterCrop(448),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {
        x: datasets.ImageFolder(base_path / x, data_transforms[x])
        for x in ['train', 'val']
    }

    # num_workers는 데이터 로딩 속도를 높여 GPU 대기 시간을 줄입니다.
    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4)
        for x in ['train', 'val']
    }
    
    class_names = image_datasets['train'].classes
    return dataloaders, class_names