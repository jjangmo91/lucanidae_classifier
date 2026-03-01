import torch
import os
import shutil
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from src.models.classifier import build_model
from tqdm import tqdm

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = "data/final/val" # 검증 데이터 대상
    output_dir = "experiments/error_analysis" # 틀린 사진 모을 곳
    weights_path = "models/weights/best_model.pth"
    
    # 데이터 변환(학습과 동일)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = datasets.ImageFolder(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    class_names = dataset.classes

    # 모델 로드
    model = build_model(num_classes=len(class_names))
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print("Error Analysis 시작...")
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(tqdm(dataloader)):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            if preds != labels:
                # 틀린 사진 복사
                img_path, _ = dataset.samples[i]
                true_label = class_names[labels]
                pred_label = class_names[preds]
                
                # 폴더 구조: error_analysis/정답_예측값/파일명
                target_folder = os.path.join(output_dir, f"{true_label}_was_{pred_label}")
                os.makedirs(target_folder, exist_ok=True)
                shutil.copy(img_path, os.path.join(target_folder, os.path.basename(img_path)))

    print(f"분석 완료! {output_dir} 폴더를 확인하세요.")

if __name__ == "__main__":
    main()