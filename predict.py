import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from pathlib import Path
from src.models.classifier import build_model

class LucanidaePredictor:
    def __init__(self, model_path, data_dir):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names = sorted(os.listdir(Path(data_dir) / "train"))
        self.num_classes = len(self.class_names)
        
        # 모델 빌드 및 가중치 로드
        self.model = build_model(num_classes=self.num_classes)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        # 전처리 설정 (448px Resolution)
        self.transform = transforms.Compose([
            transforms.Resize(512),
            transforms.CenterCrop(448),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    @torch.no_grad()
    def inference(self, image_path):
        image = Image.open(image_path).convert('RGB')
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        outputs = self.model(input_tensor)
        probabilities = F.softmax(outputs, dim=1)[0]
        
        # 상위 3개 결과 추출
        top_probs, top_indices = torch.topk(probabilities, k=min(3, self.num_classes))
        
        return top_probs.cpu().numpy(), top_indices.cpu().numpy()

def main():
    # 경로 설정
    test_dir = Path("data/test_images")
    model_path = "models/weights/best_model.pth"
    data_dir = "data/final"

    # 테스트 폴더 생성
    if not test_dir.exists():
        test_dir.mkdir(parents=True)
        print(f"Directory Created: {test_dir}")
        print("Please place your test images in this folder and run again.")
        return

    predictor = LucanidaePredictor(model_path, data_dir)
    image_extensions = ('.jpg', '.jpeg', '.png', '.JPG', '.PNG')
    image_files = [f for f in test_dir.iterdir() if f.suffix in image_extensions]

    if not image_files:
        print(f"No image files found in {test_dir}")
        return

    for img_path in image_files:
        probs, indices = predictor.inference(img_path)
        
        # 결과 출력 (네모 박스 형태)
        print("-" * 60)
        print(f" Target Image: {img_path.name}")
        print("-" * 60)
        for i in range(len(indices)):
            label = predictor.class_names[indices[i]]
            score = probs[i] * 100
            bar = "#" * int(score / 5)
            marker = "[BEST]" if i == 0 else "      "
            print(f" {marker} {label:30} | {score:6.2f}% | {bar}")
        print("-" * 60 + "\n")

if __name__ == "__main__":
    main()