import torch
import torch.nn.functional as F
from PIL import Image
from pathlib import Path

from src.ml.classifier import build_model
from src.training.dataset import EVAL_TRANSFORM, SEX_CLASSES, MALE_FORM_CLASSES


class LucanidaePredictor:
    def __init__(self, weights_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        checkpoint       = torch.load(weights_path, map_location=self.device, weights_only=False)
        self.class_names = checkpoint["class_names"]
        num_classes      = checkpoint["num_classes"]
        architecture     = checkpoint.get("architecture", "convnext_tiny")
        self.mode        = checkpoint.get("mode", "species_only")

        self.model = build_model(num_classes=num_classes, architecture=architecture, mode=self.mode)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def inference(self, image_path: str) -> dict:
        image  = Image.open(image_path).convert("RGB")
        tensor = EVAL_TRANSFORM(image).unsqueeze(0).to(self.device)

        out = self.model(tensor)

        if self.mode == "multi_task":
            sp_out, sex_out, form_out = out
        elif self.mode == "sex_only":
            sp_out, sex_out = out
            form_out = None
        elif self.mode == "form_only":
            sp_out, form_out = out
            sex_out = None
        else:
            sp_out = out
            sex_out = form_out = None

        probs = F.softmax(sp_out, dim=1)[0]
        top_probs, top_indices = torch.topk(probs, k=min(3, len(self.class_names)))

        result = {
            "top3": [
                {"species": self.class_names[idx], "confidence": float(prob)}
                for prob, idx in zip(top_probs.cpu(), top_indices.cpu())
            ],
            "species":    self.class_names[top_indices[0].item()],
            "confidence": float(top_probs[0]),
        }

        if sex_out is not None:
            sex_idx = sex_out[0].argmax().item()   # 0=male, 1=female
            result["sex"] = SEX_CLASSES[sex_idx]

        if form_out is not None:
            form_idx = form_out[0].argmax().item()  # 0=major, 1=minor, 2=intermediate
            is_male  = result.get("sex") == "male"
            result["male_form"] = MALE_FORM_CLASSES[form_idx] if is_male else "unknown"

        return result


def main():
    weights_path = "models/weights/best_model.pth"
    test_dir     = Path("data/test_images")

    if not test_dir.exists():
        test_dir.mkdir(parents=True)
        print(f"폴더 생성: {test_dir}")
        print("이미지를 넣고 다시 실행하세요.")
        return

    predictor   = LucanidaePredictor(weights_path)
    image_files = [f for f in test_dir.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"}]

    if not image_files:
        print(f"{test_dir}에 이미지가 없습니다.")
        return

    for img_path in image_files:
        result = predictor.inference(str(img_path))
        print("-" * 60)
        print(f" Target Image: {img_path.name}")
        print("-" * 60)
        for i, item in enumerate(result["top3"]):
            score  = item["confidence"] * 100
            bar    = "#" * int(score / 5)
            marker = "[BEST]" if i == 0 else "      "
            print(f" {marker} {item['species']:45s} | {score:6.2f}% | {bar}")
        if "sex" in result:
            print(f" 성별: {result['sex']}  형태: {result['male_form']}")
        print("-" * 60 + "\n")


if __name__ == "__main__":
    main()
