# Lucanidae Project: Fine-Grained Classifier

**사슴벌레 매니아들을 위한 AI 동정(Identification) 파이프라인입니다.**

사슴벌레(Lucanidae)는 큰턱(Mandible)의 내치 형태, 등갑의 미세한 점각, 그리고 특유의 광택 등 아주 미세한 차이로 종이 갈리는 까다로운 분류군입니다. 이 프로젝트는 단순한 객체 인식을 넘어, 인간의 눈으로도 헷갈리기 쉬운 형태적 차이(Fine-Grained Features)를 딥러닝 엔진이 얼마나 정교하게 잡아낼 수 있는지 검증하기 위해 시작되었습니다.

현재는 정돈된 사진에서 종의 특징을 추출해 내는 **'핵심 분류기(Stage 2)'**가 완성된 상태이며, 추후 야생 환경의 노이즈 속에서도 사슴벌레만 정확히 크롭해 내는 탐지기(YOLO 기반 Stage 1)와 결합하여 완전한 자동화 생태 분석 툴로 확장할 계획입니다.

---

## 1. Architecture Overview

This repository contains the core fine-grained classification module (Stage 2) designed to integrate with a future Object Detection module (Stage 1).

* **Core Engine**: `ConvNeXt-Tiny` (Pre-trained on ImageNet)
  * Chosen for its modern CNN architecture, which excels at capturing localized textures and minor morphological details essential for insect taxonomy.
* **Optimization Strategies**:
  * **Label Smoothing**: Applied to mitigate overconfidence and counter the inherent label noise found in crowdsourced ecological data.
  * **Class Weighting**: Dynamic weight injection to handle the imbalanced, long-tailed distribution of rare species.
* **Input Resolution**: Scaled up to `448x448` to preserve critical identification keys (e.g., mandibular teeth, pronotal shapes).

## 2. Directory Structure

```text
lucanidae_classifier/
├── data/               
│   ├── final/          # Cleaned & split dataset (Train/Val/Test)
│   └── test_images/    # Target images for inference testing
├── src/
│   ├── data_collection/# API Scrapers for raw image acquisition
│   ├── preprocessing/  # Automated taxonomic cleaners and splitters
│   └── models/         # Neural network architectures
├── main.py             # Entry point: Smart Data Pipeline
├── train.py            # Entry point: Model Training & WandB Logging
└── predict.py          # Entry point: Top-K Inference Pipeline

## 3. Quick Start (Inference)

Test the classifier's capabilities on your own stag beetle photos.

1. Place your target images (`.jpg`, `.png`) inside the `data/test_images/` directory.
2. Run the prediction script:
```bash
python predict.py
```
3. The script will output the Top-3 predicted species along with their confidence scores, formatted in a clear CLI visual layout.

## 4. Future Roadmap
- [ ] Stage 1 Integration: Implement YOLOv8 for automated background cropping in the wild.
- [ ] Data Augmentation: Introduce CutMix/Mixup for better boundary decision learning..