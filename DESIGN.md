# 한국 사슴벌레 AI 도감 — 설계 문서 v6

> 최종 수정: 2026-05-10
> 기준 코드베이스: `beetledex` / GitHub: `jjangmo91/lucanidae_classifier` (main 브랜치)

---

## 1. 프로젝트 개요

### 1.1 목표

한국 사슴벌레과(Lucanidae) 16종을 대상으로:

1. **종 분류** — 16종 fine-grained classification
2. **성별·형태 분류** — 성적 이형성 및 수컷 크기 다형성(male form) 인식
3. **미지·외래종 처리** — 한국 미서식 종 및 비사슴벌레 입력에 대한 강건한 처리
4. **커뮤니티 서비스** — 일반 사용자가 사진 한 장으로 종을 동정하는 웹 서비스
5. **연구 논문** — 위 문제를 해결하는 파이프라인을 학술적으로 검증

### 1.2 연구 노벨티

#### N1. 성적 이형성 인식 (Sexual Dimorphism-Aware Classification)
사슴벌레 수컷과 암컷은 같은 종이어도 외형이 크게 다르다.
기존 분류기는 이를 고려하지 않아 암컷 오분류율이 높다.
→ **종 분류와 성별 분류를 동시에 학습하는 Multi-task 구조** 제안.
→ SO vs SX vs FO vs MT ablation으로 각 auxiliary head 기여도 정량 검증.

#### N2. 수컷 크기 다형성 (Male Form Classification)
수컷은 크기에 따라 뿔 형태가 달라진다: **major / minor / intermediate**.
같은 종의 major와 minor는 다른 종처럼 보일 수 있어 분류기를 혼동시킨다.
→ **male_form 레이블을 추가 감독 신호로 활용**하는 계층적 분류 구조 제안.

#### N3. 전처리 전략 비교 (Preprocessing Ablation)
"Classifier에 어떤 형태로 이미지를 넘기느냐"가 성능을 결정한다는 가설.
→ **4가지 전처리 모드를 분기 구현**하고 ablation study로 검증.

#### N4. 희귀종 데이터 부족 (Rare Species / Few-shot)
일부 종은 학습 데이터가 10장 미만이다.
→ **커뮤니티 데이터 flywheel**로 점진적 해결. 시뮬레이션 실험으로 효과 사전 검증.

#### N5. 미지·경계 개체 처리 (Open-Set Recognition)
실제 서비스에서는 한국 미서식 외래종, 타 곤충, 비곤충 이미지가 입력될 수 있다.
→ **DINOv2 특징 공간 기반 OOD 탐지** 구조로 강건하게 처리.

---

## 2. 시스템 아키텍처

### 2.1 전체 구조

```
[사용자]
   │
[Next.js Frontend]
   │  /api/*  rewrites
[FastAPI Backend]
   ├─ /api/v1/predict
   ├─ /api/v1/predictions/{id}/feedback
   └─ /admin/*
   │
[ML Layer]
   ├─ ModelManager (Singleton)
   ├─ BeetleSegmenter (YOLOv8n-seg)
   ├─ Classifier (MultiTaskClassifier — best arch from sweep)
   └─ OODDetector (DINOv2 centroid 기반)
   │
[PostgreSQL]  ← specimens, admin_action_logs
```

### 2.2 미지·외래종 처리 계층 (Open-Set Recognition)

```
[이미지 입력]
      │
[Layer 1: 사슴벌레 여부]  ← Detector (YOLOv8n-seg)
      ├─ 미검출 → result_type: "no_beetle"
      ▼
[Layer 2: OOD 탐지]  ← DINOv2 feature distance
      ├─ OOD score 높음 → result_type: "uncertain"
      ▼
[Layer 3: 종 분류]  ← Classifier (Multi-task)
      ├─ confidence < 0.4 → result_type: "low_confidence"
      └─ 정상 → result_type: "identified"
```

### 2.3 result_type 정의

| result_type | 조건 | 프론트 메시지 |
|-------------|------|---------------|
| `no_beetle` | Detector 미검출 | "사슴벌레를 찾을 수 없어요" |
| `uncertain` | OOD score > 0.65 | "인식하기 어려운 사진이에요" |
| `low_confidence` | confidence < 0.4 | "확신하기 어렵습니다. 후보 종: ..." |
| `identified` | 정상 예측 | 종명 + 성별 + male_form + confidence |

### 2.4 API 응답 구조

```json
{
  "prediction_id": "uuid",
  "result_type": "identified | low_confidence | uncertain | no_beetle",
  "species": "Dorcus_hopei_binodulosus",
  "species_ko": "왕사슴벌레",
  "confidence": 0.87,
  "sex": "male",
  "male_form": "major",
  "ood_score": 0.12,
  "top3": [...]
}
```

---

## 3. ML 파이프라인

### 3.1 분류기 모드 — 4가지

```
[Backbone: best arch from sweep]
        │
   [Feature Vector]  ← extract_features()로 t-SNE/UMAP에도 활용
   ┌────┼────┐
   ▼    ▼    ▼
[종]  [성별] [male_form]
Head  Head   Head
```

| mode | 학습 신호 | 출력 | 용도 |
|------|-----------|------|------|
| `species_only` (SO) | 종 | sp_out | 기준선 |
| `sex_only` (SX) | 종 + 성별 | sp_out, sex_out | MT ablation |
| `form_only` (FO) | 종 + 형태 | sp_out, form_out | MT ablation |
| `multi_task` (MT) | 종 + 성별 + 형태 | sp_out, sex_out, form_out | 최종 모델 |

**추론 시에는 모든 모드가 이미지만 입력받고 종만 출력.** sex/form head는 학습 신호 역할만 함.

**Loss:**
```
L_total = L_species + λ_sex · L_sex + λ_form · L_male_form
```
- sex/male_form 라벨이 unknown(-1)인 샘플은 masked loss로 제외
- λ 기본값: λ_sex=0.3, λ_form=0.2 (λ ablation으로 민감도 검증)

### 3.2 5가지 아키텍처 비교 대상

| 아키텍처 | 계열 | 특징 | 입력 처리 |
|---------|------|------|-----------|
| `convnext_tiny` | CNN | ImageNet pretrain | 448×448 |
| `efficientnet_b3` | CNN | 경량, 빠른 추론 | 448×448 |
| `swin_tiny` | Transformer | Window attention | 448×448 |
| `vit_small` | Transformer | Pure ViT (ViT-B/16) | 448→224 resize (ViTWrapper) |
| `dinov2_vits14` | Self-supervised | 희귀종 특징 강점, OOD 공유 backbone | 448×448 |

### 3.3 4가지 전처리 모드

| 모드 | 설명 | 데이터 경로 |
|------|------|------------|
| `full` | 원본 그대로 | `data/final` |
| `bbox` | BBox crop (배경 포함) | `data/final_bbox` |
| `seg_hard` | Binary 마스크 + 회색 배경 | `data/final_seg_hard` |
| `seg_soft` | Alpha blending + 회색 배경 | `data/final_seg_soft` |

**전처리 ablation 결과 (3-seed 평균, val_acc 기준):**

| 모드 | ConvNeXt SO | EfficientNet SO | Swin SO |
|------|------------|----------------|---------|
| full | **81.3%** | **84.0%** | **77.6%** |
| seg_soft | 64.5% | 68.3% | 60.6% |
| seg_hard | 65.5% | 65.5% | 60.6% |
| bbox | 65.7% | 72.0% | 62.5% |

→ full이 전 아키텍처에서 15~30%p 우위. 배경/전체 맥락이 종 판별에 유효하거나, 세그멘테이션 과정에서 형태 진단 특징이 손실됨.
→ GradCAM 분석 결과: 일부 케이스(Prismognathus)에서 흰 트레이 배경 테두리에 히트맵 집중 — 배경 편향 확인. 해당 종은 학습 데이터 대부분이 동일 배경으로 구성된 것이 원인. 데이터 다양성 확보(flywheel)로 해결 예정.

### 3.4 전체 실험 설계

#### 메인 Sweep — 120개
```
5 arch × 4 prep × 2 mode(SO, MT) × 3 seed = 120
python scripts/sweep.py
```
- run_name 형식: `{arch}_{prep}_{SO|MT}_s{seed}`
- 결과 분석: `python scripts/analyze_results.py`

#### MT Ablation — 6개
```
베스트 (arch, prep) × 2 mode(SX, FO) × 3 seed = 6
python scripts/sweep_mt_ablation.py   # MLflow에서 베스트 조합 자동 선택
```
- SO/MT는 메인 sweep 결과 재사용. SX/FO만 추가 실행.
- 목적: sex head 단독 / form head 단독 기여도 분리 검증

#### λ Ablation — 11개
```
λ_sex [0.05, 0.1, 0.2, 0.3, 0.5, 1.0] × λ_form=0.2  = 6
λ_form [0.05, 0.1, 0.3, 0.5, 1.0]     × λ_sex=0.3   = 5
python scripts/sweep_lambda.py
```
- seed=42 고정 (하이퍼파라미터 민감도 분석은 단일 seed 표준)
- 목적: λ 값이 cherry-pick되지 않았음을 증명

**총 137개 실험**

**최종 실험 결과 (3-seed 평균, full 전처리, val_acc 기준):**

| 아키텍처 | SO | MT | SX | macro_f1(SO) |
|---------|----|----|-----|--------------|
| efficientnet_b3 | 84.0% | 83.2% | **85.0%** | 66.6% |
| convnext_tiny | 81.3% | 80.5% | - | 63.7% |
| swin_tiny | 77.6% | 75.8% | - | 61.9% |
| dinov2_vits14 | 72.1% | 66.0% | - | 57.2% |
| vit_small | 64.0% | 66.5% | - | 49.1% |

**MT ablation 결과 (EfficientNet, full, 3-seed 평균):**

| Mode | Val Acc | Sex Acc | Form Acc |
|------|---------|---------|---------|
| SO | 84.0% | - | - |
| SX | **85.0%** | **96.6%** | - |
| FO | 82.2% | - | 55.6% |
| MT | 83.2% | 94.3% | 55.6% |

→ 성별 보조 학습(SX)이 종 분류 val_acc 향상. Form head는 단독 시 성능 소폭 하락.
→ SX의 val 우위는 test set에서 역전됨(test: SX 70.4% < SO 72.9%) — 과적합 가능성, 데이터 확장 시 재검증 필요.

**SX ablation 전 아키텍처 비교 (full, 3-seed 평균):**

| 아키텍처 | SO | SX | Δ | 비고 |
|---------|----|----|---|------|
| efficientnet_b3 | 84.0% | **85.0%** | +1.0%p | SX 유효 |
| swin_tiny | 77.6% | **80.6%** | +3.0%p | SX 효과 가장 큼 |
| vit_small | 64.0% | **66.0%** | +2.0%p | SX 유효 |
| convnext_tiny | 81.3% | 81.1% | -0.2%p | 중립 |
| dinov2_vits14 | 72.1% | 5.7% | -66.4%p | **학습 붕괴** |

→ EfficientNet, Swin, ViT 3개 아키텍처에서 SX가 일관되게 유효 — SX 효과가 아키텍처 전반에 걸쳐 재현됨.
→ DINOv2 SX 완전 붕괴 (5.7% ≈ 랜덤 추측): 세 가지 요인 복합 작용.
  (1) Gradient 충돌: 종 head와 성별 head가 동시에 backbone으로 역전파되며 반대 방향 update 발생
  (2) Pretraining 불일치: DINOv2는 label 없는 self-supervised 학습 → supervised multi-task gradient에 민감
  (3) LR 과다: DINOv2 공식 fine-tuning 권장 LR은 1e-5 이하인데 sweep은 1e-4 고정 → SX처럼 gradient 소스 2개면 불안정
  → 다른 아키텍처(ConvNeXt/EfficientNet/Swin/ViT)는 모두 ImageNet supervised pretrain이라 동일 조건에서 안정적
  → DINOv2는 SO 전용 사용 권장. 논문 한계점: "self-supervised backbone의 multi-task fine-tuning은 별도 LR scheduling 필요"

**Test set 최종 평가 결과 (hold-out, 199장):**

| 모델 | Test Acc | Macro F1 | 95% CI | Val-Test 갭 |
|------|---------|---------|--------|------------|
| Swin SO s456 | **72.9%** | **57.2%** | [66.8, 78.9] | -6.9%p |
| EfficientNet SO s42 | 72.9% | 55.4% | [66.8, 78.9] | -12.5%p |
| ConvNeXt SO s42 | 72.4% | 55.8% | [66.3, 78.9] | -11.4%p |
| EfficientNet SX s456 | 70.4% | 54.3% | [64.3, 76.9] | -15.5%p |
| DINOv2 SO s456 | 66.8% | 54.6% | [60.3, 73.4] | -6.9%p |
| ViT SO s123 | 60.3% | 39.1% | [53.8, 67.8] | -7.4%p |

→ **Best 모델: Swin-T SO s456** — test acc 공동 1위, macro_f1 최고, val-test 갭 가장 작음.
→ 상위 3개 모델(Swin/EfficientNet/ConvNeXt) 95% CI 중첩 — 통계적 유의차 없음.
→ Val-test 갭 전반적으로 큼 — 데이터 부족에 의한 과적합. flywheel 데이터 확장으로 해결 예정.

---

## 4. 데이터 파이프라인

### 4.1 전처리 순서

```bash
python -m src.preprocessing.merger   # merged_metadata.csv (observation_id 포함)
python -m src.preprocessing.cleaner  # data/processed/ 생성
python -m src.preprocessing.splitter # data/final/ 생성 (specimen-level split)
```

### 4.2 Specimen-level Split (중요)

**문제:** 같은 개체를 여러 각도로 찍은 사진이 train과 test에 동시에 들어갈 경우 성능이 부풀려짐.

**해결:** `observation_id` 기준으로 그룹핑 후 그룹 단위 분할.
- iNaturalist: 원본 CSV의 `observation_id` 사용
- 현장(field): `field_{image_stem}` 으로 이미지당 고유 ID 부여
- `GroupShuffleSplit`으로 관찰 단위 80/10/10 분할
- 같은 observation의 사진들은 반드시 같은 split에 위치

### 4.3 레이블 체계

| Level | 정보 | 데이터 소스 | 학습 활용 |
|-------|------|-------------|-----------|
| L1 | 종 | iNaturalist, 전문가 | 기본 학습 |
| L2 | 종 + 성별 | Label Studio 라벨링 (1985장 완료) | sex_only, multi_task |
| L3 | 종 + 성별 + male_form | 관리자 검수 | form_only, multi_task |

### 4.4 Data Flywheel (Phase 5 이후)

```
사용자 업로드 → 모델 예측 (L1 자동) → 사용자 피드백 (L2) → 관리자 검수 (L3) → 재학습
```

---

## 5. 논문 지원 스크립트

sweep 완료 후 베스트 모델(best.pth)로 실행:

| 스크립트 | 목적 | 출력 |
|---------|------|------|
| `scripts/analyze_results.py` | MLflow 결과 → 논문용 figure/table 전체 | `experiments/analysis/` |
| `scripts/evaluate_test.py` | test set 최종 평가 (bootstrap CI, sex-stratified) | JSON + figure |
| `scripts/tsne_features.py` | backbone feature t-SNE / UMAP 시각화 | 종별/성별 컬러 2장 |
| `scripts/gradcam.py` | Grad-CAM 시각화 (모델이 어디를 보는지) | 종별 3-panel figure |
| `scripts/calibrate.py` | Temperature Scaling (ECE 보정) | reliability diagram + JSON |
| `scripts/simulate_flywheel.py` | Data Flywheel 효과 시뮬레이션 (N4) | accuracy recovery curve |

**analyze_results.py 생성 산출물 목록:**

| 산출물 | 설명 |
|--------|------|
| `fig_architecture_comparison_*.png` | 아키텍처별 val_acc / macro_f1 |
| `fig_preprocessing_ablation.png` | 전처리 ablation 바 차트 |
| `fig_heatmap_*.png` | 5×4 heatmap (SO/MT 각각) |
| `fig_multitask_effect.png` | MT vs SO 차이 |
| `fig_training_curves.png` | ConvNeXt-T/full 학습 곡선 |
| `fig_female_vs_male_accuracy.png` | 종별 수컷/암컷 정확도 (N1) |
| `fig_so_vs_mt_female_accuracy.png` | SO vs MT 암컷 정확도 직접 비교 (N1 핵심 증거) |
| `fig_fewshot_scatter.png` | 학습 샘플 수 vs F1 scatter (N4) |
| `table_full_results.csv/.tex` | 전체 결과 (acc, top3, F1, P, R) |
| `table_mt_ablation.csv/.tex` | SO/SX/FO/MT 4개 모드 비교 |
| `table_multitask_results.csv/.tex` | sex/form head 성능 |
| `fig_lambda_ablation.png` | λ_sex / λ_form 민감도 분석 (λ cherry-pick 아님 증명) |
| `table_significance_test.csv/.tex` | Bootstrap 95% CI 유의성 검정 (아키텍처 간) |
| `table_prep_significance.csv/.tex` | Bootstrap 95% CI 유의성 검정 (전처리 간) |
| `table_dataset_stats.csv/.tex` | 종별 train/val/test 샘플 수 |
| `table_model_efficiency.csv/.tex` | 파라미터 수 + 추론 시간 |

---

## 6. 논문 구성

> **"Sexual Dimorphism-Aware Fine-Grained Classification of Korean Lucanidae with Preprocessing Strategy Comparison"**

```
1. Introduction
2. Related Work — FGVC, Insect Classification, Sexual Dimorphism
3. Dataset
   3.1 16종 정의 및 수집 방법 (iNaturalist + 현장)
   3.2 Specimen-level train/val/test split (observation_id 기반)
   3.3 레이블 체계 (L1/L2/L3)
4. Method
   4.1 Preprocessing Pipeline (4-mode)
   4.2 Multi-task Classifier (species + sex + male_form, masked loss)
   4.3 Architecture Comparison (5 architectures)
   4.4 Calibration (Temperature Scaling)
5. Experiments
   5.1 Preprocessing Ablation (RQ1)
   5.2 Architecture Comparison (RQ2)
   5.3 Multi-task Ablation: SO vs SX vs FO vs MT (RQ3)
   5.4 λ Sensitivity Analysis
   5.5 Rare Species Analysis (RQ4)
   5.6 Feature Space Visualization (t-SNE/UMAP)
   5.7 Qualitative Analysis (Grad-CAM)
6. Results & Discussion
7. Conclusion
```

**타겟 저널:** Methods in Ecology and Evolution (primary)

---

## 7. 재시작 체크리스트

컴퓨터 재시작 또는 VS Code 재실행 후 순서:

```bash
# 터미널 1 — MLflow 서버 (항상 이 명령어 고정)
mlflow server --host 0.0.0.0 --port 5001 --backend-store-uri sqlite:///mlflow.db

# 터미널 2 — sweep 또는 작업 실행
conda activate lucanidae_classifier_venv
python scripts/sweep.py   # 완료된 run 자동 스킵
```

- `mlflow.db` : 완료된 run 영구 보존 (컴퓨터 꺼져도 유지)
- `mlruns/` : 사용 안 함 (생성돼도 무시)
- Docker : sweep 중 불필요, 웹 서비스 개발 시에만 `docker-compose up`

### VS Code 터미널 설정
VS Code 하단에 "Relaunch Terminal" 알림이 뜨면 클릭 → PowerShell 7 (pwsh) 으로 전환.
Git Bash 터미널을 기본으로 사용하면 `wc`, `grep` 등 Unix 명령어 정상 동작.

---

## 8. 개발 로드맵

### Phase 1 — 데이터 파이프라인 (완료)
- iNaturalist 수집 + cleaner + splitter
- 16종 기준 정립 (configs/taxonomy.yaml)
- merger.py: observation_id 보존
- splitter.py: specimen-level split (GroupShuffleSplit)

### Phase 2 — Baseline Classifier (완료)
- ConvNeXt-Tiny baseline, val_acc 77.8%
- FastAPI 서빙 + Next.js 프론트 연결
- DB 저장 + 피드백 UI

### Phase 3 — Detector + 전처리 분기 (완료)
- Grounded SAM 자동 어노테이션
- YOLOv8n-seg 학습 (`models/weights/best_detector.pt`)
- 4가지 전처리 모드 구현 및 ablation 완료

### Phase 4 — Full Factorial Sweep + Multi-task (완료, 2026-05-11)

#### 구현 완료
- [x] sex/male_form Label Studio 라벨링 (1985장)
- [x] Multi-task head 구현 (SO/SX/FO/MT 4개 모드, masked loss)
- [x] sex_only / form_only 모드 추가 (MT ablation용)
- [x] extract_features() 메서드 (t-SNE/UMAP용)
- [x] Specimen-level split 구현 (observation_id 기반)
- [x] sweep.py (120 experiments, MLflow 완료 run 자동 스킵)
- [x] sweep_mt_ablation.py (SX/FO 6개, MLflow 베스트 조합 자동 선택)
- [x] sweep_lambda.py (λ ablation 11개)
- [x] analyze_results.py (figure/table 전체 자동화, Bootstrap CI)
- [x] evaluate_test.py (bootstrap CI, sex-stratified, ECE)
- [x] tsne_features.py (종별/성별 컬러 t-SNE/UMAP)
- [x] gradcam.py (Grad-CAM, CNN + Transformer 통합 지원)
- [x] calibrate.py (Temperature Scaling)
- [x] simulate_flywheel.py (Data Flywheel 시뮬레이션)

#### 코드 버그 수정 완료 (2026-04-28)

| 파일 | 버그 | 영향 | 수정 내용 |
|------|------|------|-----------|
| `src/detection/generate_segmented.py` | 단일 crop을 `IMG_001_0.jpg`로 저장 | sex_labels.csv 키 불일치 → MT의 sex/form 레이블 전부 -1 | 단일 crop은 원본 파일명 보존 |
| `scripts/sweep.py` | `seg_hard` 경로가 `./data/final_segmented`로 잘못됨 | seg_hard 실험 자동 스킵 (데이터 없음으로 판단) | `./data/final_seg_hard`로 수정 |
| `scripts/sweep_mt_ablation.py` | seeds=[42,43,44] — 메인 sweep [42,123,456]과 불일치 | MT ablation seeds 정렬 불가 | [42,123,456]으로 통일 |
| `predict.py` | sex_only/form_only 모드에서 튜플 언패킹 crash | 추론 불가 | 모드별 명시적 분기 처리 |
| `scripts/evaluate_test.py` | `data["sex_gt"].any()` — NumPy에서 -1은 truthy | 전부 unknown인 배열도 레이블 있음으로 판정 | `(data["sex_gt"] >= 0).any()`로 수정 |
| `scripts/gradcam.py` | CNN 4D 텐서 전제 — Transformer는 3D (B, N, C) | ViT/Swin/DINOv2에서 crash | 차원 검사 후 patch grid reshape 분기 처리 |
| `predict.py`, `visualize_errors.py`, `scripts/build_ood_centroids.py`, `src/ml/ood_detector.py` | `torch.load()` `weights_only` 미지정 | PyTorch 경고, 향후 버전 오류 예정 | `weights_only=False` 명시 |
| `scripts/analyze_results.py` | Wilcoxon 검정 — n=3에서 p<0.05 달성 불가 (최소 p=0.25) | 유의성 검정 무효 | Bootstrap 95% CI로 교체 |
| `src/ml/classifier.py` | sex_head 3-class (unknown 뉴런이 학습되지 않음) | 미학습 뉴런이 gradient 오염 가능성 | 2-class (male/female)로 변경 |
| `src/ml/classifier.py` | form_head 4-class (unknown 뉴런이 학습되지 않음) | 동일 | 3-class (major/minor/intermediate)로 변경 |
| `predict.py` | sex/form 인덱스 클램핑 (3/4 head 대응용) | 위 head 수정으로 불필요 | 클램핑 제거, 직접 인덱스 사용 |
| `train.py` | `compute_class_weight(classes=np.unique(...))` — 클래스 누락 시 shape mismatch crash | CrossEntropyLoss weight tensor 크기 불일치 | 누락 클래스는 weight=1(중립)로 보완 |
| `configs/default.yaml` | `segmented_dir: ./data/final_segmented` 오기 | 혼동 (코드에는 영향 없음) | `./data/final_seg_hard`로 수정 |
| 전체 .py/.md/.yaml | 이모지 잔존 | cp949 인코딩 오류 가능성 | 전체 제거 |

#### 무효 run 처리 (sweep 재시작 전 필수)

`generate_segmented.py` 버그로 인해 아래 9개 run의 MT sex/form 신호가 사실상 SO와 동일하게 학습됨.
MLflow에서 삭제 후 데이터 재생성 필요:

```
convnext_bbox_MT_s42      convnext_bbox_MT_s123      convnext_bbox_MT_s456
convnext_seg_hard_MT_s42  convnext_seg_hard_MT_s123  convnext_seg_hard_MT_s456
convnext_seg_soft_MT_s42  convnext_seg_soft_MT_s123  convnext_seg_soft_MT_s456
```

#### 전체 재시작 결정 (2026-04-28)

중간 발표(2026-06-05) 전 clean baseline 확보를 위해 전체 MLflow 기록을 초기화하고 처음부터 재학습.

```bash
# 터미널 1 — MLflow 서버 종료 후 DB 초기화
del mlflow.db          # Windows
rm mlflow.db           # bash

# 전처리 데이터 전체 초기화
rm -rf data/final_bbox data/final_seg_hard data/final_seg_soft

# 전처리 데이터 재생성 (detector 필요)
python -m src.detection.generate_segmented --mode bbox
python -m src.detection.generate_segmented --mode seg_hard
python -m src.detection.generate_segmented --mode seg_soft

# 터미널 1 — MLflow 서버 재시작
mlflow server --host 0.0.0.0 --port 5001 --backend-store-uri sqlite:///mlflow.db

# 터미널 2 — 전체 sweep 처음부터 (약 1주일 소요 예상)
python scripts/sweep.py

# sweep 완료 후 순서대로
python scripts/sweep_mt_ablation.py
python scripts/sweep_lambda.py
python scripts/analyze_results.py
```

- [ ] MLflow DB 초기화 (`del mlflow.db`)
- [ ] 전처리 데이터 삭제 후 재생성 (bbox / seg_hard / seg_soft)
- [ ] 전체 sweep 처음부터 (`python scripts/sweep.py`)
- [ ] MT ablation (`python scripts/sweep_mt_ablation.py`)
- [ ] λ ablation (`python scripts/sweep_lambda.py`)
- [ ] 분석 실행 (`python scripts/analyze_results.py`)
- [x] MLflow DB 초기화 (`del mlflow.db`)
- [x] 전처리 데이터 삭제 후 재생성 (bbox / seg_hard / seg_soft)
- [x] 전체 sweep 처음부터 (`python scripts/sweep.py`) — 120개 완료
- [x] MT ablation (`python scripts/sweep_mt_ablation.py`) — 6개 완료
- [x] λ ablation (`python scripts/sweep_lambda.py`) — 11개 완료
- [x] 분석 실행 (`python scripts/analyze_results.py`)
- [x] evaluate_test.py × 6개 → best 모델 선정 (Swin SO s456)

#### Phase 4 추가 완료 (2026-05-11)
- [x] `gradcam.py` — Swin SO s456, `experiments/gradcam/` 10장. Swin channels-last 버그 수정 (features[-1] permute)
- [x] `visualize_errors.py` — 오분류 패턴 분석, `experiments/error_analysis/` 저장
- [x] SX ablation 나머지 4 arch (ConvNeXt / Swin / ViT / DINOv2 × 3 seeds = 12 runs) — 완료
- [x] `analyze_results.py` 재실행 — SX 전 arch 반영, CSV encoding utf-8-sig 수정
- [x] `tsne_features.py` — Swin SO s456, val+test(n=397), 종+성별 통합 1장. Paul Tol 16색 팔레트, 300 DPI, PDF 병행 저장
- [x] `calibrate.py` — Temperature Scaling T=0.567, ECE 0.37→0.22 (40% 개선). log_T 파라미터화로 음수 T 버그 수정. 웹서비스 배포 시 적용 예정, 논문 성능 수치는 보정 전 raw 모델 기준

**GradCAM 분석 결과 요약 (Swin SO s456, test set 10장):**
- 정분류 케이스: 머리·큰턱 부위에 히트맵 집중 → 형태학적으로 의미 있는 특징 학습 확인
- 오분류 케이스 1 (Prosopocolius): 큰턱 특징은 포착했으나 Dorcus와 혼동 → 속간 형태 유사성 문제
- 오분류 케이스 2 (Prismognathus): 흰 배경 테두리에 히트맵 집중 → 배경 편향 (데이터 부족으로 배경 패턴 암기)
- 산출물: `experiments/gradcam/gradcam_*.png` (10종 × 3-panel)

**오분류 패턴 분석 결과 (val set, Swin SO s456):**

| 원인 유형 | 대표 케이스 | 건수 |
|----------|------------|------|
| 속 내 혼동 (Dorcus spp.) | D. consentaneus → D. titanus (4건), D. titanus → D. hopei (4건) | 다수 |
| 다속 혼동 | Prosopocoilus inclinatus → 6가지 다른 종으로 분산 (총 12건) | 최다 오분류 종 |
| 소규모 속 혼동 | Lucanus, Prismognathus, Platycerus → 유사 형태 종으로 혼동 | 소수 |

→ 전 오분류 패턴이 데이터 부족에 기인. 속 내 혼동은 분류학적으로 합리적인 실수(형태 유사성 반영).
→ Prosopocoilus inclinatus가 가장 불안정 — 학습 샘플 절대 부족이 원인. flywheel 데이터 확장 시 우선 수집 대상.
→ 산출물: `experiments/error_analysis/{true}_pred_{pred}/` 폴더별 오분류 이미지

### Phase 5 — 웹서비스 배포 + Flywheel (2026-05-11~)

#### 핵심 비전
웹서비스 배포 → 커뮤니티 데이터 수집 → 모델 재학습의 flywheel 구조.
현재 1,985장 / test acc 72.9%를 기준선으로, 데이터 2배(~4,000장) 도달 시 재실험.

```
사용자 업로드
    → 모델 예측 (L1 자동)
    → 사용자 피드백 (L2: 성별)
    → 관리자 검수 (L3: male_form)
    → 재학습 → 성능 향상 → 더 많은 사용자
```

#### Phase 5 작업
- [x] 웹서비스 배포 (Docker + Contabo VPS + Cloudflare Tunnel) — beetledex.com 운영중 (2026-05-16)
- [ ] Google 로그인 OAuth Redirect URI 등록 완료 후 소셜 로그인 활성화
- [ ] 커뮤니티 데이터 수집 시작 (~4,000장 목표)
- [ ] 4,000장 도달 시 learning curve 분석 (데이터 50%/75%/100% 성능 변화)
- [ ] OOD detection 강화 (외래종 이미지 수집 후 재학습)
- [ ] Transformer vs CNN 재비교 (데이터 증가 시 순위 변동 가능)

#### 논문 기여 프레이밍
> 소규모 분류군(희귀 곤충 등)을 연구하는 수많은 연구자에게 재현 가능한 파이프라인을 제시.
> 사슴벌레과는 사례 연구이며, 동일 프레임워크를 나비·잠자리·식물 등 타 분류군에 즉시 적용 가능.

### Phase 6 — 서비스 완성 (예정)
- [ ] Cloudflare R2 이미지 저장 (현재: 서버 로컬 data/uploads/)
- [ ] GPS 분포 지도 (전체 공개 핀 — 현재는 본인 핀만)
- [ ] 관리자 대시보드 고도화
- [ ] 모델 재학습 자동화 파이프라인

---

## 9. 기술 스택

| 영역 | 스택 |
|------|------|
| Frontend | Next.js 14, Tailwind CSS, shadcn/ui |
| Backend | FastAPI, SQLAlchemy (async), Alembic |
| Database | PostgreSQL 16 |
| ML | PyTorch 2.6+cu124, 5 architectures |
| OOD | DINOv2-ViT-S/14 centroid 기반 |
| Detector | YOLOv8n-seg (class-agnostic) |
| Auto-annotation | autodistill + Grounded SAM 2 |
| Experiment tracking | MLflow (localhost:5001, SQLite backend) |
| GPU | RTX 4060 Laptop 8GB, CUDA 12.4 |
| Deploy | Docker Compose, Contabo VPS (Singapore), Cloudflare Tunnel |

---

## 10. 주요 파일 구조

```
configs/
  default.yaml          # 모든 하이퍼파라미터 중앙 관리
  taxonomy.yaml         # 16종 정의

src/
  ml/
    classifier.py       # build_model (5 arch), MultiTaskClassifier (SO/SX/FO/MT), extract_features()
    ood_detector.py     # OODDetector (centroid 기반)
    detector.py         # BeetleSegmenter (YOLOv8n-seg)
    manager.py          # ModelManager singleton (API용)
  training/
    dataset.py          # get_dataloaders, MultiTaskDataset (SO/SX/FO/MT 모두 지원)
    trainer.py          # ModelTrainer (4개 모드, MLflow logging, masked loss)
  detection/            # Grounded SAM, annotation_utils, generate_segmented
  preprocessing/
    merger.py           # observation_id 포함 merged_metadata.csv 생성
    cleaner.py          # 학명 정규화 + data/processed/ 생성
    splitter.py         # specimen-level split (observation_id 기반, GroupShuffleSplit)
  api/                  # FastAPI endpoints
  db/                   # PostgreSQL models, repository
  services/             # prediction, feedback 비즈니스 로직

scripts/
  sweep.py              # 메인 120 experiments (MLflow 스킵 포함)
  sweep_mt_ablation.py  # MT ablation SX/FO 6개 (MLflow 베스트 조합 자동 선택)
  sweep_lambda.py       # λ ablation 11개
  analyze_results.py    # 모든 figure/table 자동 생성
  evaluate_test.py      # test set 최종 평가 (bootstrap CI, sex-stratified)
  tsne_features.py      # backbone feature t-SNE / UMAP 시각화
  gradcam.py            # Grad-CAM 시각화
  calibrate.py          # Temperature Scaling (ECE 보정)
  simulate_flywheel.py  # Data Flywheel 시뮬레이션 (N4)
  build_ood_centroids.py # OOD centroid 계산·저장

train.py                # 단일 학습 실행 (sweep.py가 호출)
train_detector.py       # YOLOv8n-seg 학습
predict.py              # LucanidaePredictor (단독 추론)
pipeline.py             # 통합 추론 (Segmenter + Classifier + OOD)
main.py                 # 데이터 파이프라인 (merge → clean → split)
```
