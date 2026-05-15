# 사슴벌레 AI 동정기

**한국 사슴벌레과(Lucanidae) 16종 AI 분류 웹 서비스**

사진 한 장으로 한국 사슴벌레 16종을 즉시 동정하고, 사용자 피드백으로 데이터를 축적하는 엔드-투-엔드 파이프라인입니다.

---

## 현재 성능 (Test Set, hold-out 199장)

| 모델 | Test Acc | Macro F1 | Val-Test 갭 | 비고 |
|------|---------|---------|------------|------|
| **Swin-T SO s456** | **72.9%** | **57.2%** | **-6.9%p** | **배포 모델** |
| EfficientNet-B3 SO s42 | 72.9% | 55.4% | -12.5%p | |
| ConvNeXt-Tiny SO s42 | 72.4% | 55.8% | -11.4%p | |
| EfficientNet-B3 SX s456 | 70.4% | 54.3% | -15.5%p | |
| DINOv2-ViT-S/14 SO s456 | 66.8% | 54.6% | -6.9%p | |
| ViT-Small SO s123 | 60.3% | 39.1% | -7.4%p | |

Swin-T SO: test accuracy 공동 1위, macro F1 최고, val-test 갭 가장 작음 → 일반화 성능 최우수.

---

## 시스템 아키텍처

```
[사용자 (모바일/웹)]
        ↓ 이미지 업로드
[Next.js 14 Frontend — localhost:3000]
        ↓ /api/v1/* rewrite
[FastAPI Backend — localhost:8000]
        ↓
  ┌─────────────────────────────────────┐
  │  PredictionService                  │
  │  ├── BeetleSegmenter (YOLOv8n-seg) │  사슴벌레 검출 (full 모드 시 skip)
  │  ├── Swin-T Classifier             │  16종 분류
  │  └── OODDetector (centroid 기반)   │  학습 데이터 centroid와 cosine 거리
  └─────────────────────────────────────┘
        ↓
[PostgreSQL]  ← specimens, admin_action_logs
[data/uploads/]  ← 업로드 이미지 로컬 저장
```

### result_type 결정 로직

| result_type | 조건 | UI 메시지 |
|-------------|------|----------|
| `no_beetle` | Detector 미검출 | 사슴벌레를 찾을 수 없어요 |
| `uncertain` | OOD score ≥ 0.65 | 인식하기 어려운 사진이에요 |
| `low_confidence` | confidence < 0.40 | 확신하기 어렵습니다 |
| `identified` | 정상 | 종명 + 신뢰도 + Top3 |

---

## 디렉토리 구조

```
lucanidae_classifier/
├── src/
│   ├── api/
│   │   ├── main.py               # FastAPI 앱 진입점
│   │   ├── deps.py               # 의존성 주입 (모델·DB 세션)
│   │   └── routes/
│   │       ├── predict.py        # POST /api/v1/predict, GET /api/v1/stats
│   │       ├── feedback.py       # POST /api/v1/predictions/{id}/feedback
│   │       └── admin.py          # GET|POST /admin/*
│   ├── services/
│   │   ├── prediction.py         # 추론 흐름 (Segmenter → Classifier + OOD)
│   │   └── feedback.py           # 피드백 처리, GPS 격자 변환
│   ├── ml/
│   │   ├── manager.py            # ModelManager 싱글턴
│   │   ├── ood_detector.py       # OODDetector (centroid 기반 cosine 거리)
│   │   └── ...
│   ├── db/
│   │   ├── models.py             # SQLAlchemy ORM (Specimen, AdminActionLog)
│   │   └── repository.py         # Repository 패턴
│   ├── preprocessing/            # 데이터 파이프라인 (학습용)
│   └── training/                 # 분류기 학습
├── frontend/                     # Next.js 14 (App Router)
│   └── app/
│       ├── page.tsx              # 홈 — 사진 업로드 + 표본 스캐너 UI
│       ├── result/[id]/          # 분류 결과 + 피드백
│       ├── my/                   # 내 도감 (localStorage 기반 이력)
│       ├── species/              # 종 도감 (16종 상세)
│       └── admin/                # 관리자 — 표본 검토·교정
├── scripts/
│   ├── build_ood_centroids.py    # OOD centroid 생성 (학습 후 1회 실행)
│   └── ...
├── configs/
│   ├── default.yaml              # 모델·추론 설정 (현재: swin_tiny, full 전처리)
│   └── taxonomy.yaml             # 16종 정의
├── models/weights/
│   ├── sweep/swin_full_SO_s456.pth   # 배포 모델 체크포인트
│   └── ood_centroids.pt              # OOD 탐지용 centroid
├── data/
│   └── uploads/                  # 업로드 이미지 로컬 저장
├── Dockerfile
├── docker-compose.yml            # API + DB + Redis
└── requirements.txt
```

---

## 빠른 시작

### 환경 설정

```bash
conda activate lucanidae_classifier_venv
pip install -r requirements.txt
```

### 백엔드 실행

```bash
# .env 파일 필요 (DATABASE_URL 등)
$env:PYTHONPATH = "."; uvicorn src.api.main:app --reload --port 8000
```

### 프론트엔드 실행

```bash
cd frontend
npm run dev   # http://localhost:3000
```

### DB + Redis (Docker)

```bash
docker compose up db redis
```

### OOD centroid 생성 (모델 교체 시 재실행)

```bash
$env:PYTHONPATH = "."; python scripts/build_ood_centroids.py
# 출력: models/weights/ood_centroids.pt
```

---

## API

### `POST /api/v1/predict`

```json
// Response
[{
  "prediction_id": "uuid",
  "result_type":   "identified",
  "species":       "Dorcus_hopei_binodulosus",
  "species_ko":    "왕사슴벌레",
  "confidence":    0.729,
  "is_ood":        false,
  "ood_score":     0.31,
  "top3":          [{ "species": "...", "confidence": 0.729 }],
  "latency_ms":    234
}]
```

### `POST /api/v1/predictions/{id}/feedback`

```json
{
  "is_correct":      false,
  "correct_species": "Dorcus_hopei_binodulosus",
  "sex":             "male",
  "male_form":       "major"
}
```

### `GET /admin/queue?status=pending`

관리자 페이지: `localhost:3000/admin`

---

## 실험 설계 요약

137개 실험 (5 arch × 4 prep × mode × 3 seed + ablation)

**핵심 결과:**
- **전처리**: full이 전 아키텍처에서 15~30%p 우위 — 배경·맥락 정보가 종 판별에 유효
- **Multi-task (SX)**: EfficientNet/Swin/ViT에서 종 분류 val_acc 향상, 단 test에서 과적합 확인
- **Best 모델**: Swin-T SO s456 — val-test 갭 최소 (-6.9%p), 일반화 성능 최우수

자세한 실험 결과: [beetle_project_v6.md](beetle_project_v6.md)

---

## 인프라

| 서비스 | 용도 | 상태 |
|--------|------|------|
| Contabo VPS M | 서빙 서버 | 예정 |
| Cloudflare Tunnel | HTTPS | 예정 |
| PostgreSQL 16 | 메인 DB | Docker |
| Redis 7 | 캐시 | Docker |

---

## 주요 설계 결정

- `num_workers=0` — Windows multiprocessing hang 방지
- GPS는 1km 격자로 변환 후 저장 — raw GPS 미저장 (멸종위기종 보호)
- OOD centroid는 학습 데이터 기준으로 생성 — 모델 교체 시 재생성 필요
- `user_corrected=True` — 유저가 AI 예측에 동의하지 않고 직접 교정한 경우
