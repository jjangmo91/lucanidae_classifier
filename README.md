# 비틀덱스 (BeetleDex)

**한국 사슴벌레과(Lucanidae) 16종 AI 동정 + 채집 커뮤니티**

사진 한 장으로 한국 사슴벌레 16종을 즉시 동정하고, Google 로그인 기반 커뮤니티·게임 랭킹 시스템으로 채집 데이터를 축적하는 엔드-투-엔드 파이프라인입니다.

---

## 현재 성능 (Test Set, hold-out 199장)

| 모델 | Test Acc | Macro F1 | Val-Test 갭 | 비고 |
|------|---------|---------|------------|------|
| **Swin-T SO s456** | **72.9%** | **57.2%** | **-6.9%p** | **배포 모델** |
| EfficientNet-B3 SO s42 | 72.9% | 55.4% | -12.5%p | |
| ConvNeXt-Tiny SO s42 | 72.4% | 55.8% | -11.4%p | |
| EfficientNet-B3 SX s456 | 70.4% | 54.3% | -15.5%p | |
| DINOv2-ViT-S/14 SO s456 | 66.8% | 54.6% | -6.9%p | |

Swin-T SO: test accuracy 공동 1위, macro F1 최고, val-test 갭 최소 → 일반화 성능 최우수.

---

## 시스템 아키텍처

```
[사용자 (모바일/웹)]
        ↓ 이미지 업로드 + Google 로그인
[Next.js 14 Frontend — localhost:3000]
        ↓ /api/v1/* rewrite
[FastAPI Backend — localhost:8000]
     ├── AuthService         Google ID 토큰 검증 + JWT 발급
     ├── PredictionService
     │   ├── BeetleSegmenter (YOLOv8n-seg)   사슴벌레 검출
     │   ├── Swin-T Classifier               16종 분류
     │   └── OODDetector (centroid cosine)   분포 외 입력 탐지
     ├── FeedbackService     피드백·교정 저장 + 등급 재계산
     └── CommunityService    갤러리·랭킹·지도·프로필
        ↓
[PostgreSQL]  ← users, specimens, admin_action_logs
[data/uploads/]  ← 업로드 이미지 로컬 저장
```

### result_type 결정 로직

| result_type | 조건 | UI |
|-------------|------|-----|
| `no_beetle` | Detector 미검출 | 사슴벌레를 찾을 수 없어요 |
| `uncertain` | OOD score ≥ 0.65 | 인식하기 어려운 사진이에요 |
| `low_confidence` | confidence < 0.40 | 확신하기 어렵습니다 |
| `identified` | 정상 | 종명 + 신뢰도 + Top3 |

---

## 게임 시스템

### 등급 (채집 기준 점수제)

| 등급 | 점수 | 비고 |
|------|------|------|
| 딱린이 | 0–9 | |
| 표본수집가 | 10–29 | |
| 주간채집러 | 30–59 | |
| 야간채집러 | 60–119 | |
| 루카나이더 | 120–199 | |
| **직업 분기** | 200+ | 아래 4종 중 하나 |

### 직업 분기 (200pts 달성 후)

| 직업 | 조건 |
|------|------|
| 💎 레어헌터 | S·A종 업로드 비율 ≥ 60% |
| 🔬 분류학자 | 교정 제출 ≥ 20회 |
| 🗺️ 도감탐험가 | 16종 중 ≥ 12종 발견 |
| 👑 채집왕 | 위 조건 미해당 고량 업로더 |

### 희귀도 티어 (채집 난이도 기준)

| 티어 | 점수 | 종 |
|------|------|-----|
| S | 4pt | 왕사슴벌레, 털보왕사슴벌레, 꼬마넓적사슴벌레, 뿔꼬마사슴벌레, 엷은털왕사슴벌레, 큰꼬마사슴벌레 |
| A | 3pt | 두점박이사슴벌레 |
| B | 2pt | 사슴벌레, 톱사슴벌레, 홍다리사슴벌레, 다우리아사슴벌레, 원표애보라사슴벌레, 참넓적사슴벌레 |
| C | 1pt | 넓적사슴벌레, 애사슴벌레, 길쭉꼬마사슴벌레 |

---

## 프론트엔드 페이지

| 경로 | 설명 |
|------|------|
| `/` | 홈 — 표본 스캐너 (드래그·카메라 업로드, GPS 자동 캡처) |
| `/result/[id]` | 분류 결과 + 피드백 + 결과 공유 (Web Share / 카카오) |
| `/my` | 내 도감 — 등급 카드, 진행 바, 16종 포켓몬 격자 |
| `/gallery` | 커뮤니티 갤러리 — 희귀도 필터(S/A/B/C) + 정렬(최신/오래된/신뢰도) |
| `/rank` | 채집가 랭킹 — 내 순위 하이라이트, 위 순위까지 점수 차 표시 |
| `/map` | 내 채집 지도 — 로그인 필수, 본인 GPS 핀만 표시 |
| `/species/[slug]` | 16종 상세 도감 |
| `/users/[id]` | 유저 공개 프로필 — 등급·직업·최근 채집 12장 |
| `/admin` | 관리자 패널 — 표본 검토·승인·교정·거부 |

---

## 디렉토리 구조

```
lucanidae_classifier/
├── src/
│   ├── api/
│   │   ├── main.py               # FastAPI 앱 진입점
│   │   ├── deps.py               # 의존성 주입
│   │   └── routes/
│   │       ├── predict.py        # POST /api/v1/predict (GPS 포함)
│   │       ├── feedback.py       # POST /api/v1/predictions/{id}/feedback
│   │       ├── auth.py           # Google OAuth + JWT
│   │       ├── community.py      # 갤러리·랭킹·지도·프로필
│   │       └── admin.py          # 관리자 검토·교정
│   ├── services/
│   │   ├── prediction.py         # 추론 흐름
│   │   ├── feedback.py           # 피드백·GPS 격자 처리
│   │   └── auth.py               # Google 토큰 검증 + JWT 발급
│   ├── ml/
│   │   ├── manager.py            # ModelManager 싱글턴
│   │   ├── classifier.py         # Swin-T 분류기
│   │   ├── detector.py           # YOLOv8 세그멘터
│   │   └── ood_detector.py       # centroid cosine OOD
│   ├── db/
│   │   ├── models.py             # ORM (User, Specimen, AdminActionLog) + 등급 로직
│   │   └── repository.py         # Repository 패턴
│   ├── preprocessing/            # 데이터 파이프라인 (학습용)
│   └── training/                 # 분류기 학습
├── frontend/                     # Next.js 14 (App Router)
│   ├── app/
│   │   ├── opengraph-image.tsx   # 기본 OG 이미지 자동 생성
│   │   ├── icon.tsx              # 앱 아이콘 (512×512)
│   │   ├── apple-icon.tsx        # iOS 홈 화면 아이콘 (180×180)
│   │   ├── manifest.ts           # PWA manifest
│   │   ├── page.tsx              # 홈
│   │   ├── result/[id]/          # 결과 + 피드백 + 공유 + 동적 OG
│   │   ├── my/                   # 내 도감
│   │   ├── gallery/              # 커뮤니티 갤러리 (필터·정렬)
│   │   ├── rank/                 # 랭킹
│   │   ├── map/                  # 내 채집 지도
│   │   ├── species/              # 종 도감
│   │   ├── users/[id]/           # 유저 프로필
│   │   ├── admin/                # 관리자 패널
│   │   └── api/auth/[...nextauth]/ # NextAuth 핸들러
│   ├── components/
│   │   ├── layout/
│   │   │   ├── NavBar.tsx        # 반응형 네비게이션 (햄버거 메뉴)
│   │   │   ├── Providers.tsx     # SessionProvider + 로그인 시 localStorage 이전
│   │   │   └── LevelUpToast.tsx  # 등급 상승 알림 토스트
│   │   └── predict/
│   │       ├── UploadSection.tsx # 업로드 UI + GPS 캡처
│   │       ├── ResultCard.tsx    # 결과 카드 + 피드백 + 공유
│   │       └── GradeCard.tsx     # 등급 카드 + 포켓몬 도감 격자
│   ├── lib/
│   │   ├── auth.ts               # useAuth 훅
│   │   └── species-data.ts       # 16종 정의 + 희귀도
│   ├── Dockerfile.frontend       # Next.js standalone 빌드
│   └── .env.local.example        # 프론트엔드 환경변수 템플릿
├── alembic/                      # DB 마이그레이션
├── scripts/                      # 분석·학습 유틸
├── configs/
│   ├── default.yaml              # 모델·추론 설정
│   └── taxonomy.yaml             # 16종 정의
├── models/weights/
│   ├── sweep/swin_full_SO_s456.pth   # 배포 분류기
│   └── ood_centroids.pt              # OOD centroid
├── data/uploads/                 # 업로드 이미지
├── Dockerfile                    # FastAPI 백엔드
├── .dockerignore
├── docker-compose.yml            # API + Frontend + DB + Redis + Cloudflare Tunnel
├── .env.example                  # 환경변수 템플릿
└── DESIGN.md                     # 설계 문서 상세
```

---

## 빠른 시작

### 환경 설정

```bash
conda activate lucanidae_classifier_venv
pip install -r requirements.txt
```

### `.env` 작성

```bash
cp .env.example .env
# DATABASE_URL, GOOGLE_CLIENT_ID, JWT_SECRET, ADMIN_TOKEN 등 입력
```

### 프론트엔드 `.env.local` 작성

```bash
cp frontend/.env.local.example frontend/.env.local
# NEXTAUTH_SECRET, GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, NEXT_PUBLIC_API_URL 입력
# 선택: NEXT_PUBLIC_KAKAO_APP_KEY (카카오 공유 버튼 활성화)
```

### 백엔드 실행

```bash
$env:PYTHONPATH = "."; uvicorn src.api.main:app --reload --port 8000
```

### 프론트엔드 실행

```bash
cd frontend
npm install
npm run dev   # http://localhost:3000
```

### DB + Redis (Docker)

```bash
docker compose up db redis
# 마이그레이션
alembic upgrade head
```

### OOD centroid 생성 (모델 교체 시 재실행)

```bash
$env:PYTHONPATH = "."; python scripts/build_ood_centroids.py
```

---

## 주요 API

### 인증

| Method | URL | 설명 |
|--------|-----|------|
| POST | `/api/v1/auth/google` | Google ID 토큰 → 자체 JWT |
| GET | `/api/v1/auth/me` | 내 프로필·등급·통계 |

### 동정

| Method | URL | 설명 |
|--------|-----|------|
| POST | `/api/v1/predict` | 이미지 + GPS(선택) → 분류 결과 |
| POST | `/api/v1/predictions/{id}/feedback` | 정오답 피드백·교정 |
| GET | `/api/v1/stats` | 오늘·누적 업로드 수 |

### 커뮤니티

| Method | URL | 파라미터 | 설명 |
|--------|-----|----------|------|
| GET | `/api/v1/gallery` | `sort`, `rarity`, `limit`, `offset` | 공개 갤러리 |
| GET | `/api/v1/specimens/{id}` | — | 표본 단건 조회 (OG 메타용) |
| GET | `/api/v1/my/specimens` | — | 내 표본 목록 (인증 필요) |
| GET | `/api/v1/my/map` | — | 내 GPS 핀 (인증 필요) |
| POST | `/api/v1/my/claim-specimens` | `specimen_ids[]` | 비로그인 표본 → 내 계정 이전 (인증 필요) |
| GET | `/api/v1/ranking` | `limit` | 채집가 랭킹 |
| GET | `/api/v1/users/{id}` | — | 유저 공개 프로필 |

갤러리 `sort`: `newest`(기본) / `oldest` / `confidence`  
갤러리 `rarity`: `1`(C) / `2`(B) / `3`(A) / `4`(S)

### 관리자

| Method | URL | 설명 |
|--------|-----|------|
| GET | `/admin/queue` | 검토 대기 표본 목록 |
| POST | `/admin/items/{id}/action` | 승인·교정·거부 |

---

## 인프라

| 서비스 | 용도 | 상태 |
|--------|------|------|
| Contabo VPS (싱가포르) | 서빙 서버 | 운영중 |
| Cloudflare Tunnel | HTTPS — beetledex.com | 운영중 |
| PostgreSQL 16 | 메인 DB | Docker |
| Redis 7 | 캐시 | Docker |
| MLflow | 실험 추적 | Docker (localhost:5001) |
| Label Studio | 데이터 레이블링 | Docker (localhost:8081) |

### 프로덕션 배포

```bash
# 서버 Docker 전체 실행
docker compose up -d

# DB 마이그레이션 (최초 1회 또는 스키마 변경 시)
docker compose exec api alembic upgrade head

# 로그 확인
docker compose logs -f api
docker compose logs -f frontend
```

> Cloudflare Tunnel이 `beetledex.com → frontend:3000` 트래픽을 처리합니다.  
> API 포트(8000)와 DB 포트(5432)는 외부에 노출되지 않습니다.

---

## 주요 설계 결정

- **GPS 1km 격자 저장** — raw GPS 미저장, 멸종위기종 서식지 보호 및 개인정보 보호
- **지도 개인 핀만 공개** — 본인 업로드 핀만 조회 가능, 타인 위치 비공개
- **OOD centroid** — 학습 데이터 기준 생성, 모델 교체 시 재생성 필요
- **`user_corrected=True`** — 유저가 AI 예측에 동의하지 않고 직접 교정한 경우
- **희귀도 티어는 채집 난이도 기준** — 사육 난이도와 별개
- **결과 공유** — Web Share API(모바일 인스타·카카오 등) + 카카오 SDK(선택, 데스크톱)
- **모바일 반응형** — 햄버거 메뉴, PokedexGrid 2→4열, 랭킹 행 compact
- **PWA** — manifest + 코드 생성 아이콘(icon.tsx, apple-icon.tsx), 홈 화면 추가 지원
- **OG 이미지 동적 생성** — opengraph-image.tsx로 기본/결과별 OG 카드 자동 생성, 별도 PNG 불필요
- **비로그인 → 로그인 이전** — localStorage UUID → `/my/claim-specimens` → DB user_id 연결, 로그인 시 자동 실행

자세한 설계: [DESIGN.md](DESIGN.md)
