# 한국 사슴벌레 AI 도감 — 설계 문서 v7

> 최종 수정: 2026-09-04
> 기준: v6 (2026-05-10) + 논문 리부트 설계 v0.2 병합, 미해결 3항목 확정 반영
> 기준 코드베이스: `beetledex` / GitHub: `jjangmo91/lucanidae_classifier` (main 브랜치)
> v6 문서는 git 히스토리에서 조회 (`git show <v6-commit>:DESIGN.md`). 본 문서가 단일 기준.

---

## 0. v6 → v7 변경 요약

| 영역 | v6 | v7 | 근거 |
|------|----|----|------|
| 대상 종 수 | "16종" 단일 표기 | **16(참조) / 13(수집) / 10(모델링) 3단 분리** | §1.3 — 실측 결과 모델 출력은 10-way |
| 노벨티 구조 | N1-N5 | E0-E7 실험 축으로 재배치 + BioCLIP 2·XAI 정합 신설 | §1.2, §3.5 |
| 아키텍처 비교 | 5개 자유 비교 | 해상도·파라미터 밴드 통제 프로토콜 | §3.2 |
| 전처리 | 4모드 | 4모드 × aspect(pad/squish) = 8 + 배경 교란 검증 | §3.3 |
| male_form | 주관 3분류 (규칙 없음) | **비율 계측 + 종별 분위수 규칙 + not_applicable** | §4.4 |
| 테스트셋 | 단일 hold-out 199장 | **3층 구조 + 오염 방어.** v6 test는 특별 지위 폐지, 현장 15장만 3층 승격 | §4.5, §4.6 |
| 통계 | 단일 hold-out, 3-seed | 반복 층화 CV(5×5) + 부트스트랩 CI | §3.5 |
| v6 sweep 149개 | 최종 결과 | **전량 폐기** — 라벨 유실·조각 오염·검정력 부족으로 무효 | §3.4 |
| 라벨 단위 | 이미지 1장 = 종 1개 | **개체(crop) 단위 매니페스트로 이행** | §4.8 |
| 다중 검출 | 전 crop이 원본 라벨 상속 | 병합 후 1장 저장 + 검수 큐, 다종은 개체별 라벨 | §4.8 |
| 데이터 공개 | 계획 없음 | **매니페스트 공개 방식 확정**, 이미지 재배포에 의존하지 않음 | §11 |
| 배경·위치 | full 우위를 맥락 정보로 해석 | **배경 불변성을 요구사항으로 고정.** 위치는 모델에서 배제 | §3.3 |
| v6 실험 결과 | 예비 실험으로 보존 | **전량 폐기.** 라벨 유실·조각 오염·검정력 부족으로 무효 | §3.4 |
| 수집 목표 | 종당 test 20장 | **종당 175-370개체 3단계 목표** | §13 |

**v7에서 새로 확정된 3개 항목** (v6 초안의 미해결 목록):
1. 종 수 정합 → §1.3 — 16/13/10 3단 어휘 확정, 논문은 10-way로 서술
2. male_form 판정 규칙 → §4.4 — 계측 기반 규칙 확정, 기존 라벨 전량 재작업 대상
3. v6 test 199장 재편입 → §4.6 — 현장 15장만 3층 승격, iNaturalist 184장은 1층 풀로 환원

**v7에서 추가로 발견된 결함 2개** (둘 다 v6 실험 결과에 직접 영향):
- 성별·형태 라벨의 **32.7%가 split 키 불일치로 유실**된 채 학습됨 → §4.3
- 검출기 과분할로 **조각 이미지 309장이 종 라벨을 달고** bbox/seg 학습셋에 혼입 → §4.8
- iNaturalist 이미지의 **31.9%가 재배포 불가(ARR)**이고 라이선스가 수집조차 안 됨 → §11.2

### 0.1 이 문서가 관리하는 것과 관리하지 않는 것

**DESIGN.md는 결정과 근거 수치를 관리한다. 산출물 자체는 관리하지 않는다.**

| 구분 | 위치 | 비고 |
|------|------|------|
| 설계 결정·근거 수치 | **DESIGN.md** | 단일 기준 |
| 근거 수치를 산출한 스크립트 | `scripts/analysis/` | 각 스크립트가 대응 절을 명시 |
| 측정 출력 | `experiments/analysis_v7/` | gitignore 대상, 재생성 가능 |
| 코드 변경 | 저장소 소스 | 예: 라벨 키 수정은 `src/training/dataset.py` |
| 라이선스 역조회 결과 | `data/raw/inaturalist/photo_licenses.csv` | **gitignore 대상.** 백업 없음, API 재조회로만 복구 |
| 분류군 정의 | `configs/taxonomy.yaml` | 16종 참조 목록의 실제 기준 |
| 실험 기록 | MLflow DB | §11.5에 따라 CSV export 필요 |

**주의 1** — 본 문서는 **아직 존재하지 않는 파일을 다수 참조한다.**
`docs/adversarial_review.md`, `docs/male_form_protocol.md`, `data/labels/crops.csv`,
`LICENSE`, `environment.yml`, `src/xai/`, §5.2의 신규 스크립트 대부분이 미작성이며
Phase 7 M0의 작업 항목이다. 문서는 **현재 상태의 기술이 아니라 계획**이다.

**주의 2** — 문서에 인용된 수치는 **v7 설계 시점의 데이터 상태**를 전제한다.
재라벨링(§4.4)·재수집(§13)·detector 재학습(§4.8) 이후에는 `scripts/analysis/`를 다시 실행해
수치를 갱신해야 한다. 특히 §12.1 검정력 표와 §13 부족분 표는 수집 진행에 따라 바뀐다.

---

## 1. 프로젝트 개요

### 1.1 목표

한국 사슴벌레과(Lucanidae)를 대상으로:

1. **종 분류** — fine-grained classification (실제 모델링 대상 10종, §1.3)
2. **성별·형태 분류** — 성적 이형성 및 수컷 크기 다형성(male form) 인식
3. **미지·외래종 처리** — 한국 미서식 종, 데이터 미확보 6종, 비사슴벌레 입력의 강건한 처리
4. **커뮤니티 서비스** — 일반 사용자가 사진 한 장으로 종을 동정하는 웹 서비스 (beetledex.com)
5. **연구 논문** — **"전문가 영역 분류군(expert-domain taxa)의 분류기 구축 템플릿"**
   어떤 단일 전문가 집단도 모든 생물군을 동정할 수 없다. 데이터 빈약·롱테일·극단 이형성
   분류군의 대표 사례(한국 Lucanidae)로 **무엇을(전략), 얼마나(데이터), 왜(형질)** 준비해야
   하는지의 재현 가능한 정량 지침을 제공한다.

### 1.2 연구 노벨티 — v6 N1-N5의 재배치

| v6 | v7 위치 | 변경 |
|---|---|---|
| N1 이형성 멀티태스크 | **E5** | 유지·격상. "성판별 모델"(기존재)과 태스크 구분 명문화 — 우리는 종 분류의 이형성 취약성을 층화 평가. 성별 라벨 1,985장 자산 그대로 활용 |
| N2 male form | **E5** | **격하**. 판정 규칙 부재로 v6 form head는 다수결 기준선 미달(§4.4). 규칙 확정 후 재실험, 기여 주장은 재실험 결과가 나온 뒤에만 |
| N3 전처리 비교 | **E0** | 개정. 4모드에 **종횡비 보존 pad vs squish 축 추가**. v6 full 우위는 배경 편향 확인과 미분리 → 재검증 대상 |
| N4 희귀종/few-shot | **E2·E3** | 확장. flywheel 시뮬레이션 → **학습곡선 × 전략(BioCLIP zero/few-shot/FT) 교차점 분석**으로 격상 |
| N5 open-set | **E6c** | 유지. ood_detector(DINOv2) 기존 구현 활용, 속 수준 후퇴와 통합. **데이터 미확보 6종이 자연 OOD 평가 대상**(§1.3) |
| — | **E3 [신설]** | 생물 파운데이션 모델 축(BioCLIP 2). 2026년 투고 필수 베이스라인 |
| — | **E4 [신설]** | attention-검색표 진단 형질 정량 정합 (v6 정성 Grad-CAM의 격상) |
| — | **E6a·b [신설]** | 오류·임베딩의 분류학적 구조 (v6 t-SNE의 격상, 기술 분석 한정) |

**BioCLIP 2와의 차별 지점**: BioCLIP 2는 스스로 롱테일 편향을 인정한다. 본 연구는 그 사각지대
(지역 희귀종·극단 이형성·시민과학 소규모 데이터)의 실전 지침 + 커뮤니티 flywheel(BeetleDex)이라는
지역 데이터 생산 루프를 다룬다.

---

### 1.3 대상 종 정의 — 16 / 13 / 10 [항목 1 확정]

v6는 문서 전체에서 "16종"으로 단일 표기했으나, **모델이 실제 출력하는 클래스는 10개**다.
v7은 세 층위를 서로 다른 이름으로 분리하고, 각 층위가 어디에 쓰이는지 고정한다.

| 층위 | 종 수 | 실체 | 정의 위치 | 용도 |
|------|------|------|-----------|------|
| **참조 종목록** (reference list) | **16** | 국립생물자원관 국가생물종목록 Lucanidae | `configs/taxonomy.yaml` → `target_classes` | 학명 정규화, 제외 목록 판정, 서비스 도감 콘텐츠 |
| **수집 달성 종** (collected) | **13** | 이미지 1장 이상 확보 | `data/processed/` | 데이터 커버리지 서술, flywheel 우선순위 |
| **모델링 대상** (modeled) | **10** | `min_samples=10` 통과 | `data/final/` → `num_classes` | **모델 출력, 논문의 분류 태스크** |

#### 참조 종목록 16종 전체 (국명 대조)

| # | 학명 | 국명 | 상태 |
|---|------|------|------|
| 1 | `Dorcus_titanus_castanicolor` | 넓적사슴벌레 | 모델링 |
| 2 | `Prosopocoilus_inclinatus_inclinatus` | 톱사슴벌레 | 모델링 |
| 3 | `Lucanus_maculifemoratus_dybowskyi` | 사슴벌레 | 모델링 |
| 4 | `Dorcus_rectus_rectus` | **애사슴벌레** | 모델링 |
| 5 | `Prismognathus_dauricus` | 다우리아사슴벌레 | 모델링 |
| 6 | `Prosopocoilus_astacoides_blanchardi` | 두점박이사슴벌레 | 모델링 |
| 7 | `Dorcus_consentaneus_consentaneus` | 참넓적사슴벌레 | 모델링 |
| 8 | `Dorcus_hopei_binodulosus` | 왕사슴벌레 | 모델링 |
| 9 | `Dorcus_rubrofemoratus_rubrofemoratus` | **홍다리사슴벌레** | 모델링 |
| 10 | `Platycerus_hongwonpyoi_hongwonpyoi` | 원표애보라사슴벌레 | 모델링 |
| 11 | `Figulus_punctatus` | 길쭉꼬마사슴벌레 | 수집만 (5장) |
| 12 | `Aegus_laevicollis_subnitidus` | 꼬마넓적사슴벌레 | 수집만 (5장) |
| 13 | `Nigidius_miwai` | 뿔꼬마사슴벌레 | 수집만 (2장) |
| 14 | `Dorcus_carinulatus_koreanus` | 털보왕사슴벌레 | 미확보 |
| 15 | `Dorcus_tenuihirsutus` | 엷은털왕사슴벌레 | 미확보 |
| 16 | `Figulus_binodulus` | 큰꼬마사슴벌레 | 미확보 |

**애사슴벌레와 홍다리사슴벌레**는 목록에서 누락되기 쉬우니 주의한다. 애사슴벌레는
본 데이터셋에서 네 번째로 큰 클래스(262장)이고, 홍다리사슴벌레도 모델링 대상 10종에 포함된다.
원표애보라사슴벌레의 구 국명 **홍원표비단사슴벌레**는 `taxonomy.yaml`에서 같은 종으로 매핑된다.

#### 16 → 13: 데이터 0장인 3종

| 학명 | 국명 | 상태 |
|------|------|------|
| `Dorcus_carinulatus_koreanus` | 털보왕사슴벌레 | iNaturalist·현장 모두 0장 |
| `Dorcus_tenuihirsutus` | 엷은털왕사슴벌레 | 0장 |
| `Figulus_binodulus` | 큰꼬마사슴벌레 | 0장 |

#### 13 → 10: `min_samples=10` 미달 3종

| 학명 | 국명 | 관찰 수 |
|------|------|--------|
| `Figulus_punctatus` | 길쭉꼬마사슴벌레 | 5 |
| `Aegus_laevicollis_subnitidus` | 꼬마넓적사슴벌레 | 5 |
| `Nigidius_miwai` | 뿔꼬마사슴벌레 | 2 |

필터 위치: `src/preprocessing/splitter.py` — 관찰 그룹 단위 집계 후 `class_counts >= min_samples`
(`configs/default.yaml: min_samples: 10`). 이미지 12장이 여기서 제외되어 1,997 → **1,985장**.

#### 모델링 대상 10종의 실제 분포 (data/final)

| 학명 | train | val | test | 계 |
|------|------|-----|------|----|
| `Dorcus_titanus_castanicolor` | 596 | 66 | 67 | 729 |
| `Prosopocoilus_inclinatus_inclinatus` | 372 | 55 | 56 | 483 |
| `Lucanus_maculifemoratus_dybowskyi` | 225 | 26 | 26 | 277 |
| `Dorcus_rectus_rectus` | 213 | 24 | 25 | 262 |
| `Prismognathus_dauricus` | 104 | 11 | 11 | 126 |
| `Prosopocoilus_astacoides_blanchardi` | 25 | 5 | 5 | 35 |
| `Dorcus_consentaneus_consentaneus` | 20 | 4 | 4 | 28 |
| `Dorcus_hopei_binodulosus` | 11 | 3 | 2 | 16 |
| `Dorcus_rubrofemoratus_rubrofemoratus` | 11 | 2 | 2 | 15 |
| `Platycerus_hongwonpyoi_hongwonpyoi` | 11 | 2 | 1 | 14 |
| **계** | **1,588** | **198** | **199** | **1,985** |

상위 5종이 전체의 94.6%. 하위 5종 합계 108장(5.4%)이며 test 기여는 14장.
**극단적 롱테일이 이 데이터셋의 정의적 성질이며, 숨기지 않고 논문의 출발점으로 서술한다.**

#### v7 확정 사항

1. **논문·코드·서비스 카피에서 "16종 분류"라는 표현을 금지**한다. 분류 태스크는 **10-way**로 서술한다.
   16은 "참조 종목록", 13은 "수집 달성"으로만 등장시킨다.
2. `min_samples=10`은 **E0·E1(전처리·아키텍처 비교)에서 고정**한다. 비교 조건을 흔들지 않기 위함.
3. **E2·E3는 이 필터를 실험 변수로 다룬다.** BioCLIP zero-shot은 학습 데이터가 필요 없으므로
   13종 전체(가능하면 텍스트 프롬프트로 16종 전체)를 시도할 수 있다.
   "지역 FT는 10종밖에 못 하는데 글로벌 FM은 16종을 시도할 수 있다"는 대비가 E3의 핵심 소재다.
4. **데이터 미확보 6종(16-10)은 open-set 평가의 자연 실험 대상**으로 E6c에 편입한다.
   실제 한국 서식종인데 학습에 없는 종 → "미지"로 후퇴해야 정상. 인위적 OOD보다 설득력이 크다.
5. **flywheel 수집 우선순위 = 이 6종 + test 기여 14장뿐인 하위 5종.**
   Phase 5 커뮤니티 수집과 2층(카페/기증) 라벨링의 목표 종 목록으로 고정한다.
6. `taxonomy.yaml`은 16종 유지. 축소하지 않는다 — 정규화·제외 판정 기준이자 서비스 도감의 종 목록이다.

---

## 2. 시스템 아키텍처 [v6 유지]

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
[Layer 3: 종 분류]  ← Classifier (Multi-task, 10-way)
      ├─ confidence < 0.4 → result_type: "low_confidence"
      └─ 정상 → result_type: "identified"
```

**v7 보강**: 학습에 없는 한국 서식종 6종(§1.3)이 입력되면 Layer 2 또는 Layer 3에서 반드시
`uncertain` / `low_confidence`로 빠져야 한다. 이 6종은 E6c의 평가 세트이자 서비스 안전장치의
회귀 테스트 케이스다. 데이터 확보 시 `data/test_images/ood_korean/`에 축적한다.

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

**v7 주의**: `male_form` 필드는 §4.4 재라벨링 완료 전까지 서비스 응답에서 **비노출**한다.
현재 form head는 다수결 기준선 미달이므로 사용자에게 표시할 근거가 없다.

---

## 3. ML 파이프라인

### 3.1 분류기 모드 — 4가지 [유지]

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
- head 크기: sex_head 2-class(male/female), form_head 3-class(major/minor/intermediate).
  unknown 뉴런은 학습되지 않으므로 제외 — v6에서 수정 완료

### 3.2 아키텍처 비교 — 공정성 프로토콜 [개정]

**v6 문제점:**
(a) ViT만 448 → 224 리사이즈(ViTWrapper) → 입력 해상도 불공정
(b) 파라미터 비매칭 (EfficientNet-B3 12M vs Swin-T 28M)
(c) 계열 혼합에 통제 없음

**v7 프로토콜:**
- **입력 해상도 통일**: 전 모델 384 또는 448 native (ViT는 384 native 변형 사용, interpolate 금지)
- **파라미터 밴드 매칭**: 계열당 대표 1개, 20-30M 밴드
  (예: ConvNeXt-T 28M / Swin-T 28M / ViT-S+해상도보정) + 동일 레시피
- **DINOv2**: SO 전용 유지(v6 SX 붕괴 소견 반영), LR 1e-5 별도.
  논문에는 self-supervised backbone의 multi-task 한계 소견으로 서술
- **BioCLIP 2 트랙 추가**: zero-shot / few-shot(kNN) / linear probe / full FT
- **v6 149개 sweep 결과는 전량 폐기한다**(§3.4). 인용하지 않는다.

### 3.3 전처리 모드 — 4모드 x aspect 축 [개정]

| 모드 | 설명 | 데이터 경로 |
|------|------|------------|
| `full` | 원본 그대로 | `data/final` |
| `bbox` | BBox crop — **패딩 없음** (`src/ml/detector.py:83`) | `data/final_bbox` |
| `seg_hard` | Binary 마스크 + 회색 배경 + bbox crop | `data/final_seg_hard` |
| `seg_soft` | Alpha blending + 회색 배경 + bbox crop | `data/final_seg_soft` |

**v7 신설 축 — aspect**: `pad`(종횡비 보존 레터박스) vs `squish`(강제 정사각 리사이즈).
기존 4모드와 교차하여 **실질 2x4 = 8조건**.

**v6 구현 결함 (v7에서 교정)**: bbox/seg_hard/seg_soft 모두 YOLOv8 bbox 좌표로 패딩 없이 잘라내
사슴벌레가 프레임을 꽉 채우도록 확대된다. 자연스러운 스케일·구도가 손실되며, 이것이 full > bbox
성능 차이의 주요 원인일 수 있다. v7은 **bbox에 20-30% 패딩 옵션을 추가**하여 스케일 보존 효과와
배경 정보 효과를 분리한다.

#### 배경 불변성은 성능 옵션이 아니라 요구사항이다 [v7 확정]

**배경과 위치를 종 판별 신호로 쓰지 않는다.** 절충하지 않는다.

근거는 배포 현실이다. BeetleDex 사용자는 **집에서 사육 개체를 찍어 올린다.**
책상, 손바닥, 사육 통, 흰 종이 위의 사진이 들어온다. 야외 서식지 배경을 학습한 모델은
정확히 사용자가 실제로 쓰는 자리에서 무너진다. 학습 데이터의 배경 분포와 배포 시점의
배경 분포가 다르다는 것이 이 서비스의 전제다.

위치 정보도 같은 이유로 배제한다. 사육 개체의 촬영 좌표는 서식지가 아니라 사용자의 집이다.
좌표만으로 종을 47.2% 맞출 수 있으므로(§12.6-A) 위치를 넣으면 모델이 그 지름길을 탄다.
**모델 입력은 이미지뿐이며 메타데이터는 어떤 형태로도 특징에 들어가지 않는다.**

따라서 v7의 전처리 질문은 "무엇이 정확도가 높은가"가 아니라
**"무엇이 배경이 바뀌어도 견디는가"**로 바뀐다.

| 검증 | 방법 |
|------|------|
| 배경 교란 | 배경 셔플·중립화·무작위 합성 후 성능 낙폭 측정 (`scripts/background_probe.py`) |
| 배경 도메인 이동 | **실내·사육 배경으로 구성한 별도 테스트 분할**에서 평가 (§13.4) |
| 위치 누출 | 예측이 좌표와 상관되는지 검사. 상관이 크면 배경 지름길 신호 |
| 형질 응시 | E4 정합 지표로 몸체를 보는지 확인 |

#### 채택 규칙 (사전 등록) [v7 확정]

임의로 정한 임계값은 쓰지 않는다. **고정하는 것은 숫자가 아니라 결정 절차**이며,
그 절차의 값은 데이터가 정한다. 실험 전에 아래를 등록하고 이후 변경하지 않는다.

**1차 기준** — 실내·사육 배경 검증 분할에서의 **macro-F1 최댓값**.
배포 조건이 실내 배경이므로 그 조건의 성능이 곧 목적함수다. 상수가 등장하지 않는다.

**동률 처리** — 부트스트랩 95% CI가 겹치면 통계적으로 구분되지 않는 것으로 보고,
**더 단순한 전처리를 택한다** (full < bbox < seg_soft < seg_hard 순으로 단순).
재현과 배포가 쉬운 쪽을 택하는 것이며 성능 주장이 아니다.

**배경 의존성은 게이트가 아니라 기술 통계** — 배경 교란 낙폭은 전 조건에 대해
CI와 함께 보고한다. 낙폭 CI가 0을 배제하면 "배경 의존"으로 **표시**하되
그것만으로 탈락시키지 않는다. 탈락 여부는 1차 기준이 정한다.

이렇게 하면 "몇 %p까지 허용할 것인가"라는 임의 판단이 설계에서 사라진다.
사후에 유리한 조건을 고르는 일도 구조적으로 불가능해진다.

### 3.4 v6 실험 결과 취급 — 전량 폐기 [v7 확정]

**v6의 149개 sweep 결과, 전처리 ablation 수치, MT/SX/FO 비교, test 199장 평가 결과를
논문과 설계 판단의 근거로 일절 사용하지 않는다.** 아래 세 가지가 각각 독립적으로 결과를 무효화한다.

| 무효화 사유 | 내용 | 근거 |
|-----------|------|------|
| **라벨 유실** | 성별·형태 라벨의 32.7%가 split 키 불일치로 조용히 마스킹됨. MT/SX/FO는 의도한 감독 신호의 3분의 2만 받고 학습됨 | §4.3 |
| **조각 오염** | 검출기 과분할로 다리·더듬이 조각 309장이 종 라벨을 달고 bbox/seg 학습셋에 혼입. 전처리 비교의 대조 조건이 오염됨 | §4.8 |
| **검정력 부족** | test n=199에서 5%p 차이 검출력 35%. 아키텍처 순위를 주장할 수 있었던 적이 없음 | §12.1 |

"공정성 결함을 스스로 지적하는 예비 실험"으로 보존하는 서사도 **철회한다.**
공정성 문제라면 보존해서 대비할 가치가 있지만, 위 셋은 **다른 것을 측정한 결과**다.
잘못 측정된 수치를 논문에 실으면 방어 대상만 늘어난다.

**남기는 것은 수치가 아니라 교훈이다.** v6에서 얻은 것은 다음 세 가지이며 이것만 인용한다.
1. 라벨 키를 split에 묶으면 재분할 시 조용히 유실된다
2. 검출기 과분할을 방치하면 학습셋에 조각이 섞인다
3. 단일 hold-out 199장으로는 모델 간 차이를 판정할 수 없다

MLflow 기록은 §11.5에 따라 CSV로 export해 보존하되, **내부 개발 이력**으로만 다룬다.

### 3.5 v7 실험 설계 — E0-E7 [개정]

| ID | 내용 | 우선도 |
|----|------|--------|
| **E0** | 전처리 x aspect ablation (§3.3) — 8조건 + 배경 교란 | P0 |
| **E1** | 공정 3계열 비교 (§3.2) — 해상도·파라미터 통제 | P0 |
| **E2** | 학습곡선 x 전략 교차점: 상위 6종 1/8 → 전체 서브샘플링 x {계열 FT, BioCLIP zero/few/FT} → **종당 N장 권고표** | P0 |
| **E3** | BioCLIP 심층: 종별 훈련 노출량 표(ToL-10M grep + GBIF API, 동의어 양방향), PHON unseen 자연 실험, 희귀종 구간 글로벌 FM vs 지역 FT 격차, **10종 FT vs 13/16종 zero-shot 대비**(§1.3) | P0 |
| **E4** | attention-검색표 정합: 진단 형질 표 → 형질 영역 주석 → pointing game / energy PG / IoU. CNN=Grad-CAM++, ViT=Prompt-CAM. sanity check + deletion/insertion 충실도 병행, plausibility/faithfulness 구분 | P0 |
| **E5** | 성별·체급 층화: 층화 정확도, 성비 조작 실험, BioCLIP 종내 변이 직교 보존 주장 검증. **male_form은 §4.4 재라벨링 완료분에 한해 포함, 다수결 기준선 병기 필수** | P1 |
| **E6** | 분류학적 구조: (a) 속 내/간 혼동 구조 (b) 임베딩 덴드로그램 vs 분류체계 cophenetic (c) 확신도 기반 속 후퇴 + **미확보 6종 자연 OOD**(§1.3). **임베딩≠계통 한계 명문** | P1 |
| **E8** | **롱테일 기법 비교** — class-balanced loss / logit adjustment / LDAM / 재샘플링 / 2단계 학습. 논문이 롱테일을 문제로 걸고도 표준 해법을 시험하지 않는 구멍을 메움 (§12.6-C) | P0 |
| **E9** | **메타데이터 누출 검사(음성 대조군)** — 좌표·날짜 단독 예측(측정 완료: acc 47.2%, macro-F1 23.9%). **위치를 모델에서 배제해야 하는 근거**이자, 이미지 모델이 배경을 통해 위치를 간접 이용하는지 검사하는 기준선. 융합 모델은 만들지 않는다 (§3.3, §12.6-A) | P0 |
| **E10** | **전문가 베이스라인** — 곤충 전문가 2-3인 × test 100장. 논문 전제(전문가 영역 분류군의 난이도)를 주장에서 측정으로 바꾼다 (§12.6-B) | P0 |
| **E11** | **제2 분류군 축소 반복** — 하늘소과 또는 호랑나비과에 E2·E3만 반복. 템플릿 주장의 유일한 실질 방어 (§12.6-D) | P1 |
| **E7** | quality_grade 노이즈, **소스 ablation(카페 층 포함/제외)**. 소스 ablation은 공개 불가 데이터에 결론이 의존하지 않음을 보이는 근거이므로 조건부가 아니라 **공개 패키지 필수 구성요소**로 격상 (§11.4) | P1 |

**통계 프로토콜 (§12.1 검정력 감사 반영)**: **내부 비교(E0·E1·E8)는 반복 층화 CV(5×5)로만 수행**한다. 단일 hold-out 199장은 5%p 차이를 35% 검정력으로밖에 잡지 못하므로 아키텍처 순위 주장에 쓸 수 없다. **3층 테스트는 모델 순위가 아니라 일반화 낙폭 측정 전용**으로 쓴다. 종별 수치를 본문에 실으려면 종당 test 50장이 필요하다(20장은 CI 반폭 ±21.5%p). 부트스트랩 CI 병기. v6의 단일 hold-out 199장은 3층 테스트로
확장(§4.5). 희귀종 결론은 표본 수를 명시하고 서술 수위를 격하. accuracy와 macro-F1 병행 보고.
**클래스당 test n<10인 종은 개별 수치를 본문에 싣지 않고 부록으로 내린다** (§1.3 하위 5종 해당).

---

## 4. 데이터 파이프라인

### 4.1 전처리 순서

```bash
python -m src.preprocessing.merger   # merged_metadata.csv (observation_id 포함)
python -m src.preprocessing.cleaner  # data/processed/ 생성 (학명 정규화 + 제외 목록 적용)
python -m src.preprocessing.splitter # data/final/ 생성 (specimen-level split + min_samples)
```

현재 수량: 수집 2,153건 → 정규화 통과 1,997장(13종) → `min_samples=10` 통과 **1,985장(10종)**.

### 4.2 Specimen-level Split [유지 + 중요 보강]

**문제:** 같은 개체를 여러 각도로 찍은 사진이 train과 test에 동시에 들어가면 성능이 부풀려진다.

**해결:** `observation_id` 기준 그룹핑 후 그룹 단위 분할.
- iNaturalist: 원본 CSV의 `observation_id` 사용
- 현장(field): `field_{image_stem}`으로 이미지당 고유 ID 부여
- `GroupShuffleSplit`으로 관찰 단위 80/10/10 분할
- 같은 observation의 사진들은 반드시 같은 split에 위치
- 검증 결과 train/val/test 간 `observation_id` 교집합 0 — 누수 없음

#### v7 신규 확인 사항 — 현재 데이터에서 specimen split은 실질 무효

실측 결과 **전체 1,985장이 1,985개의 서로 다른 observation에 1:1 대응**한다
(iNaturalist 2,000 관찰 = 2,000 이미지, field 153 = 153). 다중 촬영 개체가 **0건**이다.

| split | 이미지 | 관찰 | 이미지/관찰 |
|-------|-------|------|------------|
| train | 1,588 | 1,588 | 1.00 |
| val | 198 | 198 | 1.00 |
| test | 199 | 199 | 1.00 |

즉 **현재 데이터에서 specimen-level split은 image-level random split과 결과가 동일**하다.
분할 로직은 옳고 향후 필수지만, v6 데이터에서는 방어 효과를 실제로 발휘한 적이 없다.

**v7 조치:**
1. 논문에서 specimen-level split을 **성능 부풀림을 실제로 막았다**고 서술하지 않는다.
   *"수집 방식상 관찰당 1장이어서 이번 데이터에서는 image-level과 동치였고, 다중 사진이 유입되는
   2·3층에서 비로소 구속력을 가진다"*로 정확히 기술한다. 리뷰어 지적 선점.
2. 2층(카페/기증)·3층(네이처링/현장)은 개체당 다중 사진이 정상이므로, 유입 시점부터
   `observation_id`(또는 `specimen_id`) 부여를 **필수 입력 항목**으로 강제한다.
3. **배경·촬영자 상관 통제 추가**: 동일 배경 클러스터(예: 흰 트레이)나 동일 촬영자가
   train/test 양쪽에 같은 종으로만 존재하지 않도록 검사하는 스크립트를 추가한다.
   Prismognathus 배경 편향(§3.3)이 정확히 이 문제이므로 E0의 전제 조건이다.

### 4.3 레이블 체계 [개정 — 컬럼 확장]

| Level | 정보 | 데이터 소스 | 학습 활용 |
|-------|------|-------------|-----------|
| L1 | 종 | iNaturalist, 전문가 | 기본 학습 |
| L2 | 종 + 성별 | Label Studio 라벨링 (1,985장 완료) | sex_only, multi_task |
| L3 | 종 + 성별 + male_form | 관리자 검수 (§4.4 규칙 적용) | form_only, multi_task |

**전 이미지 공통 태그 (v7 확장)**: `source`(inat/cafe/donation/naturing/field),
`observed_on`, `quality_grade`, `tier`(1/2/3), `photographer_id`, `background_cluster`.
**captive 판별은 하지 않는다** (v0.2 결정 — 소스 ablation으로 대체 방어, E7).

**신뢰도 보고 (신설, 논문 방어 핵심)**: 성별·종 라벨 각각 **제2 평가자 15-20%에 대해 Cohen's κ 보고**.
male_form은 연속 비율값이므로 κ 대신 **ICC(2,1)** 을 함께 보고한다(§4.4).

**현재 L2 라벨 실태 (data/labels/sex_labels.csv, 1,985행):**

| sex | 건수 | 비율 |
|-----|------|------|
| male | 1,200 | 60.5% |
| female | 723 | 36.4% |
| unknown | 62 | 3.1% |

sex head 정확도 96.6%(SX)로 성별 신호는 충분히 학습된다. 문제는 male_form 쪽이다.

#### 라벨 키 유실 버그 (v7에서 발견, 미수정 시 재라벨링도 무효)

`dataset.py`의 `_load_sex_labels()`는 라벨 키를 `split/species/filename`으로 만든다.
그런데 `sex_labels.csv`는 **이전 split 기준으로 작성**되었고 이후 `data/final`이 재분할되었다.
실측 결과 **1,985건 중 649건(32.7%)이 split이 달라져 키 매칭에 실패**한다.
매칭 실패분은 조용히 `unknown`(-1)으로 떨어져 masked loss에서 제외된다.

| 모드 | 키 매칭 실패 | sex 라벨 유효 | male_form 라벨 유효 |
|------|------------|-------------|-------------------|
| full | 649 / 1,985 | 1,292 (65.1%) | 776 (39.1%) |
| bbox | 1,098 / 2,294 | 1,172 (51.1%) | 698 (30.4%) |

설계 문서가 주장해 온 "1,985장 라벨 완료"는 실제 학습에 **65% 만 전달되었다.**
bbox/seg 모드에서는 §4.8의 `_N` 접미사 문제까지 겹쳐 51%까지 떨어진다.
즉 v6의 SX/FO/MT 실험 전체가 의도한 감독 신호의 3분의 2만 받은 상태로 돌아갔다.

**조치**: 키에서 split 성분을 제거하고 `species/filename`으로 매칭한다.
파일명은 `{observation_id}_{name}` 형식이라 전역 고유하며, `species/filename` 조합도
1,985건 전부 고유함을 확인했다. 재분할이 일어나도 라벨이 따라간다.
§4.8의 crop 매니페스트로 이행하면 `crop_id`가 이 역할을 대신한다.

---

### 4.4 male_form 판정 규칙 [항목 2 확정]

#### 문제 진단

v6에는 **판정 규칙 문서가 존재하지 않았다.** 유일한 기준은 Label Studio 템플릿
(`configs/label_studio_template.xml`)의 선택지 문구뿐이다:

```
major        = 대형(뿔 김)
minor        = 소형(뿔 짧음)
intermediate = 중간형
```

즉 이미지 한 장을 눈으로 보고 내리는 **기준점 없는 주관 판정**이었다. 결과:

| 지표 | 값 |
|------|----|
| 수컷 | 1,200장 |
| form 라벨 부여 | 1,173장 (97.8%) |
| major | 676장 (57.6%) |
| intermediate | 268장 (22.8%) |
| minor | 229장 (19.5%) |
| 다수결 비율 (항상 major) | 57.6% |

> v6의 form head 정확도(55.6%)는 다수결 비율에도 못 미쳤으나, **그 수치 자체는 근거로 쓰지 않는다.**
> 라벨 유실(§4.3)로 form 라벨의 상당수가 마스킹된 상태에서 학습됐기 때문이다.
> 규칙을 다시 세우는 근거는 아래의 **구조적 문제**이며, 이는 성능 수치와 무관하게 성립한다.

**종별 분포에서 드러나는 라벨 오염:**

| 종 | major | inter | minor | 계 | 문제 |
|----|------|-------|-------|----|------|
| `Dorcus_titanus_castanicolor` | 341 | 45 | 17 | 403 | major 편중 85% |
| `Prosopocoilus_inclinatus_inclinatus` | 144 | 73 | 89 | 306 | 정상 분포 |
| `Dorcus_rectus_rectus` | 79 | 42 | 39 | 160 | 정상 분포 |
| `Lucanus_maculifemoratus_dybowskyi` | 61 | 48 | 28 | 137 | 정상 분포 |
| `Prismognathus_dauricus` | 12 | 40 | 44 | 96 | **분포 역전** |
| `Dorcus_consentaneus_consentaneus` | 12 | 7 | 5 | 24 | 표본 부족 |
| `Prosopocoilus_astacoides_blanchardi` | 8 | 10 | 2 | 20 | 표본 부족 |
| `Platycerus_hongwonpyoi_hongwonpyoi` | 11 | 0 | 0 | 11 | **전량 major — 판정 불가 종** |
| `Dorcus_hopei_binodulosus` | 7 | 2 | 1 | 10 | 표본 부족 |
| `Dorcus_rubrofemoratus_rubrofemoratus` | 1 | 1 | 4 | 6 | 표본 부족 |

**근본 원인**: major/minor는 본래 **allometry(체장 대비 큰턱 성장 관계)의 구간**으로 정의되며
그 임계점은 **종마다 다르다**. 그런데 축척 없는 사진 한 장에서는 절대 체장을 잴 수 없다.
결국 라벨러는 종별 기준 없이 이미지마다 "커 보인다/작아 보인다"를 판정했고,
`D. titanus`처럼 원래 큰 종은 전부 major로, `Prismognathus`처럼 작은 종은 전부 minor 쪽으로 쏠렸다.
**종 정체성이 form 라벨로 새어 들어간 것이며, 이는 form head에 종 정보의 중복 신호를 주는 동시에
종내 변이 정보는 주지 못하는 최악의 조합이다.**

#### v7 판정 규칙 (확정)

절대 크기 판단을 버리고, **사진 안에서 측정 가능한 무차원 비율 + 종별 분위수**로 정의한다.

**1단계 — 계측 (Label Studio keypoint 도구)**
이미지에 두 개의 길이를 표시한다.

| 기호 | 정의 |
|------|------|
| `M` | 큰턱 길이 — 큰턱 선단(apex)에서 두순(clypeus) 접합부까지 직선거리 |
| `B` | 체장 대용치 — 전흉배판(pronotum) 앞 가장자리에서 초시(elytra) 끝까지 직선거리 |
| `R` | **R = M / B** (무차원. 축척·촬영거리와 무관) |

`B`를 체장 대용치로 쓰는 이유: 큰턱은 다형 형질이므로 전체 체장에 포함시키면 분모가
분자와 함께 커져 비율이 둔감해진다. 큰턱을 제외한 몸통 길이가 체급의 안정적 대리변수다.

**2단계 — 종별 분위수로 3분류**

```
species s 에 대해, train split의 R 분포에서
  major        : R >= Q70(s)
  minor        : R <= Q30(s)
  intermediate : 그 사이
```

- **임계값은 train split만으로 산출하고 동결**한다. val/test에 재적합 금지.
- 산출된 종별 Q30/Q70 값은 논문 보충자료에 표로 공개한다.
- 30/70 경계는 관례적 선택이므로 **25/75, 33/67 대안에 대한 민감도 분석을 E5에 포함**한다.

**3단계 — 제외 규칙 (unknown 처리, masked loss로 학습 제외)**

| 조건 | 라벨 |
|------|------|
| 큰턱 또는 몸통이 가려짐·프레임 밖 | `unknown` |
| 배면/측면이 아닌 각도로 단축(foreshortening) 발생 | `unknown` |
| 큰턱이 좌우 비대칭 손상(야외 개체 흔함) | `unknown` |
| 성별이 male이 아님 | `unknown` |

**클래스 수는 3개로 고정한다.** major / intermediate / minor 이며 변경하지 않는다.
`form_head = nn.Linear(in_features, 3)` 그대로다. 아래의 `not_applicable`은 **네 번째 클래스가
아니라 마스킹 값**이며, `unknown`과 동일하게 -1로 인코딩되어 masked loss에서 제외된다.
학습 대상 종을 줄이는 것이지 클래스를 줄이거나 늘리는 것이 아니다.

> 인덱스 순서 주의: `dataset.py`의 `MALE_FORM_CLASSES`는 `["major", "minor", "intermediate", ...]`
> 순이라 **1=minor, 2=intermediate**다. 크기 순서(major > intermediate > minor)와 인덱스 순서가
> 어긋나므로, 순서형(ordinal) 손실이나 혼동행렬 축 정렬을 도입할 때 반드시 재정렬해야 한다.
> `predict.py`는 같은 상수를 import하므로 현재는 일관성 문제가 없다.

**4단계 — 판정 대상 종 제한 (`not_applicable`)**

다음 종은 male_form을 **판정하지 않고** `not_applicable`로 두며, masked loss에서 제외한다.

| 기준 | 해당 종 |
|------|--------|
| 큰턱 다형성이 문헌상 뚜렷하지 않음 | `Platycerus_hongwonpyoi_hongwonpyoi` |
| 라벨 수컷 30장 미만 → 종별 분위수 추정 불가 | `D. consentaneus`(24), `P. astacoides`(20), `Platycerus`(11), `D. hopei`(10), `D. rubrofemoratus`(6) |

즉 **현 시점에서 male_form을 학습 신호로 쓰는 종은 상위 5종**
(`D. titanus`, `P. inclinatus`, `D. rectus`, `L. maculifemoratus`, `Prismognathus_dauricus`)
으로 한정한다. 이들만 종당 96장 이상의 수컷을 확보하고 있다.
이 제한이 Platycerus 11/11 전량 major 같은 인공물을 구조적으로 차단한다.

**5단계 — 신뢰도 보고**
- 계측값 `R`: 제2 평가자 15-20% 재계측 → **ICC(2,1)** 보고
- 파생 3분류: **Cohen's κ** 보고
- 두 지표 모두 논문 Dataset 절에 필수 기재

#### v7 확정 사항

1. **기존 1,173건의 male_form 라벨은 전량 폐기하고 위 규칙으로 재작업**한다.
   재계측 대상은 상위 5종 수컷 1,127장 (기존 form 라벨 보유 1,102장 + form unknown 25장).
2. `configs/label_studio_template.xml`을 **keypoint 계측 방식으로 교체**하고,
   판정 규칙 문서를 `docs/male_form_protocol.md`로 분리해 템플릿에서 링크한다.
3. **v6의 form 관련 수치는 일절 인용하지 않는다**(§3.4 전량 폐기 방침).
4. **E5에서 form 정확도를 보고할 때 다수결 기준선을 반드시 병기**한다.
   기준선 미달이면 그대로 미달로 보고한다.
5. **N2(male form)의 기여 주장은 재실험 결과가 나온 뒤에만 한다.**
   현재 논문 초안에서 N2는 novelty가 아니라 **"규칙 없는 형태 라벨링이 어떻게 실패하는가"의
   방법론적 교훈**으로 서술한다. 이 편이 방어 가능하며, 전문가 영역 분류군 템플릿이라는
   논문 주제(§1.1-5)와도 오히려 잘 맞는다.
6. 서비스 API의 `male_form` 필드는 재라벨링·재학습 완료 시까지 비노출(§2.4).

---

### 4.5 3층 테스트셋 구조 + 오염 방어 [신설]

| 층 | 출처 | 용도 |
|----|------|------|
| **1층** | GBIF/iNaturalist (기존 수집분) | 훈련 + 날짜 필터 통과분에 한해 보조 테스트 |
| **2층** | 네이버 카페 크롤링(공개 게시물, 4기둥 준수) + BeetleDex 기증 | **훈련 전용** |
| **3층** | 네이처링 협약, 기증, 현장 촬영, BioCLIP 2 스냅샷 이후 iNaturalist | **테스트 전용** |

**크롤링 4기둥**: 투명성 / 이미지 비배포 / 라벨 프로토콜 준수 / 훈련 전용 사용.

**오염 방어 (필수)**:
- 테스트셋 전체를 **MD5 + 지각해시(PDQ)** 로 TreeOfLife-200M 대비 중복 제거
- **관찰일 필터** — BioCLIP 2 학습 스냅샷 이후 관찰만 3층에 편입
- BioCLIP 2 자체 관행을 인용하고 방법론 소단락에 명시
- 스크립트: `scripts/check_tol_overlap.py`

**공개 조건 (§11.4 연동)**: 3층은 수집 단계에서 CC-BY 공개 동의를 받는다.
주 결론을 3층에서만 내기로 한 §4.6 결정과 맞물려 "주 결론의 근거 데이터는 전량 공개"가 성립한다.

**3층 수집 목표**: §13의 수집 계획을 따른다. 종당 test **50-80개체**가 기준이며,
§12.1 검정력 감사에 따라 기존의 "종당 20장" 목표는 폐기한다(CI 반폭 ±21.5%p로 판정 불가).

---

### 4.6 v6 test 199장의 처리 [항목 3 확정]

#### 실측 구성

| 구분 | 값 |
|------|----|
| 총 이미지 | 199장 (199 관찰, 1:1) |
| 출처 | iNaturalist 184 / 현장(field) 15 |
| quality_grade | research 179 / casual 4 / needs_id 1 / field 15 |
| 관찰 연도 | 2013-2025 분포, **2025년 79장, 2026년 0장** |

**날짜 컷오프별 잔존량 (현장 15장 항상 포함):**

| 컷오프 | 잔존 | iNat | field | 잔존 종 수 |
|--------|------|------|-------|-----------|
| 2024-01-01 이후 | 105장 | 90 | 15 | 9 |
| **2025-01-01 이후** | **79장** | 64 | 15 | **9** |
| 2025-06-01 이후 | 77장 | 62 | 15 | 9 |
| 2026-01-01 이후 | 0장 | 0 | 0 | 0 |

2025-01-01 컷오프 시 종별 잔존 (현장 15장 포함, 합계 79장):

| 종 | 잔존 |
|----|------|
| `P. inclinatus` | 27 |
| `D. titanus` | 23 |
| `D. rectus` | 11 |
| `L. maculifemoratus` | 10 |
| `Prismognathus_dauricus` | 3 |
| `D. rubrofemoratus` | 2 |
| `D. consentaneus` | 1 |
| `D. hopei` | 1 |
| `P. astacoides` | 1 |
| `Platycerus_hongwonpyoi` | **0 (전멸)** |

#### 판단

**날짜 필터를 통과한 79장을 새 테스트셋으로 쓰는 것은 불가능하다.**
9종만 남고 그중 5종이 3장 이하이며 Platycerus는 전멸한다. 이 표본으로는
종별 정확도도 macro-F1도 의미 있는 추정이 되지 않는다.
동시에 199장 전체를 그대로 3층에 넣는 것도 불가능하다 — 184장이 iNaturalist 연구등급이라
TreeOfLife 계열 학습 데이터와 겹칠 개연성이 높다.

#### v7 확정 사항 — 조건부 편입, 층을 나눠서

**(a) 현장 촬영 15장 → 3층으로 무조건 승격.**
iNaturalist에 게시된 적이 없어 파운데이션 모델 노출 가능성이 구조적으로 없다.
6종에 걸쳐 있다(`D. titanus` 5, `L. maculifemoratus` 4, `Prismognathus` 2, `P. inclinatus` 2,
`D. rectus` 1, `D. rubrofemoratus` 1). 3층의 초기 시드로 사용한다.

**(b) iNaturalist 184장 → 특별 지위 없이 1층 풀로 환원.**
v6 결과를 전량 폐기하기로 했으므로(§3.4) **"v6 수치와 비교"라는 용도 자체가 사라졌다.**
레거시 비교셋 `T0`는 **폐지**한다.
- 이 184장은 1층 학습·검증 풀에 되돌리고, v7 설계로 **처음부터 다시 분할**한다.
- v6의 train/val/test 경계는 보존할 이유가 없다. 라벨 키가 split 독립으로 바뀌었으므로(§4.3)
  재분할해도 성별·형태 라벨이 따라간다.

**(c) 날짜 필터는 3층 편입 심사에만 쓴다.**
1층 내부에서 `T0-recent` 같은 부분집합을 별도 관리하지 않는다.
오염 방어는 §4.5의 해시·날짜 필터로 3층 진입 시점에 일괄 적용한다.

**(d) 논문의 주 결론은 전부 3층(신규 수집)에서만 낸다.**
1층은 학습과 내부 교차검증(§12.1)에만 쓰며, 1층 수치로 일반화를 주장하지 않는다.

**(e) val 198장도 1층 풀로 환원**하고 재분할 대상에 포함한다.

#### 테스트셋 명명 정리

| 이름 | 구성 | 층 | 용도 | 주 결론 근거 |
|------|------|----|------|------------|
| `T3-seed` | v6 test 중 현장 15장 | 3 | 3층 초기 시드 | 가능 |
| `T3` | 네이처링·기증·현장·스냅샷 이후 iNat (신규 수집) | 3 | **주 평가** | **가능** |

---

### 4.7 Data Flywheel

```
사용자 업로드 → 모델 예측 (L1 자동) → 사용자 피드백 (L2: 성별)
              → 관리자 검수 (L3: male_form, §4.4 규칙) → 재학습
```

- BeetleDex 개편 시 **기증 폼 필드 = 본 라벨 스키마와 동일**(종/성별/산지/촬영일/개체 ID)
  → 전처리 없이 유입.
- 수집 우선순위는 §1.3-5 확정 목록: 미확보 6종 + test 기여 14장뿐인 하위 5종.

---

### 4.8 다중 검출·다종 이미지 처리 [신설]

한 사진에 개체가 여러 마리 들어오는 경우의 학습·추론 처리 방침.
v6는 **이미지 1장 = 종 1개**를 암묵 전제로 삼았고, 그 전제가 깨지는 지점을 다루지 않았다.

#### 4.8.1 현재 데이터 실측 — 대부분은 다종이 아니라 검출기 과분할

`generate_segmented.py`는 검출된 박스마다 crop을 저장하고, **모든 crop이 원본 이미지의 클래스
디렉토리를 그대로 상속**한다. 실측 결과:

| 항목 | 값 |
|------|----|
| 원본 이미지 | 1,985장 |
| 단일 crop (원본 파일명 유지) | 1,780장 |
| **2개 이상 검출된 원본** | **205장 (10.3%)** |
| 그 205장이 만든 crop | 514장 |
| `data/final_bbox` 총 파일 수 | 2,294장 (원본 대비 +309) |

split별 다중검출 원본: train 163 / val 21 / test 21.
원본 1장이 만든 crop 수는 최대 9개.

**육안 검증 결과 이들은 여러 마리가 아니라 한 마리를 조각낸 것이다.**
crop 9개가 나온 톱사슴벌레 이미지를 열어 보면 원본은 개체 하나가 화면을 꽉 채운 접사이고,
crop들은 다리·더듬이·딱지날개 조각이다. 다우리아사슴벌레 7-crop 사례도 동일하다.

| 지표 | 값 |
|------|----|
| crop 면적비 중앙값 (원본 대비) | 0.084 |
| 면적비 0.20 미만 crop | 375 / 514 (73%) |
| 부수 박스가 전부 최대 박스에 포함(>=0.8)되는 원본 | 68 / 205 |

confidence 임계값 상향으로는 해결되지 않는다:

| confidence | 여전히 다중검출 | 단일로 수렴 | 정상 이미지가 미검출로 손실 |
|-----------|---------------|-----------|------------------------|
| 0.25 (현재) | 205 | 0 | 2 / 200 |
| 0.40 | 177 | 28 | 4 / 200 |
| 0.50 | 154 | 51 | 5 / 200 |
| 0.60 | 137 | 67 | 6 / 200 |
| 0.70 | 111 | 91 | 8 / 200 |

0.70까지 올려도 111장이 남고 정상 이미지 손실만 늘어난다. **임계값은 해법이 아니다.**

#### 4.8.2 이 결함이 v6 결과에 미친 영향

1. **bbox/seg 학습셋에 조각 이미지 309장이 종 라벨을 달고 섞여 있다.**
   다리 사진이 `Prosopocoilus_inclinatus_inclinatus`로 학습된다.
2. **§3.3의 "full > bbox" 해석에 세 번째 후보 원인이 추가된다.**
   기존 후보는 (a) 패딩 없는 크롭의 스케일 손실, (b) full의 배경 누출이었다.
   여기에 **(c) 조각 노이즈**가 붙는다. E0는 세 원인을 분리해야 한다.
3. **성별·형태 감독 신호가 추가로 손실된다.** `_0`, `_1` 접미사 crop은
   `sex_labels.csv` 키와 매칭되지 않아 전부 `unknown`(-1)이 된다.

#### 4.8.3 확정 방침 — 라벨 단위를 이미지에서 개체로 옮긴다

**(a) 학습 데이터 생성: 원본 1장 = crop 1장을 기본값으로 한다.**
iNaturalist 관찰은 초점 개체가 하나라는 전제가 성립하므로, 검출 박스가 여럿이면
**최고 confidence 박스를 기준으로 삼고, 그 박스와 포함률 0.3 이상으로 겹치는 박스를 합집합으로 병합**한 뒤
20-30% 패딩을 넣어 하나만 저장한다. 패딩은 §3.3의 aspect 축과 동일한 파라미터를 쓴다.
- 병합 후에도 서로 떨어진 큰 박스가 남으면 **자동 저장하지 않고 검수 큐로 보낸다.**
- 검수 큐 대상은 현재 205장. 사람이 "한 마리 / 여러 마리 / 오검출"로 판정한다.

**(b) crop 매니페스트를 도입하고 ImageFolder 디렉토리=라벨 방식을 버린다.**

```
data/labels/crops.csv
  crop_id, source_image, split, box_xyxy, n_individuals,
  species, sex, male_form, review_status
```

디렉토리 구조가 라벨을 결정하는 현재 방식으로는 **한 이미지에서 나온 두 crop에 서로 다른 종을
줄 수 없다.** 매니페스트가 있으면 개체 단위 라벨이 표현된다. `MultiTaskDataset`은 ImageFolder 대신
이 CSV를 읽는다.

**(c) 진짜 다종 이미지의 처리 규칙.**

| 상황 | 처리 |
|------|------|
| 개체 여러 마리, 종이 모두 동일 | 각 개체를 개별 crop으로 저장, 전부 같은 종 라벨 |
| 개체 여러 마리, 종이 서로 다름 | **개체별로 종을 라벨링**해 각각 독립 샘플로 사용 |
| 종을 개체별로 특정할 수 없음 | 이미지 전체를 학습에서 제외 (`review_status=excluded`) |
| `full` 모드 학습 | **다종 이미지는 full 모드에서 제외.** 한 장에 정답이 둘이면 단일 라벨 분류로 표현 불가 |

`full` 모드에서 다종 이미지를 제외하는 것이 핵심이다. 이 처리를 하지 않으면 full 모드만
구조적으로 틀린 라벨을 학습하게 되어 §3.3의 전처리 비교가 불공정해진다.

**(d) split은 원본 이미지 단위로 그룹핑한다.**
같은 사진에서 나온 crop이 train과 test에 갈라지면 누수다. §4.2의 `observation_id` 그룹핑에
crop 단위가 추가되므로, 그룹 키는 `observation_id`를 유지하고 crop은 그 하위에 둔다.

**(e) 추론: 개체별 결과를 그대로 노출한다.**
`pipeline.py`는 이미 crop마다 독립 분류해 리스트로 반환하고, `/api/v1/predict`도 `list[dict]`를
반환하므로 구조는 이미 맞다. 실제 문제는 **운영 설정이 `inference.preprocessing_mode: "full"`
이라 세그멘터를 아예 우회한다**는 점이다(`prediction.py`의 full 분기에서 `crops=[image]`).
- 운영 모드를 검출 기반으로 전환하고, 응답에 `bbox`를 채워 반환한다(현재 `bbox=None` 고정).
- 프론트는 개체별 카드로 표시한다. 한 마리만 검출되면 기존 UI와 동일하게 보인다.
- 검출 0개면 기존 `no_beetle` 경로를 유지한다.

**(f) 검출기 자체를 재학습 대상으로 올린다.**
과분할은 근본적으로 Grounded SAM 자동 어노테이션 품질 문제다. 조각 박스가 정답으로 들어갔을
가능성이 높다. E0 착수 전에 어노테이션을 재검수하고 detector를 재학습한다.
재학습 전까지는 (a)의 병합 규칙으로 방어한다.

#### 4.8.4 미해결로 남기는 것

개체별 종 라벨링은 **사진만으로 두 개체의 종을 각각 확정할 수 있을 때만** 가능하다.
암컷끼리 섞인 사진은 전문가도 사진 동정이 어렵다(§6 Dataset 3.5에 선제 서술).
이 경우는 `excluded`로 두고 무리하게 라벨링하지 않는다.

---

## 5. 논문 지원 스크립트

### 5.1 기존 (v6, 유지)

| 스크립트 | 목적 | 출력 |
|---------|------|------|
| `scripts/analyze_results.py` | MLflow 결과 → 논문용 figure/table 전체 | `experiments/analysis/` |
| `scripts/evaluate_test.py` | test set 평가 (bootstrap CI, sex-stratified, ECE) | JSON + figure |
| `scripts/tsne_features.py` | backbone feature t-SNE / UMAP | 종별/성별 컬러 |
| `scripts/gradcam.py` | Grad-CAM 시각화 (CNN + Transformer 통합) | 종별 3-panel figure |
| `scripts/calibrate.py` | Temperature Scaling (ECE 보정) | reliability diagram + JSON |
| `scripts/simulate_flywheel.py` | Data Flywheel 시뮬레이션 | accuracy recovery curve |
| `scripts/build_ood_centroids.py` | OOD centroid 계산·저장 | centroid npz |
| `scripts/visualize_errors.py` | 오분류 패턴 분석 | `experiments/error_analysis/` |

### 5.2 v7 신규

| 스크립트 | 목적 | 대응 실험 |
|---------|------|----------|
| `scripts/check_tol_overlap.py` | FM 훈련 노출량 표 + MD5/PDQ dedupe | E3, §4.5 |
| `scripts/subsample_curves.py` | 학습곡선 서브샘플링 러너 | E2 |
| `scripts/background_probe.py` | 배경 셔플/중립화 교란 테스트 | E0, §3.3 |
| `scripts/check_split_confounds.py` | 배경 클러스터·촬영자의 split 상관 검사 | §4.2 |
| `scripts/compute_male_form.py` | keypoint 계측 → R 산출 → 종별 분위수 3분류 | §4.4 |
| `src/xai/` | pointing game / energy PG / IoU 지표 모듈 | E4 |
| `scripts/build_crop_manifest.py` | 박스 병합·패딩 후 crop 생성 + `crops.csv` 작성 | §4.8 |
| `scripts/review_multi_detect.py` | 다중검출 205장 검수 큐 생성 | §4.8 |
| `scripts/backfill_licenses.py` | observation_id로 사진 라이선스·출처표기 역조회 백필 | §11.5 |
| `scripts/build_release_manifest.py` | 공개용 재구성 매니페스트 + sha256 생성 | §11.3 |
| `scripts/export_mlflow.py` | MLflow run을 CSV로 export해 영구 보존 | §11.5 |

### 5.3 analyze_results.py 산출물 (유지)

`fig_architecture_comparison_*.png`, `fig_preprocessing_ablation.png`, `fig_heatmap_*.png`,
`fig_multitask_effect.png`, `fig_training_curves.png`, `fig_female_vs_male_accuracy.png`,
`fig_so_vs_mt_female_accuracy.png`, `fig_fewshot_scatter.png`, `fig_lambda_ablation.png`,
`table_full_results`, `table_mt_ablation`, `table_multitask_results`,
`table_significance_test`, `table_prep_significance`, `table_dataset_stats`,
`table_model_efficiency` (각 .csv/.tex, CSV는 utf-8-sig).

---

## 6. 논문 구성 [개정]

> 가제: **"How Much Data, Which Strategy, and What Do Models See?
> A Template for Classifying Expert-Domain Taxa, Exemplified by Korean Stag Beetles"**

```
1. Introduction — 전문가 영역 분류군 문제, FM 시대의 지역 소규모 데이터
2. Related Work — FGVC/곤충, 데이터 효율성·학습곡선, 생물 FM(BioCLIP 1·2), XAI 정합, 이형성 x DL
3. Dataset
   3.1 참조 종목록 16 → 수집 13 → 모델링 10 (§1.3, 롱테일을 출발점으로 서술)
   3.2 3층 구조와 오염 방어 (§4.5)
   3.3 라벨 체계 + κ/ICC (§4.3, §4.4)
   3.4 Specimen-level split과 그 한계 (§4.2 — 관찰당 1장 사실 명시)
   3.5 사진 동정의 한계 (암컷 조합) 선제 서술
4. Method — 공정 비교 프로토콜, 전처리 x aspect, 멀티태스크(마스킹 loss), BioCLIP 트랙, 정합 지표
5. Experiments — E0 → E1 → E2 → E3 → E4 (+E5·E6 분석 절)
6. Results & Discussion — 교차점 권고표, 형질 vs 배경, 이형성 취약성, 희귀종 사각지대,
   템플릿 이식성(주장 수위: worked example + 공개 프로토콜)
7. Conclusion
```

**타겟**: Methods in Ecology and Evolution (primary, v6와 동일). 후보 전체는 §6.6.
상향 조건: 전문가 벤치마크 확보 또는 타 분류군 축소 반복.
E4 단독으로 FGVC 워크숍 프리페이퍼 옵션.

**서술 수위 통제 (필수)**:
- "16종 분류" 금지 → 10-way (§1.3)
- specimen split이 부풀림을 막았다는 주장 금지 (§4.2)
- male_form을 기여로 주장 금지 (재실험 전) (§4.4)
- 1층 수치로 일반화·FM 우열 주장 금지 (§4.6)

### 6.5 적대적 검토 요약

> 공격 목록은 아래 표. **각 방어가 실제로 숫자를 만들어 내는지에 대한 실측 감사는 §12.**
> 상세 시나리오 문서는 `docs/adversarial_review.md` (M0 산출물 — **아직 미작성**)

14개 공격 시나리오 관리:

| # | 공격 | 방어 |
|---|------|------|
| 1 | BioCLIP 압승론 | 교차점/기제 프레임 (E2·E3) |
| 2 | 테스트 오염 | 3중 방어 (해시·날짜·층 분리, §4.5) |
| 3 | 소표본 | 반복 CV, 결론 격하 서술, n<10 종 부록행 |
| 4 | **템플릿 일반화 (최대 급소)** | 문구 통제 + 프로토콜 공개 + 다분류군 반복 옵션 |
| 5 | saliency 신뢰성 | sanity check + deletion/insertion 충실도 (E4) |
| 6 | 선행연구/스쿠핑 | 재검색 2회 + 프리페이퍼 |
| 7 | 크롤링 윤리 | 4기둥 (§4.5) |
| 8 | 셀프 라벨 | κ / ICC 보고 (§4.3) |
| 9 | 사육 개체 혼입 | 소스 ablation (E7) |
| 10 | 이형성 선행연구 구분 | 성판별 모델과 태스크 구분 명문화 (E5) |
| 11 | 임베딩 ≠ 계통 | 한계 명문 (E6b) |
| 12 | 산만함 | 본편 E0-E4로 한정, E5-E7은 분석 절 |
| 13 | **배경 누출 (v7 신설)** | v6 full 우위 재검증 (§3.3, E0) |
| 14 | **종 수 불일치 (v7 신설)** | 16/13/10 3단 어휘 고정 (§1.3) |
| 15 | **라벨 유실 (v7 신설)** | split 독립 키로 교체, 유효 라벨 수 명시 보고 (§4.3) |
| 16 | **조각 이미지 혼입 (v7 신설)** | 박스 병합 + 검수 큐, E0에서 원인 3분리 (§4.8) |
| 17 | **데이터 공개 불가 (v7 신설)** | 매니페스트+임베딩 공개, 3층은 CC-BY로 수집 (§11) |
| 18 | **재현 불가 데이터 의존 (v7 신설)** | E7 소스 ablation을 필수로 격상 (§11.4) |
| 19 | **검정력 부족 (v7 신설)** | 내부 비교는 5×5 CV, 3층은 낙폭 전용 (§12.1) |
| 20 | **롱테일 해법 미시험 (v7 신설)** | E8 신설 (§12.6-C) |
| 21 | **메타데이터 베이스라인 부재 (v7 신설)** | E9 신설, 측정 완료 (§12.6-A) |
| 22 | **난이도 전제 미측정 (v7 신설)** | E10 전문가 베이스라인 (§12.6-B) |

---

### 6.6 투고 저널 후보 [신설]

지표는 JCR 2026(2026-06-17 공개, 2025년 피인용 기준). **투고 직전 JCR에서 재확인할 것** —
아래 수치는 2차 출처 기반이며 일부는 연도 확인이 불완전하다.

#### A군 — 주 후보 (범위 적합 + Q1)

| 저널 | IF | 분면 | 출판사 | 적합성 |
|------|----|------|--------|--------|
| **Ecological Informatics** | 8.5 | Q1 | Elsevier | **범위 최적합.** 이미지 기반 모니터링·데이터 집약 생태학이 명시 범위. IF도 MEE보다 높다 |
| **Methods in Ecology and Evolution** | 5.7 | Q1 | Wiley (BES) | **프레이밍 최적합.** "템플릿·프로토콜·정량 지침"이 MEE의 편집 정체성 그 자체. §11이 이미 BES 정책 기준으로 설계됨 |

두 저널의 선택은 **IF(8.5 vs 5.7) 대 분야 내 평판**의 교환이다.
생태학 방법론 분야에서 MEE의 위상이 더 높지만, 정량 평가에서는 Ecological Informatics가 유리하다.
**권고: MEE를 1순위로 유지하되 Ecological Informatics를 즉시 대체 가능한 2순위로 준비한다.**
두 저널 모두 데이터·코드 공개를 요구하므로 §11 패키지를 그대로 재사용할 수 있어 전환 비용이 낮다.

#### B군 — 조건부 후보 (Q1이지만 재프레이밍 필요)

| 저널 | IF | 분면 | 조건 |
|------|----|------|------|
| **Scientific Data** | 7.2 | Q1 | **방법 논문이 아니라 Data Descriptor.** 데이터셋 자체를 별도 논문으로 내고 DOI를 확보한 뒤 본 논문에서 인용하는 전략. §11 아카이빙 문제를 동시에 해결한다. 단 **공개 가능한 3층 CC-BY 데이터에 한정**되므로 3층 수집 완료가 선행 조건 |
| **Ecological Indicators** | 8.7 | Q1 | 범위가 생태 지표·모니터링 지수. "데이터 빈약 분류군의 모니터링 역량"으로 재프레이밍해야 함. 적합성 중간 |
| **Insect Conservation and Diversity** | 3.3 | Q1 (곤충학) | Wiley/RES. "데이터 빈약 분류군의 보전 모니터링 도구"로 재프레이밍 시 적합. IF는 낮지만 곤충학 내 평판이 견고 |
| **Computers and Electronics in Agriculture** | 10.3 | Q1 | **IF 최고이나 범위 부적합.** 사슴벌레는 농업 해충이 아니다. 임업(forestry)이 범위에 포함되므로 산림 곤충 모니터링으로 틀면 가능하나 desk reject 위험이 크다 |

#### C군 — 대안·후퇴

| 저널 | IF | 비고 |
|------|----|------|
| **Insects** (MDPI) | 3.0 | Q1(곤충학), OA, 심사 빠름. 국내 평가에서 MDPI 할인 가능성 고려 |
| **Ecology and Evolution** | - | Q1, Wiley OA, 선별도 낮음. 최후 후퇴선 |
| **Expert Systems with Applications** | ~9.4 | Q1이나 **알고리즘 신규성**을 요구. 벤치마크·프로토콜 논문은 불리하고, 본 연구의 강점인 생물학적 프레이밍을 평가하지 않는다 |
| **Engineering Applications of AI** | ~9.0 | 위와 동일 |

#### 선택을 좌우하는 제약

1. **데이터 규모가 상한을 정한다.** 현재 10종 1,985장이고 하위 5종은 test 기여가 14장이다.
   CS 계열 고IF 저널은 이 규모에서 거의 확실히 반려된다. 반대로 생태 방법론 저널은
   생물학적 질문의 타당성과 재현성을 본다. **§11에 투자한 것이 그대로 심사 자산이 되는 쪽을 택한다.**
2. **3층 데이터 확보가 저널 선택지를 넓힌다.** 공개 가능한 테스트셋이 생기면
   Scientific Data 병행 투고가 열리고, 본 논문의 "주 결론 근거 데이터 전량 공개" 진술도 성립한다.
3. **E4(attention-검색표 정합)는 분리 가능하다.** FGVC 워크숍 프리페이퍼로 먼저 내고
   본 논문에서 인용하는 경로를 유지한다 (§6).
4. **APC**를 사전에 확인한다. MEE·Ecological Informatics는 하이브리드(구독 기반, OA 선택),
   Scientific Data·Insects는 전면 OA로 APC가 발생한다.

---

## 7. 재시작 체크리스트

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

**VS Code 터미널**: 하단에 "Relaunch Terminal" 알림이 뜨면 클릭 → PowerShell 7(pwsh) 전환.
Git Bash 터미널을 기본으로 쓰면 `wc`, `grep` 등 Unix 명령어가 정상 동작한다.

---

## 8. 개발 로드맵

### Phase 1 — 데이터 파이프라인 (완료)
- iNaturalist 수집 + cleaner + splitter
- 참조 종목록 16종 정립 (`configs/taxonomy.yaml`)
- merger.py: observation_id 보존
- splitter.py: specimen-level split (GroupShuffleSplit)

### Phase 2 — Baseline Classifier (완료)
- ConvNeXt-Tiny baseline, val_acc 77.8%
- FastAPI 서빙 + Next.js 프론트 연결, DB 저장 + 피드백 UI

### Phase 3 — Detector + 전처리 분기 (완료)
- Grounded SAM 자동 어노테이션
- YOLOv8n-seg 학습 (`models/weights/best_detector.pt`)
- 4가지 전처리 모드 구현 및 ablation

### Phase 4 — Full Factorial Sweep + Multi-task (완료, 2026-05-11)

구현 완료: sex/male_form 라벨링(1,985장), Multi-task head(SO/SX/FO/MT, masked loss),
extract_features(), specimen-level split, sweep.py(120), sweep_mt_ablation.py(6),
sweep_lambda.py(11), SX 전 아키텍처(12), analyze_results.py, evaluate_test.py,
tsne_features.py, gradcam.py, calibrate.py, simulate_flywheel.py, visualize_errors.py.

실행 완료: 전체 149 run, best 모델 선정(Swin SO s456),
Grad-CAM 10장, 오분류 분석, t-SNE, Temperature Scaling(T=0.567, ECE 0.37 → 0.22).

**Grad-CAM 소견**: 정분류는 머리·큰턱에 히트맵 집중(형태학적으로 유의미).
오분류 1 — Prosopocoilus가 Dorcus와 혼동(속간 형태 유사성).
오분류 2 — **Prismognathus는 흰 배경 테두리에 집중(배경 편향)** → v7 §3.3 재검증 대상.

**오분류 패턴**: 속 내 혼동(Dorcus spp.)이 다수, `P. inclinatus`가 6개 종으로 분산되어 최다 오분류.
전 패턴이 데이터 부족에 기인하며, 속 내 혼동은 분류학적으로 합리적인 실수다.

#### v6에서 수정된 주요 버그 (기록 보존)

| 파일 | 버그 | 수정 |
|------|------|------|
| `src/detection/generate_segmented.py` | 단일 crop을 `IMG_001_0.jpg`로 저장 → sex_labels 키 불일치로 MT 레이블 전부 -1 | 원본 파일명 보존 |
| `scripts/sweep.py` | `seg_hard` 경로 오기 → 실험 자동 스킵 | `./data/final_seg_hard` |
| `scripts/sweep_mt_ablation.py` | seeds 불일치 [42,43,44] | [42,123,456] 통일 |
| `predict.py` | sex_only/form_only 튜플 언패킹 crash | 모드별 명시적 분기 |
| `scripts/evaluate_test.py` | `data["sex_gt"].any()` — NumPy에서 -1은 truthy | `(data["sex_gt"] >= 0).any()` |
| `scripts/gradcam.py` | CNN 4D 전제 — Transformer는 3D (B,N,C) | 차원 검사 후 patch grid reshape |
| `scripts/analyze_results.py` | Wilcoxon n=3에서 p<0.05 불가 (최소 p=0.25) | Bootstrap 95% CI로 교체 |
| `src/ml/classifier.py` | sex_head 3-class / form_head 4-class — unknown 뉴런 미학습 | 2-class / 3-class로 변경 |
| `train.py` | `compute_class_weight` 클래스 누락 시 shape mismatch | 누락 클래스 weight=1 보완 |
| `scripts/gradcam.py` | Swin channels-last 미처리 | `features[-1]` permute |
| `scripts/calibrate.py` | 음수 T 발생 | `log_T` 파라미터화 |
| 전체 | `torch.load()` weights_only 미지정 / 이모지 cp949 오류 | 명시 / 전체 제거 |

### Phase 5 — 웹서비스 배포 + Flywheel (2026-05-11~, 진행중)

- [x] 웹서비스 배포 (Docker + Contabo VPS + Cloudflare Tunnel) — beetledex.com 운영중 (2026-05-16)
- [ ] Google 로그인 OAuth Redirect URI 등록 → 소셜 로그인 활성화
- [ ] 커뮤니티 데이터 수집 (~4,000장 목표)
- [ ] **수집 우선순위 종 고정** — 미확보 6종 + 하위 5종 (§1.3-5)
- [ ] OOD detection 강화 (미확보 6종 이미지 확보 후 E6c)

### Phase 6 — 서비스 완성 (예정)
- [ ] Cloudflare R2 이미지 저장 (현재: 서버 로컬 `data/uploads/`)
- [ ] GPS 분포 지도 (전체 공개 핀)
- [ ] 관리자 대시보드 고도화, 재학습 자동화 파이프라인

### Phase 7 — 논문 리부트 (M0-M4)

**M0 — 기반 정비**
- [ ] `docs/adversarial_review.md` 작성 (§6.5, 14개 시나리오)
- [ ] `docs/male_form_protocol.md` 작성 (§4.4)
- [ ] 진단 형질 표 v1 (E4·라벨링 공용)
- [ ] 라벨 스키마 컬럼 확장 마이그레이션 (`tier`, `photographer_id`, `background_cluster`)
- [ ] FM 노출량 표 1차 (`check_tol_overlap.py`)
- [ ] 문헌 재검색 1차
- [ ] **T3-seed(현장 15장) 3층 태깅 + 나머지 1층 풀 환원 후 재분할** (§4.6)
- [ ] **라벨 키를 split 독립으로 교체** (§4.3) — 재라벨링보다 먼저. 안 고치면 새 라벨도 33% 유실
- [ ] **crop 매니페스트 `data/labels/crops.csv` 도입** (§4.8)
- [ ] **다중검출 205장 검수** — 한 마리 / 여러 마리 / 오검출 판정 (§4.8)
- [ ] detector 어노테이션 재검수 및 재학습 (§4.8-f)
- [ ] **이미지 라이선스 백필** — `scraper.py` 컬럼 추가 + 기존 2,000건 역조회 (§11.5)
- [ ] **`LICENSE` 파일 추가** (코드/데이터 분리 표기) (§11.5)
- [ ] **MLflow run CSV export**를 저장소에 커밋 (§11.5)
- [ ] `environment.yml` 고정 + 하드웨어·CUDA 기록 (§11.5)
- [ ] 국내 법정보호종 지정 확인 후 좌표 격자화 방침 결정 (§11.5)
- [ ] 네이처링 협약서·기증 폼에 **CC-BY 공개 동의 조항** 반영 (§11.4)

**M1 — 데이터 확충 + E0**
- [ ] 네이처링 협약, 기증 공지, 크롤링 파이프라인(윤리 확인)
- [ ] `check_split_confounds.py` 통과 확인 (§4.2)
- [ ] E0 (전처리 x aspect + 배경 교란)

**M2 — E1·E2 + 2층 라벨링**
- [ ] E1 공정 아키텍처 비교
- [ ] E2 학습곡선 x 전략
- [ ] 2층 라벨링 + κ 보고
- [ ] **male_form 재계측** (상위 5종 수컷 1,127장, §4.4)

**M2 추가**
- [ ] **E8 롱테일 기법 비교** (§12.6-C)
- [ ] **E9 메타데이터 누출 검사** (융합 아님, 배제 근거) (§12.6-A)
- [ ] **E10 전문가 베이스라인** — 전문가 섭외가 리드타임이 길므로 M1에 착수 (§12.6-B)
- [ ] 학습곡선을 5종 고정 과제로 재설계 (§12.2)
- [ ] null 결과 보고 방식 사전 규정 (§12.7)

**M3 — E3·E4·E5·E6**
- [ ] E3 BioCLIP 심층 / E4 attention 정합 / E5 성별·체급 층화 / E6 분류학적 구조
- [ ] **E11 제2 분류군 축소 반복** (하늘소과 또는 호랑나비과, E2·E3만) (§12.6-D)
- [ ] E4 착수 전 진단 형질 **가시성 등급** 부여 (§12.5)
- [ ] 설치 가능한 파이프라인 패키지화 (§12.7)

**M4 — 마무리**
- [ ] E7 조건부 / 원고 / 재검색 2차 / 투고

### Phase 8 — BeetleDex 개편 (논문 틀 확정 후)
- 기증 폼 = 라벨 스키마 동기화 (개체 ID 필수), flywheel 재가동

---

## 9. 기술 스택

| 영역 | 스택 |
|------|------|
| Frontend | Next.js 14, Tailwind CSS, shadcn/ui |
| Backend | FastAPI, SQLAlchemy (async), Alembic |
| Database | PostgreSQL 16 |
| ML | PyTorch 2.6+cu124 |
| Foundation Model | BioCLIP 2 (v7 신규 트랙) |
| OOD | DINOv2-ViT-S/14 centroid 기반 |
| Detector | YOLOv8n-seg (class-agnostic) |
| Auto-annotation | autodistill + Grounded SAM 2 |
| Labeling | Label Studio (v7: keypoint 계측 템플릿) |
| Experiment tracking | MLflow (localhost:5001, SQLite backend) |
| GPU | RTX 4060 Laptop 8GB, CUDA 12.4 |
| Deploy | Docker Compose, Contabo VPS (Singapore), Cloudflare Tunnel |

---

## 10. 주요 파일 구조

```
configs/
  default.yaml                # 하이퍼파라미터 중앙 관리 (min_samples: 10)
  taxonomy.yaml               # 참조 종목록 16종 + 학명/국명 매핑 + 제외 목록
  label_studio_template.xml   # v7에서 keypoint 계측 방식으로 교체 예정

docs/                         # v7 신설
  adversarial_review.md       # 14개 공격 시나리오 (M0 미작성)
  male_form_protocol.md       # male_form 판정 규칙 (M0 미작성)

src/
  ml/
    classifier.py             # build_model, MultiTaskClassifier (SO/SX/FO/MT), extract_features()
    ood_detector.py           # OODDetector (centroid 기반)
    detector.py               # BeetleSegmenter (YOLOv8n-seg)
    manager.py                # ModelManager singleton (API용)
  training/
    dataset.py                # get_dataloaders, MultiTaskDataset
    trainer.py                # ModelTrainer (4개 모드, MLflow logging, masked loss)
  detection/                  # Grounded SAM, annotation_utils, generate_segmented
  preprocessing/
    merger.py                 # observation_id 포함 merged_metadata.csv
    cleaner.py                # 학명 정규화 + data/processed/
    splitter.py               # specimen-level split + min_samples 필터
  xai/                        # v7 신설 — pointing game / IoU 지표
  api/  db/  services/

scripts/
  sweep.py  sweep_mt_ablation.py  sweep_lambda.py
  analyze_results.py  evaluate_test.py  tsne_features.py
  gradcam.py  calibrate.py  simulate_flywheel.py  build_ood_centroids.py
  visualize_errors.py
  check_tol_overlap.py        # v7 신설
  subsample_curves.py         # v7 신설
  background_probe.py         # v7 신설
  check_split_confounds.py    # v7 신설
  compute_male_form.py        # v7 신설

data/
  raw/merged_metadata.csv     # 2,153건 (observation_id, source, observed_on, quality_grade)
  processed/                  # 13종 1,997장
  final/                      # 10종 1,985장 (train 1,588 / val 198 / test 199)
  final_bbox/ final_seg_hard/ final_seg_soft/
  labels/sex_labels.csv       # 1,985행 (local_path, species, split, sex, male_form)

train.py  train_detector.py  predict.py  pipeline.py  main.py
```

---

## 11. 재현성·데이터 공개 패키지 [신설]

투고 시 요구되는 데이터·코드 공개를 **사후 대응이 아니라 설계 제약으로** 앞당겨 반영한다.
이 절의 결론 한 줄: **이미지를 재배포할 수 있다는 가정에 의존하지 않는다.**

### 11.1 대상 저널 요구사항 (MEE / BES)

투고 직전 반드시 최신 author guidelines로 재확인할 것. 확인 시점 기준 요지:

| 항목 | 요구 |
|------|------|
| Data Availability Statement | 필수. 데이터베이스명 + accession number 또는 DOI 명시 |
| 아카이브 형태 | **DOI를 발급하는 영구 저장소 필요** (Zenodo, Dryad 등) |
| GitHub | **단독 아카이브 불가.** GitHub은 가변이므로 Zenodo 스냅샷으로 DOI를 받아야 함 |
| 코드 | 아카이브 대상. MEE는 방법론 저널이라 코드 비중이 특히 큼 |
| 심사 단계 | **투고 시점에 익명 형태로 데이터·코드를 심사자에게 제공** (zip 업로드 또는 Dryad/Zenodo private-for-peer-review) |
| 인용 | 데이터에 DOI가 있으면 참고문헌에 정식 데이터 인용 기재 |

BES 저널 실태 조사(Cooper et al. 2026, MEE)에 따르면 데이터 아카이빙은 97%가 이행하지만
**코드 아카이빙은 35%에 그친다.** 코드까지 갖추면 그 자체가 방어 포인트가 된다.

### 11.2 현재 상태 진단 (실측)

#### (a) 이미지 저작권 — 가장 큰 급소

`scraper.py`는 iNaturalist API 응답에서 `photos[0]`의 URL만 쓰고
**`license_code`, `attribution`, `photo_id`를 저장하지 않는다.** 즉 보유 중인 2,000장의
재배포 가능 여부를 지금은 알 수 없다. 보관 중인 `observation_id`로 API를 역조회한 결과:

| 사진 라이선스 | 장수 | 비율 | 재배포 |
|-------------|------|------|--------|
| `cc-by-nc` | 1,151 | 57.6% | 가능 (출처표기, 비상업) |
| **`ALL_RIGHTS_RESERVED`** | **638** | **31.9%** | **불가** |
| `cc-by-nc-sa` | 103 | 5.2% | 가능 (동일조건 전파) |
| `cc-by` | 66 | 3.3% | 가능 (출처표기) |
| `cc-by-nc-nd` | 34 | 1.7% | **파생물 배포 불가** (crop/seg은 파생물) |
| `cc0` | 6 | 0.3% | 가능 (제한 없음) |

- **약 3분의 1(638장)은 어떤 형태로도 재배포할 수 없다.**
- ND 34장은 원본조차 잘라서 배포할 수 없다. bbox/seg crop이 정확히 파생물이다.
- 재배포 가능한 합계는 1,326장(66.4%)이며 그중 1,254장이 비상업(NC) 조건이다.
- 조회 요청 2,000건 중 **2건은 이미 조회되지 않았다.** iNaturalist 관찰은 삭제·라이선스 변경이
  가능하므로, 시점 스냅샷과 체크섬 없이는 데이터셋이 재현되지 않는다.

#### (b) 좌표·개인정보

| 항목 | 값 |
|------|----|
| 좌표 obscured(iNat이 흐린 값 제공) | 122건 (6.1%) |
| `positional_accuracy` 결측 | 560건 |
| iNat `taxon.threatened` 플래그 | 1,998건 전부 False |
| 고유 관찰자 | 354명 (상위 10인이 36.1%) |

`user_id`(관찰자 로그인명)를 그대로 저장 중이다. 출처표기에는 필요하지만 그 외 용도로는 쓰지 않는다.
iNat의 threatened 플래그가 전부 False라도 **국내 법정보호종 지정은 별개**다.
두점박이사슴벌레를 포함해 국내 지정 현황을 확인하고, 해당 시 좌표를 격자화한다.

#### (c) 코드·실험 기록

| 항목 | 상태 |
|------|------|
| `LICENSE` 파일 | **없음** — 코드 재사용 조건 불명 |
| 의존성 고정 | `requirements.txt`만 존재. CUDA/torch 빌드·conda 환경 미고정 |
| `.gitignore` | `data/`, `models/weights/`, `experiments/`, `mlflow.db` 전부 제외 |
| MLflow 기록 | Phase 4에서 DB를 초기화한 이력. **149개 run이 파일로 보존되지 않음** |
| 라벨 CSV | `sex_labels.csv`의 `split` 컬럼이 현재 분할과 불일치 (§4.3) |

`.gitignore` 자체는 정상이지만, **공개 패키지를 별도로 만들지 않으면 아무것도 남지 않는다.**

### 11.3 확정 방침 — 픽셀이 아니라 매니페스트를 공개한다

공개물을 4계층으로 나누고, 각 계층의 공개 가능성을 독립적으로 확보한다.

**A. 재구성 매니페스트 (주력, 100% 공개)**

```
observation_id, photo_id, photo_url, license_code, attribution,
sha256, species, sex, male_form, R_ratio, split, tier, source
```

제3자가 이 CSV와 다운로드 스크립트만으로 **동일한 데이터셋을 복원**하고,
`sha256`으로 자기가 받은 파일이 우리가 쓴 파일과 같은지 검증한다.
이미지 자체를 배포하지 않으므로 저작권 문제가 발생하지 않는다.
iNaturalist 파생 데이터셋의 표준 관행이며, 라이선스가 바뀌거나 관찰이 삭제돼도
매니페스트에 기록된 시점 정보가 남는다.

**B. 파생 산출물 (100% 공개)**
라벨, keypoint 계측값과 `R` 비율, 종별 분위수 임계값, 백본 임베딩, 혼동행렬,
전 실험 결과 테이블. **전부 우리 저작물이므로 제약 없이 공개한다.**
임베딩을 공개하면 이미지 없이도 상당 부분의 분석이 재현된다.

**C. 코드 + 환경 (Zenodo DOI)**
GitHub 저장소를 Zenodo에 연결해 릴리스마다 DOI를 받는다. 포함:
`environment.yml`(conda export), torch/CUDA 빌드 정보, 하드웨어 사양, 전 seed,
MLflow 결과를 export한 CSV/parquet, 각 figure/table의 재생성 명령.

**D. 모델 가중치 (Zenodo)**
best 모델 + 아키텍처별 대표 가중치.

**E. 이미지 서브셋 (보조, 조건부)**
`cc0` / `cc-by` / `cc-by-nc` / `cc-by-nc-sa` 1,326장에 한해 출처표기를 붙여 배포 가능.
**ND 34장과 ARR 638장은 제외한다.** 단 이 서브셋만으로는 논문 수치가 재현되지 않으므로
어디까지나 A의 보조물이며, 주력 재현 경로는 A다.

### 11.4 층별 공개 전략 (§4.5 3층 구조와 연결)

| 층 | 출처 | 이미지 공개 | 대응 |
|----|------|-----------|------|
| 1층 | iNaturalist | 66.4%만 | 매니페스트 + 서브셋 |
| 2층 | 카페 크롤링 | **불가** (4기둥의 이미지 비배포) | 매니페스트에도 URL 미포함. **E7 소스 ablation으로 "2층 없이도 결론이 유지됨"을 보여 의존성을 끊는다** |
| 3층 | 네이처링·기증·현장 | **전량 공개 목표** | 수집 단계에서 CC-BY 공개 동의를 받는다 |

**2층 처리가 이 설계의 핵심 방어다.** 재현 불가능한 데이터가 결론을 떠받치면 심사에서 무너진다.
E7을 "조건부 P2"가 아니라 **공개 패키지의 필수 구성요소**로 격상한다.

**3층은 처음부터 공개 가능하게 받는다.** 기증 폼과 네이처링 협약서에 CC-BY 조항을 넣으면
**완전히 공개 가능한 테스트셋**을 확보한다. 주 결론을 3층에서만 내기로 한 §4.6 결정과 맞물려,
"주 결론의 근거 데이터는 100% 공개"라는 강한 진술이 가능해진다. 이것이 최선의 디펜스다.

### 11.5 즉시 착수 항목 (M0로 편입)

1. **`scraper.py`에 `photo_id` / `license_code` / `attribution` 컬럼 추가**, 기존 2,000건은
   `observation_id`로 역조회해 백필. (`scripts/backfill_licenses.py`)
2. **`LICENSE` 파일 추가** — 코드는 MIT 또는 GPL, 파생 데이터 산출물은 CC-BY로 분리 표기.
3. **라이선스 정책 [v7 확정]**

   학습과 재배포는 별개의 행위다. 이 구분이 결정의 핵심이다.

   | 행위 | 성격 | 라이선스 적용 |
   |------|------|-------------|
   | **학습** | 픽셀에서 모델 가중치를 유도. 가중치는 이미지의 복제물이 아니다 | 명시적으로 다투어진 바 적고, 생물 FM(BioCLIP 등) 포함 이 분야의 보편적 관행 |
   | **재배포** | 이미지 파일 자체(또는 crop = 파생물)를 공개 | 라이선스가 직접 규율. ARR 은 불가, ND 는 파생물 불가 |

   **확정 사항**
   - **ARR 640장: 학습에 사용한다. 재배포하지 않는다.**
     제외하면 전체의 32%가 날아가 데이터셋이 무너진다. 대가가 이익에 비해 지나치게 크다.
   - **ND 34장: 학습에서도 제외한다.**
     전체의 1.7%라 잃는 것이 없고, crop·seg 가 명백한 파생물이라 다툼의 여지가 있는
     유일한 범주를 통째로 제거할 수 있다. 값싸게 논쟁 하나를 없애는 선택이다.
   - **어느 쪽도 이미지를 재배포하지 않는다.** 공개는 §11.3 의 매니페스트 방식으로만 한다.
   - 논문 Dataset 절에 이 정책과 각 범주의 장수를 명시한다.

   구현: `cleaner.py` 가 `photo_license == "cc-by-nc-nd"` 인 이미지를 제외한다.
4. **MLflow 결과를 파일로 export** — DB 초기화로 소실되지 않도록 run 단위 CSV를 저장소에 커밋.
5. **`environment.yml` 고정** + 하드웨어·CUDA 버전 기록.
6. **`README`에 재현 절차** — 매니페스트 다운로드부터 figure 재생성까지 순서대로.
7. **국내 법정보호종 지정 여부 확인** 후 해당 종 좌표 격자화 방침 결정.
8. **관찰자 ID 해시화** — §4.2의 촬영자 상관 통제에는 해시 ID를 쓰고, 원문 로그인명은 출처표기에만 사용.

### 11.6 심사 단계 대응

- 투고 시 Zenodo 또는 Dryad의 **private-for-peer-review 링크**를 준비한다. 익명성이 요구되므로
  GitHub 링크를 그대로 노출하지 않는다.
- 데이터·코드 README를 심사자가 30분 안에 재현할 수 있는 수준으로 쓴다.
  BES 조사에서 아카이브물의 15%가 열리지 않거나 찾을 수 없었다는 점을 역이용한다.
- Data Availability Statement 초안에 **공개할 수 없는 부분(2층, ARR 이미지)과 그 이유,
  그리고 대체 재현 경로(매니페스트·임베딩)를 명시**한다. 숨기지 않고 먼저 밝히는 편이 강하다.

---

## 12. 최종 적대적 검토 — 방법론 감사 [신설]

§6.5가 "어떤 공격이 오는가"의 목록이라면, 이 절은 **"현재 계획이 그 공격을 실제로 막을
숫자를 만들어 내는가"**를 실측으로 감사한 결과다. 요약: **막지 못하는 곳이 세 군데 있고,
빠진 실험이 네 개 있다.**

### 12.1 검정력 감사 — E1은 설계상 결론이 나올 수 없다

test n=199에서 두 모델의 차이를 검출할 검정력 (McNemar, 불일치율 20% 가정):

| test n | 3%p 차이 | 5%p 차이 | 10%p 차이 |
|--------|---------|---------|----------|
| **199 (현재)** | 15.5% | **35.1%** | 88.4% |
| 400 | 26.8% | 60.9% | 99.4% |
| 800 | 47.5% | 88.5% | 100% |
| 1,600 | 76.5% | 99.4% | 100% |

정확도 95% CI 반폭 (p=0.73 기준):

| test n | CI 반폭 | 종당 test n | 종별 recall CI 반폭 |
|--------|--------|-----------|-------------------|
| 199 | ±6.2%p | 5 | ±42.9%p |
| 400 | ±4.4%p | 10 | ±30.4%p |
| 842 | ±3.0%p | 20 | ±21.5%p |
| 1,893 | ±2.0%p | 50 | ±13.6%p |

**해석:**
1. v6에서 모델 간 CI가 전부 겹친 것은 우연이 아니라 **구조적 결과**였다.
   5%p 차이를 잡을 검정력이 35%뿐이다. 표본 설계를 바꾸지 않으면 v7에서도 같은 일이 반복된다.
2. **§4.5의 3층 목표치 "종당 20장"(총 200장)은 여전히 부족하다.** 같은 자리다.
   아키텍처 비교로 결론을 내려면 test 800장 이상이 필요하다.
3. 종별 정확도를 논문 본문에 실으려면 **종당 50장**은 있어야 한다(±13.6%p). 20장은 ±21.5%p로
   사실상 아무 말도 못 한다.

**조치 (택일, 권고는 (b)+(c)):**
- (a) 3층 목표를 종당 50-80장으로 상향 — 수집 부담이 크다
- (b) **내부 비교(E0·E1)는 반복 층화 CV(5×5)로만 수행**하고, 3층은 **일반화 낙폭 측정 전용**으로 쓴다.
  CV는 전체 1,985장을 5회 사용하므로 검정력이 확보된다. 층이 다른 두 질문에 다른 도구를 쓰는 것이다.
- (c) **"이 데이터 규모에서 아키텍처 선택은 유의한 차이를 만들지 않는다"를 결론으로 삼는다.**
  데이터 효율성 논문에서 이것은 실패가 아니라 **본론에 부합하는 발견**이다.
  "모델을 바꾸지 말고 데이터를 늘려라"는 실무 지침이 된다.

### 12.2 E2 학습곡선 — 서브샘플링이 과제를 바꿔버린다

train 서브샘플링 시 종당 장수:

| 종 | 1/8 | 1/4 | 1/2 | 전체 |
|----|-----|-----|-----|------|
| `D. titanus` | 74 | 149 | 298 | 596 |
| `P. inclinatus` | 46 | 93 | 186 | 372 |
| `L. maculifemoratus` | 28 | 56 | 112 | 225 |
| `D. rectus` | 26 | 53 | 106 | 213 |
| `Prismognathus` | 13 | 26 | 52 | 104 |
| 하위 5종 | **1-3** | 2-6 | 5-12 | 11-25 |

**문제**: 1/8 지점에서 하위 5종은 종당 1-3장이 되어 사실상 소멸한다. 즉 곡선의 왼쪽 끝은
10-way가 아니라 "5-way + 잡음"이다. **"종당 데이터량"과 "유효 클래스 수"가 뒤섞여
곡선의 기울기를 해석할 수 없다.** 논문의 핵심 산출물인 "종당 N장 권고표"가 이 결함 위에 서 있다.

**조치**: 학습곡선은 **클래스 집합을 고정한 5종 과제**로 수행한다(각 100장 이상 보유).
하위 5종은 곡선에 섞지 말고 **few-shot 구간으로 분리 보고**한다.
BioCLIP zero/few-shot 트랙이 정확히 이 구간을 담당하므로 서사도 깔끔해진다.

### 12.3 E5 성별 층화 — 암컷 표본이 4종에만 있다

test+val 암컷 수: `D. titanus` 52, `P. inclinatus` 36, `L. maculifemoratus` 22, `D. rectus` 13,
`Prismognathus` 5, `P. astacoides` 2, `D. hopei` 2, `D. rubrofemoratus` 2,
`D. consentaneus` 0, `Platycerus` 0. **test만 놓으면 암컷 총 64장.**

v6의 N1 핵심 증거였던 `fig_so_vs_mt_female_accuracy`(종별 암컷 정확도)는
**4종에서만 계산 가능**하고, 종당 15장 내외에서 CI 반폭이 ±25%p다.

**조치**: 종별 암컷 정확도를 본문 주장에서 내리고, **암수 통합(pooled) 정확도 차이**만 주장한다.
이쪽은 64 대 135로 검정력이 있다. 종별은 부록 탐색 결과로 격하한다.
3층 수집에서 **암컷을 명시적 목표로 지정**한다(현재 전체 성비 male 60.5% / female 36.4%).

### 12.4 E6b 임베딩 덴드로그램 — 통계가 성립하지 않는다

모델링 10종의 속 구성: `Dorcus` 5종, `Prosopocoilus` 2종, `Lucanus`·`Platycerus`·`Prismognathus` 각 1종.
**5개 속 중 3개가 단일종**이다. tip이 10개인 덴드로그램에서 cophenetic 상관을 계산하면
분산이 지나치게 커서 어떤 값이 나와도 해석이 불가능하다.

**조치**: E6b는 **통계량 없이 정성 figure로만** 제시하거나 삭제한다.
E6a(속 내/간 혼동 구조)는 `Dorcus` 5종에 근거가 있으므로 유지한다.

### 12.5 E4 attention-검색표 정합 — 가장 참신하고 가장 위험하다

세 가지 실패 경로:
1. **주석 비용** — 종당 30장씩만 잡아도 300장에 진단 형질 영역을 그려야 한다.
2. **해상도 한계** — Lucanidae 검색표의 진단 형질은 큰턱 내치(內齒) 위치, 전흉배판 측연 형태,
   점각(punctation), 미모(setae) 같은 것들이다. **iNaturalist 사진과 448px 입력에서
   상당수가 물리적으로 보이지 않는다.**
3. **saliency 신뢰성** — 이미 §6.5-⑤로 관리 중.

**조치**: (2)를 약점이 아니라 **결과로 선언한다.** "사진 기반 동정의 형질 가시성 한계"는
그 자체로 보고 가치가 있고, 논문의 전문가 영역 분류군 프레이밍과 정확히 맞는다.
사전에 형질별 **가시성 등급(항상 보임 / 각도 의존 / 접사 필요)**을 매기고,
가시 형질에 한해 정합 지표를 계산한다. 비가시 형질은 "측정 불가"로 보고한다.

### 12.6 빠진 실험 네 개

#### (A) 메타데이터 베이스라인 — 이미 측정했고, 반드시 넣어야 한다

좌표와 날짜만으로 종을 예측한 결과 (동일 split, HistGradientBoosting):

| 모델 | test acc | macro F1 |
|------|---------|---------|
| 최빈 클래스 | 33.7% | 5.0% |
| 좌표만 | 44.2% | 22.4% |
| 날짜만 | 32.7% | 19.1% |
| **좌표+날짜** | **47.2%** | **23.9%** |
| 이미지 모델 | v7 재학습 후 측정 | — |

**생태 저널 심사자가 가장 먼저 물을 질문이고 현재 계획에 없다.**

다만 이 수치의 용도는 **융합이 아니라 배제의 근거**다(§3.3).
좌표만으로 47.2%가 나온다는 것은 **위치가 종 정체성을 강하게 누설한다**는 뜻이고,
사육 개체 사진에서는 그 좌표가 서식지가 아니라 사용자의 집을 가리킨다.
따라서 위치를 모델에 넣으면 배포 시점에 체계적으로 틀리는 지름길을 심는 것이 된다.

논문에서의 역할은 세 가지다.
- "이미지 모델이 위치·날짜 기준선 대비 얼마를 더하는가"의 기준점
- **위치를 배제한 설계 결정의 정당화**
- 이미지 모델의 예측이 좌표와 상관되면 **배경을 통한 간접 위치 이용**을 의심하는 진단 도구

**융합 모델은 만들지 않는다.** 정확도는 오르겠지만 배포 조건에서 무너지는 종류의 상승이다.

#### (B) 전문가 베이스라인 — 논문의 전제가 아직 측정되지 않았다

논문 제목이 "expert-domain taxa"인데 **이 분류군이 실제로 어려운지를 측정한 적이 없다.**
모델 정확도가 몇이 나오든 그것이 좋은 값인지 판단할 기준점이 없다.

**조치**: 곤충 전문가 2-3인에게 test 이미지 100장을 동정하게 한다.
비용이 매우 낮고 효과가 크다.
- 전문가도 어려워하면 → 논문의 전제가 **증명**된다(현재는 주장일 뿐)
- 모델이 전문가에 근접하면 → 강력한 결과
- 전문가 간 불일치가 크면 → 라벨 상한(label ceiling)을 정량화하게 되고,
  "모델 성능이 라벨 노이즈 상한에 근접했다"는 방어가 가능해진다

#### (C) 롱테일 기법 비교 — 가장 큰 내용 구멍

논문 전체가 **롱테일·데이터 빈약**을 문제로 걸어놓고, **롱테일 기법을 하나도 시험하지 않는다.**
현재 대응은 `compute_class_weight` 하나뿐이다. 심사자는 반드시 묻는다.
"문제를 진단해 놓고 표준 해법은 왜 안 써봤는가."

**조치**: 클래스 불균형 전략을 1급 실험 축으로 추가한다.
class-balanced loss, logit adjustment, LDAM, 재샘플링, 2단계 학습(decoupled) 비교.
GPU 비용이 낮고(백본 고정 가능) 롱테일 서사에 직결된다.

#### (D) 제2 분류군 반복 — 템플릿 주장의 유일한 실질 방어

§6.5-④가 스스로 "최대 급소"로 지목한 항목이다. 현재 방어는 "문구 통제 + 프로토콜 공개"뿐인데,
**n=1 분류군으로 템플릿을 주장하는 구조 자체는 그대로다.**

**조치**: `scraper.py`의 `taxon_id`만 바꾸면 제2 분류군 수집이 즉시 가능하다.
하늘소과(Cerambycidae)나 호랑나비과(Papilionidae) 같은 한국 분류군을 동일 프로토콜로 축소 반복한다.
전체 실험을 다 돌릴 필요 없이 **E2(학습곡선)와 E3(BioCLIP 노출량)만 반복해도**
"worked example"이 "replicated protocol"로 바뀐다. **투입 대비 효과가 가장 큰 추가 항목이다.**

### 12.7 그밖의 보완

| 항목 | 내용 |
|------|------|
| 단순 베이스라인 | 동결 ImageNet 특징 + kNN / linear probe. 값싸고 전체 수치에 맥락을 준다 |
| 선택적 예측 | Temperature Scaling까지 했으니 risk-coverage 곡선까지 간다. "커버리지 80%에서 정확도 X%"가 서비스에 직접 쓰이는 수치 |
| 종 라벨 신뢰도 | iNaturalist research-grade는 **커뮤니티 합의이지 전문가 검증이 아니다.** 논문에 명시하고, (B)의 전문가 검토로 표본 검증한다 |
| null 결과 사전 규정 | **아래 12.9에서 확정** |
| 실질 산출물 | MEE는 방법론지다. 설치 가능한 파이프라인 패키지를 내지 않으면 "템플릿"은 산문에 그친다. §11 코드 공개와 묶어 CLI/라이브러리 형태로 정리한다 |

### 12.9 E5 null 결과 사전 규정 [v7 확정]

**null 이란 무엇인가**: E5는 성별·형태 보조 감독(multi-task)이 **종 분류를 개선하는가**를 묻는다.
null 은 "개선하지 않는다"가 아니라 **"개선한다고 말할 만한 증거가 없다"**는 뜻이다.
효과가 0이라는 증명이 아니라, 이 표본으로는 구분되지 않는다는 진술이다.

v6에서 이미 그 조짐이 있었다. 성별 보조(SX)가 val 에서는 올랐는데 test 에서 역전됐고,
form head 는 다수결 비율에도 못 미쳤다. 다만 그 수치들은 라벨 32.7% 유실 상태에서 나온 것이라
근거로 쓰지 않는다(§3.4). v7에서 제대로 다시 측정한다.

**사후에 정하면 p-hacking 이 된다.** 결과를 보고 나서 "이건 성공" 또는 "이건 예외"를
고르면 어떤 결과든 원하는 결론으로 끌 수 있다. 그래서 지금 고정한다.

| 항목 | 사전 규정 |
|------|----------|
| 주 지표 | **macro-F1** (accuracy 병기). 롱테일이므로 macro 가 주 지표다 |
| 비교 | MT vs SO, SX vs SO, FO vs SO. 각각 독립 비교 |
| 검정 | 5×5 반복 층화 CV 의 **폴드 쌍대 비교** + 부트스트랩 CI |
| 다중비교 | 3개 비교에 Holm 보정 |
| 판정 | 효과크기와 CI를 **부호에 관계없이 그대로 보고**한다 |

**보고 방식** (결과가 어느 쪽이든 이 형식을 지킨다):
- CI 가 0을 포함하면 → "검출 가능한 효과 없음"으로 보고하고 **기여로 주장하지 않는다.**
  동시에 **그 표본에서 검출 가능했던 최소 효과크기**를 함께 밝힌다.
  "효과가 없다"와 "작은 효과를 잡을 힘이 없었다"는 다르므로 구분해서 쓴다.
- CI 가 0을 배제하고 양수면 → 효과크기와 함께 보고한다. 유의성만 말하지 않는다.
- CI 가 0을 배제하고 음수면 → **보조 감독이 해가 된다는 결과로 그대로 보고한다.**
  숨기거나 부록으로 내리지 않는다.

**null 이어도 논문은 성립한다.** 이형성이 심한 분류군에서 보조 감독이 듣지 않는다는 것은
그 자체로 보고 가치가 있고, "전문가 영역 분류군 템플릿"이라는 주제(§1.1-5)와도 맞는다.
다만 그렇게 쓰려면 **검정력이 충분해야** 한다. §12.1의 표본 설계가 전제 조건이다.

동일 규정을 **E1 아키텍처 비교와 E8 롱테일 기법 비교에도 적용**한다.

---

### 12.8 우선순위

| 순위 | 항목 | 근거 |
|------|------|------|
| 1 | **(C) 롱테일 기법 비교** | 가장 큰 내용 구멍. 심사에서 확실히 지적됨 |
| 2 | **(A) 메타데이터 누출 검사** | 이미 측정 완료. 위치 배제의 정량 근거 |
| 3 | **(B) 전문가 베이스라인** | 논문 전제의 증명. 비용 최저, 효과 최고 |
| 4 | **12.1(b) 검정력 재설계** | 결론을 낼 수 있는 실험으로 만드는 전제 조건 |
| 5 | **12.2 학습곡선 클래스 고정** | 핵심 산출물의 해석 가능성 |
| 6 | **(D) 제2 분류군 반복** | 템플릿 주장의 유일한 실질 방어 |
| 7 | 12.3 / 12.4 / 12.5 서술 수위 조정 | 과잉 주장 제거 |

---

## 13. 데이터 수집 계획 [신설]

§12.1 검정력 감사에서 역산한 수집 목표. **v6 보유량이나 v6 성능과 무관하게
통계 요건에서만 도출**했다.

### 13.1 단위는 사진이 아니라 개체다

가장 중요한 원칙이다. **split은 개체(observation) 단위로 나뉜다.**
한 마리를 열 각도로 찍으면 사진은 10장이지만 **통계적 단위는 1개체**다.
열 장 전부 같은 split에 들어가므로 test 표본 수도, CI 폭도, 검정력도 전혀 개선되지 않는다.

현재 데이터가 정확히 이 함정을 피해 있다. 1,985장이 1,985개체에 1대 1로 대응한다(§4.2).
현장 촬영을 시작하면 이 성질이 깨지므로 **처음부터 개체 ID를 부여**해야 한다.

| 목적 | 무엇을 늘려야 하나 |
|------|------------------|
| test 검정력, CI 축소 | **개체 수** (사진 수는 무관) |
| 학습 다양성, 배경 불변성 | 개체당 사진 수 (각도·배경 변화) |

**권장**: 개체당 3-5장, 각도와 배경을 바꿔 촬영. 단 **목표 수치는 항상 개체 수로 센다.**

### 13.2 3단계 목표 (종당 개체 수)

| 단계 | train | val | test | 종당 계 | 10종 합계 |
|------|------|-----|------|--------|----------|
| **최소** | 100 | 25 | 50 | **175** | 1,750 |
| **권장** | 250 | 40 | 80 | **370** | 3,700 |
| 충분 | 400 | 50 | 100 | 550 | 5,500 |

근거가 되는 통계량:

| 종당 test 개체 | 종별 recall CI 반폭 | 10종 합계 | 전체 정확도 CI | 5%p 차이 검출력 |
|--------------|-------------------|----------|--------------|---------------|
| 30 | ±16.4%p | 300 | ±4.9%p | 49.1% |
| **50** | **±12.7%p** | 500 | ±3.8%p | 70.5% |
| **80** | **±10.0%p** | 800 | ±3.0%p | 88.5% |
| 100 | ±9.0%p | 1,000 | ±2.7%p | 94.2% |

- **종당 test 50개체가 하한선.** 이하로는 종별 정확도를 본문에 실을 수 없다.
- **종당 test 80개체면 아키텍처 비교까지 성립한다**(검출력 88.5%).
- train 100은 전이학습 기준으로 종이 서로 뚜렷할 때 쓸 만한 성능이 나오기 시작하는 지점이며,
  250-400 구간에서 수확체감이 온다. **정확한 변곡점을 찾는 것이 E2의 목적**이므로
  이 수치는 목표치이지 결론이 아니다.

### 13.3 종별 부족분 (개체 수)

| 국명 | 현재 | 최소(175)까지 | 권장(370)까지 | 등급 |
|------|-----|-------------|-------------|------|
| 넓적사슴벌레 | 729 | 0 | 0 | A |
| 톱사슴벌레 | 483 | 0 | 0 | A |
| 사슴벌레 | 277 | 0 | 93 | A |
| 애사슴벌레 | 262 | 0 | 108 | A |
| 다우리아사슴벌레 | 126 | **49** | 244 | B |
| 두점박이사슴벌레 | 35 | **140** | 335 | B |
| 참넓적사슴벌레 | 28 | **147** | 342 | B |
| 왕사슴벌레 | 16 | **159** | 354 | B |
| 홍다리사슴벌레 | 15 | **160** | 355 | B |
| 원표애보라사슴벌레 | 14 | **161** | 356 | B |
| 길쭉꼬마사슴벌레 | 5 | 170 | 365 | C |
| 꼬마넓적사슴벌레 | 5 | 170 | 365 | C |
| 뿔꼬마사슴벌레 | 2 | 173 | 368 | C |
| 털보왕사슴벌레 | 0 | 175 | 370 | C |
| 엷은털왕사슴벌레 | 0 | 175 | 370 | C |
| 큰꼬마사슴벌레 | 0 | 175 | 370 | C |

**A등급 4종**: 최소 요건 충족. 추가 수집의 목적은 양이 아니라 **3층 테스트 품질**(§13.4)과
**성비·배경 다양성**이다.

**B등급 6종 — 여기가 실제 목표다. 합계 816개체.**
이 816개체를 채우면 **모델링 대상 10종 전부가 최소 요건을 충족**하고,
종별 정확도를 본문에 실을 수 있게 된다. 논문의 병목이 정확히 여기다.

**C등급 6종**: 야외 채집으로 채우는 것은 비현실적이다. **폐쇄집합 클래스로 넣지 않는다.**
대신 §1.3-4와 §2.2에 따라 **개방집합(OOD) 평가 대상**으로 쓴다.
이 용도로는 **종당 10-30개체면 충분**하므로 목표가 훨씬 낮아진다.

### 13.4 촬영 프로토콜 — 배경 불변성을 데이터로 만든다

§3.3에서 배경을 신호로 쓰지 않기로 했으므로, **수집 단계에서 배경을 통제 변수로 설계**한다.
이것이 현장 촬영을 단순 수집이 아니라 실험으로 만든다.

| 항목 | 지침 |
|------|------|
| **개체당 배경 교차** | 같은 개체를 **서로 다른 배경 2-3곳**에서 촬영 (자연 기질 / 손·중립 배경 / 실내) |
| **실내·사육 사진 필수** | 배포 조건과 동일. 책상, 사육 통, 흰 종이 위. **오염이 아니라 필수 도메인** |
| **한 종을 한 장소에서 몰아 찍지 않기** | 종과 배경이 상관되면 §4.2의 교란이 재생산된다 |
| **개체 ID** | `{날짜}_{장소코드}_{일련번호}`. 같은 개체의 모든 사진에 동일 ID |
| **촬영자 ID** | 촬영자 상관 통제용(§4.2). 해시로 저장 |
| **각도** | 배면(dorsal) 필수. male_form 계측(§4.4)을 위해 **큰턱과 몸통이 한 프레임에 온전히** 들어와야 함 |

**성비**: 현재 male 60.5% / female 36.4%. 성별 층화 평가(E5)에 **종당 test 암컷 15개체 이상**이
필요하므로 **암컷 비율 40%를 목표**로 한다. 암컷은 동정이 어렵고 덜 눈에 띄어 자연히 과소표집되므로
의식적으로 채워야 한다.

**male_form용 크기 분포**: §4.4의 종별 분위수 규칙은 **종당 라벨된 수컷 30개체 이상**을 요구한다.
더 중요한 것은 **크기 스펙트럼**이다. 큰 개체만 찍으면 major에 쏠려 분위수가 무너진다.
작은 수컷(minor)을 의도적으로 함께 촬영한다.

**법적 주의**: 국내 법정보호종 지정 여부를 사전에 확인한다(§11.5). 촬영은 문제되지 않으나
채집·취급에는 허가가 필요할 수 있고, 보호종의 정밀 좌표는 공개 시 격자화한다.

### 13.5 수집 경로 배분

| 경로 | 적합 종 | 비고 |
|------|--------|------|
| **사육 개체 기증·협조** | 넓적, 톱, 애, 왕, 두점박이 | 애완 사육이 활발한 종. **현장 답사보다 훨씬 효율적**이며 실내 배경 도메인을 동시에 확보 |
| **현장 답사** | 다우리아, 원표애보라, 홍다리, 참넓적 | 사육 유통이 적어 직접 촬영이 필요. 발생 시기·지역을 조사해 계획 |
| **네이처링·커뮤니티** | 전 종 | 3층 테스트 전용. **CC-BY 공개 동의 필수**(§11.4) |
| **iNaturalist 추가** | 전 종 | BioCLIP 스냅샷 이후 관찰만 3층 편입(§4.5) |

**우선순위**: B등급 6종 816개체가 1순위다. 그중에서도 **사육 유통이 있는 왕사슴벌레와
두점박이사슴벌레(299개체)는 기증 경로로 먼저 해결**하고, 현장 답사 역량은
다우리아·원표애보라·홍다리·참넓적(517개체)에 집중하는 편이 효율적이다.

### 13.6 중간 점검 기준

수집 도중 다음을 주기적으로 확인한다.

- [ ] 개체 수 기준으로 세고 있는가 (사진 수로 세면 목표를 잘못 판단한다)
- [ ] 종별 암컷 비율이 35% 이상인가
- [ ] 종별 수컷의 크기 분포가 한쪽으로 쏠리지 않았는가
- [ ] 같은 종의 개체들이 특정 장소·배경에 몰려 있지 않은가
- [ ] 실내·사육 배경 사진이 확보되고 있는가
- [ ] 개체 ID와 촬영자 ID가 빠짐없이 기록되는가
