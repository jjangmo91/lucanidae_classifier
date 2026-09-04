# 한국 사슴벌레 AI 도감 — 설계 문서 v7 (병합 초안)

> 작성: 2026-09-04 · 기준: v6 (2026-05-10) + 논문 리부트 설계 v0.2 병합
> 사용법: v6의 시스템/서비스 파트(§2, §4.1–4.2, §7–10)는 유지. 아래는 변경·신설 섹션의 전문과 유지 섹션의 지시.
> 표기: [유지] v6 그대로 / [개정] 교체 / [신설] 추가

---

## 1. 프로젝트 개요 [개정]

### 1.1 목표 — 유지하되 5번 항목 확장
1–4. (v6 유지: 종 분류 / 성별·형태 / 미지·외래종 / 커뮤니티 서비스)
5. **연구 논문** — "전문가 영역 분류군(expert-domain taxa)의 분류기 구축 템플릿"으로 확장.
   어떤 단일 전문가 집단도 모든 생물군을 동정할 수 없다. 데이터 빈약·롱테일·극단 이형성 분류군의
   대표 사례(한국 Lucanidae)로 **무엇을(전략), 얼마나(데이터), 왜(형질)** 준비해야 하는지의
   재현 가능한 정량 지침을 제공한다.

### 1.2 연구 노벨티 — v6 N1–N5 유지 + 재배치 + 신설
| v6 | v7 위치 | 변경 |
|---|---|---|
| N1 이형성 멀티태스크 | **E5** | 유지·격상. "성판별 모델"(기존재)과의 태스크 구분 명문화: 우리는 종 분류의 이형성 취약성을 층화 평가. 성별 라벨 1,985장 자산 그대로 활용 |
| N2 male form | **E5** | 유지. major/minor/intermediate 판정 규칙 문서화 필요(§5) |
| N3 전처리 비교 | **E0** | 개정. 4모드에 **종횡비 보존 pad vs squish 축 추가**. v6 결과(full 우위)는 배경 편향 확인으로 재검증 대상(§3.4 재해석) |
| N4 희귀종/few-shot | **E2·E3** | 확장. flywheel 시뮬레이션 → **학습곡선 × 전략(BioCLIP zero/few-shot/FT) 교차점 분석**으로 격상 |
| N5 open-set | **E6c** | 유지. ood_detector(DINOv2) 기존 구현 활용, 속 수준 후퇴와 통합 |
| — | **E3 [신설]** | 생물 파운데이션 모델 축(BioCLIP 2). 2026년 투고 필수 베이스라인 |
| — | **E4 [신설]** | attention–검색표 진단 형질 정량 정합 (v6 §5.7 정성 Grad-CAM의 격상) |
| — | **E6a·b [신설]** | 오류·임베딩의 분류학적 구조 (v6 §5.6 t-SNE의 격상, 기술 분석 한정) |

**BioCLIP 2와의 차별 지점**: BioCLIP 2는 스스로 롱테일 편향을 인정. 본 연구는 그 사각지대(지역 희귀종·극단 이형성·시민과학 소규모 데이터)의 실전 지침 + 커뮤니티 flywheel(BeetleDex)이라는 지역 데이터 생산 루프를 다룬다.

---

## 2. 시스템 아키텍처 [유지 — v6 §2 전체]

---

## 3. ML 파이프라인 [개정]

### 3.1 분류기 모드 [유지 — SO/SX/FO/MT]

### 3.2 아키텍처 비교 — 공정성 프로토콜 [개정]
v6 문제점: (a) ViT만 448→224 리사이즈(ViTWrapper) → 입력 해상도 불공정, (b) 파라미터 비매칭(EffNet-B3 12M vs Swin-T 28M), (c) 계열 혼합에 통제 없음.
v7 프로토콜:
- **입력 해상도 통일**: 전 모델 384 또는 448 native (ViT는 384 native 변형 사용, interpolate 금지)
- **파라미터 밴드 매칭**: 계열당 대표 1개, 20–30M 밴드 (예: ConvNeXt-T 28M / Swin-T 28M / ViT-S+해상도보정) + 동일 레시피 (Bai et al. 2021 방식)
- DINOv2: SO 전용 유지(v6 SX 붕괴 소견 반영), LR 1e-5 별도 — 논문에는 self-supervised 한계 소견으로 서술
- **BioCLIP 2 트랙 추가**: zero-shot / few-shot(kNN) / linear probe / full FT
- v6 137개 sweep 결과: 폐기하지 않고 "예비 실험"으로 보존 — 공정성 결함을 스스로 지적하고 v7 재실험과 대비하는 서사로 활용 (리뷰어 선제 방어)

### 3.3 전처리 모드 [개정]
- 기존 4모드(full/bbox/seg_hard/seg_soft) + **aspect 축(pad vs squish) 교차** → 실질 2×4
- **v6 결과 재해석 의무**: "full 15–30%p 우위"는 Prismognathus 흰 트레이 배경 편향 확인과 동시 보고됨 → full 우위가 맥락 정보인지 배경 누출(shortcut)인지 미분리. v7 검증:
  (a) 배경 교란 테스트(배경 셔플/중립화 후 성능 낙폭 측정),
  (b) 출처·배경 다양화된 3층 테스트셋에서 재평가,
  (c) E4 정합 지표로 "형질을 보는가 배경을 보는가" 정량화.
- 이 재검증 자체가 논문 소재: "세그멘테이션이 형태 진단 특징을 훼손하는가 vs full이 배경을 컨닝하는가"

### 3.4 전체 실험 설계 [개정 — v0.2 E0–E7 편입]
- **E0** 전처리×aspect ablation (§3.3) — P0
- **E1** 공정 3계열 비교 (§3.2) — P0
- **E2** 학습곡선 × 전략 교차점: 상위 6종 1/8→전체 서브샘플링 × {계열 FT, BioCLIP zero/few/FT} → "종당 N장" 권고표 — P0
- **E3** BioCLIP 심층: 종별 훈련 노출량 표(ToL-10M grep + GBIF API, 동의어 양방향), PHON unseen 자연 실험, 희귀종 구간 글로벌 FM vs 지역 FT 격차 — P0
- **E4** attention–검색표 정합: 진단 형질 표 → 형질 영역 주석 → pointing game/energy PG/IoU. CNN=Grad-CAM++, ViT=Prompt-CAM. sanity check + deletion/insertion 충실도 병행, plausibility/faithfulness 구분 — P0
- **E5** 성별·체급 층화 (v6 N1/N2 재활용): 층화 정확도, 성비 조작 실험, BioCLIP "종내 변이 직교 보존" 주장 검증 — P1
- **E6** 분류학적 구조: (a) 속 내/간 혼동 구조 (b) 임베딩 덴드로그램 vs 분류체계 cophenetic (+새날개나비 congruence 선행 인용, 지역 재현 한정) (c) 확신도 기반 속 후퇴(기존 OOD와 통합). **임베딩≠계통 한계 명문** — P1
- **E7** 조건부: quality_grade 노이즈, 소스 ablation(카페 층 포함/제외) — P2
- 통계: 반복 층화 CV(5×5) + 부트스트랩 CI (v6 단일 hold-out 199장 → 3층 테스트로 확장), 희귀종 결론 격하 서술, macro-F1 병행

---

## 4. 데이터 파이프라인 [개정]

### 4.1 전처리 순서 / 4.2 Specimen-level Split [유지 + 보강]
- specimen split 유지하되 **배경·촬영자 상관 통제 추가**: 동일 배경 클러스터(예: 흰 트레이)가 train/test 양쪽에 같은 종으로만 존재하지 않도록 검사 스크립트 추가.

### 4.3 레이블 체계 [개정 — 컬럼 확장]
L1/L2/L3 유지 + 전 이미지 공통 태그: `source`(inat/cafe/donation/naturing/field), `observed_on`, `quality_grade`, `captive 판별은 하지 않음`(v0.2 결정 — 소스 ablation으로 대체 방어).
Label Studio 템플릿에 male_form 판정 규칙 문서 링크. **성별·종 라벨 각각 제2 평가자 15–20% κ 보고**(신설, 논문 방어 핵심).

### 4.4 Data Flywheel [유지 + 3층 구조 연결]
| 층 | 출처 | 용도 |
|---|---|---|
| 1층 | GBIF/iNat (기존) | 훈련 + 날짜 필터 통과분 테스트 |
| 2층 | 네이버 카페 크롤링(공개 게시물, 4기둥 준수: 투명성/이미지 비배포/라벨 프로토콜/훈련 전용) + BeetleDex 기증 | 훈련 전용 |
| 3층 | 네이처링 협약, 기증, 현장 촬영, BioCLIP 2 스냅샷 이후 iNat | 테스트 전용 |
- **오염 방어(신설, 필수)**: 테스트셋 전체 MD5+지각해시(PDQ)로 TreeOfLife-200M 대비 중복 제거 + 날짜 필터. BioCLIP 2 자체 관행 인용. 방법론 소단락 명시.
- BeetleDex 개편 시 기증 폼 필드 = 본 라벨 스키마와 동일(종/성별/산지/촬영일) → 무전처리 유입.

---

## 5. 논문 지원 스크립트 [유지 + 추가]
신규: `scripts/check_tol_overlap.py`(FM 노출량 표+해시 dedupe), `scripts/subsample_curves.py`(E2), `src/xai/`(pointing game 지표), `scripts/background_probe.py`(§3.3 배경 교란).

## 6. 논문 구성 [개정]
> 가제: **"How Much Data, Which Strategy, and What Do Models See? A Template for Classifying Expert-Domain Taxa, Exemplified by Korean Stag Beetles"** (부제에 dimorphism-aware 유지 가능)

1. Introduction — 전문가 영역 분류군 문제, FM 시대의 지역 소규모 데이터
2. Related Work — FGVC/곤충, 데이터 효율성·학습곡선, 생물 FM(BioCLIP 1·2), XAI 정합, 이형성×DL
3. Dataset — 3층 구조, 라벨 체계(κ), 오염 방어, 사진 동정 한계(암컷 조합) 선제 서술
4. Method — 공정 비교 프로토콜, 전처리×aspect, 멀티태스크(마스킹 loss), BioCLIP 트랙, 정합 지표
5. Experiments — E0→E1→E2→E3→E4 (+E5·E6 분석 절)
6. Results & Discussion — 교차점 권고표, 형질 vs 배경, 이형성 취약성, 희귀종 사각지대, 템플릿 이식성(주장 수위: worked example + 공개 프로토콜)
7. Conclusion
**타겟**: MEE (primary, v6와 동일). 상향 조건: 전문가 벤치마크 또는 타 분류군 축소 반복. E4 단독 FGVC 워크숍 프리페이퍼 옵션.

## 6.5 적대적 검토 요약 [신설 — 상세는 docs/adversarial_review.md]
13개 공격 시나리오 관리: ①BioCLIP 압승론(→교차점/기제 프레임) ②오염(→3중 방어) ③소표본(→반복CV·격하서술) ④**템플릿 일반화(최대 급소** →문구 통제+프로토콜 공개+다분류군 옵션) ⑤saliency 신뢰성(→sanity check+충실도) ⑥A안 선행/스쿠핑(→재검색 2회+프리페이퍼) ⑦크롤링 윤리(→4기둥) ⑧셀프 라벨(→κ) ⑨사육 혼입(→소스 ablation) ⑩이형성 선행 구분 ⑪임베딩≠계통 ⑫산만함(→본편 E0–E4) ⑬**배경 누출(v7 신설** — v6 full 우위 재검증, §3.3).

## 7–10. [유지 — v6 체크리스트/로드맵/스택/파일구조] + 로드맵 추가:
### Phase 7 — 논문 리부트 (M0–M4)
- M0: FM 노출량 표, 진단 형질 표 v1(E4·라벨링 공용), 라벨 스키마 컬럼 확장 마이그레이션, 문헌 재검색 1차
- M1: 네이처링 협약·기증 공지·크롤링 파이프라인(+윤리 확인), E0
- M2: E1·E2, 2층 라벨링(κ)
- M3: E3·E4·E5, 다분류군 반복 결정, E6
- M4: E7 조건부, 원고, 재검색 2차, 투고
### Phase 8 — BeetleDex 개편 (논문 틀 확정 후): 기증 폼=라벨 스키마 동기화, flywheel 재가동

---
*미해결 정리 항목: (1) 종 수 16(v6) vs 13(모델링 대상) 정합 — taxonomy.yaml 기준으로 §3에서 명시, (2) male_form 판정 규칙 문서화, (3) v6 test 199장의 3층 테스트 편입 여부(오염 필터 통과분만).*
