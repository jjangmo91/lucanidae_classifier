# 서드파티 라이선스 고지

이 저장소의 자체 코드는 MIT다 (`LICENSE`). 의존성은 별개이며,
그중 **하나가 배포 조건에 실질적인 제약을 건다.**

설치된 186개 패키지를 검사한 결과 카피레프트 계열은 6개다.

| 패키지 | 라이선스 | 전염성 | 판단 |
|--------|---------|--------|------|
| **ultralytics** | **AGPL-3.0+** | **강함 (네트워크 사용 포함)** | **주의 필요** |
| **ultralytics-thop** | **AGPL-3.0+** | 위와 동일 | 위와 동일 |
| pi_heif | LGPL-3.0 | 약함 | 라이브러리로 사용하는 한 문제 없음 |
| psycopg2-binary | LGPL | 약함 | 문제 없음 |
| certifi | MPL-2.0 | 파일 단위 | 문제 없음 |
| tqdm | MPL-2.0 AND MIT | 파일 단위 | 문제 없음 |

LGPL과 MPL은 해당 라이브러리 자체를 수정해 배포할 때만 조건이 붙는다.
그냥 import 해서 쓰는 우리 코드에는 전염되지 않는다.

---

## ultralytics (AGPL-3.0) — 무엇이 문제인가

YOLOv8 검출기를 `ultralytics` 패키지로 쓰고 있다.

```
src/ml/detector.py          BeetleSegmenter
src/detection/segmenter.py
train_detector.py
scripts/review_multi_detect.py
```

그리고 이 경로는 **웹서비스 응답 경로에 들어간다.**
`src/ml/manager.py` 가 `BeetleSegmenter` 를 로드하고
`src/services/prediction.py` 가 `segmenter.segment()` 를 호출한다.

AGPL-3.0 의 핵심은 13조다. 일반 GPL은 **배포**할 때 소스 공개 의무가 생기지만,
AGPL은 **네트워크로 사용자와 상호작용하기만 해도** 그 사용자에게 대응 소스를
제공해야 한다. beetledex.com 이 정확히 이 경우다.

### 결과적으로 두 가지가 걸린다

**1. 서비스 운영 (beetledex.com) — 충족**
서비스 이용자에게 전체 애플리케이션의 대응 소스를 제공할 수 있어야 한다.
**저장소가 public 임을 확인했다.** 소스가 공개되어 있으므로 AGPL 13조는 사실상 충족된다.
다만 저장소를 비공개로 전환하면 그 시점부터 미충족 상태가 되므로, 앞으로도
public 을 유지하거나 검출기를 교체해야 한다.

**2. 논문과 함께 배포하는 코드**
MIT로 표시하더라도, ultralytics 를 필요로 하는 **결합 저작물 전체**를 배포하면
AGPL 조건이 따라붙는다. 우리 코드만 떼어 보면 MIT가 맞지만,
"이 저장소를 통째로 받아 실행하는 사람"에게는 AGPL이 적용된다.

**저널 요구와는 충돌하지 않는다.** MEE/BES가 요구하는 것은 코드를 DOI 발급
저장소에 아카이브해 공개하라는 것이고, AGPL은 OSI 승인 오픈소스 라이선스다.
AGPL로 공개해도 투고 요건은 그대로 충족된다.

---

## 선택지

| 안 | 내용 | 비용 | 비고 |
|----|------|------|------|
| **A. 검출기 교체** | ultralytics 를 허용적 라이선스 모델로 대체 | 재학습 1회 | **권장** |
| B. 전체 AGPL 채택 | 자체 코드도 AGPL-3.0 으로 배포 | 없음 | 재사용자에게 전염됨 |
| C. 저장소 공개 유지 | 현행 유지, 저장소를 public 으로 | 없음 | 논문 코드는 여전히 AGPL 결합물 |
| D. 상용 라이선스 구매 | Ultralytics Enterprise | 유료 | 연구 규모에 과함 |

### A안을 권하는 이유

DESIGN.md 4.8 에서 이미 **검출기 재학습을 M0 필수 항목으로 잡아두었다.**
과분할 때문에 어차피 다시 학습시켜야 한다. 그 시점에 백본을 바꾸면
추가 비용이 사실상 없다.

허용적 라이선스 대안:

| 모델 | 라이선스 | 비고 |
|------|---------|------|
| torchvision `maskrcnn_resnet50_fpn` / `fasterrcnn` | BSD-3 | 이미 torchvision 의존 중, 추가 설치 없음 |
| HuggingFace `transformers` DETR / RT-DETR | Apache-2.0 | 학습 코드가 단순함 |
| YOLOX | Apache-2.0 | YOLO 계열 유지 원할 때 |

### 교체 시 반드시 지켜야 할 조건 — 인스턴스 분할이어야 한다

**단순 검출기로 바꾸면 안 된다.** 전처리 모드 중 `seg_hard` 와 `seg_soft` 는
박스가 아니라 **마스크**를 쓴다 (`src/ml/detector.py` 의 `_extract_crops`).
마스크가 없으면 코드가 조용히 bbox crop 으로 폴백하도록 되어 있어서,
박스만 주는 모델로 갈아끼우면 **4개 전처리 모드 중 2개가 bbox 와 동일해진다.**
E0(전처리 비교)의 절반이 사라지는 것이므로 반드시 인스턴스 분할 모델을 골라야 한다.

| 모델 | 라이선스 | 마스크 | 판단 |
|------|---------|--------|------|
| torchvision `maskrcnn_resnet50_fpn` | BSD-3 | **제공** | **권장.** 이미 torchvision 의존 중이라 추가 설치 없음 |
| Detectron2 Mask R-CNN | Apache-2.0 | 제공 | 설치가 다소 무거움 |
| HuggingFace `transformers` Mask2Former | Apache-2.0 | 제공 | 가능 |
| torchvision `fasterrcnn` | BSD-3 | 없음 | **부적합** |
| YOLOX | Apache-2.0 | 없음(검출 전용) | **부적합** |

### 교체에 따르는 작업

| 항목 | 내용 | 비고 |
|------|------|------|
| 어노테이션 변환 | 현재 `data/annotations/` 는 YOLO 폴리곤 형식. COCO 폴리곤으로 변환 필요 | 기계적 변환, 정보 손실 없음 |
| 학습 스크립트 | `train_detector.py` 가 전부 ultralytics 기반이라 재작성 | |
| 가중치 | `models/weights/best_detector.pt` 폐기 | 어차피 과분할로 재학습 예정 |
| crop 데이터 재생성 | `data/final_bbox`, `final_seg_hard`, `final_seg_soft` 전부 | 어차피 재생성 예정 (DESIGN.md 4.8) |
| `detector.py` | ultralytics `result.boxes` / `result.masks` 접근부를 교체 | 인터페이스는 유지 가능 |

### 기존 설계와 충돌하지 않는다

- **v6 crop 데이터와의 비교 불가?** 문제되지 않는다. v6 결과는 이미 전량 폐기했다 (DESIGN.md 3.4).
- **전처리 비교(E0)가 흔들리나?** 오히려 반대다. 모든 crop 기반 모드가 같은 검출기에
  의존하므로, 검출기 품질이 올라가면 비교가 더 공정해진다. 과분할 조각이 빠지는 것도 E0 에 이득이다.
- **어노테이션 도구는?** autodistill / Grounded SAM 은 설치 환경 검사에서 강한 카피레프트가
  아니었다. 오프라인 어노테이션 도구라 배포물에 포함되지도 않는다.

우리 용도는 **클래스 무관(class-agnostic) 단일 객체 분할**이라 난이도가 낮다.
YOLOv8n-seg 의 성능이 꼭 필요한 상황이 아니다.

---

## 조치 상태

- [x] 의존성 카피레프트 전수 검사
- [x] MIT `LICENSE` 작성 + 적용 범위 명시
- [x] **저장소 public 확인** — AGPL 13조(네트워크 사용) 충족
- [ ] 검출기 교체 여부 결정 (A안 권장, 재학습 시점에 함께)
      -> 서비스 쪽 급한 불은 껐다. 남은 이유는 **논문 배포물이 AGPL 결합 저작물이 되는 것**이다.
      -> MIT 로 내놓겠다면 교체가 필요하고, AGPL 로 내놓겠다면 현행 유지도 가능하다.
- [ ] 교체 시 `requirements.txt` 에서 ultralytics 제거

재검사:

```bash
python scripts/check_licenses.py
```
