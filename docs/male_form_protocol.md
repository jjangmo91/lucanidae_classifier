# male_form 판정 프로토콜 v1

> 근거와 배경은 `DESIGN.md` §4.4. 이 문서는 **라벨러가 보는 실무 지침**이다.
> 이 프로토콜 이전에 매겨진 male_form 라벨은 전량 폐기하고 처음부터 다시 매긴다.

## 0. 왜 바꾸는가

기존 라벨은 "대형(뿔 김) / 소형(뿔 짧음) / 중간형" 이라는 문구만 보고 눈대중으로 매겼다.
기준점이 없으니 라벨러가 **종의 크기를 형태 등급으로 착각**했다.
원래 큰 넓적사슴벌레는 거의 전부 major가 되고, 작은 다우리아사슴벌레는 minor 쪽으로 쏠렸다.
종 정체성이 형태 라벨로 새어 들어간 것이라 학습 신호로 쓸 수 없다.

major와 minor는 본래 **체장 대비 큰턱의 성장 관계(allometry)** 에서 나오는 구간이고
그 임계점은 종마다 다르다. 사진 한 장에서는 절대 체장을 잴 수 없으므로,
**사진 안에서 잴 수 있는 무차원 비율**로 바꾸고 **종별로** 구간을 나눈다.

## 1. 클래스는 3개, 바뀌지 않는다

`major` / `intermediate` / `minor` 세 개다. 늘리지도 줄이지도 않는다.
아래의 `unknown`과 `not_applicable`은 **클래스가 아니라 학습 제외 표시**이며
둘 다 -1로 인코딩되어 masked loss에서 빠진다.

## 2. 계측 — 점 4개를 찍는다

이미지에 아래 네 점을 찍는다. 순서는 상관없다.

| 점 이름 | 위치 |
|---------|------|
| `mandible_apex` | 큰턱 **선단**(끝). 좌우 중 더 온전히 보이는 쪽 하나만 |
| `mandible_base` | 그 큰턱이 머리에 붙는 **두순(clypeus) 접합부** |
| `body_front` | **전흉배판(pronotum) 앞 가장자리** 중앙 |
| `body_rear` | **초시(elytra) 끝** 중앙 |

두 길이를 계산한다.

```
M = |mandible_apex - mandible_base|     큰턱 길이
B = |body_front - body_rear|            체장 대용치 (큰턱 제외 몸통)
R = M / B                                무차원 비율
```

**왜 큰턱을 뺀 몸통을 쓰나**: 큰턱은 그 자체가 다형 형질이다. 전체 체장에 포함시키면
분모가 분자와 같이 커져서 비율이 둔해진다. 큰턱을 뺀 몸통이 체급의 안정적인 대리변수다.

**왜 비율인가**: R은 축척과 촬영 거리에 영향받지 않는다. 자를 대지 않아도 되고
사진마다 배율이 달라도 비교가 된다.

## 3. 등급은 자동으로 매겨진다

라벨러는 **점만 찍는다.** major / intermediate / minor 판정은 사람이 하지 않는다.
`scripts/compute_male_form.py` 가 종별 분포에서 자동으로 나눈다.

```
종 s 에 대해, train split의 R 분포에서
  major        : R >= Q70(s)
  minor        : R <= Q30(s)
  intermediate : 그 사이
```

- 임계값은 **train split만으로** 계산하고 동결한다. val/test에는 그 값을 그대로 적용한다.
- 산출된 종별 Q30/Q70은 `data/labels/male_form_thresholds.csv` 에 저장되고 논문 보충자료로 공개한다.
- 30/70 경계는 관례적 선택이므로 25/75, 33/67 대안에 대한 민감도 분석을 함께 보고한다.

## 4. 찍지 말아야 할 때 — `unknown`

아래에 하나라도 해당하면 점을 찍지 말고 `unknown`으로 표시한다.
억지로 찍은 점이 분위수를 망가뜨리는 것이 라벨을 비우는 것보다 나쁘다.

- 큰턱이나 몸통 일부가 **가려졌거나 프레임 밖**으로 나감
- **배면(dorsal)이나 측면이 아닌 각도**여서 단축(foreshortening)이 생김
  — 비스듬히 찍혀 길이가 실제보다 짧아 보이는 경우
- 큰턱이 **좌우 비대칭으로 손상**됨 (야외 개체에 흔하다)
- 성별이 수컷이 아님

## 5. 아예 판정하지 않는 종 — `not_applicable`

다음 종은 male_form을 매기지 않는다. 점도 찍지 않는다.

| 기준 | 해당 종 |
|------|---------|
| 큰턱 다형성이 문헌상 뚜렷하지 않음 | 원표애보라사슴벌레 |
| 라벨된 수컷이 30개체 미만 -> 종별 분위수 추정 불가 | 참넓적사슴벌레, 두점박이사슴벌레, 왕사슴벌레, 홍다리사슴벌레 |

현재 male_form을 **학습 신호로 쓰는 종은 상위 5종**뿐이다.

- 넓적사슴벌레, 톱사슴벌레, 애사슴벌레, 사슴벌레, 다우리아사슴벌레

수집이 진행되어 어떤 종의 라벨된 수컷이 30개체를 넘으면 그때 대상에 추가한다.
`not_applicable` 목록은 `configs/male_form_scope.yaml` 에서 관리한다.

## 6. 신뢰도 보고

논문 Dataset 절에 필수로 들어간다.

- **전체의 15-20%를 제2 평가자가 재계측**한다.
- 계측값 `R` 의 일치도는 **ICC(2,1)** 로 보고한다 (연속값이므로 kappa가 아니다).
- 자동 산출된 3분류의 일치도는 **Cohen's kappa** 로 보고한다.

## 7. 작업 순서

```bash
# 1. Label Studio 프로젝트 생성, configs/label_studio_male_form.xml 을 템플릿으로 사용
# 2. 대상 이미지 export (상위 5종 수컷)
python scripts/export_for_labeling.py --scope male_form

# 3. 라벨링 (점 4개 또는 unknown)

# 4. Label Studio JSON export -> R 계산 -> 종별 분위수 -> 3분류
python scripts/compute_male_form.py \
    --annotations data/labels/male_form_export.json \
    --out data/labels/male_form.csv
```

산출물:
- `data/labels/male_form.csv` — 개체별 R 값과 등급
- `data/labels/male_form_thresholds.csv` — 종별 Q30/Q70 (동결, 공개 대상)
