# 진단 형질 표 — 작성 지침과 참고문헌

> 용도: E4(attention-검색표 정합)의 정답 정의, male_form 라벨링 시 부위 식별 보조.
> 근거: DESIGN.md §3.5 E4, §12.5
> 템플릿: `docs/templates/diagnostic_traits.csv` (추적됨)
> 작업본: `data/labels/diagnostic_traits.csv` (gitignore. 완성되면 템플릿 자리로 커밋)

## 왜 필요한가

E4는 "모델이 분류학자가 보는 곳을 보는가"를 정량화한다.
그러려면 **분류학자가 어디를 보는지**가 먼저 문서로 있어야 한다.
이 표가 없으면 E4는 시작할 수 없다.

**형질은 반드시 출판된 검색표에서 가져온다.** 우리가 임의로 "여기가 중요해 보인다"고
정하면 E4 전체가 순환논증이 된다. 모든 행에 출처를 단다.

## 표 구조

`data/labels/diagnostic_traits.csv`

| 컬럼 | 설명 |
|------|------|
| `species` | canonical 학명 (`configs/taxonomy.yaml` 기준) |
| `character` | 형질 이름 (예: `mandible_inner_tooth`) |
| `character_ko` | 한국어 형질명 (예: 큰턱 내치) |
| `body_region` | 부위. `head` / `mandible` / `pronotum` / `elytra` / `legs` / `antenna` |
| `state` | 이 종에서의 상태 (검색표 기술 그대로) |
| `contrast_with` | 이 형질로 구분되는 상대 종 (쉼표 구분) |
| `sex` | 형질이 유효한 성별. `male` / `female` / `both` |
| `visibility` | **가시성 등급** (아래 참조) |
| `source` | 출처 약칭 (예: `Kim1998`, `NIBR2014`) |
| `source_page` | 쪽수 또는 검색표 항목 번호 |
| `note` | 비고 |

## 가시성 등급 — E4의 성패가 여기 달려 있다

Lucanidae 검색표의 진단 형질에는 **접사 표본 사진이라야 보이는 것**이 많다.
점각(punctation), 미모(setae), 큰턱 내치의 미세 위치 같은 것들이다.
iNaturalist 사진과 448px 입력에서 상당수가 물리적으로 보이지 않는다.

| 등급 | 뜻 | E4 처리 |
|------|----|---------|
| `always` | 통상적인 배면 사진에서 항상 보임 | 정합 지표 계산 대상 |
| `angle` | 각도에 따라 보임 (측면, 정면 등) | 해당 각도 사진에 한해 계산 |
| `macro` | 접사·표본 사진이라야 보임 | **계산에서 제외, "측정 불가"로 보고** |
| `dissection` | 해부(생식기 등)가 필요 | 제외 |

**`macro`와 `dissection` 이 많다는 사실 자체가 결과다.**
"사진 기반 동정으로는 진단 형질의 몇 %에 접근할 수 없다"는 정량 진술이 되고,
이는 논문의 전문가 영역 분류군 프레이밍(DESIGN.md §1.1-5)과 정확히 맞는다.
약점으로 숨기지 말고 먼저 보고한다.

## 참고문헌

**확인된 1차 문헌**

| 약칭 | 서지 |
|------|------|
| `Kim1998` | Kim, J.I. & Kim, S.Y. (1998) Taxonomic Review of Korean Lucanidae (Coleoptera: Scarabaeoidea). *Animal Systematics, Evolution and Diversity* 14(1): 21-33. |
| `NIBR2014` | Kim, J.-I. & Kim, S.I. (2014) *Insect Fauna of Korea: Lucanidae and Passalidae.* Insect Fauna of Korea Vol. 12, No. 15. National Institute of Biological Resources, Incheon. 58 pp. |

**확인 필요 (검색 결과에 나타났으나 서지 미확정)**

- Review of family Lucanidae (Insecta: Coleoptera) in Korea with the description of one new species
- A new species of the *Dorcus velutinus*-group (Coleoptera, Lucanidae) from Korea (2019 무렵)
  -> 엷은털왕사슴벌레(`Dorcus_tenuihirsutus`) 관련일 가능성. 확인 요망

## 종 수 불일치 확인 과제

`Kim1998` 은 한국산 Lucanidae 를 **10속 14종**으로 정리했다.
`configs/taxonomy.yaml` 은 국립생물자원관 국가생물종목록 기준 **16종**이다.

차이는 1998년 이후 추가된 종(신종 기재 또는 미기록종 추가)에서 오는 것으로 보이나
**확인되지 않았다.** DESIGN.md §1.3 의 16종 목록은 다음을 대조해 검증해야 한다.

- [ ] 국가생물종목록 최신판의 Lucanidae 항목
- [ ] `NIBR2014` 의 종 목록
- [ ] 1998년 이후 신종 기재 논문

특히 데이터가 0장인 3종(털보왕사슴벌레, 엷은털왕사슴벌레, 큰꼬마사슴벌레)이
현재도 유효한 한국 서식종인지 확인한다. 목록이 바뀌면 §1.3 을 갱신한다.

## 작성 순서

1. `NIBR2014` 의 검색표를 확보한다 (가장 최신이고 국명·학명이 정리되어 있음).
2. 검색표의 각 분기점에서 쓰이는 형질을 행으로 옮긴다. 분기 = 대비이므로
   `contrast_with` 가 자연스럽게 채워진다.
3. `Kim1998` 로 보완한다. 두 문헌이 다르면 둘 다 기재하고 `note` 에 차이를 적는다.
4. 각 형질에 가시성 등급을 매긴다. **판단이 서지 않으면 실제 보유 사진 몇 장을 열어
   확인한다.** 이 등급이 E4 결과를 좌우하므로 추측하지 않는다.
5. 최소 모델링 대상 10종을 채운다. 나머지 6종은 데이터 확보 후.

## 검수

- 형질 기재는 **검색표 원문에 근거**해야 한다. 의역은 `note` 에 남긴다.
- 가시성 등급은 **제2 평가자가 15~20% 재판정**한다 (DESIGN.md §4.3 의 신뢰도 보고와 동일 원칙).
