# scripts/analysis - v7 설계 근거 측정 스크립트

DESIGN.md v7에 인용된 수치를 산출한 스크립트다.
일회성 진단용이며 파이프라인의 일부가 아니다. 수치를 재확인하거나
데이터가 바뀐 뒤 다시 재려면 실행한다.

| 스크립트 | 측정 내용 | DESIGN.md |
|---|---|---|
| `backfill_licenses.py` | iNaturalist 사진 라이선스·출처표기 역조회 백필. | §11.2 |
| `probe_multi_detect.py` | 다중검출 원본의 박스 겹침 구조 측정 (한 개체 과분할 vs 실제 다개체). | §4.8.1 |
| `sweep_detector_conf.py` | detector confidence 임계값별 다중검출 감소 효과 측정. | §4.8.1 |
| `metadata_leakage.py` | 좌표·날짜 단독 종 예측 (위치 배제의 근거가 되는 음성 대조군). | §12.6-A / E9 |
| `power_analysis.py` | 테스트셋 크기별 검정력·신뢰구간 계산. | §12.1 |
| `collection_targets.py` | 통계 요건에서 역산한 종별 수집 목표와 부족분. | §13 |
| `check_label_coverage.py` | 성별·형태 라벨의 실제 매칭률 점검 (split 키 유실 진단). | §4.3 |
| `figure_eligible.py` | 논문 그림에 실을 수 있는 사진 집계 + 후보 목록 | §11.7 |

출력은 `experiments/analysis_v7/` 에 저장된다 (gitignore 대상).

주의: 이 스크립트들은 v7 설계 시점의 데이터 상태를 전제한다.
재라벨링·재수집 이후에는 결과가 달라진다.