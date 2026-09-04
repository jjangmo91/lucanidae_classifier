# MLflow run export

`mlflow.db` 에서 뜬 실험 기록. `scripts/export_mlflow.py` 로 재생성한다.

- experiments: 2
- runs: 149
- metric 기록: 28,787

| 파일 | 내용 |
|---|---|
| `runs_summary.csv` | run 한 줄에 params + 최종 metric |
| `metrics_history.csv` | epoch 단위 metric 전체 |
| `params_long.csv` | params long-format |
| `experiments.csv` | experiment 목록 |

**주의**: v6 시점 기록은 라벨 유실·조각 오염·검정력 부족으로
논문 근거에서 폐기되었다(DESIGN.md §3.4). 내부 개발 이력으로만 참조한다.