"""MLflow SQLite DB의 run 기록을 CSV로 export한다.

mlflow.db는 gitignore 대상이고 과거에 초기화된 이력이 있어(DESIGN.md Phase 4)
DB만 두면 실험 기록이 유실된다. DESIGN.md §11.5에 따라 CSV로 떠서 저장소에 남긴다.

MLflow 서버가 떠 있지 않아도 동작한다 (SQLite 직접 읽기).

사용법:
    python scripts/export_mlflow.py
    python scripts/export_mlflow.py --db mlflow.db --out results/v6_mlflow
"""

import argparse
import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def export(db_path: Path, out_dir: Path) -> None:
    if not db_path.exists():
        raise SystemExit(f"MLflow DB를 찾을 수 없습니다: {db_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(db_path))

    experiments = pd.read_sql_query(
        "SELECT experiment_id, name, lifecycle_stage FROM experiments", con
    )

    runs = pd.read_sql_query(
        """
        SELECT run_uuid, name AS run_name, experiment_id, status,
               start_time, end_time, lifecycle_stage
        FROM runs
        """,
        con,
    )

    params = pd.read_sql_query("SELECT run_uuid, key, value FROM params", con)
    latest = pd.read_sql_query("SELECT run_uuid, key, value, step FROM latest_metrics", con)
    history = pd.read_sql_query(
        "SELECT run_uuid, key, value, step, timestamp FROM metrics ORDER BY run_uuid, key, step",
        con,
    )
    con.close()

    # run 한 줄에 params + 최종 metric 을 펼친다
    p_wide = params.pivot_table(
        index="run_uuid", columns="key", values="value", aggfunc="last"
    ).add_prefix("param.")
    m_wide = latest.pivot_table(
        index="run_uuid", columns="key", values="value", aggfunc="last"
    ).add_prefix("metric.")

    summary = (
        runs.merge(experiments, on="experiment_id", how="left", suffixes=("", "_exp"))
        .merge(p_wide, on="run_uuid", how="left")
        .merge(m_wide, on="run_uuid", how="left")
    )
    for col in ("start_time", "end_time"):
        summary[col] = pd.to_datetime(summary[col], unit="ms", errors="coerce")
    summary = summary.sort_values("start_time")

    files = {
        "runs_summary.csv": summary,
        "metrics_history.csv": history,
        "params_long.csv": params,
        "experiments.csv": experiments,
    }
    for name, df in files.items():
        df.to_csv(out_dir / name, index=False, encoding="utf-8-sig")
        print(f"  {name:22s} {len(df):7,d} rows")

    readme = out_dir / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# MLflow run export",
                "",
                f"`{db_path.name}` 에서 뜬 실험 기록. `scripts/export_mlflow.py` 로 재생성한다.",
                "",
                f"- experiments: {len(experiments)}",
                f"- runs: {len(runs)}",
                f"- metric 기록: {len(history):,}",
                "",
                "| 파일 | 내용 |",
                "|---|---|",
                "| `runs_summary.csv` | run 한 줄에 params + 최종 metric |",
                "| `metrics_history.csv` | epoch 단위 metric 전체 |",
                "| `params_long.csv` | params long-format |",
                "| `experiments.csv` | experiment 목록 |",
                "",
                "**주의**: v6 시점 기록은 라벨 유실·조각 오염·검정력 부족으로",
                "논문 근거에서 폐기되었다(DESIGN.md §3.4). 내부 개발 이력으로만 참조한다.",
            ]
        ),
        encoding="utf-8",
        newline="",
    )
    print(f"\n저장 위치: {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(ROOT / "mlflow.db"))
    ap.add_argument("--out", default=str(ROOT / "results" / "v6_mlflow"))
    a = ap.parse_args()
    export(Path(a.db), Path(a.out))


if __name__ == "__main__":
    main()
