"""
Label Studio export JSON을 파싱해 data/labels/sex_labels.csv 로 저장한다.
이후 dataset.py가 이 CSV를 읽어 multi-task 학습에 활용한다.

사용법:
    python scripts/merge_sex_labels.py
    python scripts/merge_sex_labels.py --export data/labels/sex_labels_export.json
"""

import argparse
import json
import csv
from pathlib import Path


OUTPUT_CSV = Path("data/labels/sex_labels.csv")

SEX_MAP = {
    "수컷": "male", "암컷": "female", "모름": "unknown",
    "male": "male", "female": "female", "unknown": "unknown",
}
FORM_MAP = {
    "대형(뿔 김)": "major", "소형(뿔 짧음)": "minor", "중간형": "intermediate", "모름": "unknown",
    "major": "major", "minor": "minor", "intermediate": "intermediate", "unknown": "unknown",
}


def main(export_path: str = "data/labels/sex_labels_export.json"):
    with open(export_path, "r", encoding="utf-8") as f:
        tasks = json.load(f)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    skipped = 0

    for task in tasks:
        local_path = task.get("data", {}).get("local_path", "")
        species    = task.get("data", {}).get("species", "")
        split      = task.get("data", {}).get("split", "")

        annotations = task.get("annotations", [])
        if not annotations:
            skipped += 1
            continue

        result = annotations[0].get("result", [])
        sex, male_form = "unknown", "unknown"

        for r in result:
            label_name = r.get("from_name", "")
            raw_value  = r.get("value", {}).get("choices", ["unknown"])[0]
            if label_name == "sex":
                sex = SEX_MAP.get(raw_value, "unknown")
            elif label_name == "male_form":
                male_form = FORM_MAP.get(raw_value, "unknown")

        rows.append({
            "local_path": local_path,
            "species":    species,
            "split":      split,
            "sex":        sex,
            "male_form":  male_form if sex == "male" else "unknown",
        })

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["local_path", "species", "split", "sex", "male_form"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"라벨 통합 완료: {len(rows)}개 → {OUTPUT_CSV}")
    if skipped:
        print(f"라벨 없음(스킵): {skipped}개")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--export", default="data/labels/sex_labels_export.json")
    args = parser.parse_args()
    main(export_path=args.export)
