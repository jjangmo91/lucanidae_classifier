"""
data/final/ 이미지 목록을 Label Studio import용 JSON으로 내보낸다.

Label Studio에서:
  1. 새 프로젝트 생성
  2. Labeling Interface → Custom template 에 configs/label_studio_template.xml 붙여넣기
  3. Import → JSON → 이 스크립트 출력 파일 업로드
  4. 라벨링 완료 후 Export → JSON → data/labels/sex_labels_export.json 저장
  5. python scripts/merge_sex_labels.py 실행

사용법:
    python scripts/export_for_labeling.py
"""

import json
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
SOURCE_DIR = Path("data/final")
OUTPUT_FILE = Path("data/labels/label_studio_import.json")


def main():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    tasks = []
    for split_dir in sorted(SOURCE_DIR.iterdir()):
        if not split_dir.is_dir():
            continue
        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            species = class_dir.name
            split   = split_dir.name

            for img_path in sorted(class_dir.iterdir()):
                if img_path.suffix not in IMAGE_EXTENSIONS:
                    continue
                # 브라우저에서 직접 접근하는 URL (localhost)
                ls_path = f"http://localhost:8082/{split}/{species}/{img_path.name}"
                tasks.append({
                    "data": {
                        "image":        ls_path,
                        "species":      species,
                        "split":        split,
                        "local_path":   str(img_path),
                    }
                })

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(tasks, f, ensure_ascii=False, indent=2)

    print(f"총 {len(tasks)}개 이미지 → {OUTPUT_FILE}")
    print("Label Studio에서 이 파일을 Import 하세요.")


if __name__ == "__main__":
    main()
