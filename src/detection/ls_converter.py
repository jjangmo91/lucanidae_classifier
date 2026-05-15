"""
Label Studio ↔ YOLO 변환 유틸리티

[어노테이션 검토 워크플로우]

  Step A) YOLO → Label Studio JSON (검토 전 import용)
    python -m src.detection.ls_converter to_ls

  Step B) Label Studio export zip → YOLO labels (검토 후 복원)
    python -m src.detection.ls_converter from_ls --zip path/to/export.zip

[Label Studio 검토 절차]
  1. label-studio start                        # 로컬 Label Studio 실행
  2. 새 프로젝트 생성 → Polygon labeling (이미지 세그먼테이션) 템플릿
  3. Settings → Labeling Interface → 클래스 "stag_beetle" 추가
  4. Import → "Upload Files" → data/annotations/images/ 의 이미지 전체 선택
  5. 이 스크립트로 생성된 ls_predictions.json 을 Import → "Import predictions" 로 업로드
  6. 어노테이션 검토·수정
  7. Export → "YOLO with Polygons" 선택 → zip 다운로드
  8. 이 스크립트로 zip → YOLO 복원 (Step B)
"""

import argparse
import json
import shutil
import zipfile
from pathlib import Path

import yaml

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


# ──────────────────────────────────────────────
# Step A: YOLO txt → Label Studio JSON
# ──────────────────────────────────────────────

def yolo_to_ls(config_path: str = "configs/default.yaml") -> None:
    """
    data/annotations/labels/*.txt → data/annotations/ls_predictions.json

    Label Studio 에서 "Import predictions" 로 업로드하면
    Grounded SAM 2 자동 어노테이션이 pre-annotation으로 표시된다.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    annotation_dir = Path(config["detection"]["annotation_dir"])
    images_dir     = annotation_dir / "images"
    labels_dir     = annotation_dir / "labels"
    output_path    = annotation_dir / "ls_predictions.json"

    if not labels_dir.exists():
        raise FileNotFoundError(f"labels 디렉토리가 없습니다: {labels_dir}")

    tasks = []
    for label_path in sorted(labels_dir.glob("*.txt")):
        if label_path.stat().st_size == 0:
            continue

        # 대응하는 이미지 파일 탐색
        img_path = _find_image(images_dir, label_path.stem)
        if img_path is None:
            continue

        predictions = _parse_yolo_seg(label_path, img_path)
        if not predictions:
            continue

        tasks.append({
            "data":        {"image": f"/data/local-files/?d={img_path.resolve()}"},
            "predictions": [{"result": predictions, "score": 1.0}],
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(tasks, f, ensure_ascii=False, indent=2)

    print(f"Label Studio predictions JSON 생성: {output_path}")
    print(f"  변환 완료: {len(tasks)}개 이미지")
    print()
    print("Label Studio Import 방법:")
    print("  1. 프로젝트 → Import → 'Upload Files' → images/ 폴더 전체 선택")
    print(f"  2. 프로젝트 → Import → 'Import predictions' → {output_path.name} 업로드")
    print("  ※ Label Studio Settings → Local Storage 를 먼저 설정해야 파일 경로가 작동합니다.")


def _parse_yolo_seg(label_path: Path, img_path: Path) -> list[dict]:
    """YOLO seg txt → Label Studio polygon result 형식."""
    from PIL import Image as PILImage

    try:
        w, h = PILImage.open(img_path).size
    except Exception:
        return []

    results = []
    for line in label_path.read_text().strip().splitlines():
        parts = line.strip().split()
        if len(parts) < 7:          # class + 최소 3점(x,y)×3 = 7
            continue
        # class_id x1 y1 x2 y2 ... (normalized)
        coords = [float(v) for v in parts[1:]]
        points = [
            [coords[i] * 100, coords[i + 1] * 100]   # Label Studio는 퍼센트 단위
            for i in range(0, len(coords) - 1, 2)
        ]
        results.append({
            "type":        "polygonlabels",
            "from_name":   "label",
            "to_name":     "image",
            "original_width":  w,
            "original_height": h,
            "value": {
                "points":        points,
                "polygonlabels": ["stag_beetle"],
            },
        })
    return results


# ──────────────────────────────────────────────
# Step B: Label Studio export zip → YOLO labels
# ──────────────────────────────────────────────

def ls_to_yolo(zip_path: str, config_path: str = "configs/default.yaml") -> None:
    """
    Label Studio 'YOLO with Polygons' export zip 을 받아
    data/annotations/labels/ 를 갱신한다.

    Label Studio export zip 구조:
        images/        ← 이미지 (원본 동일)
        labels/        ← YOLO seg txt (flat)
        classes.txt
        notes.json
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    annotation_dir = Path(config["detection"]["annotation_dir"])
    labels_dir     = annotation_dir / "labels"

    zip_path = Path(zip_path)
    if not zip_path.exists():
        raise FileNotFoundError(f"zip 파일을 찾을 수 없습니다: {zip_path}")

    # zip 압축 해제 → 임시 디렉토리
    tmp_dir = annotation_dir / "_ls_export_tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(tmp_dir)

    # labels/ 위치 탐색 (flat 또는 한 단계 아래에 있을 수 있음)
    exported_labels = _find_labels_dir(tmp_dir)
    if exported_labels is None:
        raise RuntimeError(
            f"zip에서 labels/ 디렉토리를 찾을 수 없습니다.\n"
            f"압축 해제 경로: {tmp_dir}\n"
            "Label Studio에서 'YOLO with Polygons' 형식으로 export했는지 확인하세요."
        )

    # 기존 labels/ 백업 후 교체
    backup_dir = annotation_dir / "labels_backup"
    if labels_dir.exists():
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        shutil.copytree(labels_dir, backup_dir)
        print(f"기존 labels 백업: {backup_dir}")

    if labels_dir.exists():
        shutil.rmtree(labels_dir)
    shutil.copytree(exported_labels, labels_dir)

    # 정리
    shutil.rmtree(tmp_dir)

    n_labels = len(list(labels_dir.glob("*.txt")))
    print(f"Labels 갱신 완료: {labels_dir}  ({n_labels}개)")
    print("다음 단계: python -m src.detection.annotation_utils")


def _find_labels_dir(root: Path) -> Path | None:
    # 1) root/labels/
    candidate = root / "labels"
    if candidate.is_dir():
        return candidate
    # 2) root/*/labels/  (export가 하위 폴더에 풀린 경우)
    for sub in root.iterdir():
        if sub.is_dir():
            candidate = sub / "labels"
            if candidate.is_dir():
                return candidate
    return None


def _find_image(images_dir: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXTENSIONS:
        p = images_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Label Studio ↔ YOLO 변환 유틸리티",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("to_ls",   help="YOLO → Label Studio JSON (검토 전)")
    p_from = sub.add_parser("from_ls", help="Label Studio zip → YOLO labels (검토 후)")
    p_from.add_argument("--zip", required=True, help="Label Studio export zip 경로")

    args = parser.parse_args()

    if args.cmd == "to_ls":
        yolo_to_ls()
    elif args.cmd == "from_ls":
        ls_to_yolo(args.zip)


if __name__ == "__main__":
    main()
