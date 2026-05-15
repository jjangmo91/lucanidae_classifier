"""
어노테이션 → YOLO 학습 데이터셋 split 유틸리티

autodistill 출력 형식(train/ + valid/ 서브디렉토리) 또는
Label Studio export 후 flat 형식(images/ + labels/) 모두 처리한다.

사용법:
    python -m src.detection.annotation_utils
"""

import random
import shutil
from pathlib import Path

import yaml


def build_detector_dataset(config_path: str = "configs/default.yaml") -> None:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    cfg            = config["detection"]
    annotation_dir = Path(cfg["annotation_dir"])
    detector_dir   = Path(cfg["detector_train"])
    val_split      = float(cfg.get("val_split", 0.15))

    # autodistill 출력 형식 감지: train/ + valid/ 서브디렉토리
    if _is_autodistill_format(annotation_dir):
        print("autodistill 출력 형식 감지 (train/ + valid/ 구조)")
        _build_from_autodistill(annotation_dir, detector_dir)
    else:
        # Label Studio export 후 flat 형식: images/ + labels/
        print("flat 형식 감지 (images/ + labels/ 구조)")
        _build_from_flat(annotation_dir, detector_dir, val_split)

    # dataset.yaml 생성 (상대 경로 — portable)
    dataset_yaml = {
        "path":  str(detector_dir.resolve()),
        "train": "train/images",
        "val":   "val/images",
        "nc":    1,
        "names": ["stag_beetle"],
    }
    yaml_path = detector_dir / "dataset.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(dataset_yaml, f, default_flow_style=False, allow_unicode=True)

    # 결과 리포트
    n_train = len(list((detector_dir / "train" / "images").glob("*")))
    n_val   = len(list((detector_dir / "val"   / "images").glob("*")))
    n_empty = sum(
        1 for p in (detector_dir / "train" / "labels").glob("*.txt")
        if p.stat().st_size == 0
    )

    print(f"dataset.yaml 생성: {yaml_path}")
    print(f"  Train: {n_train}장  |  Val: {n_val}장  |  빈 라벨(미검출): {n_empty}개")
    print("다음 단계: python train_detector.py")


def _is_autodistill_format(annotation_dir: Path) -> bool:
    """autodistill이 생성하는 train/ + valid/ 구조인지 판별."""
    return (
        (annotation_dir / "train" / "images").is_dir() and
        (annotation_dir / "valid" / "images").is_dir()
    )


def _build_from_autodistill(annotation_dir: Path, detector_dir: Path) -> None:
    """autodistill train/valid → detector_train/train + detector_train/val 복사."""
    if detector_dir.exists():
        shutil.rmtree(detector_dir)

    # train → train
    _copy_split(
        src_images = annotation_dir / "train" / "images",
        src_labels = annotation_dir / "train" / "labels",
        dst_dir    = detector_dir / "train",
    )
    # valid → val (YOLO 표준 명칭)
    _copy_split(
        src_images = annotation_dir / "valid" / "images",
        src_labels = annotation_dir / "valid" / "labels",
        dst_dir    = detector_dir / "val",
    )


def _build_from_flat(annotation_dir: Path, detector_dir: Path, val_split: float) -> None:
    """
    flat images/ + labels/ 구조에서 유효 샘플을 수집하고
    train/val 분할 후 detector_train/ 에 저장.
    Label Studio export 후 사용하는 경로.
    """
    images_dir = annotation_dir / "images"
    labels_dir = annotation_dir / "labels"

    if not images_dir.exists() or not labels_dir.exists():
        raise FileNotFoundError(
            f"images/ 또는 labels/ 디렉토리가 없습니다: {annotation_dir}\n"
            "먼저 python -m src.detection.grounded_sam_annotate 를 실행하세요."
        )

    # 라벨이 존재하고 비어 있지 않은 이미지만 사용
    valid_stems = []
    empty_count = 0
    for label_path in sorted(labels_dir.glob("*.txt")):
        if label_path.stat().st_size == 0:
            empty_count += 1
            continue
        if _find_image(images_dir, label_path.stem) is not None:
            valid_stems.append(label_path.stem)

    print(f"유효 샘플: {len(valid_stems)}개  (빈 라벨 제외: {empty_count}개)")

    random.seed(42)
    random.shuffle(valid_stems)
    n_val       = max(1, int(len(valid_stems) * val_split))
    val_stems   = set(valid_stems[:n_val])
    train_stems = set(valid_stems[n_val:])

    print(f"Train: {len(train_stems)}개  /  Val: {len(val_stems)}개")

    if detector_dir.exists():
        shutil.rmtree(detector_dir)

    for split, stems in [("train", train_stems), ("val", val_stems)]:
        img_out = detector_dir / split / "images"
        lbl_out = detector_dir / split / "labels"
        img_out.mkdir(parents=True)
        lbl_out.mkdir(parents=True)

        for stem in stems:
            src_img = _find_image(images_dir, stem)
            src_lbl = labels_dir / f"{stem}.txt"
            if src_img is not None:
                shutil.copy2(src_img, img_out / src_img.name)
                shutil.copy2(src_lbl, lbl_out / f"{stem}.txt")


def _copy_split(src_images: Path, src_labels: Path, dst_dir: Path) -> None:
    """이미지와 라벨을 dst_dir/images, dst_dir/labels 로 복사."""
    dst_images = dst_dir / "images"
    dst_labels = dst_dir / "labels"
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)

    for img_path in src_images.iterdir():
        if img_path.is_file():
            shutil.copy2(img_path, dst_images / img_path.name)

    if src_labels.exists():
        for lbl_path in src_labels.iterdir():
            if lbl_path.suffix == ".txt":
                shutil.copy2(lbl_path, dst_labels / lbl_path.name)


def _find_image(images_dir: Path, stem: str) -> Path | None:
    for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
        p = images_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def validate_annotations(config_path: str = "configs/default.yaml") -> None:
    """어노테이션 품질 검사."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    annotation_dir = Path(config["detection"]["annotation_dir"])

    if _is_autodistill_format(annotation_dir):
        labels_dir = annotation_dir / "train" / "labels"
    else:
        labels_dir = annotation_dir / "labels"

    if not labels_dir.exists():
        print("라벨 디렉토리가 없습니다.")
        return

    total, empty, counts = 0, 0, []
    for label_path in sorted(labels_dir.glob("*.txt")):
        total += 1
        lines = label_path.read_text().strip().splitlines()
        if not lines:
            empty += 1
        else:
            counts.append(len(lines))

    print(f"총 라벨 파일:    {total}개")
    print(f"빈 라벨(미검출): {empty}개  ({empty / total * 100:.1f}%)")
    if counts:
        print(f"인스턴스/이미지: min={min(counts)}  max={max(counts)}  avg={sum(counts)/len(counts):.1f}")


if __name__ == "__main__":
    build_detector_dataset()
