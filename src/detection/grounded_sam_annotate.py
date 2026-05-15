"""
Grounded SAM 2 자동 어노테이션 스크립트

data/final/ (ImageFolder 구조) 의 모든 이미지를 수집하여
Grounded SAM 2로 class-agnostic 사슴벌레 세그먼테이션 어노테이션 생성.

출력: data/annotations/
  images/  - 어노테이션된 이미지 (평면 구조)
  labels/  - YOLO seg 형식 .txt 파일 (class_id x1 y1 x2 y2 ...)

사용법:
    python -m src.detection.grounded_sam_annotate
"""

import shutil
from pathlib import Path

import yaml
from tqdm import tqdm

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def collect_images(source_dir: Path, flat_dir: Path) -> list[tuple[Path, Path]]:
    """
    ImageFolder 구조(split/class/image.jpg)에서 이미지를 수집해
    flat_dir에 고유한 이름으로 복사한다.

    반환: [(원본 경로, 복사 경로), ...]
    """
    flat_dir.mkdir(parents=True, exist_ok=True)
    collected = []

    all_images = [p for p in source_dir.rglob("*") if p.suffix in IMAGE_EXTENSIONS]

    for img_path in tqdm(all_images, desc="이미지 수집"):
        # split/class/filename.jpg → split__class__filename.jpg
        relative = img_path.relative_to(source_dir)
        flat_name = "__".join(relative.parts)
        dst = flat_dir / flat_name
        if not dst.exists():
            shutil.copy2(img_path, dst)
        collected.append((img_path, dst))

    return collected


def annotate(config_path: str = "configs/default.yaml") -> None:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    cfg = config["detection"]
    source_dir     = Path(cfg["annotation_source"])
    annotation_dir = Path(cfg["annotation_dir"])
    prompt         = cfg.get("prompt", "stag beetle")

    # autodistill import — v1 우선 시도, v2 fallback
    # Windows: flash-attn 경로 제한으로 v2 설치 실패 시 v1 사용
    #   pip install autodistill autodistill-grounded-sam
    try:
        from autodistill_grounded_sam import GroundedSAM
        from autodistill.detection import CaptionOntology
        _use_v2 = False
    except ImportError:
        try:
            from autodistill_grounded_sam_2 import GroundedSAM2
            from autodistill.detection import CaptionOntology
            _use_v2 = True
        except ImportError as e:
            raise ImportError(
                "autodistill 모델이 설치되지 않았습니다.\n"
                "pip install autodistill autodistill-grounded-sam"
            ) from e

    # 1단계: 이미지 수집 (평면 구조로 복사)
    flat_dir = annotation_dir / "_flat_images"
    print(f"[1/3] 이미지 수집: {source_dir} → {flat_dir}")
    image_map = collect_images(source_dir, flat_dir)
    print(f"      총 {len(image_map)}장 수집 완료")

    # 2단계: Grounded SAM 어노테이션
    version = "Grounded SAM 2" if _use_v2 else "Grounded SAM"
    print(f"[2/3] {version} 어노테이션 (프롬프트: '{prompt}')")
    if _use_v2:
        base_model = GroundedSAM2(ontology=CaptionOntology({prompt: "stag_beetle"}))
    else:
        base_model = GroundedSAM(ontology=CaptionOntology({prompt: "stag_beetle"}))

    # autodistill label() 은 images/, labels/ 를 annotation_dir 아래에 생성
    base_model.label(
        input_folder=str(flat_dir),
        output_folder=str(annotation_dir),
    )

    # 임시 flat 디렉토리 정리
    if flat_dir.exists():
        shutil.rmtree(flat_dir)

    # 3단계: 결과 리포트
    images_dir = annotation_dir / "images"
    labels_dir = annotation_dir / "labels"

    n_images = len(list(images_dir.glob("*"))) if images_dir.exists() else 0
    n_labels = len(list(labels_dir.glob("*.txt"))) if labels_dir.exists() else 0
    n_empty  = sum(
        1 for p in labels_dir.glob("*.txt") if p.stat().st_size == 0
    ) if labels_dir.exists() else 0

    print(f"[3/3] 어노테이션 완료")
    print(f"      이미지:          {n_images}장  →  {images_dir}")
    print(f"      라벨:            {n_labels}개  →  {labels_dir}")
    print(f"      빈 라벨(미검출): {n_empty}개  ({n_empty / n_labels * 100:.1f}%)" if n_labels else "")
    print()
    print("──── 다음 단계 선택 ────────────────────────────────────────────")
    print()
    print("[A] Label Studio 검토 없이 바로 진행 (권장: 첫 실험)")
    print("    python -m src.detection.annotation_utils")
    print()
    print("[B] Label Studio 검토 후 진행 (권장: 품질 향상)")
    print("    1. label-studio start")
    print("    2. python -m src.detection.ls_converter to_ls  # pre-annotation JSON 생성")
    print("    3. Label Studio에서 이미지 import → predictions import → 검토·수정")
    print("    4. Label Studio에서 'YOLO with Polygons' 형식으로 export → zip 다운로드")
    print("    5. python -m src.detection.ls_converter from_ls --zip <export.zip>")
    print("    6. python -m src.detection.annotation_utils")
    print("────────────────────────────────────────────────────────────────")


if __name__ == "__main__":
    annotate()
