import pandas as pd
import shutil

# 파생물 배포가 금지되어 학습에서도 제외하는 라이선스 (DESIGN.md 11.5)
EXCLUDED_LICENSES = {"cc-by-nc-nd", "cc-by-nd"}
import logging
import yaml
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class DataCleaner:
    def __init__(self, config_path: str = "configs/default.yaml", taxonomy_path: str = "configs/taxonomy.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        with open(taxonomy_path, "r", encoding="utf-8") as f:
            taxonomy = yaml.safe_load(f)

        prep = config["data_preprocessing"]
        self.merged_metadata_path = Path(prep["merged_metadata"])
        self.processed_dir        = Path(prep["processed_dir"])

        self.target_classes    = set(taxonomy["target_classes"])
        self.name_mapping      = taxonomy["scientific_name_mapping"]
        self.exclusion_list    = set(taxonomy["exclusion_list"])

    def _resolve_class(self, raw_name: str) -> str | None:
        """학명 → canonical 클래스명 변환. 제외 대상이면 None 반환."""
        if raw_name in self.exclusion_list:
            return None

        # 직접 매핑
        if raw_name in self.name_mapping:
            return self.name_mapping[raw_name]

        # 공백을 언더스코어로 변환한 형태가 target_classes에 있으면 그대로 사용
        normalized = raw_name.replace(" ", "_")
        if normalized in self.target_classes:
            return normalized

        # 매핑 없음
        return None

    def process(self):
        if not self.merged_metadata_path.exists():
            logger.error(f"Merged metadata not found: {self.merged_metadata_path}")
            logger.error("먼저 merger.py를 실행하세요.")
            return

        df = pd.read_csv(self.merged_metadata_path, encoding="utf-8-sig")
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        success, skipped_excluded, skipped_unmapped, skipped_missing = 0, 0, 0, 0
        skipped_license = 0
        unmapped_names: set[str] = set()

        logger.info(f"총 {len(df)}건 처리 시작...")

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Cleaning"):
            raw_name   = str(row["scientific_name"]).strip()
            image_path = Path(str(row["image_path"]))

            # 제외 목록 확인
            if raw_name in self.exclusion_list:
                skipped_excluded += 1
                continue

            # 라이선스 제외 (DESIGN.md 11.5)
            # ND(NoDerivatives)는 파생물 배포가 금지된다. crop/seg 가 명백한
            # 파생물이므로 다툼의 여지를 없애기 위해 학습에서도 제외한다.
            # 전체의 1.7% 라 잃는 것이 없다.
            # ARR 은 학습에 쓰되 재배포하지 않는다 (여기서 거르지 않는다).
            if str(row.get("photo_license", "")).strip().lower() in EXCLUDED_LICENSES:
                skipped_license += 1
                continue

            # canonical 클래스명 결정
            canonical = self._resolve_class(raw_name)
            if canonical is None:
                skipped_unmapped += 1
                unmapped_names.add(raw_name)
                continue

            # target_classes 외 클래스 경고
            if canonical not in self.target_classes:
                logger.warning(f"target_classes 외 클래스 생성됨: '{canonical}' (원본: '{raw_name}')")

            # 이미지 복사
            if not image_path.exists():
                skipped_missing += 1
                continue

            target_dir = self.processed_dir / canonical
            target_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(image_path, target_dir / image_path.name)
            success += 1

        logger.info(
            f"완료 — 복사: {success}건 | "
            f"제외(exclusion): {skipped_excluded}건 | "
            f"제외(라이선스 ND): {skipped_license}건 | "
            f"매핑 없음: {skipped_unmapped}건 | "
            f"파일 없음: {skipped_missing}건"
        )

        if unmapped_names:
            logger.warning(f"매핑되지 않은 학명 목록 ({len(unmapped_names)}개):")
            for name in sorted(unmapped_names):
                logger.warning(f"  - '{name}'")


if __name__ == "__main__":
    cleaner = DataCleaner()
    cleaner.process()
