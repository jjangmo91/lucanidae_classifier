import pandas as pd
import shutil
import logging
import yaml
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class DataCleaner:
    def __init__(self, config_path: str = "configs/default.yaml", taxonomy_path: str = "configs/taxonomy.yaml"):
        self.base_path = Path(".")
        
        # 설정 파일 로드
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        with open(taxonomy_path, "r", encoding="utf-8") as f:
            self.taxonomy_config = yaml.safe_load(f)

        self.raw_metadata_path = Path(self.config['data_collection']['output_dir']) / "metadata.csv"
        self.processed_dir = self.base_path / "data/processed"
        
        self.taxonomy_map = self.taxonomy_config['taxonomy_mapping']
        self.unidentified_list = self.taxonomy_config['unidentified_groups']

    def process(self):
        """데이터 정제 및 물리적 분류 실행"""
        if not self.raw_metadata_path.exists():
            logger.error(f"Metadata not found at {self.raw_metadata_path}")
            return

        df = pd.read_csv(self.raw_metadata_path)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        success_count = 0
        logger.info("Starting taxonomic data consolidation using YAML config...")

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Classifying"):
            raw_name = row['scientific_name']
            
            # Taxonomy mapping 적용
            target_name = self.taxonomy_map.get(raw_name, raw_name.replace(" ", "_"))

            # Unidentified 그룹 처리
            if target_name in self.unidentified_list:
                target_name = "Unidentified_Lucanidae"

            target_path = self.processed_dir / target_name
            target_path.mkdir(parents=True, exist_ok=True)

            source_img = Path(row['image_path'])
            if source_img.exists():
                shutil.copy(source_img, target_path / source_img.name)
                success_count += 1

        logger.info(f"Consolidation completed. Processed: {success_count} images.")

if __name__ == "__main__":
    cleaner = DataCleaner()
    cleaner.process()