"""
Main Entry Point - Smart Pipeline Edition
이미 완료된 단계는 자동으로 스킵하고, 변경사항이 필요한 단계만 실행합니다.
"""

import logging
from pathlib import Path
from src.data_collection.scraper import main as run_scraper
from src.preprocessing.cleaner import DataCleaner
from src.preprocessing.splitter import DatasetSplitter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

def run_pipeline(force_update=False):
    logger.info("=== Starting Lucanidae Data Pipeline ===")

    # 설정 경로 정의
    metadata_csv = Path("data/raw/inaturalist/metadata.csv")
    processed_dir = Path("data/processed")
    final_dir = Path("data/final")

    # Step 1: Data Collection
    # [CRITICAL] 수동 데이터 정제 후에는 절대 scraper를 실행하지 않습니다.
    # 기존: if not metadata_csv.exists() or force_update:
    # 수정: 아래처럼 조건을 강제로 False로 처리합니다.
    if False: # 수동 정제 데이터 보호를 위해 일시적으로 비활성화
        logger.info("Step 1: Metadata not found. Starting Scraper...")
        run_scraper()
    else:
        logger.info("Step 1: Manual cleaning mode. Skipping Scraper to protect raw data.")

    # Step 2: Taxonomic Cleaning
    # 정제된 데이터를 다시 분류 체계에 맞게 배치합니다.
    # 기존 processed_dir를 삭제하고 돌리는 것이 안전합니다.
    logger.info("Step 2: Starting Taxonomic Cleaning...")
    cleaner = DataCleaner()
    cleaner.process()

    # Step 3: Dataset Splitting
    # 정제된 데이터를 바탕으로 train/val/test 세트를 다시 구성합니다.
    logger.info("Step 3: Starting Dataset Splitting...")
    splitter = DatasetSplitter(min_samples=10)
    splitter.split()

    logger.info("=== Pipeline Execution Completed ===")

if __name__ == "__main__":
    # [주의] 수동으로 사진을 지운 후에는 force_update와 상관없이 
    # Step 1이 실행되지 않도록 위 로직에서 제어해야 합니다.
    run_pipeline(force_update=True)