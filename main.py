"""
Main Entry Point — Lucanidae Data Pipeline
16종 분류 기준: 국립생물자원관 국가생물종목록

Step 1  : Scraper          — 수동 데이터 보호를 위해 비활성화 유지
Step 1.5: DataMerger       — iNaturalist + field_data → merged_metadata.csv
Step 2  : DataCleaner      — merged_metadata.csv → data/processed/ (16-class)
Step 3  : DatasetSplitter  — data/processed/ → data/final/ (train/val/test)
"""

import logging
import shutil
from pathlib import Path

from src.data_collection.scraper import main as run_scraper
from src.preprocessing.merger import DataMerger
from src.preprocessing.cleaner import DataCleaner
from src.preprocessing.splitter import DatasetSplitter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_pipeline():
    logger.info("=== Lucanidae Data Pipeline 시작 ===")

    # Step 1: Data Collection (비활성화)
    # [중요] 수동 정제 데이터 보호를 위해 Scraper를 실행하지 않습니다.
    logger.info("Step 1: Scraper — 수동 정제 모드, 건너뜀")

    # Step 1.5: Data Merge
    logger.info("Step 1.5: DataMerger 시작 — iNaturalist + field_data 통합")
    merger = DataMerger()
    merger.merge()

    # Step 2: Taxonomic Cleaning
    # data/processed/ 를 초기화 후 재구성합니다.
    processed_dir = Path("data/processed")
    if processed_dir.exists():
        logger.info("Step 2: 기존 data/processed/ 초기화 중...")
        shutil.rmtree(processed_dir)

    logger.info("Step 2: DataCleaner 시작 — 16-class 기준 분류")
    cleaner = DataCleaner()
    cleaner.process()

    # Step 3: Dataset Splitting
    # data/final/ 을 초기화 후 재구성합니다.
    final_dir = Path("data/final")
    if final_dir.exists():
        logger.info("Step 3: 기존 data/final/ 초기화 중...")
        shutil.rmtree(final_dir)

    logger.info("Step 3: DatasetSplitter 시작 — train/val/test 분할")
    splitter = DatasetSplitter(min_samples=10)
    splitter.split()

    logger.info("=== Pipeline 완료 ===")


if __name__ == "__main__":
    run_pipeline()
