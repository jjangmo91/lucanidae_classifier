"""
iNaturalist Data Scraper (Professional Version)
- Raw Data (JSON) & Processed Data (CSV) separation.
- Fixed Latitude/Longitude parsing issue.
- Enhanced Metadata Field Extraction.
"""

import time
import json
import logging
import yaml
from pathlib import Path
from typing import Dict, List

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm

# 로깅 설정 (현업 표준)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class INaturalistScraper:
    API_URL = "https://api.inaturalist.org/v1/observations"

    def __init__(self, taxon_id: int, place_id: int, output_dir: Path):
        self.taxon_id = taxon_id
        self.place_id = place_id
        self.output_dir = Path(output_dir)
        self.image_dir = self.output_dir / "images"
        self.metadata_path = self.output_dir / "metadata.csv"
        self.raw_json_path = self.output_dir / "raw_metadata.json"  # 원본 보존용
        
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.session = self._build_robust_session()

    def _build_robust_session(self) -> requests.Session:
        """네트워크 불안정 대응용 Retry 세션 구축"""
        session = requests.Session()
        retry = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504]
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    def fetch_metadata(self, max_pages: int = 50) -> List[Dict]:
        """API Pagination 처리 및 메타데이터 수집"""
        observations = []
        page = 1

        logger.info(f"Fetching metadata for Taxon: {self.taxon_id}, Place: {self.place_id}")
        
        while page <= max_pages:
            params = {
                "taxon_id": self.taxon_id,
                "place_id": self.place_id,
                "photos": "true",
                "per_page": 200,
                "page": page,
                "order_by": "observed_on",
                "order": "desc"
            }

            try:
                response = self.session.get(self.API_URL, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()
                
                results = data.get("results", [])
                if not results:
                    break

                observations.extend(results)
                logger.info(f"Page {page} fetched: {len(results)} records.")
                
                page += 1
                time.sleep(1)  # API Rate Limit 준수

            except Exception as e:
                logger.error(f"Failed to fetch page {page}: {e}")
                break

        return observations

    def parse_and_download(self, observations: List[Dict]) -> None:
        """메타데이터 파싱 및 원본 이미지 다운로드"""
        parsed_data = []

        logger.info(f"Starting process for {len(observations)} observations...")
        
        for obs in tqdm(observations, desc="Processing Observations"):
            obs_id = obs.get("id")
            taxon_info = obs.get("taxon", {})
            taxon_name = taxon_info.get("name", "unknown")
            common_name = taxon_info.get("preferred_common_name", "unknown")
            photos = obs.get("photos", [])

            if not photos:
                continue

            # 고해상도(large) 이미지 URL 추출
            img_url = photos[0].get("url", "").replace("square", "large")
            if not img_url:
                continue

            img_name = f"{obs_id}_{taxon_name.replace(' ', '_')}.jpg"
            img_path = self.image_dir / img_name

            # 이미지 다운로드 (존재하면 건너뜀)
            if not img_path.exists():
                try:
                    res = self.session.get(img_url, timeout=10)
                    res.raise_for_status()
                    with open(img_path, "wb") as f:
                        f.write(res.content)
                    time.sleep(0.5)
                except Exception as e:
                    logger.warning(f"Download failed for {img_url}: {e}")
                    continue

            # 위경도 문자열 파싱 (lat, lng 분리)
            location = obs.get("location")
            lat, lng = location.split(',') if location else (None, None)

            # 피처 추출
            parsed_data.append({
                "observation_id": obs_id,
                "scientific_name": taxon_name,
                "common_name": common_name,
                "latitude": lat,
                "longitude": lng,
                "observed_on": obs.get("observed_on_details", {}).get("date"),
                "quality_grade": obs.get("quality_grade"),             # 검증 등급
                "positional_accuracy": obs.get("positional_accuracy"), # 위치 오차(m)
                "user_id": obs.get("user", {}).get("login"),           # 관찰자
                "image_path": str(img_path.resolve())
            })

        # 메타데이터 CSV 저장 (Silver Layer)
        if parsed_data:
            df = pd.DataFrame(parsed_data)
            df.to_csv(self.metadata_path, index=False, encoding="utf-8-sig")
            logger.info(f"Saved refined metadata to {self.metadata_path}")

def main():
    # YAML 설정 로드
    with open("configs/default.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    dc_config = config['data_collection']

    scraper = INaturalistScraper(
        taxon_id=dc_config['taxon_id'],
        place_id=dc_config['place_id'],
        output_dir=dc_config['output_dir']
    )
    
    # API 데이터 수집
    raw_observations = scraper.fetch_metadata(max_pages=dc_config['max_pages'])
    
    # 원본 JSON 데이터 보존 (Bronze Layer)
    if raw_observations:
        with open(scraper.raw_json_path, "w", encoding="utf-8") as f:
            json.dump(raw_observations, f, ensure_ascii=False, indent=4)
        logger.info(f"Saved RAW JSON (Bronze) to {scraper.raw_json_path}")
    
    # 이미지 체크 및 정제된 CSV 저장 (Silver Layer)
    scraper.parse_and_download(raw_observations)

if __name__ == "__main__":
    main()