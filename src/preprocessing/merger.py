import pandas as pd
import logging
import yaml
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

UNIFIED_COLUMNS = ["observation_id", "scientific_name", "image_path", "latitude", "longitude", "observed_on", "source", "quality_grade"]


class DataMerger:
    def __init__(self, config_path: str = "configs/default.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        prep = config["data_preprocessing"]
        self.inat_path    = Path(prep["inaturalist_metadata"])
        self.field_path   = Path(prep["field_metadata"])
        self.output_path  = Path(prep["merged_metadata"])

    def _load_inaturalist(self) -> pd.DataFrame:
        df = pd.read_csv(self.inat_path, encoding="utf-8")
        return pd.DataFrame({
            "observation_id":  df["observation_id"].astype(str),
            "scientific_name": df["scientific_name"],
            "image_path":      df["image_path"],
            "latitude":        df["latitude"],
            "longitude":       df["longitude"],
            "observed_on":     df["observed_on"],
            "source":          "inaturalist",
            "quality_grade":   df.get("quality_grade", pd.Series(["unknown"] * len(df))),

            # 저작권 정보는 반드시 끝까지 끌고 간다 (DESIGN.md 11.2).
            # 여기서 흘리면 어떤 이미지를 공개할 수 있는지 판단할 수 없게 된다.
            "photo_id":            df.get("photo_id", pd.Series([None] * len(df))),
            "photo_license":       df.get("photo_license", pd.Series(["UNKNOWN"] * len(df))),
            "photo_attribution":   df.get("photo_attribution", pd.Series([None] * len(df))),
            "observation_license": df.get("observation_license", pd.Series(["UNKNOWN"] * len(df))),
            "obscured":            df.get("obscured", pd.Series([None] * len(df))),
        })

    def _load_field(self) -> pd.DataFrame:
        df = pd.read_csv(self.field_path, encoding="utf-8-sig")
        # Each field image is treated as a unique observation
        obs_ids = [f"field_{Path(p).stem}" for p in df["image_path"]]
        return pd.DataFrame({
            "observation_id":  obs_ids,
            "scientific_name": df["scientific_name"],
            "image_path":      df["image_path"],
            "latitude":        df["latitude"],
            "longitude":       df["longitude"],
            "observed_on":     df["observed_on"],
            "source":          "field",
            "quality_grade":   "field",

            # 현장 촬영분은 직접 촬영이므로 배포 조건을 우리가 정한다.
            # 3층 테스트셋 공개를 위해 CC-BY 를 기본값으로 둔다 (DESIGN.md 11.4).
            "photo_id":            None,
            "photo_license":       df.get("photo_license", pd.Series(["cc-by"] * len(df))),
            "photo_attribution":   df.get("photo_attribution", pd.Series([None] * len(df))),
            "observation_license": df.get("photo_license", pd.Series(["cc-by"] * len(df))),
            "obscured":            False,
        })

    def merge(self):
        if not self.inat_path.exists():
            logger.error(f"iNaturalist metadata not found: {self.inat_path}")
            return
        if not self.field_path.exists():
            logger.error(f"Field metadata not found: {self.field_path}")
            return

        inat_df  = self._load_inaturalist()
        field_df = self._load_field()

        logger.info(f"iNaturalist: {len(inat_df)}건")
        logger.info(f"Field data:  {len(field_df)}건")

        merged = pd.concat([inat_df, field_df], ignore_index=True)

        # 중복 이미지 제거 (image_path 기준)
        before = len(merged)
        merged = merged.drop_duplicates(subset=["image_path"])
        dropped = before - len(merged)
        if dropped:
            logger.warning(f"중복 이미지 {dropped}건 제거됨")

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(self.output_path, index=False, encoding="utf-8-sig")
        logger.info(f"병합 완료: 총 {len(merged)}건 -> {self.output_path}")

        # 소스별 종 분포 요약
        summary = merged.groupby(["source", "scientific_name"]).size().reset_index(name="count")
        logger.info(f"\n{summary.to_string(index=False)}")


if __name__ == "__main__":
    merger = DataMerger()
    merger.merge()
