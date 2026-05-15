import shutil
import logging
import yaml
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class DatasetSplitter:
    def __init__(
        self,
        min_samples: int = 10,
        config_path: str = "configs/default.yaml",
        taxonomy_path: str = "configs/taxonomy.yaml",
    ):
        self.min_samples = min_samples
        self.src_dir  = Path("data/processed")
        self.dest_dir = Path("data/final")

        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        with open(taxonomy_path, "r", encoding="utf-8") as f:
            tax = yaml.safe_load(f)

        self.merged_meta = Path(cfg["data_preprocessing"]["merged_metadata"])
        self.target_classes = set(tax["target_classes"])
        self.name_mapping   = tax["scientific_name_mapping"]
        self.exclusion_list = set(tax["exclusion_list"])

    # ── Class-name resolution (mirrors cleaner.py) ──────────────────────────

    def _resolve_class(self, raw_name: str) -> str | None:
        if raw_name in self.exclusion_list:
            return None
        if raw_name in self.name_mapping:
            return self.name_mapping[raw_name]
        normalized = raw_name.replace(" ", "_")
        if normalized in self.target_classes:
            return normalized
        return None

    # ── Specimen-level split (preferred) ────────────────────────────────────

    def _split_specimen_level(self, df_meta: pd.DataFrame) -> None:
        """
        Group images by observation_id, split groups 80/10/10.
        Prevents the same individual from appearing in both train and test.
        """
        records = []
        for _, row in df_meta.iterrows():
            canonical = self._resolve_class(str(row["scientific_name"]).strip())
            if canonical is None:
                continue
            img_name = Path(str(row["image_path"])).name
            src = self.src_dir / canonical / img_name
            if not src.exists():
                continue
            records.append({
                "obs_id":    str(row["observation_id"]),
                "class":     canonical,
                "src":       src,
            })

        if not records:
            logger.warning("메타데이터에서 유효한 이미지를 찾을 수 없습니다. image-level 분할로 전환합니다.")
            self._split_image_level()
            return

        df = pd.DataFrame(records)

        # Class-level sample count filter (at observation group level)
        obs_class = df.groupby("obs_id")["class"].agg(lambda x: x.mode()[0]).reset_index()
        obs_class.columns = ["obs_id", "class"]
        class_counts = obs_class["class"].value_counts()
        valid_classes = class_counts[class_counts >= self.min_samples].index
        invalid = class_counts[class_counts < self.min_samples]
        if not invalid.empty:
            logger.warning(
                f"최소 관찰 수({self.min_samples}) 미달 클래스 제외: "
                f"{list(invalid.index)}"
            )
        obs_class = obs_class[obs_class["class"].isin(valid_classes)]
        df = df[df["obs_id"].isin(obs_class["obs_id"])]

        obs_ids    = obs_class["obs_id"].values
        obs_labels = obs_class["class"].values

        # 80 / 20 first split (group-aware, stratified)
        gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
        train_obs_idx, temp_obs_idx = next(gss.split(obs_ids, obs_labels, groups=obs_ids))
        train_obs = set(obs_ids[train_obs_idx])
        temp_obs  = set(obs_ids[temp_obs_idx])

        # 50/50 split of temp → val / test
        temp_obs_arr    = obs_ids[temp_obs_idx]
        temp_labels_arr = obs_labels[temp_obs_idx]
        val_obs_sub, test_obs_sub = train_test_split(
            temp_obs_arr,
            test_size=0.50,
            stratify=temp_labels_arr,
            random_state=42,
        )
        val_obs  = set(val_obs_sub)
        test_obs = set(test_obs_sub)

        split_map = {"train": train_obs, "val": val_obs, "test": test_obs}
        counts = {s: 0 for s in split_map}

        for split_name, obs_set in split_map.items():
            subset = df[df["obs_id"].isin(obs_set)]
            logger.info(f"[{split_name}] {len(obs_set)} observations, {len(subset)} images")
            for _, row in tqdm(subset.iterrows(), total=len(subset), desc=split_name):
                dst_dir = self.dest_dir / split_name / row["class"]
                dst_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy(row["src"], dst_dir / row["src"].name)
                counts[split_name] += 1

        logger.info(f"Specimen-level split 완료 — " + " | ".join(f"{k}: {v}" for k, v in counts.items()))
        logger.info(f"결과 저장: {self.dest_dir}")

    # ── Image-level split (fallback) ─────────────────────────────────────────

    def _split_image_level(self) -> None:
        """Image-level fallback (no observation_id available)."""
        logger.warning(
            "Image-level 분할 사용 중 — 같은 개체의 사진이 train/test에 모두 포함될 수 있습니다."
        )
        all_paths, all_labels = [], []
        for cls_dir in self.src_dir.iterdir():
            if not cls_dir.is_dir():
                continue
            imgs = list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.jpeg")) + list(cls_dir.glob("*.png"))
            if len(imgs) < self.min_samples:
                continue
            all_paths.extend(imgs)
            all_labels.extend([cls_dir.name] * len(imgs))

        train_idx, temp_idx = train_test_split(
            range(len(all_paths)), test_size=0.2, stratify=all_labels, random_state=42
        )
        temp_labels = [all_labels[i] for i in temp_idx]
        val_sub, test_sub = train_test_split(
            range(len(temp_idx)), test_size=0.5, stratify=temp_labels, random_state=42
        )
        val_idx  = [temp_idx[i] for i in val_sub]
        test_idx = [temp_idx[i] for i in test_sub]

        for split_name, indices in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
            for idx in tqdm(indices, desc=split_name):
                src = all_paths[idx]
                dst = self.dest_dir / split_name / all_labels[idx]
                dst.mkdir(parents=True, exist_ok=True)
                shutil.copy(src, dst / src.name)

        logger.info(f"Image-level split 완료. 결과 저장: {self.dest_dir}")

    # ── Entry point ──────────────────────────────────────────────────────────

    def split(self) -> None:
        if self.merged_meta.exists():
            df_meta = pd.read_csv(self.merged_meta, encoding="utf-8-sig")
            if "observation_id" in df_meta.columns:
                logger.info("observation_id 컬럼 감지 → specimen-level 분할 사용")
                self._split_specimen_level(df_meta)
                return
            else:
                logger.warning("merged_metadata.csv에 observation_id 없음 → image-level 분할로 전환")
        else:
            logger.warning(f"merged_metadata.csv 없음 ({self.merged_meta}) → image-level 분할로 전환")

        self._split_image_level()


if __name__ == "__main__":
    splitter = DatasetSplitter(min_samples=10)
    splitter.split()
