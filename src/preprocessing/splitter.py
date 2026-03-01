import os
import shutil
import logging
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class DatasetSplitter:
    def __init__(self, min_samples: int = 10):
        self.base_path = Path(".")
        self.src_dir = self.base_path / "data/processed"
        self.dest_dir = self.base_path / "data/final"
        self.min_samples = min_samples

    def get_dataset_summary(self) -> pd.DataFrame:
        """현재 클래스별 데이터 수량 파악"""
        data = []
        # src_dir 내의 모든 폴더를 순회
        for class_dir in self.src_dir.iterdir():
            if class_dir.is_dir():
                # 이미지 파일 목록 추출
                images = list(class_dir.glob("*.jpg"))
                if len(images) > 0:
                    data.append({
                        "class_name": class_dir.name, 
                        "count": len(images), 
                        "paths": images
                    })
        
        # 데이터가 없을 경우 빈 데이터프레임에 컬럼명만 정의해서 반환
        if not data:
            logger.warning(f"No image data found in {self.src_dir}. Check if Step 2 was successful.")
            return pd.DataFrame(columns=["class_name", "count", "paths"])
            
        return pd.DataFrame(data)

    def split(self):
        """데이터 분할 및 물리적 이동"""
        df = self.get_dataset_summary()
        
        # 최소 수량 미달 클래스 필터링
        valid_df = df[df['count'] >= self.min_samples].copy()
        invalid_df = df[df['count'] < self.min_samples]
        
        if not invalid_df.empty:
            logger.warning(f"Excluded {len(invalid_df)} classes due to insufficient data (min: {self.min_samples}).")

        all_paths = []
        all_labels = []
        
        for _, row in valid_df.iterrows():
            all_paths.extend(row['paths'])
            all_labels.extend([row['class_name']] * row['count'])

        # Stratified Split (8:1:1)
        # Train : Temp (8:2)
        train_idx, temp_idx = train_test_split(
            range(len(all_paths)), 
            test_size=0.2, 
            stratify=all_labels, 
            random_state=42
        )
        
        # Temp -> Val : Test (1:1)
        temp_labels = [all_labels[i] for i in temp_idx]
        val_idx_sub, test_idx_sub = train_test_split(
            range(len(temp_idx)), 
            test_size=0.5, 
            stratify=temp_labels, 
            random_state=42
        )
        
        val_idx = [temp_idx[i] for i in val_idx_sub]
        test_idx = [temp_idx[i] for i in test_idx_sub]

        # 파일 복사 실행
        split_map = {'train': train_idx, 'val': val_idx, 'test': test_idx}
        
        for split_name, indices in split_map.items():
            logger.info(f"Organizing {split_name} set...")
            for idx in tqdm(indices):
                src_path = all_paths[idx]
                label = all_labels[idx]
                
                target_dir = self.dest_dir / split_name / label
                target_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy(src_path, target_dir / src_path.name)

        logger.info(f"Dataset split completed. Results saved to {self.dest_dir}")

if __name__ == "__main__":
    splitter = DatasetSplitter(min_samples=10)
    splitter.split()