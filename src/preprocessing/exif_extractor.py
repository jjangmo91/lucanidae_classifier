# src/preprocessing/exif_extractor.py
import os
import pandas as pd
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from pathlib import Path

def get_decimal_from_dms(dms, ref):
    """도분초(DMS) 포맷을 십진수 위경도로 변환"""
    degrees, minutes, seconds = float(dms[0]), float(dms[1]), float(dms[2])
    decimal = degrees + minutes / 60 + seconds / 3600
    if ref in ['S', 'W']:
        decimal = -decimal
    return round(decimal, 6)

def extract_metadata(image_path):
    """단일 이미지에서 시간 및 GPS 메타데이터 추출"""
    metadata = {
        'file_name': os.path.basename(image_path),
        'file_path': str(image_path).replace('\\', '/'),
        'date_taken': None,
        'latitude': None,
        'longitude': None,
        'species': ''
    }
    
    try:
        image = Image.open(image_path)
        exif_data = image._getexif()
        if not exif_data:
            return metadata
            
        for tag_id, value in exif_data.items():
            tag_name = TAGS.get(tag_id, tag_id)
            if tag_name == 'DateTimeOriginal':
                metadata['date_taken'] = value
            elif tag_name == 'GPSInfo':
                gps_data = {GPSTAGS.get(t, t): value[t] for t in value}
                if 'GPSLatitude' in gps_data and 'GPSLongitude' in gps_data:
                    metadata['latitude'] = get_decimal_from_dms(gps_data['GPSLatitude'], gps_data.get('GPSLatitudeRef', 'N'))
                    metadata['longitude'] = get_decimal_from_dms(gps_data['GPSLongitude'], gps_data.get('GPSLongitudeRef', 'E'))
    except Exception:
        pass
        
    return metadata

def update_field_metadata(data_dir):
    """기존 수동 입력 데이터를 보존하며 신규 이미지 데이터만 CSV에 병합"""
    output_path = Path(data_dir) / 'field_metadata.csv'
    image_extensions = {'.jpg', '.jpeg', '.png'}
    
    # 기존 데이터 로드 (수기 입력 보존용)
    existing_files = set()
    if output_path.exists():
        existing_df = pd.read_csv(output_path)
        existing_files = set(existing_df['file_name'].tolist())
    else:
        existing_df = pd.DataFrame()

    # 신규 추가된 이미지 스캔
    new_metadata = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if Path(file).suffix.lower() in image_extensions:
                if file not in existing_files:
                    img_path = Path(root) / file
                    new_metadata.append(extract_metadata(img_path))
    
    # 신규 데이터 병합 및 저장
    if new_metadata:
        new_df = pd.DataFrame(new_metadata)
        final_df = pd.concat([existing_df, new_df], ignore_index=True)
        final_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"Added {len(new_metadata)} new records to {output_path.name}.")
    else:
        print("No new images found. Metadata is up to date.")

if __name__ == "__main__":
    FIELD_DATA_DIR = "data/raw/field_data"
    update_field_metadata(FIELD_DATA_DIR)