from ultralytics import YOLO
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2

model = YOLO("models/weights/best_detector.pt")

# 여러 사진 테스트해서 검출되는 거 찾기
import os
imgs = [f for f in os.listdir("data/raw/field_data") if f.endswith(".jpg")][:10]
for img_name in imgs:
    img_path = f"data/raw/field_data/{img_name}"
    results = model(img_path, conf=0.2, verbose=False)
    r = results[0]
    if len(r.boxes) > 0:
        print(f"FOUND: {img_name}, conf={r.boxes.conf.tolist()}")
        break
    else:
        print(f"no detection: {img_name}")
