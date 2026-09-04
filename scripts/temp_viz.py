from ultralytics import YOLO
import numpy as np
from PIL import Image, ImageDraw
import cv2, os

model = YOLO("models/weights/best_detector.pt")
img_path = "data/raw/field_data/IMG_20250521_211957.jpg"
out_dir  = "experiments/detection_viz"
os.makedirs(out_dir, exist_ok=True)

# 1) 원본 저장 (리사이즈)
orig = Image.open(img_path).convert("RGB")
W, H = orig.size
scale = 800 / max(W, H)
nw, nh = int(W*scale), int(H*scale)
orig_r = orig.resize((nw, nh), Image.LANCZOS)
orig_r.save(f"{out_dir}/01_original.jpg")

# 2) YOLOv8 결과 (bbox + mask 오버레이)
results = model(img_path, conf=0.2, verbose=False)
r = results[0]

img_np = np.array(orig_r)
vis = img_np.copy()

if r.masks is not None:
    # mask 오버레이 (초록 반투명)
    mask_full = r.masks.data[0].cpu().numpy()
    mask_res = cv2.resize(mask_full, (nw, nh))
    overlay = vis.copy()
    overlay[mask_res > 0.5] = [0, 220, 80]
    vis = cv2.addWeighted(vis, 0.45, overlay, 0.55, 0)

# bbox 그리기
box = r.boxes.xyxy[0].cpu().numpy() * scale
x1,y1,x2,y2 = int(box[0]),int(box[1]),int(box[2]),int(box[3])
conf = float(r.boxes.conf[0])
cv2.rectangle(vis, (x1,y1),(x2,y2),(0,220,80), 3)
# 레이블 배경
label = f"stag beetle  {conf:.2f}"
(tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
cv2.rectangle(vis,(x1,y1-th-10),(x1+tw+8,y1),(0,220,80),-1)
cv2.putText(vis, label,(x1+4,y1-5),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,0),2)

Image.fromarray(vis).save(f"{out_dir}/02_yolo_detection.jpg")

# 3) 마스크만 (seg_hard 스타일)
if r.masks is not None:
    seg = img_np.copy()
    seg[mask_res <= 0.5] = [100,100,100]
    Image.fromarray(seg).save(f"{out_dir}/03_seg_hard.jpg")

print("saved:", out_dir)
print(f"bbox: ({x1},{y1}) ~ ({x2},{y2}), conf={conf:.3f}")
