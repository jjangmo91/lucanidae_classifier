#!/usr/bin/env bash
# HuggingFace Hub에서 최신 모델을 pull 하고 hot-swap API를 호출한다
set -euo pipefail

MODEL_PATH="./models/weights/best_model.pth"
API_URL="${API_URL:-http://localhost:8000}"

huggingface-cli download \
    jjangmo91/lucanidae-models \
    best_model.pth \
    --local-dir ./models/weights/

curl -sf -X POST "${API_URL}/admin/reload-model" \
    -H "Content-Type: application/json" \
    -d "{\"weights_path\": \"${MODEL_PATH}\"}"

echo "[reload_model] 완료: $(date)"
