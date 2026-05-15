#!/usr/bin/env bash
# 이미지 볼륨 → Hugging Face Datasets rsync 백업
set -euo pipefail

huggingface-cli upload \
    --repo-type dataset \
    jjangmo91/lucanidae-backups \
    /app/data/images \
    "images/"

echo "[backup_images] 완료: $(date)"
