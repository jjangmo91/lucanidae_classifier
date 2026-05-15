#!/usr/bin/env bash
# PostgreSQL → Hugging Face Datasets 일일 백업
set -euo pipefail

DUMP_FILE="/tmp/lucanidae_$(date +%Y%m%d).sql.gz"

pg_dump "$DATABASE_URL" | gzip > "$DUMP_FILE"

huggingface-cli upload \
    --repo-type dataset \
    jjangmo91/lucanidae-backups \
    "$DUMP_FILE" \
    "db/$(basename "$DUMP_FILE")"

rm "$DUMP_FILE"
echo "[backup_db] 완료: $(date)"
