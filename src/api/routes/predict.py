from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from PIL import Image
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps import get_prediction_service, get_session
from src.db.models import Specimen
from src.db.repository import SpecimenRepository
from src.services.prediction import PredictionResult, PredictionService

router = APIRouter(prefix="/api/v1", tags=["predict"])

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/heic", "image/heif"}
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20MB


@router.post("/predict", response_model=list[dict])
async def predict(
    file: UploadFile = File(...),
    service: PredictionService = Depends(get_prediction_service),
    session: AsyncSession = Depends(get_session),
):
    if file.content_type and file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=415, detail="이미지 파일만 업로드 가능합니다 (JPG, PNG, WebP)")

    raw = await file.read()
    if len(raw) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="파일 크기는 20MB 이하여야 합니다")

    import io
    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=422, detail="이미지를 열 수 없습니다")
    results = await service.predict_image(image)

    repo     = SpecimenRepository(session)
    response = []
    for r in results:
        specimen = await repo.create({
            "result_type":          r.result_type,
            "predicted_species":    r.species,
            "predicted_confidence": r.confidence,
            "is_ood":               r.is_ood,
            "ood_score":            r.ood_score,
            "top3_json":            json.dumps([{"species": p.species, "confidence": round(p.confidence, 4)} for p in r.top3]),
            "preprocessing_mode":   r.preprocessing_mode,
        })

        # 이미지 저장
        img_path = UPLOAD_DIR / f"{specimen.id}.jpg"
        image.save(img_path, format="JPEG", quality=85)
        await repo.update(specimen.id, {"image_url": f"/uploads/{specimen.id}.jpg"})

        await session.commit()
        response.append(_to_response(r, str(specimen.id)))
    return response


@router.get("/stats")
async def stats(session: AsyncSession = Depends(get_session)):
    today = datetime.now(timezone.utc).date()
    result = await session.execute(
        select(func.count()).select_from(Specimen)
        .where(func.date(Specimen.upload_time) == today)
    )
    count_today = result.scalar_one()
    total = await session.execute(select(func.count()).select_from(Specimen))
    return {"today_count": count_today, "total_count": total.scalar_one()}


def _to_response(r: PredictionResult, prediction_id: str) -> dict:
    return {
        "prediction_id":     prediction_id,
        "result_type":       r.result_type,
        "species":           r.species,
        "species_ko":        r.species_ko,
        "confidence":        round(r.confidence, 4),
        "is_ood":            r.is_ood,
        "ood_score":         round(r.ood_score, 4),
        "top3":              [{"species": p.species, "confidence": round(p.confidence, 4)} for p in r.top3],
        "is_fallback":       r.is_fallback,
        "preprocessing_mode": r.preprocessing_mode,
        "latency_ms":        round(r.latency_ms, 1),
    }
