from __future__ import annotations

import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from src.api.deps import get_feedback_service
from src.services.feedback import FeedbackInput, FeedbackService

router = APIRouter(prefix="/api/v1", tags=["feedback"])


class FeedbackRequest(BaseModel):
    is_correct: bool
    correct_species: Optional[str] = None
    sex: Optional[str] = None
    male_form: Optional[str] = None
    gps_optional: Optional[list[float]] = None
    user_nickname: Optional[str] = None


@router.post("/predictions/{prediction_id}/feedback")
async def submit_feedback(
    prediction_id: uuid.UUID,
    body: FeedbackRequest,
    service: FeedbackService = Depends(get_feedback_service),
):
    ok = await service.submit(
        specimen_id = prediction_id,
        feedback    = FeedbackInput(**body.model_dump()),
    )
    if not ok:
        raise HTTPException(status_code=404, detail="prediction not found")
    return {"status": "ok"}
