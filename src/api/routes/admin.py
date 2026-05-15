from __future__ import annotations

import os
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps import get_session
from src.db.repository import AdminActionLogRepository, SpecimenRepository

router = APIRouter(prefix="/admin", tags=["admin"])

ADMIN_ID = "admin"
_ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")


def require_admin(request: Request) -> None:
    """ADMIN_TOKEN 환경변수가 설정된 경우 Bearer 토큰으로 검증."""
    if not _ADMIN_TOKEN:
        return  # 토큰 미설정 시 로컬 개발 모드 (Cloudflare Access로 대체 예정)
    auth = request.headers.get("Authorization", "")
    if auth != f"Bearer {_ADMIN_TOKEN}":
        raise HTTPException(status_code=401, detail="Unauthorized")


class AdminAction(BaseModel):
    action: str                      # approve / reject / correct / foreign / review
    correct_species: Optional[str] = None
    sex: Optional[str] = None
    male_form: Optional[str] = None
    note: Optional[str] = None


@router.get("/queue")
async def get_queue(
    request: Request,
    limit:  int = 100,
    status: str = "pending",
    session: AsyncSession = Depends(get_session),
):
    require_admin(request)
    repo  = SpecimenRepository(session)
    if status == "all":
        items = await repo.list_all(limit=limit)
    else:
        items = await repo.list_by_status(status=status, limit=limit)
    return [
        {
            "id":                str(s.id),
            "upload_time":       s.upload_time.isoformat() if s.upload_time else None,
            "result_type":       s.result_type,
            "predicted_species": s.predicted_species,
            "confidence":        s.predicted_confidence,
            "user_corrected":    s.user_corrected,
            "correct_species":   s.correct_species,
            "sex":               s.sex,
            "admin_status":      s.admin_status,
            "is_trainable":      s.is_trainable,
            "image_url":         s.image_url,
        }
        for s in items
    ]


@router.post("/items/{specimen_id}/action")
async def admin_action(
    request: Request,
    specimen_id: uuid.UUID,
    body: AdminAction,
    session: AsyncSession = Depends(get_session),
):
    require_admin(request)
    specimens = SpecimenRepository(session)
    logs      = AdminActionLogRepository(session)

    specimen = await specimens.get_by_id(specimen_id)
    if specimen is None:
        raise HTTPException(status_code=404, detail="specimen not found")

    update: dict = {"admin_status": body.action}

    if body.action == "approve":
        update["is_trainable"]  = True
        update["label_source"]  = "expert"
    elif body.action == "correct":
        update["is_trainable"]  = True
        update["label_source"]  = "expert"
        update["correct_species"] = body.correct_species
        update["sex"]           = body.sex
        update["male_form"]     = body.male_form
    elif body.action in ("reject", "foreign"):
        update["is_trainable"]  = False

    await specimens.update(specimen_id, update)
    await logs.log(specimen_id, body.action, ADMIN_ID, body.note)
    await session.commit()
    return {"status": "ok"}
