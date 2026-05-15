from __future__ import annotations

import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from sqlalchemy import select
from src.api.deps import get_session
from src.db.models import Specimen
from src.db.models import SPECIES_RARITY_SCORE, DEFAULT_RARITY_SCORE
from src.db.repository import UserRepository
from src.services.auth import create_jwt, decode_jwt, verify_google_token

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])


class GoogleLoginBody(BaseModel):
    id_token: str


def get_current_user_id(request: Request) -> Optional[str]:
    """Authorization: Bearer <jwt> 헤더에서 user_id 추출. 없으면 None."""
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        return None
    try:
        return decode_jwt(auth[7:])
    except ValueError:
        return None


def require_user(request: Request) -> str:
    user_id = get_current_user_id(request)
    if user_id is None:
        raise HTTPException(status_code=401, detail="로그인이 필요합니다")
    return user_id


@router.post("/google")
async def google_login(
    body: GoogleLoginBody,
    session: AsyncSession = Depends(get_session),
):
    try:
        info = await verify_google_token(body.id_token)
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))

    repo = UserRepository(session)
    user = await repo.upsert(
        google_id  = info["google_id"],
        email      = info["email"],
        username   = info["username"],
        avatar_url = info["avatar_url"],
    )
    await session.commit()

    token = create_jwt(str(user.id))
    return {
        "token":    token,
        "user_id":  str(user.id),
        "username": user.username,
        "avatar":   user.avatar_url,
        "grade":    user.grade,
        "specialty": user.specialty,
        "score":    user.total_score,
    }


@router.get("/me")
async def me(
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    user_id = require_user(request)
    uid     = uuid.UUID(user_id)
    repo    = UserRepository(session)
    user    = await repo.get_by_id(uid)
    if user is None:
        raise HTTPException(status_code=404, detail="user not found")

    # 통계 계산
    rows = await session.execute(
        select(Specimen.predicted_species, Specimen.user_corrected)
        .where(Specimen.user_id == uid)
    )
    specimens = rows.all()

    found_species:   set[str] = set()
    rare_count       = 0
    correction_count = 0
    for sp, corrected in specimens:
        if sp:
            found_species.add(sp)
            if SPECIES_RARITY_SCORE.get(sp, DEFAULT_RARITY_SCORE) >= 3:
                rare_count += 1
        if corrected:
            correction_count += 1

    total        = len(specimens)
    rare_ratio   = round(rare_count / total, 3) if total else 0.0

    return {
        "user_id":          str(user.id),
        "username":         user.username,
        "avatar":           user.avatar_url,
        "email":            user.email,
        "grade":            user.grade,
        "specialty":        user.specialty,
        "score":            user.total_score,
        "created_at":       user.created_at.isoformat() if user.created_at else None,
        # 진행도 stats
        "total_uploads":    total,
        "found_species":    sorted(found_species),
        "rare_ratio":       rare_ratio,
        "correction_count": correction_count,
        "species_count":    len(found_species),
    }
