"""갤러리 · 랭킹 · 지도 · 프로필 API."""

from __future__ import annotations

import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel
from sqlalchemy import asc, desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps import get_session
from src.api.routes.auth import get_current_user_id, require_user
from src.db.models import Specimen, User, SPECIES_RARITY_SCORE
from src.db.repository import UserRepository

router = APIRouter(prefix="/api/v1", tags=["community"])

# 희귀도 → 종 키 목록
_RARITY_SPECIES: dict[int, list[str]] = {}
for _sp, _r in SPECIES_RARITY_SCORE.items():
    _RARITY_SPECIES.setdefault(_r, []).append(_sp)


# ── 공개 갤러리 ─────────────────────────────────────────────────────────────

@router.get("/gallery")
async def gallery(
    limit: int = 30,
    offset: int = 0,
    sort: str = Query("newest", pattern="^(newest|oldest|confidence)$"),
    rarity: Optional[int] = Query(None, ge=1, le=4),
    session: AsyncSession = Depends(get_session),
):
    """이미지가 있는 표본 공개 피드. sort=newest|oldest|confidence, rarity=1-4."""
    q = select(Specimen).where(Specimen.image_url.isnot(None))

    if rarity is not None:
        keys = _RARITY_SPECIES.get(rarity, [])
        q = q.where(Specimen.predicted_species.in_(keys))

    if sort == "oldest":
        q = q.order_by(asc(Specimen.upload_time))
    elif sort == "confidence":
        q = q.order_by(desc(Specimen.predicted_confidence))
    else:
        q = q.order_by(desc(Specimen.upload_time))

    result = await session.execute(q.offset(offset).limit(limit))
    return [_specimen_card(s) for s in result.scalars().all()]


# ── 표본 단건 조회 (OG 메타 태그용) ────────────────────────────────────────

@router.get("/specimens/{specimen_id}")
async def get_specimen(
    specimen_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
):
    """공개 표본 단건 조회 — OG 메타 태그 생성용."""
    result = await session.execute(
        select(Specimen).where(Specimen.id == specimen_id)
    )
    s = result.scalar_one_or_none()
    if s is None:
        raise HTTPException(status_code=404, detail="not found")
    return _specimen_card(s)


# ── 비로그인 표본 소유권 이전 ────────────────────────────────────────────────

class ClaimRequest(BaseModel):
    specimen_ids: list[uuid.UUID]

@router.post("/my/claim-specimens")
async def claim_specimens(
    body: ClaimRequest,
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    """비로그인 상태로 업로드한 표본을 로그인 후 내 계정으로 연결."""
    user_id = require_user(request)
    user_uuid = uuid.UUID(user_id)

    if not body.specimen_ids:
        return {"claimed": 0}

    # user_id가 없는 표본만 소유권 이전 (이미 남의 것은 건드리지 않음)
    result = await session.execute(
        select(Specimen).where(
            Specimen.id.in_(body.specimen_ids),
            Specimen.user_id.is_(None),
        )
    )
    specimens = result.scalars().all()

    for s in specimens:
        s.user_id = user_uuid

    if specimens:
        await session.commit()
        repo = UserRepository(session)
        await repo.recalculate_grade(user_uuid)
        await session.commit()

    return {"claimed": len(specimens)}


# ── 내 표본 목록 ────────────────────────────────────────────────────────────

@router.get("/my/specimens")
async def my_specimens(
    request: Request,
    limit: int = 50,
    session: AsyncSession = Depends(get_session),
):
    user_id = require_user(request)
    result = await session.execute(
        select(Specimen)
        .where(Specimen.user_id == uuid.UUID(user_id))
        .order_by(Specimen.upload_time.desc())
        .limit(limit)
    )
    return [_specimen_card(s) for s in result.scalars().all()]


# ── 내 지도 핀 ──────────────────────────────────────────────────────────────

@router.get("/my/map")
async def my_map(
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    """gps_grid 있는 내 표본만 반환 (남의 위치는 공개 안 함)."""
    user_id = require_user(request)
    result = await session.execute(
        select(Specimen)
        .where(
            Specimen.user_id == uuid.UUID(user_id),
            Specimen.gps_grid.isnot(None),
        )
    )
    pins = []
    for s in result.scalars().all():
        lat, lng = _parse_gps_grid(s.gps_grid)
        if lat is None:
            continue
        pins.append({
            "id":       str(s.id),
            "lat":      lat,
            "lng":      lng,
            "species":  s.predicted_species,
            "image_url": s.image_url,
            "upload_time": s.upload_time.isoformat() if s.upload_time else None,
        })
    return pins


# ── 랭킹 ───────────────────────────────────────────────────────────────────

@router.get("/ranking")
async def ranking(
    limit: int = 50,
    session: AsyncSession = Depends(get_session),
):
    repo = UserRepository(session)
    users = await repo.ranking(limit=limit)
    return [
        {
            "rank":      i + 1,
            "user_id":   str(u.id),
            "username":  u.username,
            "avatar":    u.avatar_url,
            "grade":     u.grade,
            "specialty": u.specialty,
            "score":     u.total_score,
        }
        for i, u in enumerate(users)
    ]


# ── 유저 프로필 (공개) ───────────────────────────────────────────────────────

@router.get("/users/{user_id}")
async def user_profile(
    user_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
):
    repo = UserRepository(session)
    user = await repo.get_by_id(user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="user not found")

    result = await session.execute(
        select(Specimen)
        .where(Specimen.user_id == user_id, Specimen.image_url.isnot(None))
        .order_by(Specimen.upload_time.desc())
        .limit(12)
    )
    recent = [_specimen_card(s) for s in result.scalars().all()]

    return {
        "user_id":   str(user.id),
        "username":  user.username,
        "avatar":    user.avatar_url,
        "grade":     user.grade,
        "specialty": user.specialty,
        "score":     user.total_score,
        "recent_specimens": recent,
    }


# ── helpers ─────────────────────────────────────────────────────────────────

def _specimen_card(s: Specimen) -> dict:
    return {
        "id":         str(s.id),
        "image_url":  s.image_url,
        "species":    s.predicted_species,
        "confidence": round(s.predicted_confidence or 0, 4),
        "result_type": s.result_type,
        "upload_time": s.upload_time.isoformat() if s.upload_time else None,
        "user_id":    str(s.user_id) if s.user_id else None,
    }


def _parse_gps_grid(gps_grid: str) -> tuple[float | None, float | None]:
    """'lat:37.123,lng:127.456' 형식 파싱."""
    try:
        parts = dict(p.split(":") for p in gps_grid.split(","))
        return float(parts["lat"]), float(parts["lng"])
    except Exception:
        return None, None
