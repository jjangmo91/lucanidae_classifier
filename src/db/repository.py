"""
Repository 패턴 — DB 접근을 서비스 레이어에서 완전히 분리한다.
서비스는 이 인터페이스만 사용하며 SQLAlchemy를 직접 노출하지 않는다.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import AdminActionLog, Specimen, User, SPECIES_RARITY_SCORE, DEFAULT_RARITY_SCORE, compute_grade


class SpecimenRepository:

    def __init__(self, session: AsyncSession):
        self._session = session

    async def create(self, data: dict) -> Specimen:
        specimen = Specimen(**data)
        self._session.add(specimen)
        await self._session.flush()
        return specimen

    async def get_by_id(self, specimen_id: uuid.UUID) -> Optional[Specimen]:
        result = await self._session.execute(
            select(Specimen).where(Specimen.id == specimen_id)
        )
        return result.scalar_one_or_none()

    async def update(self, specimen_id: uuid.UUID, data: dict) -> Optional[Specimen]:
        specimen = await self.get_by_id(specimen_id)
        if specimen is None:
            return None
        for key, value in data.items():
            setattr(specimen, key, value)
        await self._session.flush()
        return specimen

    async def list_pending_admin(self, limit: int = 50) -> list[Specimen]:
        result = await self._session.execute(
            select(Specimen)
            .where(Specimen.admin_status == "pending")
            .order_by(Specimen.upload_time)
            .limit(limit)
        )
        return list(result.scalars().all())

    async def list_by_status(self, status: str, limit: int = 100) -> list[Specimen]:
        result = await self._session.execute(
            select(Specimen)
            .where(Specimen.admin_status == status)
            .order_by(Specimen.upload_time.desc())
            .limit(limit)
        )
        return list(result.scalars().all())

    async def list_all(self, limit: int = 100) -> list[Specimen]:
        result = await self._session.execute(
            select(Specimen)
            .order_by(Specimen.upload_time.desc())
            .limit(limit)
        )
        return list(result.scalars().all())

    async def list_trainable(self) -> list[Specimen]:
        result = await self._session.execute(
            select(Specimen).where(Specimen.is_trainable == True)
        )
        return list(result.scalars().all())


class UserRepository:

    def __init__(self, session: AsyncSession):
        self._session = session

    async def get_by_google_id(self, google_id: str) -> Optional[User]:
        result = await self._session.execute(
            select(User).where(User.google_id == google_id)
        )
        return result.scalar_one_or_none()

    async def get_by_id(self, user_id: uuid.UUID) -> Optional[User]:
        result = await self._session.execute(
            select(User).where(User.id == user_id)
        )
        return result.scalar_one_or_none()

    async def create(self, google_id: str, email: str, username: str, avatar_url: Optional[str]) -> User:
        user = User(
            google_id  = google_id,
            email      = email,
            username   = username,
            avatar_url = avatar_url,
        )
        self._session.add(user)
        await self._session.flush()
        return user

    async def upsert(self, google_id: str, email: str, username: str, avatar_url: Optional[str]) -> User:
        user = await self.get_by_google_id(google_id)
        if user is None:
            return await self.create(google_id, email, username, avatar_url)
        user.email      = email
        user.username   = username
        user.avatar_url = avatar_url
        await self._session.flush()
        return user

    async def recalculate_grade(self, user_id: uuid.UUID) -> User:
        """업로드 통계 기반으로 점수·등급·specialty 재계산."""
        from sqlalchemy import func
        user = await self.get_by_id(user_id)
        if user is None:
            raise ValueError(f"user {user_id} not found")

        rows = await self._session.execute(
            select(Specimen.predicted_species, Specimen.user_corrected)
            .where(Specimen.user_id == user_id)
        )
        specimens = rows.all()

        total_score    = 0
        rare_count     = 0
        correction_count = 0
        species_set: set[str] = set()

        for species, corrected in specimens:
            score = SPECIES_RARITY_SCORE.get(species or "", DEFAULT_RARITY_SCORE)
            total_score += score
            if score >= 3:
                rare_count += 1
            if corrected:
                correction_count += 1
            if species:
                species_set.add(species)

        rare_ratio = rare_count / len(specimens) if specimens else 0.0
        grade, specialty = compute_grade(total_score, rare_ratio, correction_count, len(species_set))

        user.total_score = total_score
        user.grade       = grade
        user.specialty   = specialty
        await self._session.flush()
        return user

    async def ranking(self, limit: int = 50) -> list[User]:
        result = await self._session.execute(
            select(User).order_by(User.total_score.desc()).limit(limit)
        )
        return list(result.scalars().all())


class AdminActionLogRepository:

    def __init__(self, session: AsyncSession):
        self._session = session

    async def log(
        self,
        specimen_id: uuid.UUID,
        action: str,
        admin_id: str,
        note: Optional[str] = None,
    ) -> AdminActionLog:
        entry = AdminActionLog(
            specimen_id = specimen_id,
            action      = action,
            admin_id    = admin_id,
            note        = note,
        )
        self._session.add(entry)
        await self._session.flush()
        return entry
