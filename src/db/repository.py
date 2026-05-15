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

from src.db.models import AdminActionLog, Specimen


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
