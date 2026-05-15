"""
피드백 서비스 — 사용자 피드백 처리 및 학습 가능 여부 판정.

골든 라벨 기준 (v5):
    - label_source='expert'            → 관리자 직접 승인
    - label_source='community_verified' → user_corrected=True AND 관리자 approved
    - label_source='pseudo'            → calibrated_confidence >= 0.95 (별도 격리)
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from src.db.repository import AdminActionLogRepository, SpecimenRepository


@dataclass
class FeedbackInput:
    is_correct: bool
    correct_species: Optional[str]
    sex: Optional[str]             # male / female / unknown
    male_form: Optional[str]       # major / minor / intermediate / unknown
    gps_optional: Optional[list[float]]  # [lat, lon] — 1km 격자로 변환 후 저장
    user_nickname: Optional[str]


class FeedbackService:

    def __init__(self, session: AsyncSession):
        self._specimens = SpecimenRepository(session)
        self._logs      = AdminActionLogRepository(session)
        self._session   = session

    async def submit(self, specimen_id: uuid.UUID, feedback: FeedbackInput) -> bool:
        specimen = await self._specimens.get_by_id(specimen_id)
        if specimen is None:
            return False

        update_data: dict = {
            "user_corrected":  not feedback.is_correct,
            "correct_species": feedback.correct_species,
            "sex":             feedback.sex,
            "male_form":       feedback.male_form,
        }

        if feedback.gps_optional:
            update_data["gps_grid"] = _to_grid(*feedback.gps_optional)

        await self._specimens.update(specimen_id, update_data)
        await self._session.commit()
        return True


def _to_grid(lat: float, lon: float, km: float = 1.0) -> str:
    """위경도를 1km 격자 문자열로 변환 (raw GPS 저장 안 함)."""
    grid_lat = round(lat / (km / 111.0)) * (km / 111.0)
    grid_lon = round(lon / (km / 111.0 / abs(lat or 1))) * (km / 111.0 / abs(lat or 1))
    return f"{grid_lat:.4f},{grid_lon:.4f}"
