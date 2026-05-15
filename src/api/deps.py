"""
FastAPI 의존성 주입 — 모델·DB 세션을 라우터에 주입한다.
모든 라우터는 이 모듈을 통해서만 의존성을 받는다.
"""

from __future__ import annotations

from typing import AsyncGenerator

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.ml.manager import ModelManager
from src.services.feedback import FeedbackService
from src.services.prediction import PredictionService

# ------------------------------------------------------------------ #
# DB                                                                   #
# ------------------------------------------------------------------ #

_engine: create_async_engine | None = None
_SessionLocal: async_sessionmaker | None = None


def init_db(database_url: str) -> None:
    global _engine, _SessionLocal
    _engine       = create_async_engine(database_url, echo=False)
    _SessionLocal = async_sessionmaker(_engine, expire_on_commit=False)


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    async with _SessionLocal() as session:
        yield session


# ------------------------------------------------------------------ #
# ML                                                                   #
# ------------------------------------------------------------------ #

def get_model_manager() -> ModelManager:
    return ModelManager.get_instance()


def get_prediction_service(
    manager: ModelManager = Depends(get_model_manager),
) -> PredictionService:
    preprocessing_mode = manager._config.get("inference", {}).get("preprocessing_mode", "full")
    return PredictionService(manager, preprocessing_mode=preprocessing_mode)


def get_feedback_service(
    session: AsyncSession = Depends(get_session),
) -> FeedbackService:
    return FeedbackService(session)
