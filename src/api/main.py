"""
FastAPI 앱 진입점.

실행:
    uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
load_dotenv()

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.api.deps import init_db
from src.api.routes import admin, feedback, predict
from src.ml.manager import ModelManager

_cors_origins = [o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")]

app = FastAPI(
    title="Lucanidae Classifier API",
    version="1.0.0",
    docs_url="/docs" if os.getenv("ENV", "development") == "development" else None,
    redoc_url=None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict.router)
app.include_router(feedback.router)
app.include_router(admin.router)

upload_dir = Path("data/uploads")
upload_dir.mkdir(parents=True, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=str(upload_dir)), name="uploads")


@app.on_event("startup")
async def startup() -> None:
    db_url = os.environ["DATABASE_URL"]
    init_db(db_url)
    ModelManager.get_instance()          # 앱 시작 시 모델 1회 로드


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}
