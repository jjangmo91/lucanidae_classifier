"""
SQLAlchemy ORM 모델 — v6 설계문서 DB 스키마 기준.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, Float, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class Specimen(Base):
    """
    업로드된 사슴벌레 이미지 1건 = 표본(specimen) 1행.
    is_trainable=True 인 행만 학습 풀로 사용한다.
    """

    __tablename__ = "specimens"

    id           = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id   = Column(String(16),  nullable=True)   # pHash + date_hour + gps_grid
    image_hash   = Column(String(16),  nullable=True)   # perceptual hash
    image_url    = Column(String(512), nullable=True)   # Cloudflare R2 URL
    upload_time  = Column(DateTime(timezone=True), default=datetime.utcnow)
    gps_grid     = Column(String(32),  nullable=True)   # 1km 격자 좌표 (raw GPS 저장 안 함)
    season       = Column(String(10),  nullable=True)   # spring | summer | autumn | winter

    # ── 모델 예측 ──────────────────────────────────────────────────────────
    result_type                    = Column(String(20),  nullable=True)
    # identified | low_confidence | uncertain | foreign | no_beetle

    predicted_species              = Column(String(80),  nullable=True)
    predicted_confidence           = Column(Float,       nullable=True)
    predicted_calibrated_confidence = Column(Float,      nullable=True)
    predicted_sex                  = Column(String(10),  nullable=True)   # male | female | unknown
    predicted_male_form            = Column(String(15),  nullable=True)   # major | minor | intermediate | unknown
    top3_json                      = Column(Text,        nullable=True)   # JSON: [{species, confidence}, ...]
    is_ood                         = Column(Boolean,     default=False)
    ood_score                      = Column(Float,       nullable=True)
    preprocessing_mode             = Column(String(20),  nullable=True)   # full | bbox | seg_hard | seg_soft

    # ── 사용자 피드백 ──────────────────────────────────────────────────────
    user_corrected   = Column(Boolean,    nullable=True)
    correct_species  = Column(String(80), nullable=True)
    sex              = Column(String(10), nullable=True)   # male | female | unknown
    male_form        = Column(String(15), nullable=True)   # major | minor | intermediate | female | unknown

    # ── 레이블 품질 ────────────────────────────────────────────────────────
    label_level   = Column(Integer,     nullable=True)
    # 0=없음 | 1=종만 | 2=종+성별 | 3=종+성별+male_form
    is_trainable  = Column(Boolean,     default=False)
    label_source  = Column(String(20),  nullable=True)
    # expert | community_verified | pseudo

    # ── 관리자 ─────────────────────────────────────────────────────────────
    admin_status      = Column(String(20), default="pending")
    # pending | approve | correct | reject | foreign | review
    admin_note        = Column(Text,       nullable=True)
    admin_reviewed_by = Column(String(50), nullable=True)
    admin_reviewed_at = Column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        Index("idx_spec_session",     "session_id"),
        Index("idx_spec_admin",       "admin_status"),
        Index("idx_spec_trainable",   "is_trainable", "label_source"),
        Index("idx_spec_result_type", "result_type"),
        Index("idx_spec_label_level", "label_level"),
    )


class AdminActionLog(Base):
    """관리자 모든 행동을 감사(audit) 추적."""

    __tablename__ = "admin_action_logs"

    id          = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    specimen_id = Column(UUID(as_uuid=True), nullable=False)
    action      = Column(String(30), nullable=False)
    # approve | reject | correct | foreign | review
    admin_id    = Column(String(50), nullable=False)
    note        = Column(Text,       nullable=True)
    created_at  = Column(DateTime(timezone=True), default=datetime.utcnow)
