"""
SQLAlchemy ORM 모델 — v6 설계문서 DB 스키마 기준.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


GRADE_THRESHOLDS = [
    (200, None),       # 6티어: 분기 (specialty 필드로 결정)
    (120, "루카나이더"),
    (60,  "야간채집러"),
    (30,  "주간채집러"),
    (10,  "표본수집가"),
    (0,   "딱린이"),
]

SPECIES_RARITY_SCORE: dict[str, int] = {
    # S티어 (4점) — 채집 난이도 기준
    "Dorcus_hopei_binodulosus":           4,  # 왕사슴벌레
    "Dorcus_carinulatus_koreanus":         4,  # 털보왕사슴벌레
    "Aegus_laevicollis_subnitidus":        4,  # 꼬마넓적사슴벌레
    "Nigidius_miwai":                      4,  # 뿔꼬마사슴벌레
    "Dorcus_tenuihirsutus":                4,  # 엷은털왕사슴벌레
    "Figulus_binodulus":                   4,  # 큰꼬마사슴벌레
    # A티어 (3점)
    "Prosopocoilus_astacoides_blanchardi": 3,  # 두점박이사슴벌레
    # B티어 (2점)
    "Lucanus_maculifemoratus_dybowskyi":   2,  # 사슴벌레
    "Prosopocoilus_inclinatus_inclinatus": 2,  # 톱사슴벌레
    "Dorcus_rubrofemoratus_rubrofemoratus":2,  # 홍다리사슴벌레
    "Prismognathus_dauricus":              2,  # 다우리아사슴벌레
    "Platycerus_hongwonpyoi_hongwonpyoi":  2,  # 원표애보라사슴벌레
    "Dorcus_consentaneus_consentaneus":    2,  # 참넓적사슴벌레
    # C티어 (1점)
    "Dorcus_titanus_castanicolor":         1,  # 넓적사슴벌레
    "Dorcus_rectus_rectus":                1,  # 애사슴벌레
    "Figulus_punctatus":                   1,  # 길쭉꼬마사슴벌레
}
DEFAULT_RARITY_SCORE = 2  # 목록에 없는 종 기본값 (B티어)


def compute_grade(score: int, rare_ratio: float, correction_count: int, species_count: int) -> tuple[str, str | None]:
    """(grade, specialty) 반환. specialty는 6티어일 때만 설정."""
    if score >= 200:
        if rare_ratio >= 0.6:
            return "6티어", "레어헌터"
        if correction_count >= 20:
            return "6티어", "분류학자"
        if species_count >= 12:
            return "6티어", "도감탐험가"
        return "6티어", "채집왕"
    for threshold, name in GRADE_THRESHOLDS[1:]:
        if score >= threshold:
            return name, None
    return "딱린이", None


class User(Base):
    """Google OAuth 유저."""

    __tablename__ = "users"

    id          = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    google_id   = Column(String(64),  nullable=False, unique=True)
    email       = Column(String(255), nullable=False, unique=True)
    username    = Column(String(80),  nullable=False)
    avatar_url  = Column(String(512), nullable=True)
    created_at  = Column(DateTime(timezone=True), default=datetime.utcnow)
    total_score = Column(Integer, nullable=False, default=0)
    grade       = Column(String(30),  nullable=False, default="딱린이")
    specialty   = Column(String(30),  nullable=True)
    # specialty: 레어헌터 | 분류학자 | 도감탐험가 | 채집왕


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

    # ── 유저 연결 ──────────────────────────────────────────────────────────
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)

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
