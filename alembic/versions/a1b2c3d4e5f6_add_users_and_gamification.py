"""add users and gamification

Revision ID: a1b2c3d4e5f6
Revises: 0529c0f5a20c
Create Date: 2026-05-15 00:00:00.000000
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

revision = "a1b2c3d4e5f6"
down_revision = "0529c0f5a20c"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("google_id", sa.String(64), nullable=False, unique=True),
        sa.Column("email", sa.String(255), nullable=False, unique=True),
        sa.Column("username", sa.String(80), nullable=False),
        sa.Column("avatar_url", sa.String(512), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("total_score", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("grade", sa.String(30), nullable=False, server_default="딱린이"),
        sa.Column("specialty", sa.String(30), nullable=True),
        # specialty: 레어헌터 | 분류학자 | 도감탐험가 | 채집왕
    )

    op.add_column("specimens", sa.Column("user_id", UUID(as_uuid=True), nullable=True))
    op.create_foreign_key(
        "fk_specimens_user_id",
        "specimens", "users",
        ["user_id"], ["id"],
        ondelete="SET NULL",
    )
    op.create_index("idx_spec_user", "specimens", ["user_id"])


def downgrade() -> None:
    op.drop_index("idx_spec_user", table_name="specimens")
    op.drop_constraint("fk_specimens_user_id", "specimens", type_="foreignkey")
    op.drop_column("specimens", "user_id")
    op.drop_table("users")
