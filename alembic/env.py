import os
from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool
from alembic import context
from dotenv import load_dotenv

load_dotenv()

# src/db/models.py 의 Base.metadata 를 가져와서
# Alembic이 테이블 구조를 자동으로 파악하게 함
from src.db.models import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata

# .env 의 DATABASE_URL 을 읽어서 alembic.ini 의 sqlalchemy.url 을 덮어씀
# asyncpg(비동기) → psycopg2(동기) 로 변환 (Alembic은 동기 드라이버 필요)
def get_sync_url() -> str:
    url = os.environ.get("DATABASE_URL", "")
    return url.replace("postgresql+asyncpg://", "postgresql://")


def run_migrations_offline() -> None:
    context.configure(
        url=get_sync_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    cfg = config.get_section(config.config_ini_section, {})
    cfg["sqlalchemy.url"] = get_sync_url()

    connectable = engine_from_config(cfg, prefix="sqlalchemy.", poolclass=pool.NullPool)

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
