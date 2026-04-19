"""
Test database fixtures.

Overrides get_db with an in-memory SQLite engine so tests run without
a real CockroachDB instance.  StaticPool shares one connection across
all async operations so the in-memory DB persists for the life of the test.
"""

import asyncio
import os
import tempfile

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

# Set required env vars before any app module is imported.
os.environ.setdefault("HELIUM_DATA_DIR", tempfile.mkdtemp())
os.environ.setdefault("HELIUM_MIN_IMAGES", "2")
os.environ.setdefault("HELIUM_DATABASE_URL", "sqlite+aiosqlite://")

from app.db.engine import get_db  # noqa: E402
from app.db.models import Base  # noqa: E402
from app.main import app  # noqa: E402

_TEST_ENGINE = create_async_engine(
    "sqlite+aiosqlite://",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)


def pytest_configure(config):
    asyncio.run(_create_tables())


async def _create_tables():
    async with _TEST_ENGINE.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def _get_test_db():
    factory = async_sessionmaker(_TEST_ENGINE, class_=AsyncSession, expire_on_commit=False)
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


@pytest.fixture(autouse=True)
def override_get_db():
    app.dependency_overrides[get_db] = _get_test_db
    yield
    app.dependency_overrides.pop(get_db, None)
