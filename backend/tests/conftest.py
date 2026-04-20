"""
Test database fixtures.

Uses a file-based SQLite DB so the connection isn't scoped to a single
event loop.  Tables are created synchronously via a plain SQLAlchemy engine
(no asyncio involved) before pytest starts any test.  The async engine then
connects to the same file for all test requests.
"""

import os
import tempfile

import pytest
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

# --- env vars must be set before any app module is imported ---
os.environ.setdefault("HELIUM_DATA_DIR", tempfile.mkdtemp())
os.environ.setdefault("HELIUM_MIN_IMAGES", "2")

_DB_PATH = os.path.join(tempfile.mkdtemp(), "test.db")
os.environ.setdefault("HELIUM_DATABASE_URL", f"sqlite+aiosqlite:///{_DB_PATH}")

from app.db.engine import get_db  # noqa: E402
from app.db.models import Base  # noqa: E402
from app.main import app  # noqa: E402

_TEST_ENGINE = create_async_engine(f"sqlite+aiosqlite:///{_DB_PATH}")


def pytest_configure(config):
    # Sync engine: no event loop, no cross-loop lifetime issues
    sync_engine = create_engine(f"sqlite:///{_DB_PATH}")
    Base.metadata.create_all(sync_engine)
    sync_engine.dispose()


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
