"""Async database engine and session factory for the AI service.

DATABASE_URL comes from the environment (docker-compose injects the
in-network postgres URL); the fallback targets the published port for
running uvicorn on the host.
"""

import os
from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

DEFAULT_DATABASE_URL = "postgresql+asyncpg://estimator:estimator@localhost:5432/estimator"

# create_async_engine does not connect — connections open lazily on first
# use, so importing this module never blocks on the database being up.
engine = create_async_engine(
    os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL),
    pool_pre_ping=True,
)

# expire_on_commit=False keeps ORM attributes (like Document.id) readable
# after commit without triggering an implicit refresh query.
session_factory = async_sessionmaker(engine, expire_on_commit=False)


async def get_session() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency: one session per request, closed on exit."""
    async with session_factory() as session:
        yield session
