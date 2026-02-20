"""
DecisionMesh Database Layer
SQLAlchemy 2.0 async engine + session factory.
Supports SQLite (default) and PostgreSQL (optional).
"""
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from decisionmesh.config import settings


class Base(DeclarativeBase):
    """Shared declarative base for all ORM models."""
    pass


def create_engine(database_url: str | None = None) -> AsyncEngine:
    """Create the async SQLAlchemy engine."""
    url = database_url or settings.DATABASE_URL

    connect_args = {}
    if url.startswith("sqlite"):
        # SQLite requires check_same_thread=False for async use
        connect_args["check_same_thread"] = False

    return create_async_engine(
        url,
        echo=False,
        connect_args=connect_args,
    )


# Module-level engine and session factory
engine: AsyncEngine = create_engine()

async_session_factory: async_sessionmaker[AsyncSession] = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Async context manager for database sessions."""
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def init_db() -> None:
    """Create all tables. Use for development/testing; prefer Alembic for production."""
    async with engine.begin() as conn:
        from decisionmesh.models import (  # noqa: F401 — import all ORM models
            agent_log,
            condition,
            counterfactual,
            decision,
            divergence,
            observation,
            premise,
        )
        await conn.run_sync(Base.metadata.create_all)


async def drop_db() -> None:
    """Drop all tables. USE WITH CAUTION."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
