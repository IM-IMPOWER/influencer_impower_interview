"""Database connection and session management."""

import asyncio
from typing import AsyncGenerator

import structlog
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.pool import NullPool

from kol_api.config import settings

logger = structlog.get_logger()

# AIDEV-NOTE: Global database engine and session factory
engine = None
async_session_factory = None


class Base(DeclarativeBase):
    """SQLAlchemy declarative base class."""
    pass


async def init_database() -> None:
    """Initialize database engine and session factory."""
    global engine, async_session_factory
    
    # AIDEV-NOTE: Create async engine with connection pooling
    engine = create_async_engine(
        str(settings.database_url),
        echo=settings.database_echo,
        pool_size=settings.database_pool_size,
        max_overflow=settings.database_max_overflow,
        pool_pre_ping=True,
        # AIDEV-NOTE: Use NullPool in development to avoid connection issues
        poolclass=NullPool if settings.environment == "development" else None,
    )
    
    # AIDEV-NOTE: Create session factory
    async_session_factory = async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    
    # AIDEV-NOTE: Enable pgvector extension if not exists
    try:
        async with engine.begin() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm"))
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS btree_gin"))
        logger.info("PostgreSQL extensions initialized successfully")
    except Exception as e:
        logger.error("Failed to initialize PostgreSQL extensions", error=str(e))
        raise


async def close_database() -> None:
    """Close database connections."""
    global engine
    
    if engine:
        await engine.dispose()
        logger.info("Database connections closed")


async def get_database_session() -> AsyncGenerator[AsyncSession, None]:
    """Get async database session with proper cleanup."""
    if not async_session_factory:
        raise RuntimeError("Database not initialized. Call init_database() first.")
    
    async with async_session_factory() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def get_session() -> AsyncSession:
    """Get database session for use in dependency injection."""
    if not async_session_factory:
        raise RuntimeError("Database not initialized. Call init_database() first.")
    
    return async_session_factory()


# AIDEV-NOTE: Health check function for database connectivity
async def check_database_health() -> bool:
    """Check database connectivity and return health status."""
    try:
        async with get_session() as session:
            await session.execute(text("SELECT 1"))
            return True
    except Exception as e:
        logger.error("Database health check failed", error=str(e))
        return False