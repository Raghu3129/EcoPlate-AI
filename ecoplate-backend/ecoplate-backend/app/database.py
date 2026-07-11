"""
Database engine and session management for the EcoPlate AI backend.

Uses SQLAlchemy with a SQLite backend. Provides a FastAPI-compatible
dependency (`get_db`) that yields a scoped session per request.
"""

from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.config import DATABASE_URL

# `check_same_thread` must be disabled for SQLite when used with a
# multi-threaded server such as uvicorn.
connect_args = (
    {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
)

engine = create_engine(DATABASE_URL, connect_args=connect_args, echo=False)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    """Declarative base class for all ORM models."""

    pass


def get_db() -> Generator:
    """
    FastAPI dependency that provides a database session.

    Ensures the session is closed after the request completes, even if
    an exception is raised during handling.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create all database tables if they do not already exist."""
    # Import models here to ensure they are registered on Base.metadata
    # before create_all is invoked.
    from app.models import prediction  # noqa: F401

    Base.metadata.create_all(bind=engine)
