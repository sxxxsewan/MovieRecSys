"""Shared SQLAlchemy declarative metadata."""

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Base for all persistent database models."""
