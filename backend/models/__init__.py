#!/usr/bin/env python3
"""
SCIO - Data Models
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from backend.config import Config

# SQLAlchemy Base
Base = declarative_base()

# Database Engine (echo=False for production, no SQL logging)
engine = create_engine(Config.DATABASE_URL, echo=False)

# Session Factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    """Dependency für Database Session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Initialisiert alle Tabellen"""
    Base.metadata.create_all(bind=engine)


# Importiere alle Models
from .job import Job
from .api_key import APIKey
from .earnings import Earning

__all__ = ['Base', 'engine', 'SessionLocal', 'get_db', 'init_db', 'Job', 'APIKey', 'Earning']
