"""
Database configuration - Simplified for Railway
"""

import os
from sqlalchemy import create_engine, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

# Get database URL
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./moi_biometric.db")

if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

is_sqlite = DATABASE_URL.startswith("sqlite")
print(f"📊 DB: {'SQLite' if is_sqlite else 'PostgreSQL'}", flush=True)

Base = declarative_base()
engine = None
SessionLocal = None

def get_engine():
    global engine
    if engine is None:
        if is_sqlite:
            engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
        else:
            engine = create_engine(DATABASE_URL, poolclass=NullPool)
    return engine

def get_session_local():
    global SessionLocal
    if SessionLocal is None:
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=get_engine())
    return SessionLocal

def get_db():
    db = get_session_local()()
    try:
        yield db
    finally:
        db.close()

def init_db():
    """Initialize database - called on startup"""
    print("🔄 init_db...", flush=True)
    try:
        from models import Visitor
        eng = get_engine()
        Base.metadata.create_all(bind=eng)
        print("✅ Tables created", flush=True)
    except Exception as e:
        print(f"⚠️ init_db: {e}", flush=True)

def get_db_info():
    return {"type": "SQLite" if is_sqlite else "PostgreSQL", "ok": engine is not None}
