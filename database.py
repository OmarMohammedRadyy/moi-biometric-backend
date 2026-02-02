"""
Database configuration for MOI Biometric System
Supports both SQLite (development) and PostgreSQL (production)
"""

import os
import sys
from sqlalchemy import create_engine, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

# Get database URL from environment
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./moi_biometric.db")

# Handle Railway's PostgreSQL URL format (postgres:// -> postgresql://)
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# Determine if using SQLite
is_sqlite = DATABASE_URL.startswith("sqlite")

# Print masked URL for debugging
if "@" in DATABASE_URL:
    masked_url = DATABASE_URL.split("@")[-1]
else:
    masked_url = DATABASE_URL
print(f"📊 Database: {'SQLite' if is_sqlite else 'PostgreSQL'} -> {masked_url}", flush=True)

# Create base class for models FIRST
Base = declarative_base()

# Create engine
engine = None
SessionLocal = None

try:
    if is_sqlite:
        engine = create_engine(
            DATABASE_URL,
            connect_args={"check_same_thread": False},
            echo=False
        )
    else:
        # PostgreSQL with connection timeout
        engine = create_engine(
            DATABASE_URL,
            poolclass=NullPool,
            connect_args={
                "connect_timeout": 5,  # 5 second timeout
            },
            echo=False
        )
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    print("✅ Engine created", flush=True)
except Exception as e:
    print(f"❌ Engine error: {e}", flush=True)


def get_db():
    if SessionLocal is None:
        raise Exception("No database")
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    global engine, SessionLocal
    
    print("🔄 init_db...", flush=True)
    
    if engine is None:
        print("⚠️ No engine", flush=True)
        return
    
    try:
        # Quick import
        from models import Visitor
        print("📥 Models OK", flush=True)
        
        # Test with timeout
        print("🔗 Testing...", flush=True)
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            print(f"✅ DB OK: {result.scalar()}", flush=True)
        
        # Create tables
        print("📋 Creating tables...", flush=True)
        Base.metadata.create_all(bind=engine)
        print("✅ Ready!", flush=True)
        
    except Exception as e:
        print(f"❌ DB Error: {type(e).__name__}: {e}", flush=True)


def get_db_info():
    return {
        "type": "SQLite" if is_sqlite else "PostgreSQL",
        "connected": engine is not None
    }
