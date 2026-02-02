"""
Database configuration for MOI Biometric System
Supports both SQLite (development) and PostgreSQL (production)
"""

import os
import sys
from sqlalchemy import create_engine, event, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool, NullPool

# Get database URL from environment
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./moi_biometric.db")

# Handle Railway's PostgreSQL URL format (postgres:// -> postgresql://)
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# Determine if using SQLite
is_sqlite = DATABASE_URL.startswith("sqlite")

print(f"📊 Database type: {'SQLite' if is_sqlite else 'PostgreSQL'}", flush=True)

# Create base class for models FIRST
Base = declarative_base()

# Create engine based on database type
engine = None

def create_db_engine():
    global engine
    try:
        if is_sqlite:
            # SQLite configuration
            engine = create_engine(
                DATABASE_URL,
                connect_args={"check_same_thread": False},
                poolclass=StaticPool,
                echo=False
            )
        else:
            # PostgreSQL configuration - use NullPool to avoid connection issues
            engine = create_engine(
                DATABASE_URL,
                poolclass=NullPool,  # Create new connection each time
                echo=False
            )
        print("✅ Database engine created", flush=True)
        return engine
    except Exception as e:
        print(f"❌ Failed to create engine: {e}", flush=True)
        return None

# Create engine immediately
engine = create_db_engine()

# Create session factory
if engine:
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
else:
    SessionLocal = None


def get_db():
    """
    Dependency for getting database session.
    """
    if SessionLocal is None:
        raise Exception("Database not initialized")
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """
    Initialize database tables.
    """
    global engine, SessionLocal
    
    print("🔄 init_db() called...", flush=True)
    
    if engine is None:
        print("⚠️ Engine is None, trying to create...", flush=True)
        engine = create_db_engine()
        if engine:
            SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    if engine is None:
        print("❌ Cannot initialize - no engine", flush=True)
        return
    
    try:
        print("📥 Importing models...", flush=True)
        from models import Visitor
        
        print("🔗 Testing connection...", flush=True)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print("✅ Connection OK", flush=True)
        
        print("📋 Creating tables...", flush=True)
        Base.metadata.create_all(bind=engine)
        print(f"✅ Database ready: {'SQLite' if is_sqlite else 'PostgreSQL'}", flush=True)
        
    except Exception as e:
        print(f"❌ init_db error: {e}", flush=True)


def get_db_info():
    """
    Get database connection info for debugging.
    """
    return {
        "type": "SQLite" if is_sqlite else "PostgreSQL",
        "url": DATABASE_URL.split("@")[-1] if "@" in DATABASE_URL else DATABASE_URL,
        "connected": engine is not None
    }
