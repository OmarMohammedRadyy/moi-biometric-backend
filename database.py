"""
Database configuration for MOI Biometric System
Supports both SQLite (development) and PostgreSQL (production)
"""

import os
import sys
from sqlalchemy import create_engine, event, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

# Get database URL from environment
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./moi_biometric.db")

# Handle Railway's PostgreSQL URL format (postgres:// -> postgresql://)
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# Determine if using SQLite
is_sqlite = DATABASE_URL.startswith("sqlite")

print(f"📊 Database type: {'SQLite' if is_sqlite else 'PostgreSQL'}")

# Create engine based on database type
try:
    if is_sqlite:
        # SQLite configuration
        engine = create_engine(
            DATABASE_URL,
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
            echo=False
        )
        
        # Enable foreign keys for SQLite
        @event.listens_for(engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()
    else:
        # PostgreSQL configuration
        engine = create_engine(
            DATABASE_URL,
            pool_size=5,
            max_overflow=10,
            pool_timeout=30,
            pool_recycle=1800,
            pool_pre_ping=True,  # Test connections before use
            echo=False
        )
    print("✅ Database engine created successfully")
except Exception as e:
    print(f"❌ Failed to create database engine: {e}")
    sys.exit(1)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create base class for models
Base = declarative_base()


def get_db():
    """
    Dependency for getting database session.
    Yields a session and ensures it's closed after use.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """
    Initialize database tables.
    Creates all tables defined in the models.
    """
    try:
        print("🔄 Initializing database...")
        from models import Visitor  # Import here to avoid circular imports
        
        # Test connection first
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            print("✅ Database connection test passed")
        
        # Create tables
        Base.metadata.create_all(bind=engine)
        print(f"✅ Database initialized: {'SQLite' if is_sqlite else 'PostgreSQL'}")
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        # Don't crash - let the app start anyway
        # The database might come up later


def get_db_info():
    """
    Get database connection info for debugging.
    """
    return {
        "type": "SQLite" if is_sqlite else "PostgreSQL",
        "url": DATABASE_URL.split("@")[-1] if "@" in DATABASE_URL else DATABASE_URL
    }
