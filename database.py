# -*- coding: utf-8 -*-
import os
import time
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

# ❗ Force PostgreSQL Connection
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("❌ FATAL: DATABASE_URL is not set! This system requires PostgreSQL.")

# Fix legacy postgres:// schema for SQLAlchemy and force pg8000 driver for stability
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+pg8000://", 1)
elif DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+pg8000://", 1)

print("� Connecting to PostgreSQL...", flush=True)

# PostgreSQL Engine
engine = create_engine(DATABASE_URL, poolclass=NullPool)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    """Wait for DB and Create Tables"""
    print("⏳ Waiting for Database...", flush=True)
    retries = 10
    while retries > 0:
        try:
            # Try to connect
            with engine.connect() as connection:
                print("✅ Database Connected!", flush=True)
                # Import models to create tables
                from models import Visitor
                Base.metadata.create_all(bind=engine)
                print("✅ Tables initialized successfully.", flush=True)
                return
        except Exception as e:
            print(f"⚠️ DB Connection failed: {e}. Retrying in 5s... ({retries} left)", flush=True)
            retries -= 1
            time.sleep(5)
    
    print("❌ Could not connect to Database after retries.", flush=True)


def run_migrations():
    """Run database migrations for new columns and tables"""
    try:
        from sqlalchemy import text
        with engine.begin() as connection:
            # Add photo_base64 column if it doesn't exist
            connection.execute(text("""
                DO $$ 
                BEGIN 
                    IF NOT EXISTS (
                        SELECT 1 FROM information_schema.columns 
                        WHERE table_name='visitors' AND column_name='photo_base64'
                    ) THEN 
                        ALTER TABLE visitors ADD COLUMN photo_base64 TEXT;
                    END IF;
                END $$;
            """))
            
            # Create notifications table if it doesn't exist
            connection.execute(text("""
                CREATE TABLE IF NOT EXISTS notifications (
                    id SERIAL PRIMARY KEY,
                    type VARCHAR(50) NOT NULL,
                    title VARCHAR(255) NOT NULL,
                    message TEXT NOT NULL,
                    user_id INTEGER REFERENCES users(id),
                    metadata JSON,
                    is_read BOOLEAN DEFAULT FALSE NOT NULL,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS ix_notifications_id ON notifications(id);
                CREATE INDEX IF NOT EXISTS ix_notifications_type ON notifications(type);
                CREATE INDEX IF NOT EXISTS ix_notifications_created_at ON notifications(created_at);
            """))
            
        print("✅ Migrations completed.", flush=True)
    except Exception as e:
        print(f"⚠️ Migration note: {e}", flush=True)

    

def get_db_info():
    return {"type": "PostgreSQL", "status": "Connected"}
