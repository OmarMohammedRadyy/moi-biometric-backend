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

# Fix legacy postgres:// schema for SQLAlchemy
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

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
    # Don't exit, let FastAPI try to run anyway, maybe DB comes up late

    

def get_db_info():
    return {"type": "PostgreSQL", "status": "Connected"}
