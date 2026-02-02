"""
SQLAlchemy Models for MOI Biometric System
"""

from sqlalchemy import Column, Integer, String, DateTime, JSON
from sqlalchemy.sql import func
from database import Base


class Visitor(Base):
    """
    Visitor model for storing registered visitors and their face encodings.
    
    Attributes:
        id: Primary key
        full_name: Visitor's full name
        passport_number: Unique passport number
        visa_status: Current visa status (e.g., "Valid", "Expired", "Tourist", "Work")
        photo_path: Path to the stored photo file
        face_encoding: 128-dimensional face encoding stored as JSON array
        created_at: Timestamp when record was created
        updated_at: Timestamp when record was last updated
    """
    __tablename__ = "visitors"

    id = Column(Integer, primary_key=True, index=True)
    full_name = Column(String(255), nullable=False)
    passport_number = Column(String(50), unique=True, nullable=False, index=True)
    visa_status = Column(String(100), nullable=False)
    photo_path = Column(String(500), nullable=False)
    face_encoding = Column(JSON, nullable=False)  # Stores 128-d vector as JSON array
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    def __repr__(self):
        return f"<Visitor(id={self.id}, name={self.full_name}, passport={self.passport_number})>"
