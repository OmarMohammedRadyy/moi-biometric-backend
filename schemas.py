"""
Pydantic Schemas for API Request/Response validation
"""

from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


# ==================== Visitor Schemas ====================

class VisitorBase(BaseModel):
    """Base schema for visitor data"""
    full_name: str = Field(..., min_length=2, max_length=255, description="Visitor's full name")
    passport_number: str = Field(..., min_length=5, max_length=50, description="Passport number")
    visa_status: str = Field(..., min_length=2, max_length=100, description="Visa status")


class VisitorCreate(VisitorBase):
    """Schema for creating a new visitor (photo sent separately as file)"""
    pass


class VisitorResponse(VisitorBase):
    """Schema for visitor response"""
    id: int
    photo_path: str
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class VisitorList(BaseModel):
    """Schema for list of visitors"""
    visitors: List[VisitorResponse]
    total: int


# ==================== Verification Schemas ====================

class VerificationResult(BaseModel):
    """Schema for face verification result"""
    match_found: bool = Field(..., description="Whether a match was found")
    confidence: Optional[float] = Field(None, description="Match confidence (0-100)")
    visitor: Optional[VisitorResponse] = Field(None, description="Matched visitor data if found")
    message: str = Field(..., description="Result message")
    captured_photo: Optional[str] = Field(None, description="Base64 of captured photo for comparison")


class FaceDetectionResult(BaseModel):
    """Schema for face detection during registration"""
    success: bool
    message: str
    face_count: int = 0


# ==================== General Response Schemas ====================

class MessageResponse(BaseModel):
    """Generic message response"""
    success: bool
    message: str


class ErrorResponse(BaseModel):
    """Error response schema"""
    success: bool = False
    error: str
    detail: Optional[str] = None
