"""
Pydantic Schemas for API Request/Response validation
"""

from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


# ==================== User Schemas ====================

class UserBase(BaseModel):
    """Base schema for user data"""
    username: str = Field(..., min_length=3, max_length=100, description="Unique username")
    full_name: str = Field(..., min_length=2, max_length=255, description="User's full name")
    role: str = Field(default="officer", description="User role: admin or officer")


class UserCreate(UserBase):
    """Schema for creating a new user"""
    password: str = Field(..., min_length=4, max_length=100, description="User password")


class UserResponse(UserBase):
    """Schema for user response"""
    id: int
    is_active: bool
    created_at: datetime
    created_by: Optional[int] = None

    class Config:
        from_attributes = True


class UserList(BaseModel):
    """Schema for list of users"""
    users: List[UserResponse]
    total: int


class UserToggleResponse(BaseModel):
    """Schema for toggle user active status response"""
    success: bool
    user_id: int
    is_active: bool
    message: str


# ==================== Auth Log Schemas ====================

class AuthLogResponse(BaseModel):
    """Schema for auth log response"""
    id: int
    user_id: int
    username: str
    full_name: str
    action: str  # "login" or "logout"
    timestamp: datetime
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    location: Optional[str] = None

    class Config:
        from_attributes = True


class AuthLogList(BaseModel):
    """Schema for list of auth logs"""
    logs: List[AuthLogResponse]
    total: int
    page: int
    per_page: int


# ==================== Scan Log Schemas ====================

class ScanLogVisitor(BaseModel):
    """Minimal visitor info for scan log"""
    id: int
    full_name: str
    passport_number: str
    visa_status: str
    photo_path: Optional[str] = None
    photo_base64: Optional[str] = None

    class Config:
        from_attributes = True


class ScanLogResponse(BaseModel):
    """Schema for scan log response"""
    id: int
    officer_id: int
    officer_name: str
    officer_username: str
    visitor: Optional[ScanLogVisitor] = None
    match_found: bool
    confidence: Optional[float] = None
    timestamp: datetime
    ip_address: Optional[str] = None
    location: Optional[str] = None
    captured_photo_path: Optional[str] = None

    class Config:
        from_attributes = True


class ScanLogList(BaseModel):
    """Schema for list of scan logs"""
    logs: List[ScanLogResponse]
    total: int
    page: int
    per_page: int


# ==================== Visitor Schemas ====================

class VisitorBase(BaseModel):
    """Base schema for visitor data"""
    full_name: str = Field(..., min_length=2, max_length=255, description="Visitor's full name")
    passport_number: str = Field(..., min_length=1, max_length=50, description="Passport number")
    visa_status: str = Field(..., min_length=1, max_length=100, description="Visa status")


class VisitorCreate(VisitorBase):
    """Schema for creating a new visitor (photo sent separately as file)"""
    pass


class VisitorResponse(VisitorBase):
    """Schema for visitor response"""
    id: int
    photo_path: Optional[str] = None
    photo_base64: Optional[str] = None
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


# ==================== Dashboard Statistics ====================

class TopOfficer(BaseModel):
    """Schema for top active officer"""
    id: int
    username: str
    full_name: str
    scan_count: int

class DashboardStats(BaseModel):
    """Schema for admin dashboard statistics"""
    total_visitors: int
    total_officers: int
    total_scans_today: int
    total_matches_today: int
    total_scans_week: int
    total_matches_week: int
    total_scans_month: int
    total_matches_month: int
    match_rate_today: float
    match_rate_week: float
    top_officers: List[TopOfficer]
    recent_scans: List[ScanLogResponse]
    system_status: str


# ==================== Notifications ====================

class NotificationResponse(BaseModel):
    """Schema for notification"""
    id: int
    type: str  # "account_disabled", "failed_login", "system_alert"
    title: str
    message: str
    timestamp: datetime
    is_read: bool
    user_id: Optional[int] = None
    extra_data: Optional[dict] = None

    class Config:
        from_attributes = True


class NotificationList(BaseModel):
    """Schema for list of notifications"""
    notifications: List[NotificationResponse]
    total: int
    unread_count: int


# ==================== Export Schemas ====================

class ExportRequest(BaseModel):
    """Schema for export request"""
    type: str = Field(..., description="Export type: auth_logs, scan_logs, visitors")
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    officer_id: Optional[int] = None
    match_only: Optional[bool] = None
