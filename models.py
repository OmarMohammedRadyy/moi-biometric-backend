"""
SQLAlchemy Models for MOI Biometric System
"""

from sqlalchemy import Column, Integer, String, DateTime, JSON, Boolean, ForeignKey, Float, Text
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from database import Base


# Available pages in the system
SYSTEM_PAGES = [
    {"id": "scanner", "name": "الماسح الأمني", "description": "صفحة المسح البيومتري"},
    {"id": "visitors", "name": "إدارة الزوار", "description": "إضافة وتعديل وحذف الزوار"},
    {"id": "users", "name": "إدارة المستخدمين", "description": "إدارة حسابات العساكر والمديرين"},
    {"id": "auth_logs", "name": "سجل الدخول", "description": "سجل عمليات تسجيل الدخول والخروج"},
    {"id": "scan_logs", "name": "سجل المسح", "description": "سجل عمليات المسح الأمني"},
    {"id": "notifications", "name": "الإشعارات", "description": "إشعارات النظام"},
    {"id": "dashboard", "name": "لوحة المعلومات", "description": "الإحصائيات والرسوم البيانية"},
    {"id": "permissions", "name": "إدارة الصلاحيات", "description": "تحديد صلاحيات المستخدمين"},
]

# Default permissions for officers (scanner only)
DEFAULT_OFFICER_PERMISSIONS = ["scanner"]


class User(Base):
    """
    User model for system users (admins and officers).
    
    Attributes:
        id: Primary key
        username: Unique username for login
        password_hash: Hashed password
        full_name: User's full name (اسم العسكري)
        role: User role - "admin" or "officer"
        is_active: Whether the account is active
        permissions: JSON array of allowed page IDs
        created_at: Timestamp when record was created
        created_by: ID of the admin who created this user
    """
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(100), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    full_name = Column(String(255), nullable=False)
    role = Column(String(20), nullable=False, default="officer")  # "admin" or "officer"
    is_active = Column(Boolean, default=True, nullable=False)
    permissions = Column(JSON, nullable=True, default=list)  # List of allowed page IDs
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    created_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    
    # Relationships
    auth_logs = relationship("AuthLog", back_populates="user", foreign_keys="AuthLog.user_id")
    scan_logs = relationship("ScanLog", back_populates="officer", foreign_keys="ScanLog.officer_id")

    def __repr__(self):
        return f"<User(id={self.id}, username={self.username}, role={self.role})>"
    
    def get_permissions(self):
        """Get user's permissions. Admins have all permissions."""
        if self.role == "admin":
            return [p["id"] for p in SYSTEM_PAGES]
        return self.permissions or DEFAULT_OFFICER_PERMISSIONS
    
    def has_permission(self, page_id: str) -> bool:
        """Check if user has permission to access a page."""
        if self.role == "admin":
            return True
        perms = self.permissions or DEFAULT_OFFICER_PERMISSIONS
        return page_id in perms


class AuthLog(Base):
    """
    Authentication log for tracking user login/logout activities.
    
    Attributes:
        id: Primary key
        user_id: Foreign key to User
        action: "login" or "logout"
        timestamp: When the action occurred
        ip_address: Client IP address
        user_agent: Browser/device information
        location: Geographic location if available
    """
    __tablename__ = "auth_logs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    action = Column(String(20), nullable=False)  # "login" or "logout"
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    ip_address = Column(String(50), nullable=True)
    user_agent = Column(String(500), nullable=True)
    location = Column(String(255), nullable=True)
    
    # Relationships
    user = relationship("User", back_populates="auth_logs", foreign_keys=[user_id])

    def __repr__(self):
        return f"<AuthLog(id={self.id}, user_id={self.user_id}, action={self.action})>"


class ScanLog(Base):
    """
    Security scan log for tracking all biometric scans.
    
    Attributes:
        id: Primary key
        officer_id: Foreign key to User (the officer who performed the scan)
        visitor_id: Foreign key to Visitor (matched visitor, null if no match)
        match_found: Whether a match was found
        confidence: Match confidence percentage
        timestamp: When the scan occurred
        ip_address: Client IP address
        location: Geographic location if available
        captured_photo_path: Path to the captured photo during scan
    """
    __tablename__ = "scan_logs"

    id = Column(Integer, primary_key=True, index=True)
    officer_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    visitor_id = Column(Integer, ForeignKey("visitors.id"), nullable=True, index=True)
    match_found = Column(Boolean, nullable=False)
    confidence = Column(Float, nullable=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    ip_address = Column(String(50), nullable=True)
    location = Column(String(255), nullable=True)
    captured_photo_path = Column(String(500), nullable=True)
    
    # Relationships
    officer = relationship("User", back_populates="scan_logs", foreign_keys=[officer_id])
    visitor = relationship("Visitor", backref="scan_logs")

    def __repr__(self):
        return f"<ScanLog(id={self.id}, officer_id={self.officer_id}, match={self.match_found})>"


class Visitor(Base):
    """
    Visitor model for storing registered visitors and their face encodings.
    
    Attributes:
        id: Primary key
        full_name: Visitor's full name
        passport_number: Unique passport number
        visa_status: Current visa status (e.g., "Valid", "Expired", "Tourist", "Work")
        photo_path: Path to the stored photo file (legacy, kept for compatibility)
        photo_base64: Photo stored as base64 string (for cloud deployment)
        face_encoding: 512-dimensional face encoding stored as JSON array
        created_at: Timestamp when record was created
        updated_at: Timestamp when record was last updated
    """
    __tablename__ = "visitors"

    id = Column(Integer, primary_key=True, index=True)
    full_name = Column(String(255), nullable=False)
    passport_number = Column(String(50), unique=True, nullable=False, index=True)
    visa_status = Column(String(100), nullable=False)
    photo_path = Column(String(500), nullable=True)  # Legacy field
    photo_base64 = Column(Text, nullable=True)  # New: stores photo as base64
    face_encoding = Column(JSON, nullable=False)  # Stores 512-d vector (Facenet512) as JSON array
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    def __repr__(self):
        return f"<Visitor(id={self.id}, name={self.full_name}, passport={self.passport_number})>"


class Notification(Base):
    """
    Notification model for system notifications to admins.
    
    Attributes:
        id: Primary key
        type: Notification type (account_disabled, failed_login, system_alert)
        title: Notification title
        message: Notification message
        user_id: Related user ID (optional)
        extra_data: Additional data as JSON
        is_read: Whether the notification has been read
        created_at: Timestamp when notification was created
    """
    __tablename__ = "notifications"

    id = Column(Integer, primary_key=True, index=True)
    type = Column(String(50), nullable=False, index=True)
    title = Column(String(255), nullable=False)
    message = Column(Text, nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    extra_data = Column(JSON, nullable=True)  # renamed from 'metadata' (reserved)
    is_read = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)

    # Relationships
    user = relationship("User", backref="notifications")

    def __repr__(self):
        return f"<Notification(id={self.id}, type={self.type}, is_read={self.is_read})>"
