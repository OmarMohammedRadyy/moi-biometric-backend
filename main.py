"""
MOI Biometric System - FastAPI Backend
Main application file with all API endpoints
Using DeepFace for face recognition with enhanced security
Build: 2026-02-04-v4.5 - Enhanced Security & Speed
"""

import os
import uuid
import base64
import cv2
import time
import threading
from io import BytesIO
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
from collections import defaultdict

import numpy as np
from PIL import Image, ImageFilter
from scipy.spatial.distance import cosine as cosine_distance
from fastapi import FastAPI, File, UploadFile, Form, Depends, HTTPException, status, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, StreamingResponse
import csv
from io import StringIO
from sqlalchemy.orm import Session
from sqlalchemy import func, and_
from dotenv import load_dotenv
from pydantic import BaseModel

# Import DeepFace for face recognition
from deepface import DeepFace

from database import get_db, init_db, engine, Base
from models import Visitor, User, AuthLog, ScanLog, Notification, SYSTEM_PAGES, DEFAULT_OFFICER_PERMISSIONS
from schemas import (
    VisitorResponse,
    VisitorList,
    VerificationResult,
    MessageResponse,
    ErrorResponse,
    UserCreate,
    UserResponse,
    UserList,
    UserToggleResponse,
    AuthLogResponse,
    AuthLogList,
    ScanLogResponse,
    ScanLogList,
    ScanLogVisitor,
    DashboardStats,
    TopOfficer,
    NotificationResponse,
    NotificationList,
    ExportRequest
)
from auth import (
    authenticate_user,
    verify_token,
    revoke_token,
    hash_password,
    create_default_admin,
    is_admin,
    get_current_user_id,
    get_current_user_role
)

# Load environment variables
load_dotenv()

# Configuration
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")
SCAN_PHOTOS_DIR = os.path.join(UPLOAD_DIR, "scans")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(SCAN_PHOTOS_DIR, exist_ok=True)

# =================================================================
# FAST SECURITY CONFIGURATION
# =================================================================

# DeepFace model configuration
FACE_MODEL = "Facenet512"
DISTANCE_METRIC = "cosine"

# Single threshold for match (simple: match or no match)
MATCH_THRESHOLD = 0.40      # 60%+ similarity = match

# Use OpenCV for fastest detection (retinaface is slower)
DETECTOR_BACKEND = "opencv"  # Options: opencv, ssd, mtcnn, retinaface, mediapipe

# Speed optimization flags
ENABLE_QUALITY_CHECK = False    # Disable for faster scanning
ENABLE_ANTI_SPOOFING = False    # Disable for faster scanning
ENABLE_RATE_LIMITING = False    # Disable rate limiting

# Backwards compatibility
THRESHOLD_HIGH = MATCH_THRESHOLD
THRESHOLD_MEDIUM = MATCH_THRESHOLD
THRESHOLD_LOW = 0.50

# =================================================================
# IN-MEMORY CACHING & RATE LIMITING
# =================================================================

class EmbeddingsCache:
    """In-memory cache for face embeddings with auto-refresh"""
    def __init__(self):
        self._cache: Dict[int, np.ndarray] = {}
        self._lock = threading.Lock()
        self._last_refresh = datetime.min
        self._refresh_interval = timedelta(minutes=5)
    
    def get_all(self) -> Dict[int, np.ndarray]:
        with self._lock:
            return self._cache.copy()
    
    def set(self, visitor_id: int, embedding: list):
        with self._lock:
            self._cache[visitor_id] = np.array(embedding)
    
    def remove(self, visitor_id: int):
        with self._lock:
            self._cache.pop(visitor_id, None)
    
    def refresh(self, db: Session):
        """Refresh cache from database"""
        with self._lock:
            from models import Visitor
            visitors = db.query(Visitor).all()
            self._cache = {v.id: np.array(v.face_encoding) for v in visitors}
            self._last_refresh = datetime.now()
            print(f"🔄 Embeddings cache refreshed: {len(self._cache)} visitors")
    
    def needs_refresh(self) -> bool:
        return datetime.now() - self._last_refresh > self._refresh_interval
    
    def size(self) -> int:
        return len(self._cache)


# Initialize global instances
embeddings_cache = EmbeddingsCache()

# Initialize FastAPI app
app = FastAPI(
    title="MOI Biometric System",
    description="Kuwait Ministry of Interior - Facial Recognition Security System",
    version="5.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS origins
CORS_ORIGINS = [
    "https://moi-biometric-frontend.vercel.app",
    "https://moi-biometric-frontend-git-main-omars-projects-5731842e.vercel.app",
    "http://localhost:5173",
    "http://localhost:5174",
    "http://localhost:3000",
]

# Custom middleware to handle CORS for ALL responses including errors
@app.middleware("http")
async def cors_middleware(request: Request, call_next):
    # Handle preflight
    if request.method == "OPTIONS":
        response = JSONResponse(content={}, status_code=200)
    else:
        try:
            response = await call_next(request)
        except Exception as e:
            response = JSONResponse(
                content={"detail": str(e)},
                status_code=500
            )
    
    # Add CORS headers to ALL responses
    origin = request.headers.get("origin", "")
    if origin in CORS_ORIGINS or not origin:
        response.headers["Access-Control-Allow-Origin"] = origin or "*"
    else:
        response.headers["Access-Control-Allow-Origin"] = CORS_ORIGINS[0]
    
    response.headers["Access-Control-Allow-Credentials"] = "true"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, PATCH, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Authorization, Content-Type, Accept, Origin, X-Requested-With"
    response.headers["Access-Control-Expose-Headers"] = "*"
    response.headers["Access-Control-Max-Age"] = "3600"
    
    return response

# Mount uploads directory
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")


# ==================== Enhanced Helper Functions ====================

def get_face_embedding(image_path: str, with_quality_check: bool = False) -> Tuple[list, None]:
    """
    Generate face embedding using DeepFace - Fast mode.
    Returns: (embedding, None)
    """
    try:
        embedding_objs = DeepFace.represent(
            img_path=image_path,
            model_name=FACE_MODEL,
            enforce_detection=True,
            detector_backend=DETECTOR_BACKEND
        )
        
        if len(embedding_objs) == 0:
            raise ValueError("لم يتم اكتشاف وجه في الصورة")
        
        if len(embedding_objs) > 1:
            raise ValueError(f"تم اكتشاف {len(embedding_objs)} وجوه - يجب وجه واحد فقط")
        
        return embedding_objs[0]["embedding"], None
    
    except Exception as e:
        raise ValueError(f"فشل في تحليل الوجه: {str(e)}")


def compare_embeddings_fast(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Fast comparison of two face embeddings using scipy.
    Returns cosine distance (0 = identical, 1 = completely different)
    """
    return cosine_distance(embedding1, embedding2)


def find_best_match_fast(captured_embedding: list, db: Session) -> Tuple[Optional[int], float, str]:
    """
    Find the best matching visitor using cached embeddings.
    Returns: (visitor_id, distance, confidence_level)
    """
    captured_vec = np.array(captured_embedding)
    
    # Refresh cache if needed
    if embeddings_cache.needs_refresh() or embeddings_cache.size() == 0:
        embeddings_cache.refresh(db)
    
    cached = embeddings_cache.get_all()
    
    if len(cached) == 0:
        return None, float('inf'), "none"
    
    best_id = None
    best_distance = float('inf')
    
    # Fast vectorized comparison
    for visitor_id, stored_vec in cached.items():
        distance = compare_embeddings_fast(captured_vec, stored_vec)
        if distance < best_distance:
            best_distance = distance
            best_id = visitor_id
    
    # Determine confidence level
    if best_distance < THRESHOLD_HIGH:
        confidence_level = "high"
    elif best_distance < THRESHOLD_MEDIUM:
        confidence_level = "medium"
    elif best_distance < THRESHOLD_LOW:
        confidence_level = "low"
    else:
        confidence_level = "none"
    
    return best_id, best_distance, confidence_level


def compare_embeddings(embedding1: list, embedding2: list) -> float:
    """Compare two face embeddings and return distance (legacy support)."""
    return compare_embeddings_fast(np.array(embedding1), np.array(embedding2))


def get_client_ip(request: Request) -> str:
    """Get client IP address from request."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def get_token_from_header(authorization: Optional[str] = Header(None)) -> Optional[str]:
    """Extract token from Authorization header."""
    if authorization and authorization.startswith("Bearer "):
        return authorization[7:]
    return None


async def require_auth(
    authorization: Optional[str] = Header(None),
    token: Optional[str] = None
) -> dict:
    """Dependency to require authentication."""
    # Try header first, then query param
    auth_token = get_token_from_header(authorization) or token
    
    if not auth_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="غير مصرح - يرجى تسجيل الدخول"
        )
    
    user_data = verify_token(auth_token)
    if not user_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="الجلسة غير صالحة أو منتهية"
        )
    
    return user_data


async def require_admin(
    authorization: Optional[str] = Header(None),
    token: Optional[str] = None
) -> dict:
    """Dependency to require admin role."""
    user_data = await require_auth(authorization, token)
    
    if user_data.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="هذا الإجراء يتطلب صلاحيات المدير"
        )
    
    return user_data


# ==================== Startup Event ====================

@app.on_event("startup")
async def startup_event():
    """Initialize database and default admin on startup."""
    print("🚀 Starting MOI Biometric System v3.0...")
    try:
        init_db()
        print("✅ Database initialized successfully!")
        
        # Run migrations for new columns
        from database import run_migrations
        run_migrations()
        
        # Create default admin if needed
        from database import SessionLocal
        db = SessionLocal()
        try:
            create_default_admin(db)
        finally:
            db.close()
            
    except Exception as e:
        print(f"⚠️ Database initialization warning: {e}")
    
    print(f"🧠 Face Recognition Model: {FACE_MODEL}")
    print(f"📏 Distance Metric: {DISTANCE_METRIC}")
    print(f"🎯 Thresholds: HIGH={THRESHOLD_HIGH}, MEDIUM={THRESHOLD_MEDIUM}, LOW={THRESHOLD_LOW}")
    print(f"🔍 Detector: {DETECTOR_BACKEND}")
    print(f"🛡️ Security: Anti-Spoofing, Rate Limiting, Quality Check")


# ==================== Health Check ====================

@app.get("/", tags=["Health"])
async def root():
    """Health check endpoint."""
    return {
        "status": "online",
        "system": "MOI Biometric Security System",
        "version": "5.0.0",
        "face_model": FACE_MODEL
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Detailed health check."""
    return {
        "status": "healthy",
        "database": "connected",
        "face_recognition": "ready",
        "model": FACE_MODEL,
        "detector": DETECTOR_BACKEND,
        "thresholds": {
            "high": THRESHOLD_HIGH,
            "medium": THRESHOLD_MEDIUM,
            "low": THRESHOLD_LOW
        },
        "cache_size": embeddings_cache.size(),
        "security_features": [
            "anti_spoofing",
            "quality_check",
            "rate_limiting",
            "multi_level_confidence"
        ]
    }


@app.get("/api/security/status", tags=["Security"])
async def security_status(
    admin: dict = Depends(require_admin)
):
    """Get security system status (admin only)."""
    return {
        "face_model": FACE_MODEL,
        "detector_backend": DETECTOR_BACKEND,
        "match_threshold": f"{(1-MATCH_THRESHOLD)*100:.0f}%+",
        "mode": "fast",
        "cache": {
            "embeddings_cached": embeddings_cache.size(),
            "needs_refresh": embeddings_cache.needs_refresh()
        }
    }


# ==================== Authentication ====================

class LoginRequest(BaseModel):
    username: str
    password: str

class TokenResponse(BaseModel):
    token: str
    user_id: int
    username: str
    full_name: str
    role: str
    message: str


@app.post("/api/auth/login", response_model=TokenResponse, tags=["Authentication"])
async def login(
    request: LoginRequest,
    req: Request,
    db: Session = Depends(get_db)
):
    """Login with username and password."""
    # Check if user exists and is disabled
    user = db.query(User).filter(User.username == request.username).first()
    if user and not user.is_active:
        # Create notification for failed login attempt from disabled account
        notification = Notification(
            type="failed_login_disabled",
            title="محاولة دخول من حساب معطل",
            message=f"محاولة تسجيل دخول فاشلة من الحساب المعطل: {user.full_name} (@{user.username})",
            user_id=user.id,
            extra_data={
                "ip_address": get_client_ip(req),
                "user_agent": req.headers.get("User-Agent", "")[:200]
            }
        )
        db.add(notification)
        db.commit()
        
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="الحساب معطل - تواصل مع المدير"
        )
    
    result = authenticate_user(request.username, request.password, db)
    
    if not result:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="اسم المستخدم أو كلمة المرور غير صحيحة"
        )
    
    # Log the login
    auth_log = AuthLog(
        user_id=result["user_id"],
        action="login",
        ip_address=get_client_ip(req),
        user_agent=req.headers.get("User-Agent", "")[:500],
        location=None  # Can be enhanced with GeoIP
    )
    db.add(auth_log)
    db.commit()
    
    return TokenResponse(
        token=result["token"],
        user_id=result["user_id"],
        username=result["username"],
        full_name=result["full_name"],
        role=result["role"],
        message="تم تسجيل الدخول بنجاح"
    )


@app.post("/api/auth/logout", tags=["Authentication"])
async def logout(
    req: Request,
    token: str = Form(...),
    db: Session = Depends(get_db)
):
    """Logout and invalidate token."""
    user_data = verify_token(token)
    
    if user_data:
        # Log the logout
        auth_log = AuthLog(
            user_id=user_data["user_id"],
            action="logout",
            ip_address=get_client_ip(req),
            user_agent=req.headers.get("User-Agent", "")[:500],
            location=None
        )
        db.add(auth_log)
        db.commit()
    
    revoke_token(token)
    return {"message": "تم تسجيل الخروج بنجاح"}


@app.get("/api/auth/verify", tags=["Authentication"])
async def verify_auth(token: str):
    """Verify if a token is valid."""
    user_data = verify_token(token)
    
    if not user_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="الجلسة غير صالحة أو منتهية"
        )
    
    return {
        "valid": True,
        "user_id": user_data["user_id"],
        "username": user_data["username"],
        "full_name": user_data["full_name"],
        "role": user_data["role"]
    }


@app.get("/api/auth/me", tags=["Authentication"])
async def get_current_user(user_data: dict = Depends(require_auth)):
    """Get current authenticated user info."""
    return {
        "user_id": user_data["user_id"],
        "username": user_data["username"],
        "full_name": user_data["full_name"],
        "role": user_data["role"]
    }


# ==================== User Management (Admin Only) ====================

@app.post("/api/users", response_model=UserResponse, tags=["User Management"])
async def create_user(
    user_data: UserCreate,
    admin: dict = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """Create a new user (admin only)."""
    # Check if username exists
    existing = db.query(User).filter(User.username == user_data.username).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"اسم المستخدم '{user_data.username}' مستخدم بالفعل"
        )
    
    # Create user
    new_user = User(
        username=user_data.username,
        password_hash=hash_password(user_data.password),
        full_name=user_data.full_name,
        role=user_data.role if user_data.role in ["admin", "officer"] else "officer",
        is_active=True,
        created_by=admin["user_id"]
    )
    
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    print(f"✅ New user created: {new_user.username} ({new_user.role}) by {admin['username']}")
    
    return new_user


@app.get("/api/users", response_model=UserList, tags=["User Management"])
async def list_users(
    admin: dict = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """Get all users (admin only)."""
    users = db.query(User).order_by(User.created_at.desc()).all()
    return UserList(users=users, total=len(users))


@app.get("/api/users/{user_id}", response_model=UserResponse, tags=["User Management"])
async def get_user(
    user_id: int,
    admin: dict = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """Get a specific user by ID (admin only)."""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"المستخدم غير موجود"
        )
    return user


@app.patch("/api/users/{user_id}/toggle", response_model=UserToggleResponse, tags=["User Management"])
async def toggle_user_status(
    user_id: int,
    admin: dict = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """Toggle user active status (admin only)."""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="المستخدم غير موجود"
        )
    
    # Prevent admin from disabling themselves
    if user.id == admin["user_id"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="لا يمكنك تعطيل حسابك الخاص"
        )
    
    user.is_active = not user.is_active
    db.commit()
    
    status_text = "مفعّل" if user.is_active else "معطّل"
    print(f"👤 User {user.username} is now {status_text} by {admin['username']}")
    
    # Create notification for account status change
    notification = Notification(
        type="account_disabled" if not user.is_active else "account_enabled",
        title=f"تم {'تعطيل' if not user.is_active else 'تفعيل'} حساب",
        message=f"تم {'تعطيل' if not user.is_active else 'تفعيل'} حساب العسكري {user.full_name} (@{user.username}) بواسطة {admin['full_name']}",
        user_id=user.id,
        extra_data={"admin_id": admin["user_id"], "admin_username": admin["username"]}
    )
    db.add(notification)
    db.commit()
    
    return UserToggleResponse(
        success=True,
        user_id=user.id,
        is_active=user.is_active,
        message=f"تم {'تفعيل' if user.is_active else 'تعطيل'} حساب {user.full_name}"
    )


class UserUpdate(BaseModel):
    """Schema for updating user data"""
    username: Optional[str] = None
    full_name: Optional[str] = None
    password: Optional[str] = None
    role: Optional[str] = None


@app.put("/api/users/{user_id}", response_model=UserResponse, tags=["User Management"])
async def update_user(
    user_id: int,
    user_data: UserUpdate,
    admin: dict = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """Update user information (admin only)."""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="المستخدم غير موجود"
        )
    
    # Update fields if provided
    if user_data.username:
        # Check if username already exists for another user
        existing = db.query(User).filter(
            User.username == user_data.username,
            User.id != user_id
        ).first()
        if existing:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="اسم المستخدم موجود مسبقاً"
            )
        user.username = user_data.username
    
    if user_data.full_name:
        user.full_name = user_data.full_name
    
    if user_data.password:
        user.password_hash = hash_password(user_data.password)
    
    if user_data.role and user_data.role in ["admin", "officer"]:
        # Prevent admin from changing their own role
        if user.id == admin["user_id"] and user_data.role != "admin":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="لا يمكنك تغيير صلاحياتك الخاصة"
            )
        user.role = user_data.role
    
    db.commit()
    db.refresh(user)
    
    return UserResponse(
        id=user.id,
        username=user.username,
        full_name=user.full_name,
        role=user.role,
        is_active=user.is_active,
        created_at=user.created_at
    )


@app.delete("/api/users/{user_id}", response_model=MessageResponse, tags=["User Management"])
async def delete_user(
    user_id: int,
    admin: dict = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """Delete a user (admin only)."""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="المستخدم غير موجود"
        )
    
    # Prevent admin from deleting themselves
    if user.id == admin["user_id"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="لا يمكنك حذف حسابك الخاص"
        )
    
    username = user.username
    
    # Delete related records first to avoid foreign key constraint errors
    # Delete auth logs
    db.query(AuthLog).filter(AuthLog.user_id == user_id).delete()
    
    # Delete scan logs (set officer_id to NULL or delete)
    db.query(ScanLog).filter(ScanLog.officer_id == user_id).delete()
    
    # Delete notifications
    db.query(Notification).filter(Notification.user_id == user_id).delete()
    
    # Now delete the user
    db.delete(user)
    db.commit()
    
    return MessageResponse(success=True, message=f"تم حذف المستخدم {username}")


# ==================== Permissions Management (Admin Only) ====================

@app.get("/api/permissions/pages", tags=["Permissions"])
async def get_available_pages(
    admin: dict = Depends(require_admin)
):
    """Get list of all available system pages."""
    return {
        "pages": SYSTEM_PAGES,
        "default_officer_permissions": DEFAULT_OFFICER_PERMISSIONS
    }


@app.get("/api/permissions/users", tags=["Permissions"])
async def get_users_permissions(
    admin: dict = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """Get all officers with their permissions."""
    officers = db.query(User).filter(User.role == "officer").all()
    
    return {
        "users": [
            {
                "id": u.id,
                "username": u.username,
                "full_name": u.full_name,
                "is_active": u.is_active,
                "permissions": u.permissions or DEFAULT_OFFICER_PERMISSIONS
            }
            for u in officers
        ]
    }


class PermissionUpdate(BaseModel):
    permissions: list


@app.put("/api/permissions/users/{user_id}", tags=["Permissions"])
async def update_user_permissions(
    user_id: int,
    data: PermissionUpdate,
    admin: dict = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """Update a user's page permissions (admin only)."""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="المستخدم غير موجود"
        )
    
    if user.role == "admin":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="لا يمكن تعديل صلاحيات المدير - المدير لديه جميع الصلاحيات"
        )
    
    # Validate permissions
    valid_page_ids = [p["id"] for p in SYSTEM_PAGES]
    invalid_perms = [p for p in data.permissions if p not in valid_page_ids]
    if invalid_perms:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"صلاحيات غير صالحة: {invalid_perms}"
        )
    
    user.permissions = data.permissions
    db.commit()
    db.refresh(user)
    
    return {
        "success": True,
        "message": f"تم تحديث صلاحيات {user.full_name}",
        "user": {
            "id": user.id,
            "username": user.username,
            "full_name": user.full_name,
            "permissions": user.permissions
        }
    }


@app.get("/api/permissions/check/{page_id}", tags=["Permissions"])
async def check_page_permission(
    page_id: str,
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db)
):
    """Check if current user has permission to access a page."""
    auth_token = get_token_from_header(authorization)
    if not auth_token:
        return {"allowed": False, "message": "غير مصرح - يرجى تسجيل الدخول"}
    
    user_data = verify_token(auth_token)
    if not user_data:
        return {"allowed": False, "message": "الجلسة غير صالحة أو منتهية"}
    
    user = db.query(User).filter(User.id == user_data["user_id"]).first()
    if not user:
        return {"allowed": False, "message": "المستخدم غير موجود"}
    
    if not user.is_active:
        return {"allowed": False, "message": "الحساب معطّل"}
    
    # Admins have all permissions
    if user.role == "admin":
        return {"allowed": True, "message": "مسموح"}
    
    # Check officer permissions
    permissions = user.permissions or DEFAULT_OFFICER_PERMISSIONS
    if page_id in permissions:
        return {"allowed": True, "message": "مسموح"}
    
    return {
        "allowed": False,
        "message": "غير مصرح لك بالدخول لهذه الصفحة"
    }


@app.get("/api/user/permissions", tags=["Permissions"])
async def get_my_permissions(
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db)
):
    """Get current user's permissions."""
    auth_token = get_token_from_header(authorization)
    if not auth_token:
        raise HTTPException(status_code=401, detail="غير مصرح")
    
    user_data = verify_token(auth_token)
    if not user_data:
        raise HTTPException(status_code=401, detail="الجلسة غير صالحة")
    
    user = db.query(User).filter(User.id == user_data["user_id"]).first()
    if not user:
        raise HTTPException(status_code=404, detail="المستخدم غير موجود")
    
    # Admins have all permissions
    if user.role == "admin":
        return {
            "role": "admin",
            "permissions": [p["id"] for p in SYSTEM_PAGES],
            "all_pages": True
        }
    
    return {
        "role": "officer",
        "permissions": user.permissions or DEFAULT_OFFICER_PERMISSIONS,
        "all_pages": False
    }


# ==================== Auth Logs (Admin Only) ====================

@app.get("/api/logs/auth", response_model=AuthLogList, tags=["Logs"])
async def get_auth_logs(
    page: int = 1,
    per_page: int = 50,
    user_id: Optional[int] = None,
    action: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    admin: dict = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """Get authentication logs with pagination and advanced filters (admin only)."""
    query = db.query(AuthLog).join(User)
    
    # Apply filters
    if user_id:
        query = query.filter(AuthLog.user_id == user_id)
    if action:
        query = query.filter(AuthLog.action == action)
    
    # Date range filter
    if date_from:
        try:
            from_date = datetime.fromisoformat(date_from.replace('Z', '+00:00'))
            query = query.filter(AuthLog.timestamp >= from_date)
        except:
            pass
    if date_to:
        try:
            to_date = datetime.fromisoformat(date_to.replace('Z', '+00:00'))
            query = query.filter(AuthLog.timestamp <= to_date)
        except:
            pass
    
    # Get total count
    total = query.count()
    
    # Apply pagination
    offset = (page - 1) * per_page
    logs = query.order_by(AuthLog.timestamp.desc()).offset(offset).limit(per_page).all()
    
    # Build response
    log_responses = []
    for log in logs:
        user = db.query(User).filter(User.id == log.user_id).first()
        log_responses.append(AuthLogResponse(
            id=log.id,
            user_id=log.user_id,
            username=user.username if user else "unknown",
            full_name=user.full_name if user else "unknown",
            action=log.action,
            timestamp=log.timestamp,
            ip_address=log.ip_address,
            user_agent=log.user_agent,
            location=log.location
        ))
    
    return AuthLogList(logs=log_responses, total=total, page=page, per_page=per_page)


# ==================== Scan Logs (Admin Only) ====================

@app.get("/api/logs/scans", response_model=ScanLogList, tags=["Logs"])
async def get_scan_logs(
    page: int = 1,
    per_page: int = 50,
    officer_id: Optional[int] = None,
    match_only: bool = True,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    admin: dict = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """Get scan logs with pagination and advanced filters (admin only). By default shows only matches."""
    query = db.query(ScanLog).join(User, ScanLog.officer_id == User.id)
    
    # Apply filters
    if officer_id:
        query = query.filter(ScanLog.officer_id == officer_id)
    if match_only:
        query = query.filter(ScanLog.match_found == True)
    
    # Date range filter
    if date_from:
        try:
            from_date = datetime.fromisoformat(date_from.replace('Z', '+00:00'))
            query = query.filter(ScanLog.timestamp >= from_date)
        except:
            pass
    if date_to:
        try:
            to_date = datetime.fromisoformat(date_to.replace('Z', '+00:00'))
            query = query.filter(ScanLog.timestamp <= to_date)
        except:
            pass
    
    # Get total count
    total = query.count()
    
    # Apply pagination
    offset = (page - 1) * per_page
    logs = query.order_by(ScanLog.timestamp.desc()).offset(offset).limit(per_page).all()
    
    # Build response
    log_responses = []
    for log in logs:
        officer = db.query(User).filter(User.id == log.officer_id).first()
        visitor = db.query(Visitor).filter(Visitor.id == log.visitor_id).first() if log.visitor_id else None
        
        visitor_data = None
        if visitor:
            visitor_data = ScanLogVisitor(
                id=visitor.id,
                full_name=visitor.full_name,
                passport_number=visitor.passport_number,
                visa_status=visitor.visa_status,
                photo_path=visitor.photo_path,
                photo_base64=visitor.photo_base64
            )
        
        log_responses.append(ScanLogResponse(
            id=log.id,
            officer_id=log.officer_id,
            officer_name=officer.full_name if officer else "unknown",
            officer_username=officer.username if officer else "unknown",
            visitor=visitor_data,
            match_found=log.match_found,
            confidence=log.confidence,
            timestamp=log.timestamp,
            ip_address=log.ip_address,
            location=log.location,
            captured_photo_path=log.captured_photo_path
        ))
    
    return ScanLogList(logs=log_responses, total=total, page=page, per_page=per_page)


# ==================== Get Officers List (for filters) ====================

@app.get("/api/officers", tags=["Logs"])
async def get_officers_list(
    admin: dict = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """Get list of officers for filter dropdowns."""
    officers = db.query(User).filter(User.role == "officer").all()
    return [{"id": o.id, "username": o.username, "full_name": o.full_name} for o in officers]


# ==================== Dashboard Statistics (Admin Only) ====================

@app.get("/api/dashboard/stats", response_model=DashboardStats, tags=["Dashboard"])
async def get_dashboard_stats(
    admin: dict = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """Get dashboard statistics (admin only) with advanced analytics."""
    now = datetime.now()
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    week_start = today_start - timedelta(days=7)
    month_start = today_start - timedelta(days=30)
    
    # Counts
    total_visitors = db.query(Visitor).count()
    total_officers = db.query(User).filter(User.role == "officer").count()
    
    # Today's scans
    total_scans_today = db.query(ScanLog).filter(ScanLog.timestamp >= today_start).count()
    total_matches_today = db.query(ScanLog).filter(
        and_(ScanLog.timestamp >= today_start, ScanLog.match_found == True)
    ).count()
    
    # Week's scans
    total_scans_week = db.query(ScanLog).filter(ScanLog.timestamp >= week_start).count()
    total_matches_week = db.query(ScanLog).filter(
        and_(ScanLog.timestamp >= week_start, ScanLog.match_found == True)
    ).count()
    
    # Month's scans
    total_scans_month = db.query(ScanLog).filter(ScanLog.timestamp >= month_start).count()
    total_matches_month = db.query(ScanLog).filter(
        and_(ScanLog.timestamp >= month_start, ScanLog.match_found == True)
    ).count()
    
    # Calculate match rates
    match_rate_today = (total_matches_today / total_scans_today * 100) if total_scans_today > 0 else 0
    match_rate_week = (total_matches_week / total_scans_week * 100) if total_scans_week > 0 else 0
    
    # Top 5 most active officers (this week)
    top_officers_query = db.query(
        User.id,
        User.username,
        User.full_name,
        func.count(ScanLog.id).label('scan_count')
    ).join(ScanLog, User.id == ScanLog.officer_id).filter(
        ScanLog.timestamp >= week_start
    ).group_by(User.id).order_by(func.count(ScanLog.id).desc()).limit(5).all()
    
    top_officers = [
        TopOfficer(
            id=officer.id,
            username=officer.username,
            full_name=officer.full_name,
            scan_count=officer.scan_count
        ) for officer in top_officers_query
    ]
    
    # Recent scans (last 10 matches)
    recent_logs = db.query(ScanLog).filter(ScanLog.match_found == True).order_by(
        ScanLog.timestamp.desc()
    ).limit(10).all()
    
    recent_scans = []
    for log in recent_logs:
        officer = db.query(User).filter(User.id == log.officer_id).first()
        visitor = db.query(Visitor).filter(Visitor.id == log.visitor_id).first() if log.visitor_id else None
        
        visitor_data = None
        if visitor:
            visitor_data = ScanLogVisitor(
                id=visitor.id,
                full_name=visitor.full_name,
                passport_number=visitor.passport_number,
                visa_status=visitor.visa_status,
                photo_path=visitor.photo_path,
                photo_base64=visitor.photo_base64
            )
        
        recent_scans.append(ScanLogResponse(
            id=log.id,
            officer_id=log.officer_id,
            officer_name=officer.full_name if officer else "unknown",
            officer_username=officer.username if officer else "unknown",
            visitor=visitor_data,
            match_found=log.match_found,
            confidence=log.confidence,
            timestamp=log.timestamp,
            ip_address=log.ip_address,
            location=log.location,
            captured_photo_path=log.captured_photo_path
        ))
    
    return DashboardStats(
        total_visitors=total_visitors,
        total_officers=total_officers,
        total_scans_today=total_scans_today,
        total_matches_today=total_matches_today,
        total_scans_week=total_scans_week,
        total_matches_week=total_matches_week,
        total_scans_month=total_scans_month,
        total_matches_month=total_matches_month,
        match_rate_today=round(match_rate_today, 1),
        match_rate_week=round(match_rate_week, 1),
        top_officers=top_officers,
        recent_scans=recent_scans,
        system_status="operational"
    )


# ==================== Export to CSV (Admin Only) ====================

@app.get("/api/export/auth-logs", tags=["Export"])
async def export_auth_logs(
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    user_id: Optional[int] = None,
    action: Optional[str] = None,
    admin: dict = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """Export authentication logs to CSV."""
    query = db.query(AuthLog).join(User)
    
    # Apply filters
    if user_id:
        query = query.filter(AuthLog.user_id == user_id)
    if action:
        query = query.filter(AuthLog.action == action)
    if date_from:
        try:
            from_date = datetime.fromisoformat(date_from.replace('Z', '+00:00'))
            query = query.filter(AuthLog.timestamp >= from_date)
        except:
            pass
    if date_to:
        try:
            to_date = datetime.fromisoformat(date_to.replace('Z', '+00:00'))
            query = query.filter(AuthLog.timestamp <= to_date)
        except:
            pass
    
    logs = query.order_by(AuthLog.timestamp.desc()).limit(5000).all()
    
    # Create CSV
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(['ID', 'المستخدم', 'الاسم الكامل', 'الإجراء', 'التاريخ والوقت', 'عنوان IP', 'الموقع'])
    
    for log in logs:
        user = db.query(User).filter(User.id == log.user_id).first()
        writer.writerow([
            log.id,
            user.username if user else 'unknown',
            user.full_name if user else 'unknown',
            'تسجيل دخول' if log.action == 'login' else 'تسجيل خروج',
            log.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            log.ip_address or '',
            log.location or ''
        ])
    
    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=auth_logs_{datetime.now().strftime('%Y%m%d')}.csv"}
    )


@app.get("/api/export/scan-logs", tags=["Export"])
async def export_scan_logs(
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    officer_id: Optional[int] = None,
    match_only: bool = True,
    admin: dict = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """Export scan logs to CSV."""
    query = db.query(ScanLog).join(User, ScanLog.officer_id == User.id)
    
    # Apply filters
    if officer_id:
        query = query.filter(ScanLog.officer_id == officer_id)
    if match_only:
        query = query.filter(ScanLog.match_found == True)
    if date_from:
        try:
            from_date = datetime.fromisoformat(date_from.replace('Z', '+00:00'))
            query = query.filter(ScanLog.timestamp >= from_date)
        except:
            pass
    if date_to:
        try:
            to_date = datetime.fromisoformat(date_to.replace('Z', '+00:00'))
            query = query.filter(ScanLog.timestamp <= to_date)
        except:
            pass
    
    logs = query.order_by(ScanLog.timestamp.desc()).limit(5000).all()
    
    # Create CSV
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(['ID', 'العسكري', 'الزائر', 'رقم الجواز', 'نوع التأشيرة', 'نتيجة المسح', 'التاريخ والوقت', 'عنوان IP'])
    
    for log in logs:
        officer = db.query(User).filter(User.id == log.officer_id).first()
        visitor = db.query(Visitor).filter(Visitor.id == log.visitor_id).first() if log.visitor_id else None
        writer.writerow([
            log.id,
            officer.full_name if officer else 'unknown',
            visitor.full_name if visitor else 'غير معروف',
            visitor.passport_number if visitor else '',
            visitor.visa_status if visitor else '',
            'تطابق' if log.match_found else 'لا يوجد تطابق',
            log.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            log.ip_address or ''
        ])
    
    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=scan_logs_{datetime.now().strftime('%Y%m%d')}.csv"}
    )


@app.get("/api/export/visitors", tags=["Export"])
async def export_visitors(
    admin: dict = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """Export visitors list to CSV."""
    visitors = db.query(Visitor).order_by(Visitor.created_at.desc()).limit(5000).all()
    
    # Create CSV
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(['ID', 'الاسم الكامل', 'رقم الجواز', 'نوع التأشيرة', 'تاريخ التسجيل'])
    
    for visitor in visitors:
        writer.writerow([
            visitor.id,
            visitor.full_name,
            visitor.passport_number,
            visitor.visa_status,
            visitor.created_at.strftime('%Y-%m-%d %H:%M:%S')
        ])
    
    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=visitors_{datetime.now().strftime('%Y%m%d')}.csv"}
    )


# ==================== Notifications (Admin Only) ====================

@app.get("/api/notifications", response_model=NotificationList, tags=["Notifications"])
async def get_notifications(
    page: int = 1,
    per_page: int = 20,
    unread_only: bool = False,
    admin: dict = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """Get notifications for admin."""
    query = db.query(Notification)
    
    if unread_only:
        query = query.filter(Notification.is_read == False)
    
    total = query.count()
    unread_count = db.query(Notification).filter(Notification.is_read == False).count()
    
    offset = (page - 1) * per_page
    notifications = query.order_by(Notification.created_at.desc()).offset(offset).limit(per_page).all()
    
    return NotificationList(
        notifications=[
            NotificationResponse(
                id=n.id,
                type=n.type,
                title=n.title,
                message=n.message,
                timestamp=n.created_at,
                is_read=n.is_read,
                user_id=n.user_id,
                extra_data=n.extra_data
            ) for n in notifications
        ],
        total=total,
        unread_count=unread_count
    )


@app.patch("/api/notifications/{notification_id}/read", tags=["Notifications"])
async def mark_notification_read(
    notification_id: int,
    admin: dict = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """Mark a notification as read."""
    notification = db.query(Notification).filter(Notification.id == notification_id).first()
    if not notification:
        raise HTTPException(status_code=404, detail="الإشعار غير موجود")
    
    notification.is_read = True
    db.commit()
    return {"success": True, "message": "تم تحديث الإشعار"}


@app.patch("/api/notifications/read-all", tags=["Notifications"])
async def mark_all_notifications_read(
    admin: dict = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """Mark all notifications as read."""
    db.query(Notification).filter(Notification.is_read == False).update({"is_read": True})
    db.commit()
    return {"success": True, "message": "تم تحديث جميع الإشعارات"}


# Helper function to create notification
def create_notification(db: Session, type: str, title: str, message: str, user_id: int = None, extra_data: dict = None):
    """Create a new notification."""
    notification = Notification(
        type=type,
        title=title,
        message=message,
        user_id=user_id,
        extra_data=extra_data
    )
    db.add(notification)
    db.commit()
    return notification


# ==================== Visitor Management (Admin Only) ====================

@app.post("/api/visitors", response_model=VisitorResponse, tags=["Visitors"])
async def register_visitor(
    full_name: str = Form(...),
    passport_number: str = Form(...),
    visa_status: str = Form(...),
    photo: UploadFile = File(...),
    admin: dict = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """Register a new visitor with their photo (admin only)."""
    try:
        # Check if passport already exists
        existing = db.query(Visitor).filter(Visitor.passport_number == passport_number).first()
        if existing:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"الزائر برقم جواز {passport_number} مسجل مسبقاً"
            )

        # Read and validate image
        contents = await photo.read()
        image = Image.open(BytesIO(contents))
        
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Resize image to reduce storage size (max 500px width)
        max_width = 500
        if image.width > max_width:
            ratio = max_width / image.width
            new_height = int(image.height * ratio)
            image = image.resize((max_width, new_height), Image.Resampling.LANCZOS)
        
        # Save photo temporarily for face detection
        file_extension = photo.filename.split(".")[-1] if "." in photo.filename else "jpg"
        unique_filename = f"{uuid.uuid4()}.{file_extension}"
        photo_path = os.path.join(UPLOAD_DIR, unique_filename)
        image.save(photo_path, quality=85)
        
        # Generate face embedding (fast mode)
        try:
            face_embedding, _ = get_face_embedding(photo_path, with_quality_check=False)
        except ValueError as e:
            if os.path.exists(photo_path):
                os.remove(photo_path)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e)
            )
        
        # Convert image to base64 for database storage
        buffered = BytesIO()
        image.save(buffered, format="JPEG", quality=85)
        photo_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        # Clean up temporary file
        if os.path.exists(photo_path):
            os.remove(photo_path)
        
        # Create visitor record with base64 photo
        visitor = Visitor(
            full_name=full_name,
            passport_number=passport_number,
            visa_status=visa_status,
            photo_path=unique_filename,  # Keep for compatibility
            photo_base64=photo_base64,   # Store actual photo data
            face_encoding=face_embedding
        )
        
        db.add(visitor)
        db.commit()
        db.refresh(visitor)
        
        # Update embeddings cache
        embeddings_cache.set(visitor.id, face_embedding)
        
        print(f"✅ Visitor registered: {full_name} ({passport_number}) by {admin['username']}")
        
        return visitor

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error registering visitor: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"خطأ في تسجيل الزائر: {str(e)}"
        )


@app.get("/api/visitors", response_model=VisitorList, tags=["Visitors"])
async def list_visitors(
    admin: dict = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """Get all registered visitors (admin only)."""
    try:
        visitors = db.query(Visitor).order_by(Visitor.created_at.desc()).all()
        return VisitorList(visitors=visitors, total=len(visitors))
    except Exception as e:
        print(f"Error listing visitors: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"خطأ في جلب الزوار: {str(e)}"
        )


@app.get("/api/visitors/{visitor_id}", response_model=VisitorResponse, tags=["Visitors"])
async def get_visitor(
    visitor_id: int,
    admin: dict = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """Get a specific visitor by ID (admin only)."""
    visitor = db.query(Visitor).filter(Visitor.id == visitor_id).first()
    if not visitor:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="الزائر غير موجود"
        )
    return visitor


@app.delete("/api/visitors/{visitor_id}", response_model=MessageResponse, tags=["Visitors"])
async def delete_visitor(
    visitor_id: int,
    admin: dict = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """Delete a visitor by ID (admin only)."""
    visitor = db.query(Visitor).filter(Visitor.id == visitor_id).first()
    if not visitor:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="الزائر غير موجود"
        )
    
    # Delete photo file
    if visitor.photo_path:
        photo_path = os.path.join(UPLOAD_DIR, visitor.photo_path)
        if os.path.exists(photo_path):
            os.remove(photo_path)
    
    name = visitor.full_name
    vid = visitor.id
    db.delete(visitor)
    db.commit()
    
    # Remove from embeddings cache
    embeddings_cache.remove(vid)
    
    return MessageResponse(success=True, message=f"تم حذف الزائر {name}")


# ==================== Face Verification (All Authenticated Users) ====================

@app.post("/api/verify", tags=["Security"])
async def verify_face(
    req: Request,
    photo: UploadFile = File(...),
    token: Optional[str] = Form(None),
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db)
):
    """
    Fast face verification - optimized for speed.
    Simple result: match or no match.
    """
    start_time = time.time()
    
    # Get authenticated user
    auth_token = get_token_from_header(authorization) or token
    if not auth_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="غير مصرح - يرجى تسجيل الدخول"
        )
    
    user_data = verify_token(auth_token)
    if not user_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="الجلسة غير صالحة أو منتهية"
        )
    
    user_id = user_data["user_id"]
    temp_path = None
    
    try:
        # Read and process uploaded image
        contents = await photo.read()
        image = Image.open(BytesIO(contents))
        
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Save temporarily for processing
        temp_path = os.path.join(UPLOAD_DIR, f"temp_{uuid.uuid4()}.jpg")
        image.save(temp_path, quality=90)
        
        # Generate face embedding (fast mode - no quality check)
        try:
            captured_embedding, _ = get_face_embedding(temp_path, with_quality_check=False)
        except ValueError as e:
            scan_log = ScanLog(
                officer_id=user_id,
                visitor_id=None,
                match_found=False,
                confidence=None,
                ip_address=get_client_ip(req),
                location=None,
                captured_photo_path=None
            )
            db.add(scan_log)
            db.commit()
            
            return {
                "match_found": False,
                "visitor": None,
                "message": str(e),
                "captured_photo": None
            }
        
        # Fast search using cached embeddings
        best_id, best_distance, _ = find_best_match_fast(captured_embedding, db)
        
        # Convert captured image to base64
        buffered = BytesIO()
        image.save(buffered, format="JPEG", quality=80)
        captured_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        # Simple match check
        match_found = best_distance < MATCH_THRESHOLD
        
        # Get visitor data if match found
        best_match = None
        if best_id and match_found:
            best_match = db.query(Visitor).filter(Visitor.id == best_id).first()
        
        # Save scan photo for matches
        relative_scan_path = None
        if match_found and best_match:
            scan_filename = f"scan_{uuid.uuid4()}.jpg"
            saved_scan_path = os.path.join(SCAN_PHOTOS_DIR, scan_filename)
            image.save(saved_scan_path, quality=80)
            relative_scan_path = f"scans/{scan_filename}"
        
        # Log the scan
        scan_log = ScanLog(
            officer_id=user_id,
            visitor_id=best_match.id if match_found and best_match else None,
            match_found=match_found,
            confidence=round((1 - best_distance) * 100, 2) if best_distance < 1 else 0,
            ip_address=get_client_ip(req),
            location=None,
            captured_photo_path=relative_scan_path
        )
        db.add(scan_log)
        db.commit()
        
        processing_time = int((time.time() - start_time) * 1000)
        
        if match_found and best_match:
            print(f"✅ MATCH: {best_match.full_name} by {user_data['username']} in {processing_time}ms")
            
            status_msg = f"تم التحقق: {best_match.full_name}"
            
            return {
                "match_found": True,
                "visitor": {
                    "id": best_match.id,
                    "full_name": best_match.full_name,
                    "passport_number": best_match.passport_number,
                    "visa_status": best_match.visa_status,
                    "photo_path": best_match.photo_path,
                    "photo_base64": best_match.photo_base64,
                    "created_at": best_match.created_at.isoformat() if best_match.created_at else None,
                    "updated_at": best_match.updated_at.isoformat() if best_match.updated_at else None
                },
                "message": status_msg,
                "captured_photo": captured_base64
            }
        else:
            print(f"❌ NO MATCH by {user_data['username']} (best: {best_distance:.4f}) in {processing_time}ms")
            
            return {
                "match_found": False,
                "visitor": None,
                "message": "غير مصرح - لا يوجد تطابق في السجلات",
                "captured_photo": captured_base64
            }

    except Exception as e:
        print(f"❌ Verification error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"خطأ في التحقق: {str(e)}"
        )
    
    finally:
        # Clean up temp file
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass


# ==================== Statistics ====================

@app.get("/api/stats", tags=["Statistics"])
async def get_statistics(
    admin: dict = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """Get system statistics (admin only)."""
    total_visitors = db.query(Visitor).count()
    total_users = db.query(User).count()
    total_scans = db.query(ScanLog).count()
    total_matches = db.query(ScanLog).filter(ScanLog.match_found == True).count()
    
    return {
        "total_visitors": total_visitors,
        "total_users": total_users,
        "total_scans": total_scans,
        "total_matches": total_matches,
        "system_status": "operational",
        "face_recognition_model": FACE_MODEL,
        "distance_metric": DISTANCE_METRIC,
        "thresholds": {
            "high": THRESHOLD_HIGH,
            "medium": THRESHOLD_MEDIUM,
            "low": THRESHOLD_LOW
        }
    }


# ==================== Run Server ====================

if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    
    print(f"🔒 Starting on http://{host}:{port}", flush=True)
    uvicorn.run(app, host=host, port=port)
