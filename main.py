"""
MOI Biometric System - FastAPI Backend
Main application file with all API endpoints
Using DeepFace for face recognition
"""

import os
import uuid
import base64
from io import BytesIO
from typing import List, Optional
from datetime import datetime, timedelta

import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, Depends, HTTPException, status, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from sqlalchemy import func, and_
from dotenv import load_dotenv
from pydantic import BaseModel

# Import DeepFace for face recognition
from deepface import DeepFace

from database import get_db, init_db, engine, Base
from models import Visitor, User, AuthLog, ScanLog
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
    DashboardStats
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

# DeepFace model configuration
FACE_MODEL = "Facenet512"
DISTANCE_METRIC = "cosine"
MATCH_THRESHOLD = 0.40

# Initialize FastAPI app
app = FastAPI(
    title="MOI Biometric System",
    description="Kuwait Ministry of Interior - Facial Recognition Security System",
    version="3.5.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Custom CORS middleware to ensure headers are always sent
@app.middleware("http")
async def add_cors_headers(request: Request, call_next):
    # Handle preflight OPTIONS requests
    if request.method == "OPTIONS":
        response = JSONResponse(content={}, status_code=200)
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, PATCH, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "*"
        response.headers["Access-Control-Max-Age"] = "3600"
        return response
    
    # Process request and add CORS headers to response
    response = await call_next(request)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, PATCH, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "*"
    return response

# Mount uploads directory
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")


# ==================== Helper Functions ====================

def get_face_embedding(image_path: str) -> list:
    """Generate face embedding using DeepFace."""
    try:
        embedding_objs = DeepFace.represent(
            img_path=image_path,
            model_name=FACE_MODEL,
            enforce_detection=True,
            detector_backend="opencv"
        )
        
        if len(embedding_objs) == 0:
            raise ValueError("No face detected in image")
        
        if len(embedding_objs) > 1:
            raise ValueError(f"Multiple faces ({len(embedding_objs)}) detected in image")
        
        return embedding_objs[0]["embedding"]
    
    except Exception as e:
        raise ValueError(f"Face detection failed: {str(e)}")


def compare_embeddings(embedding1: list, embedding2: list) -> float:
    """Compare two face embeddings and return distance."""
    vec1 = np.array(embedding1)
    vec2 = np.array(embedding2)
    
    if DISTANCE_METRIC == "cosine":
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        similarity = dot_product / (norm1 * norm2)
        distance = 1 - similarity
    else:
        distance = np.linalg.norm(vec1 - vec2)
    
    return float(distance)


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
    print(f"🎯 Match Threshold: {MATCH_THRESHOLD}")


# ==================== Health Check ====================

@app.get("/", tags=["Health"])
async def root():
    """Health check endpoint."""
    return {
        "status": "online",
        "system": "MOI Biometric Security System",
        "version": "2.0.0",
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
        "threshold": MATCH_THRESHOLD
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
    result = authenticate_user(request.username, request.password, db)
    
    if not result:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="اسم المستخدم أو كلمة المرور غير صحيحة أو الحساب معطل"
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
    
    return UserToggleResponse(
        success=True,
        user_id=user.id,
        is_active=user.is_active,
        message=f"تم {'تفعيل' if user.is_active else 'تعطيل'} حساب {user.full_name}"
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
    db.delete(user)
    db.commit()
    
    return MessageResponse(success=True, message=f"تم حذف المستخدم {username}")


# ==================== Auth Logs (Admin Only) ====================

@app.get("/api/logs/auth", response_model=AuthLogList, tags=["Logs"])
async def get_auth_logs(
    page: int = 1,
    per_page: int = 50,
    user_id: Optional[int] = None,
    action: Optional[str] = None,
    admin: dict = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """Get authentication logs with pagination (admin only)."""
    query = db.query(AuthLog).join(User)
    
    # Apply filters
    if user_id:
        query = query.filter(AuthLog.user_id == user_id)
    if action:
        query = query.filter(AuthLog.action == action)
    
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
    admin: dict = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """Get scan logs with pagination (admin only). By default shows only matches."""
    query = db.query(ScanLog).join(User, ScanLog.officer_id == User.id)
    
    # Apply filters
    if officer_id:
        query = query.filter(ScanLog.officer_id == officer_id)
    if match_only:
        query = query.filter(ScanLog.match_found == True)
    
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


# ==================== Dashboard Statistics (Admin Only) ====================

@app.get("/api/dashboard/stats", response_model=DashboardStats, tags=["Dashboard"])
async def get_dashboard_stats(
    admin: dict = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """Get dashboard statistics (admin only)."""
    now = datetime.now()
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    week_start = today_start - timedelta(days=7)
    
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
        recent_scans=recent_scans,
        system_status="operational"
    )


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
        
        # Generate face embedding
        try:
            face_embedding = get_face_embedding(photo_path)
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
    visitors = db.query(Visitor).order_by(Visitor.created_at.desc()).all()
    return VisitorList(visitors=visitors, total=len(visitors))


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
    photo_path = os.path.join(UPLOAD_DIR, visitor.photo_path)
    if os.path.exists(photo_path):
        os.remove(photo_path)
    
    name = visitor.full_name
    db.delete(visitor)
    db.commit()
    
    return MessageResponse(success=True, message=f"تم حذف الزائر {name}")


# ==================== Face Verification (All Authenticated Users) ====================

@app.post("/api/verify", response_model=VerificationResult, tags=["Security"])
async def verify_face(
    req: Request,
    photo: UploadFile = File(...),
    token: Optional[str] = Form(None),
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db)
):
    """
    Verify a face against all registered visitors.
    Logs the scan with officer information.
    """
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
    
    temp_path = None
    saved_scan_path = None
    
    try:
        # Read and process uploaded image
        contents = await photo.read()
        image = Image.open(BytesIO(contents))
        
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Save temporarily for DeepFace processing
        temp_path = os.path.join(UPLOAD_DIR, f"temp_{uuid.uuid4()}.jpg")
        image.save(temp_path, quality=95)
        
        # Generate face embedding
        try:
            captured_embedding = get_face_embedding(temp_path)
        except ValueError as e:
            # Log failed scan (no face detected)
            scan_log = ScanLog(
                officer_id=user_data["user_id"],
                visitor_id=None,
                match_found=False,
                confidence=None,
                ip_address=get_client_ip(req),
                location=None,
                captured_photo_path=None
            )
            db.add(scan_log)
            db.commit()
            
            return VerificationResult(
                match_found=False,
                confidence=None,
                visitor=None,
                message=str(e),
                captured_photo=None
            )
        
        # Get all visitors
        visitors = db.query(Visitor).all()
        
        if len(visitors) == 0:
            return VerificationResult(
                match_found=False,
                confidence=None,
                visitor=None,
                message="لا يوجد زوار مسجلين في قاعدة البيانات",
                captured_photo=None
            )
        
        # Compare against all stored embeddings
        best_match = None
        best_distance = float('inf')
        
        for visitor in visitors:
            stored_embedding = visitor.face_encoding
            distance = compare_embeddings(captured_embedding, stored_embedding)
            
            if distance < best_distance:
                best_distance = distance
                best_match = visitor
        
        # Convert captured image to base64
        buffered = BytesIO()
        image.save(buffered, format="JPEG", quality=85)
        captured_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        # Check if match found
        match_found = best_match and best_distance < MATCH_THRESHOLD
        confidence = round((1 - best_distance) * 100, 2) if best_distance < 1 else 0
        
        # Save scan photo only for matches
        if match_found:
            scan_filename = f"scan_{uuid.uuid4()}.jpg"
            saved_scan_path = os.path.join(SCAN_PHOTOS_DIR, scan_filename)
            image.save(saved_scan_path, quality=85)
            relative_scan_path = f"scans/{scan_filename}"
        else:
            relative_scan_path = None
        
        # Log the scan
        scan_log = ScanLog(
            officer_id=user_data["user_id"],
            visitor_id=best_match.id if match_found else None,
            match_found=match_found,
            confidence=confidence,
            ip_address=get_client_ip(req),
            location=None,
            captured_photo_path=relative_scan_path
        )
        db.add(scan_log)
        db.commit()
        
        if match_found:
            print(f"✅ MATCH: {best_match.full_name} by {user_data['username']} (Confidence: {confidence}%)")
            
            return VerificationResult(
                match_found=True,
                confidence=confidence,
                visitor=VisitorResponse(
                    id=best_match.id,
                    full_name=best_match.full_name,
                    passport_number=best_match.passport_number,
                    visa_status=best_match.visa_status,
                    photo_path=best_match.photo_path,
                    photo_base64=best_match.photo_base64,
                    created_at=best_match.created_at,
                    updated_at=best_match.updated_at
                ),
                message=f"تم التحقق من الهوية: {best_match.full_name}",
                captured_photo=captured_base64
            )
        else:
            print(f"❌ NO MATCH by {user_data['username']} (Best distance: {best_distance:.4f})")
            
            return VerificationResult(
                match_found=False,
                confidence=confidence,
                visitor=None,
                message="غير مصرح - لا يوجد تطابق في السجلات",
                captured_photo=captured_base64
            )

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
        "match_threshold": MATCH_THRESHOLD
    }


# ==================== Run Server ====================

if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    
    print(f"🔒 Starting on http://{host}:{port}", flush=True)
    uvicorn.run(app, host=host, port=port)
