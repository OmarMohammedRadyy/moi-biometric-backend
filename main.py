"""
MOI Biometric System - FastAPI Backend
Main application file with all API endpoints
Using DeepFace for face recognition (compatible with Python 3.13)
"""

import os
import uuid
import base64
from io import BytesIO
from typing import List
import json

import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from dotenv import load_dotenv

# Import DeepFace for face recognition
from deepface import DeepFace

from database import get_db, init_db, engine, Base
from models import Visitor
from schemas import (
    VisitorResponse,
    VisitorList,
    VerificationResult,
    MessageResponse,
    ErrorResponse
)

# Load environment variables
load_dotenv()

# Configuration
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# DeepFace model to use (options: VGG-Face, Facenet, Facenet512, OpenFace, DeepFace, DeepID, ArcFace, Dlib, SFace)
FACE_MODEL = "Facenet512"  # Good balance of speed and accuracy
DISTANCE_METRIC = "cosine"  # cosine, euclidean, euclidean_l2
MATCH_THRESHOLD = 0.40  # Lower = more strict matching

# Initialize FastAPI app
app = FastAPI(
    title="MOI Biometric System",
    description="Kuwait Ministry of Interior - Facial Recognition Security System",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Configuration - Allow frontend to connect
# Read allowed origins from environment variable (comma-separated)
cors_origins_env = os.getenv("CORS_ORIGINS", "")
cors_origins = [origin.strip() for origin in cors_origins_env.split(",") if origin.strip()]
# Add default development origins
cors_origins.extend(["http://localhost:5173", "http://localhost:5174", "http://localhost:3000"])

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins if cors_origins else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount uploads directory for serving photos
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")


# ==================== Helper Functions ====================

def get_face_embedding(image_path: str) -> list:
    """
    Generate face embedding using DeepFace.
    Returns a list of floats representing the face encoding.
    """
    try:
        # Get embedding using DeepFace
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
        
        # Return the embedding vector
        return embedding_objs[0]["embedding"]
    
    except Exception as e:
        raise ValueError(f"Face detection failed: {str(e)}")


def compare_embeddings(embedding1: list, embedding2: list) -> float:
    """
    Compare two face embeddings and return distance.
    Lower distance = more similar faces.
    """
    vec1 = np.array(embedding1)
    vec2 = np.array(embedding2)
    
    if DISTANCE_METRIC == "cosine":
        # Cosine distance
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        similarity = dot_product / (norm1 * norm2)
        distance = 1 - similarity
    else:
        # Euclidean distance
        distance = np.linalg.norm(vec1 - vec2)
    
    return float(distance)


# ==================== Startup Event ====================

@app.on_event("startup")
async def startup_event():
    """Initialize database tables on startup"""
    print("🚀 Starting MOI Biometric System...")
    try:
        init_db()
        print("✅ Database initialized successfully!")
    except Exception as e:
        print(f"⚠️ Database initialization warning: {e}")
        print("⚠️ App will continue - database might come up later")
    print(f"🧠 Face Recognition Model: {FACE_MODEL}")
    print(f"📏 Distance Metric: {DISTANCE_METRIC}")
    print(f"🎯 Match Threshold: {MATCH_THRESHOLD}")



# ==================== Health Check ====================

@app.get("/", tags=["Health"])
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "system": "MOI Biometric Security System",
        "version": "1.0.0",
        "face_model": FACE_MODEL
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "database": "connected",
        "face_recognition": "ready",
        "model": FACE_MODEL,
        "threshold": MATCH_THRESHOLD
    }


# ==================== Authentication ====================

from auth import authenticate_user, verify_token, revoke_token
from pydantic import BaseModel

class LoginRequest(BaseModel):
    username: str
    password: str

class TokenResponse(BaseModel):
    token: str
    username: str
    message: str


@app.post("/api/auth/login", response_model=TokenResponse, tags=["Authentication"])
async def login(request: LoginRequest):
    """
    Login with username and password.
    Returns a token for authenticated requests.
    """
    token = authenticate_user(request.username, request.password)
    
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="اسم المستخدم أو كلمة المرور غير صحيحة"
        )
    
    return {
        "token": token,
        "username": request.username,
        "message": "تم تسجيل الدخول بنجاح"
    }


@app.post("/api/auth/logout", tags=["Authentication"])
async def logout(token: str = Form(...)):
    """Logout and invalidate token"""
    revoke_token(token)
    return {"message": "تم تسجيل الخروج بنجاح"}


@app.get("/api/auth/verify", tags=["Authentication"])
async def verify_auth(token: str):
    """Verify if a token is valid"""
    username = verify_token(token)
    
    if not username:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="الجلسة غير صالحة أو منتهية"
        )
    
    return {
        "valid": True,
        "username": username
    }


@app.get("/api/auth/me", tags=["Authentication"])
async def get_current_user(token: str):
    """Get current authenticated user info"""
    username = verify_token(token)
    
    if not username:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="غير مصرح"
        )
    
    return {
        "username": username,
        "role": "admin"
    }


# ==================== Visitor Management (Admin) ====================

@app.post("/api/visitors", response_model=VisitorResponse, tags=["Admin"])
async def register_visitor(
    full_name: str = Form(...),
    passport_number: str = Form(...),
    visa_status: str = Form(...),
    photo: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Register a new visitor with their photo.
    
    - Detects face in the uploaded photo
    - Generates face embedding using DeepFace
    - Stores visitor data and embedding in database
    """
    try:
        # Check if passport already exists
        existing = db.query(Visitor).filter(Visitor.passport_number == passport_number).first()
        if existing:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Visitor with passport {passport_number} already registered"
            )

        # Read and validate image
        contents = await photo.read()
        image = Image.open(BytesIO(contents))
        
        # Convert to RGB if necessary
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Save photo temporarily for processing
        file_extension = photo.filename.split(".")[-1] if "." in photo.filename else "jpg"
        unique_filename = f"{uuid.uuid4()}.{file_extension}"
        photo_path = os.path.join(UPLOAD_DIR, unique_filename)
        
        # Save the image
        image.save(photo_path, quality=95)
        
        # Generate face embedding
        try:
            face_embedding = get_face_embedding(photo_path)
        except ValueError as e:
            # Delete the saved photo if face detection fails
            if os.path.exists(photo_path):
                os.remove(photo_path)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e)
            )
        
        # Create visitor record
        visitor = Visitor(
            full_name=full_name,
            passport_number=passport_number,
            visa_status=visa_status,
            photo_path=unique_filename,
            face_encoding=face_embedding
        )
        
        db.add(visitor)
        db.commit()
        db.refresh(visitor)
        
        print(f"✅ Visitor registered: {full_name} ({passport_number})")
        
        return visitor

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error registering visitor: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing registration: {str(e)}"
        )


@app.get("/api/visitors", response_model=VisitorList, tags=["Admin"])
async def list_visitors(db: Session = Depends(get_db)):
    """Get all registered visitors"""
    visitors = db.query(Visitor).order_by(Visitor.created_at.desc()).all()
    return VisitorList(visitors=visitors, total=len(visitors))


@app.get("/api/visitors/{visitor_id}", response_model=VisitorResponse, tags=["Admin"])
async def get_visitor(visitor_id: int, db: Session = Depends(get_db)):
    """Get a specific visitor by ID"""
    visitor = db.query(Visitor).filter(Visitor.id == visitor_id).first()
    if not visitor:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Visitor with ID {visitor_id} not found"
        )
    return visitor


@app.delete("/api/visitors/{visitor_id}", response_model=MessageResponse, tags=["Admin"])
async def delete_visitor(visitor_id: int, db: Session = Depends(get_db)):
    """Delete a visitor by ID"""
    visitor = db.query(Visitor).filter(Visitor.id == visitor_id).first()
    if not visitor:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Visitor with ID {visitor_id} not found"
        )
    
    # Delete photo file if exists
    photo_path = os.path.join(UPLOAD_DIR, visitor.photo_path)
    if os.path.exists(photo_path):
        os.remove(photo_path)
    
    db.delete(visitor)
    db.commit()
    
    return MessageResponse(success=True, message=f"Visitor {visitor.full_name} deleted successfully")


# ==================== Face Verification (Security Scanner) ====================

@app.post("/api/verify", response_model=VerificationResult, tags=["Security"])
async def verify_face(
    photo: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Verify a face against all registered visitors.
    
    - Accepts an image upload from the security scanner
    - Detects face and generates embedding
    - Compares against all stored embeddings in database
    - Returns match result with visitor details if found
    """
    temp_path = None
    
    try:
        # Read and process uploaded image
        contents = await photo.read()
        image = Image.open(BytesIO(contents))
        
        # Convert to RGB if necessary
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Save temporarily for DeepFace processing
        temp_path = os.path.join(UPLOAD_DIR, f"temp_{uuid.uuid4()}.jpg")
        image.save(temp_path, quality=95)
        
        # Generate face embedding for captured image
        try:
            captured_embedding = get_face_embedding(temp_path)
        except ValueError as e:
            return VerificationResult(
                match_found=False,
                confidence=None,
                visitor=None,
                message=str(e),
                captured_photo=None
            )
        
        # Get all visitors from database
        visitors = db.query(Visitor).all()
        
        if len(visitors) == 0:
            return VerificationResult(
                match_found=False,
                confidence=None,
                visitor=None,
                message="No registered visitors in database",
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
        
        # Convert captured image to base64 for frontend display
        buffered = BytesIO()
        image.save(buffered, format="JPEG", quality=85)
        captured_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        # Check if best match is within threshold
        if best_match and best_distance < MATCH_THRESHOLD:
            # Convert distance to confidence percentage (inverse relationship)
            confidence = round((1 - best_distance) * 100, 2)
            
            print(f"✅ MATCH FOUND: {best_match.full_name} (Distance: {best_distance:.4f}, Confidence: {confidence}%)")
            
            return VerificationResult(
                match_found=True,
                confidence=confidence,
                visitor=VisitorResponse(
                    id=best_match.id,
                    full_name=best_match.full_name,
                    passport_number=best_match.passport_number,
                    visa_status=best_match.visa_status,
                    photo_path=best_match.photo_path,
                    created_at=best_match.created_at,
                    updated_at=best_match.updated_at
                ),
                message=f"Identity Verified: {best_match.full_name}",
                captured_photo=captured_base64
            )
        else:
            print(f"❌ NO MATCH: Best distance was {best_distance:.4f} (threshold: {MATCH_THRESHOLD})")
            
            return VerificationResult(
                match_found=False,
                confidence=round((1 - best_distance) * 100, 2) if best_distance < 1 else 0,
                visitor=None,
                message="NOT AUTHORIZED - No matching record found",
                captured_photo=captured_base64
            )

    except Exception as e:
        print(f"❌ Verification error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error during verification: {str(e)}"
        )
    
    finally:
        # Clean up temporary file
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass


# ==================== Statistics ====================

@app.get("/api/stats", tags=["Admin"])
async def get_statistics(db: Session = Depends(get_db)):
    """Get system statistics"""
    total_visitors = db.query(Visitor).count()
    
    return {
        "total_visitors": total_visitors,
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
    
    # Run directly - no reload in production
    uvicorn.run(app, host=host, port=port)


