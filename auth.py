"""
Authentication module for MOI Biometric System
Database-backed authentication with role-based access control
"""

import os
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from sqlalchemy.orm import Session

# JWT token settings
SECRET_KEY = os.getenv("SECRET_KEY", "moi-biometric-secret-key-change-in-production")
TOKEN_EXPIRE_HOURS = 24

# In-memory token storage with user data
# Structure: {token: {"user_id": int, "username": str, "role": str, "full_name": str, "expiry": datetime}}
active_tokens: Dict[str, Dict[str, Any]] = {}


def hash_password(password: str) -> str:
    """Hash a password using SHA256"""
    return hashlib.sha256(password.encode()).hexdigest()


def verify_password(password: str, password_hash: str) -> bool:
    """Verify a password against its hash"""
    return hash_password(password) == password_hash


def create_token(user_id: int, username: str, role: str, full_name: str) -> str:
    """Create a new authentication token with user data"""
    token = secrets.token_urlsafe(32)
    expiry = datetime.now() + timedelta(hours=TOKEN_EXPIRE_HOURS)
    active_tokens[token] = {
        "user_id": user_id,
        "username": username,
        "role": role,
        "full_name": full_name,
        "expiry": expiry
    }
    return token


def verify_token(token: str) -> Optional[Dict[str, Any]]:
    """
    Verify a token and return user data if valid.
    Returns: {"user_id": int, "username": str, "role": str, "full_name": str} or None
    """
    if token not in active_tokens:
        return None
    
    token_data = active_tokens[token]
    if datetime.now() > token_data["expiry"]:
        # Token expired, remove it
        del active_tokens[token]
        return None
    
    return {
        "user_id": token_data["user_id"],
        "username": token_data["username"],
        "role": token_data["role"],
        "full_name": token_data["full_name"]
    }


def get_token_username(token: str) -> Optional[str]:
    """Get username from token (for backward compatibility)"""
    data = verify_token(token)
    return data["username"] if data else None


def revoke_token(token: str) -> bool:
    """Revoke (logout) a token"""
    if token in active_tokens:
        del active_tokens[token]
        return True
    return False


def authenticate_user(username: str, password: str, db: Session) -> Optional[Dict[str, Any]]:
    """
    Authenticate user from database and return token with user data if successful.
    Returns: {"token": str, "user_id": int, "username": str, "role": str, "full_name": str} or None
    """
    from models import User
    
    # Find user in database
    user = db.query(User).filter(User.username == username).first()
    
    if not user:
        return None
    
    # Check if account is active
    if not user.is_active:
        return None
    
    # Verify password
    if not verify_password(password, user.password_hash):
        return None
    
    # Create token
    token = create_token(user.id, user.username, user.role, user.full_name)
    
    return {
        "token": token,
        "user_id": user.id,
        "username": user.username,
        "role": user.role,
        "full_name": user.full_name
    }


def create_default_admin(db: Session) -> bool:
    """
    Create default admin user if no users exist in database.
    Returns True if admin was created, False otherwise.
    """
    from models import User
    
    # Check if any users exist
    user_count = db.query(User).count()
    if user_count > 0:
        return False
    
    # Create default admin
    default_admin = User(
        username=os.getenv("ADMIN_USERNAME", "admin"),
        password_hash=hash_password(os.getenv("ADMIN_PASSWORD", "moi2024")),
        full_name="مدير النظام",
        role="admin",
        is_active=True,
        created_by=None
    )
    
    db.add(default_admin)
    db.commit()
    print("✅ Default admin user created: admin / moi2024")
    return True


def is_admin(token: str) -> bool:
    """Check if token belongs to an admin user"""
    data = verify_token(token)
    return data is not None and data.get("role") == "admin"


def is_active_user(token: str) -> bool:
    """Check if token is valid (user is authenticated)"""
    return verify_token(token) is not None


def get_current_user_id(token: str) -> Optional[int]:
    """Get user ID from token"""
    data = verify_token(token)
    return data["user_id"] if data else None


def get_current_user_role(token: str) -> Optional[str]:
    """Get user role from token"""
    data = verify_token(token)
    return data["role"] if data else None
