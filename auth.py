"""
Authentication module for MOI Biometric System
Simple JWT-based authentication
"""

import os
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional

# JWT token settings
SECRET_KEY = os.getenv("SECRET_KEY", "moi-biometric-secret-key-change-in-production")
TOKEN_EXPIRE_HOURS = 24

# Default admin credentials (change in production!)
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD_HASH = hashlib.sha256(
    os.getenv("ADMIN_PASSWORD", "moi2024").encode()
).hexdigest()

# In-memory token storage (for simplicity)
# In production, use Redis or database
active_tokens = {}


def hash_password(password: str) -> str:
    """Hash a password using SHA256"""
    return hashlib.sha256(password.encode()).hexdigest()


def verify_password(password: str, password_hash: str) -> bool:
    """Verify a password against its hash"""
    return hash_password(password) == password_hash


def create_token(username: str) -> str:
    """Create a new authentication token"""
    token = secrets.token_urlsafe(32)
    expiry = datetime.now() + timedelta(hours=TOKEN_EXPIRE_HOURS)
    active_tokens[token] = {
        "username": username,
        "expiry": expiry
    }
    return token


def verify_token(token: str) -> Optional[str]:
    """Verify a token and return the username if valid"""
    if token not in active_tokens:
        return None
    
    token_data = active_tokens[token]
    if datetime.now() > token_data["expiry"]:
        # Token expired, remove it
        del active_tokens[token]
        return None
    
    return token_data["username"]


def revoke_token(token: str) -> bool:
    """Revoke (logout) a token"""
    if token in active_tokens:
        del active_tokens[token]
        return True
    return False


def authenticate_user(username: str, password: str) -> Optional[str]:
    """
    Authenticate user and return token if successful
    """
    # Check credentials
    if username == ADMIN_USERNAME and verify_password(password, ADMIN_PASSWORD_HASH):
        return create_token(username)
    
    return None
