from passlib.context import CryptContext
from datetime import datetime, timedelta
from jose import JWTError, jwt

# Password Hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT Configuration
SECRET_KEY = "YOUR_SUPER_SECRET_KEY"  # This should be in an env variable
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

import re

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def validate_password_strength(password: str) -> bool:
    """
    Validates that the password meets the following criteria:
    - Minimum 8 characters
    - At least one number
    - At least one letter
    - (Ideally mixed case and symbols, but per requirement: alphanumeric, 8 chars, combination)
    Strict requirement from prompt: "Minimum 8 chars, combination of numbers and letters, at least one char and one number"
    """
    if len(password) < 8:
        return False
    if not re.search(r"[a-zA-Z]", password):
        return False
    if not re.search(r"\d", password):
        return False
    return True
