"""Authentication and authorization REST endpoints."""

from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, status, Response, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr
from typing import Optional
import jwt
import bcrypt
import structlog

from kol_api.config import settings
from kol_api.database.connection import get_session
from kol_api.database.models.auth import User, UserRole

router = APIRouter(prefix="/auth", tags=["Authentication"])
security = HTTPBearer(auto_error=False)
logger = structlog.get_logger()


class LoginRequest(BaseModel):
    """Login request model."""
    email: EmailStr
    password: str
    remember_me: bool = False


class RegisterRequest(BaseModel):
    """User registration request model."""
    email: EmailStr
    password: str
    first_name: str
    last_name: str
    role: UserRole = UserRole.VIEWER


class TokenResponse(BaseModel):
    """Authentication token response."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    user: dict


class RefreshRequest(BaseModel):
    """Token refresh request."""
    refresh_token: str


class PasswordResetRequest(BaseModel):
    """Password reset request."""
    email: EmailStr


class PasswordUpdateRequest(BaseModel):
    """Password update request."""
    current_password: str
    new_password: str


def create_access_token(user_data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token."""
    to_encode = user_data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.access_token_expire_minutes)
    
    to_encode.update({"exp": expire, "type": "access"})
    
    return jwt.encode(to_encode, settings.secret_key, algorithm=settings.algorithm)


def create_refresh_token(user_data: dict) -> str:
    """Create JWT refresh token."""
    to_encode = user_data.copy()
    expire = datetime.utcnow() + timedelta(days=settings.refresh_token_expire_days)
    to_encode.update({"exp": expire, "type": "refresh"})
    
    return jwt.encode(to_encode, settings.secret_key, algorithm=settings.algorithm)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash."""
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))


def hash_password(password: str) -> str:
    """Hash password using bcrypt."""
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')


async def get_user_by_email(db_session, email: str) -> Optional[User]:
    """Get user by email address."""
    # AIDEV-NOTE: This would be implemented with actual database query
    # For now, return None - implement with SQLAlchemy queries
    return None


@router.post("/login", response_model=TokenResponse)
async def login(
    request: LoginRequest,
    response: Response,
    db_session=Depends(get_session)
):
    """Authenticate user and return tokens."""
    
    try:
        # AIDEV-NOTE: Get user from database
        user = await get_user_by_email(db_session, request.email)
        
        if not user or not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials"
            )
        
        # AIDEV-NOTE: Verify password
        if not verify_password(request.password, user.hashed_password):
            # AIDEV-NOTE: Log failed login attempt
            logger.warning(
                "Failed login attempt",
                email=request.email,
                user_id=user.id
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials"
            )
        
        # AIDEV-NOTE: Check if account is locked
        if user.is_locked:
            raise HTTPException(
                status_code=status.HTTP_423_LOCKED,
                detail="Account temporarily locked due to failed login attempts"
            )
        
        # AIDEV-NOTE: Create token payload
        token_data = {
            "sub": str(user.id),
            "email": user.email,
            "role": user.role.value,
        }
        
        # AIDEV-NOTE: Generate tokens
        access_token_expires = timedelta(
            minutes=settings.access_token_expire_minutes * (30 if request.remember_me else 1)
        )
        access_token = create_access_token(token_data, access_token_expires)
        refresh_token = create_refresh_token(token_data)
        
        # AIDEV-NOTE: Set HTTP-only cookie for web clients
        response.set_cookie(
            key="access_token",
            value=access_token,
            httponly=True,
            secure=settings.environment == "production",
            samesite="lax",
            max_age=int(access_token_expires.total_seconds())
        )
        
        # AIDEV-NOTE: Update last login time
        user.last_login = datetime.utcnow()
        user.failed_login_attempts = 0
        await db_session.commit()
        
        logger.info("User logged in successfully", user_id=user.id, email=user.email)
        
        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=int(access_token_expires.total_seconds()),
            user={
                "id": str(user.id),
                "email": user.email,
                "first_name": user.first_name,
                "last_name": user.last_name,
                "role": user.role.value,
                "is_verified": user.is_verified,
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Login error", error=str(e), email=request.email)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )


@router.post("/register", response_model=dict)
async def register(
    request: RegisterRequest,
    db_session=Depends(get_session)
):
    """Register new user account."""
    
    try:
        # AIDEV-NOTE: Check if user already exists
        existing_user = await get_user_by_email(db_session, request.email)
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Email already registered"
            )
        
        # AIDEV-NOTE: Hash password
        hashed_password = hash_password(request.password)
        
        # AIDEV-NOTE: Create new user (would use SQLAlchemy)
        # This is placeholder - implement with actual database operations
        logger.info("User registration attempt", email=request.email)
        
        return {
            "message": "Registration successful",
            "email": request.email,
            "verification_required": True
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Registration error", error=str(e), email=request.email)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed"
        )


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(
    request: RefreshRequest,
    db_session=Depends(get_session)
):
    """Refresh access token using refresh token."""
    
    try:
        # AIDEV-NOTE: Decode and validate refresh token
        payload = jwt.decode(
            request.refresh_token,
            settings.secret_key,
            algorithms=[settings.algorithm]
        )
        
        if payload.get("type") != "refresh":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token type"
            )
        
        user_id = payload.get("sub")
        email = payload.get("email")
        
        # AIDEV-NOTE: Verify user still exists and is active
        user = await get_user_by_email(db_session, email)
        if not user or not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found or inactive"
            )
        
        # AIDEV-NOTE: Create new tokens
        token_data = {
            "sub": str(user.id),
            "email": user.email,
            "role": user.role.value,
        }
        
        access_token = create_access_token(token_data)
        new_refresh_token = create_refresh_token(token_data)
        
        return TokenResponse(
            access_token=access_token,
            refresh_token=new_refresh_token,
            expires_in=settings.access_token_expire_minutes * 60,
            user={
                "id": str(user.id),
                "email": user.email,
                "first_name": user.first_name,
                "last_name": user.last_name,
                "role": user.role.value,
            }
        )
        
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token expired"
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token"
        )
    except Exception as e:
        logger.error("Token refresh error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Token refresh failed"
        )


@router.post("/logout")
async def logout(response: Response):
    """Logout user by clearing cookies."""
    
    # AIDEV-NOTE: Clear authentication cookies
    response.delete_cookie(key="access_token")
    
    return {"message": "Logged out successfully"}


@router.get("/me")
async def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Get current authenticated user information."""
    
    # AIDEV-NOTE: User information is available from auth middleware
    user = getattr(request.state, "user", None)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated"
        )
    
    return {
        "id": user["id"],
        "email": user["email"],
        "role": user["role"],
        "authenticated": True
    }