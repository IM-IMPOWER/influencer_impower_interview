"""Authentication and user models."""

from datetime import datetime
from enum import Enum
from typing import List, Optional

from sqlalchemy import Boolean, String, Text, DateTime, Integer
from sqlalchemy.orm import Mapped, mapped_column, relationship

from kol_api.database.models.base import BaseModel


class UserRole(str, Enum):
    """User role enumeration."""
    ADMIN = "admin"
    MANAGER = "manager"
    ANALYST = "analyst"
    VIEWER = "viewer"


class User(BaseModel):
    """User authentication and profile model."""
    
    __tablename__ = "users"
    
    # AIDEV-NOTE: Authentication fields
    email: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        nullable=False,
        index=True,
    )
    hashed_password: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
    )
    is_verified: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )
    
    # AIDEV-NOTE: Profile information
    first_name: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )
    last_name: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )
    role: Mapped[UserRole] = mapped_column(
        String(50),
        default=UserRole.VIEWER,
        nullable=False,
    )
    
    # AIDEV-NOTE: Additional profile fields
    avatar_url: Mapped[Optional[str]] = mapped_column(
        String(500),
        nullable=True,
    )
    bio: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )
    phone: Mapped[Optional[str]] = mapped_column(
        String(20),
        nullable=True,
    )
    
    # AIDEV-NOTE: Authentication tracking
    last_login: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    failed_login_attempts: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )
    locked_until: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    
    # AIDEV-NOTE: Email verification
    verification_token: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )
    verification_token_expires: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    
    # AIDEV-NOTE: Password reset
    reset_token: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )
    reset_token_expires: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    
    # AIDEV-NOTE: Relationships to campaigns created by user
    campaigns = relationship(
        "Campaign",
        back_populates="created_by",
        cascade="all, delete-orphan"
    )
    
    @property
    def full_name(self) -> str:
        """Get user's full name."""
        return f"{self.first_name} {self.last_name}"
    
    @property
    def is_locked(self) -> bool:
        """Check if user account is locked."""
        if not self.locked_until:
            return False
        return datetime.utcnow() < self.locked_until
    
    def __repr__(self) -> str:
        """String representation of user."""
        return f"<User(email={self.email}, role={self.role})>"