"""Base resolver class with common functionality."""

from typing import Optional, Dict, Any
from fastapi import HTTPException, status
import structlog

logger = structlog.get_logger()


class BaseResolver:
    """Base class for all GraphQL resolvers."""
    
    @staticmethod
    def require_authentication(context: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure user is authenticated and return user info."""
        if not context.get("is_authenticated") or not context.get("user"):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required"
            )
        return context["user"]
    
    @staticmethod
    def require_role(context: Dict[str, Any], allowed_roles: list[str]) -> Dict[str, Any]:
        """Ensure user has required role."""
        user = BaseResolver.require_authentication(context)
        if user.get("role") not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions"
            )
        return user
    
    @staticmethod
    def get_database_session(context: Dict[str, Any]):
        """Get database session from context."""
        request = context.get("request")
        if not request:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Database session not available"
            )
        return request.state.get("db_session")
    
    @staticmethod
    def log_resolver_error(
        operation: str,
        error: Exception,
        context: Dict[str, Any],
        **kwargs
    ) -> None:
        """Log resolver errors with context."""
        user = context.get("user")
        logger.error(
            f"Resolver error: {operation}",
            error=str(error),
            error_type=type(error).__name__,
            user_id=user.get("id") if user else None,
            **kwargs
        )