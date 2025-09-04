"""Authentication middleware for JWT token validation."""

import jwt
from typing import Optional, Tuple
from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
import structlog

from kol_api.config import settings

logger = structlog.get_logger()


class AuthMiddleware(BaseHTTPMiddleware):
    """JWT authentication middleware."""
    
    # AIDEV-NOTE: Public endpoints that don't require authentication
    PUBLIC_PATHS = {
        "/",
        "/api/health",
        "/api/v1/auth/login",
        "/api/v1/auth/register",
        "/api/v1/auth/refresh",
        "/docs",
        "/redoc",
        "/openapi.json",
        "/metrics",
    }
    
    def __init__(self, app):
        super().__init__(app)
    
    async def dispatch(self, request: Request, call_next) -> Response:
        """Process authentication for each request."""
        
        # AIDEV-NOTE: Skip auth for public endpoints
        if self._is_public_path(request.url.path):
            return await call_next(request)
        
        # AIDEV-NOTE: Extract and validate JWT token
        user_info = await self._extract_user_from_token(request)
        if user_info:
            request.state.user = user_info
            request.state.is_authenticated = True
        else:
            request.state.user = None
            request.state.is_authenticated = False
            
            # AIDEV-NOTE: Return 401 for protected endpoints without valid auth
            if not self._is_public_path(request.url.path):
                return Response(
                    content='{"error": "Authentication required"}',
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    headers={"Content-Type": "application/json"}
                )
        
        response = await call_next(request)
        return response
    
    def _is_public_path(self, path: str) -> bool:
        """Check if path is public and doesn't require authentication."""
        # AIDEV-NOTE: Exact match for public paths
        if path in self.PUBLIC_PATHS:
            return True
        
        # AIDEV-NOTE: Pattern matching for dynamic paths
        if path.startswith("/api/v1/auth/"):
            return True
        if path.startswith("/docs"):
            return True
        if path.startswith("/redoc"):
            return True
        
        return False
    
    async def _extract_user_from_token(self, request: Request) -> Optional[dict]:
        """Extract and validate user information from JWT token."""
        
        # AIDEV-NOTE: Try to get token from Authorization header
        token = self._get_token_from_header(request)
        
        # AIDEV-NOTE: Fallback to cookie-based token
        if not token:
            token = self._get_token_from_cookie(request)
        
        if not token:
            return None
        
        try:
            # AIDEV-NOTE: Decode and validate JWT token
            payload = jwt.decode(
                token,
                settings.secret_key,
                algorithms=[settings.algorithm]
            )
            
            # AIDEV-NOTE: Extract user information from token
            user_id = payload.get("sub")
            email = payload.get("email")
            role = payload.get("role", "viewer")
            
            if not user_id or not email:
                logger.warning("Invalid token payload", payload=payload)
                return None
            
            return {
                "id": user_id,
                "email": email,
                "role": role,
                "token_payload": payload
            }
            
        except jwt.ExpiredSignatureError:
            logger.info("Token expired", path=request.url.path)
            return None
        except jwt.InvalidTokenError as e:
            logger.warning("Invalid token", error=str(e), path=request.url.path)
            return None
        except Exception as e:
            logger.error("Token validation error", error=str(e), path=request.url.path)
            return None
    
    def _get_token_from_header(self, request: Request) -> Optional[str]:
        """Extract token from Authorization header."""
        auth_header = request.headers.get("Authorization")
        if not auth_header:
            return None
        
        try:
            scheme, token = auth_header.split(" ", 1)
            if scheme.lower() != "bearer":
                return None
            return token
        except ValueError:
            return None
    
    def _get_token_from_cookie(self, request: Request) -> Optional[str]:
        """Extract token from HTTP-only cookie."""
        return request.cookies.get("access_token")