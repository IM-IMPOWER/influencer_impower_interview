"""Request/Response logging middleware with audit trails."""

import time
import uuid
from typing import Callable
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
import structlog

logger = structlog.get_logger()


class LoggingMiddleware(BaseHTTPMiddleware):
    """Request and response logging middleware with audit capabilities."""
    
    def __init__(self, app):
        super().__init__(app)
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Log request and response with audit trail information."""
        
        # AIDEV-NOTE: Generate unique request ID for tracing
        request_id = str(uuid.uuid4())
        start_time = time.time()
        
        # AIDEV-NOTE: Extract request information
        client_ip = self._get_client_ip(request)
        user_agent = request.headers.get("User-Agent", "")
        user_info = getattr(request.state, "user", None)
        
        # AIDEV-NOTE: Log incoming request
        logger.info(
            "Request started",
            request_id=request_id,
            method=request.method,
            url=str(request.url),
            path=request.url.path,
            query_params=dict(request.query_params),
            client_ip=client_ip,
            user_agent=user_agent,
            user_id=user_info.get("id") if user_info else None,
            user_email=user_info.get("email") if user_info else None,
        )
        
        # AIDEV-NOTE: Add request ID to request state for downstream use
        request.state.request_id = request_id
        
        try:
            # AIDEV-NOTE: Process request
            response = await call_next(request)
            
            # AIDEV-NOTE: Calculate processing time
            process_time = time.time() - start_time
            
            # AIDEV-NOTE: Log successful response
            logger.info(
                "Request completed",
                request_id=request_id,
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                process_time=round(process_time, 4),
                client_ip=client_ip,
                user_id=user_info.get("id") if user_info else None,
            )
            
            # AIDEV-NOTE: Add request ID to response headers for debugging
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time"] = str(round(process_time, 4))
            
            return response
            
        except Exception as e:
            # AIDEV-NOTE: Log request errors with full context
            process_time = time.time() - start_time
            
            logger.error(
                "Request failed",
                request_id=request_id,
                method=request.method,
                path=request.url.path,
                error=str(e),
                error_type=type(e).__name__,
                process_time=round(process_time, 4),
                client_ip=client_ip,
                user_id=user_info.get("id") if user_info else None,
            )
            
            # AIDEV-NOTE: Re-raise exception for FastAPI error handling
            raise
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address from request headers."""
        # AIDEV-NOTE: Check common proxy headers first
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            # Return first IP in the chain
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        # AIDEV-NOTE: Fall back to direct client connection
        if hasattr(request, "client") and request.client:
            return request.client.host
        
        return "unknown"