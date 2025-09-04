"""Rate limiting middleware using Redis for distributed rate limiting."""

import time
from typing import Optional
from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
import redis.asyncio as redis
import structlog

from kol_api.config import settings

logger = structlog.get_logger()


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Token bucket rate limiting middleware with Redis backend."""
    
    def __init__(self, app):
        super().__init__(app)
        self.redis_client: Optional[redis.Redis] = None
    
    async def dispatch(self, request: Request, call_next) -> Response:
        """Apply rate limiting to requests."""
        
        # AIDEV-NOTE: Initialize Redis client if not done
        if not self.redis_client:
            try:
                self.redis_client = redis.from_url(
                    settings.redis_url,
                    password=settings.redis_password,
                    max_connections=settings.redis_max_connections
                )
                # Test connection
                await self.redis_client.ping()
            except Exception as e:
                logger.warning("Redis connection failed, rate limiting disabled", error=str(e))
                # Continue without rate limiting if Redis is unavailable
                return await call_next(request)
        
        # AIDEV-NOTE: Skip rate limiting for health checks and static assets
        if self._should_skip_rate_limit(request):
            return await call_next(request)
        
        # AIDEV-NOTE: Determine rate limit key based on user or IP
        rate_limit_key = self._get_rate_limit_key(request)
        
        # AIDEV-NOTE: Check and update rate limit
        is_allowed, reset_time, remaining = await self._check_rate_limit(rate_limit_key)
        
        if not is_allowed:
            # AIDEV-NOTE: Log rate limit violation
            user_info = getattr(request.state, "user", None)
            logger.warning(
                "Rate limit exceeded",
                key=rate_limit_key,
                path=request.url.path,
                user_id=user_info.get("id") if user_info else None,
                reset_time=reset_time
            )
            
            # AIDEV-NOTE: Return 429 Too Many Requests
            headers = {
                "X-RateLimit-Limit": str(settings.rate_limit_requests),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(int(reset_time)),
                "Retry-After": str(max(1, int(reset_time - time.time()))),
            }
            
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded. Too many requests.",
                headers=headers
            )
        
        # AIDEV-NOTE: Process request
        response = await call_next(request)
        
        # AIDEV-NOTE: Add rate limit headers to response
        response.headers["X-RateLimit-Limit"] = str(settings.rate_limit_requests)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(int(reset_time))
        
        return response
    
    def _should_skip_rate_limit(self, request: Request) -> bool:
        """Determine if request should skip rate limiting."""
        path = request.url.path.lower()
        
        # AIDEV-NOTE: Skip rate limiting for these endpoints
        skip_paths = [
            "/api/health",
            "/metrics",
            "/docs",
            "/redoc",
            "/openapi.json",
        ]
        
        return any(path.startswith(skip_path) for skip_path in skip_paths)
    
    def _get_rate_limit_key(self, request: Request) -> str:
        """Generate rate limit key based on user or IP."""
        user_info = getattr(request.state, "user", None)
        
        # AIDEV-NOTE: Use user ID for authenticated requests
        if user_info and user_info.get("id"):
            return f"rate_limit:user:{user_info['id']}"
        
        # AIDEV-NOTE: Use client IP for anonymous requests
        client_ip = self._get_client_ip(request)
        return f"rate_limit:ip:{client_ip}"
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP for rate limiting."""
        # AIDEV-NOTE: Check proxy headers
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        # AIDEV-NOTE: Direct client IP
        if hasattr(request, "client") and request.client:
            return request.client.host
        
        return "unknown"
    
    async def _check_rate_limit(self, key: str) -> tuple[bool, float, int]:
        """
        Check rate limit using token bucket algorithm.
        
        Returns:
            Tuple of (is_allowed, reset_time, remaining_requests)
        """
        try:
            current_time = time.time()
            window_start = int(current_time // settings.rate_limit_window) * settings.rate_limit_window
            window_key = f"{key}:{window_start}"
            
            # AIDEV-NOTE: Use Redis pipeline for atomic operations
            pipe = self.redis_client.pipeline()
            
            # AIDEV-NOTE: Increment request count
            pipe.incr(window_key)
            pipe.expire(window_key, settings.rate_limit_window)
            
            results = await pipe.execute()
            current_requests = results[0]
            
            # AIDEV-NOTE: Calculate remaining requests and reset time
            remaining = max(0, settings.rate_limit_requests - current_requests)
            reset_time = window_start + settings.rate_limit_window
            is_allowed = current_requests <= settings.rate_limit_requests
            
            return is_allowed, reset_time, remaining
            
        except Exception as e:
            # AIDEV-NOTE: If Redis fails, allow request but log error
            logger.error("Rate limit check failed", error=str(e), key=key)
            return True, time.time() + settings.rate_limit_window, settings.rate_limit_requests