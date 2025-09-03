"""Health check endpoints for monitoring and load balancing."""

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from typing import Dict, Any
import asyncio
import time

from kol_api.database.connection import check_database_health
from kol_api.config import settings

router = APIRouter(tags=["Health"])


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    timestamp: float
    version: str
    environment: str
    checks: Dict[str, Any]


class ReadinessResponse(BaseModel):
    """Readiness check response model."""
    ready: bool
    timestamp: float
    services: Dict[str, bool]


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Basic health check endpoint.
    Returns 200 if service is running.
    """
    return HealthResponse(
        status="healthy",
        timestamp=time.time(),
        version=settings.api_version,
        environment=settings.environment,
        checks={
            "api": "healthy",
            "uptime": time.time(),
        }
    )


@router.get("/health/ready", response_model=ReadinessResponse)
async def readiness_check():
    """
    Readiness check for Kubernetes/Docker deployments.
    Verifies all critical dependencies are available.
    """
    services = {}
    all_ready = True
    
    # AIDEV-NOTE: Check database connectivity
    try:
        db_healthy = await check_database_health()
        services["database"] = db_healthy
        if not db_healthy:
            all_ready = False
    except Exception:
        services["database"] = False
        all_ready = False
    
    # AIDEV-NOTE: Check Redis connectivity (optional for caching)
    try:
        import redis.asyncio as redis
        redis_client = redis.from_url(settings.redis_url)
        await redis_client.ping()
        services["redis"] = True
        await redis_client.close()
    except Exception:
        services["redis"] = False
        # Redis is optional, don't fail readiness for it
    
    # AIDEV-NOTE: Check Go scraper service (integration point)
    try:
        import httpx
        async with httpx.AsyncClient(timeout=2.0) as client:
            response = await client.get(f"{settings.go_scraper_service_url}/health")
            services["scraper_service"] = response.status_code == 200
    except Exception:
        services["scraper_service"] = False
        # Scraper service is optional for core functionality
    
    if not all_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not ready"
        )
    
    return ReadinessResponse(
        ready=all_ready,
        timestamp=time.time(),
        services=services
    )


@router.get("/health/live")
async def liveness_check():
    """
    Liveness check for Kubernetes/Docker deployments.
    Simple check to verify the process is alive.
    """
    return {"alive": True, "timestamp": time.time()}