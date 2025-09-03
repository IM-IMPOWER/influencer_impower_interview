"""FastAPI main application entry point."""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import ORJSONResponse
from prometheus_client import make_asgi_app
from strawberry.fastapi import GraphQLRouter

from kol_api.config import settings
from kol_api.database.connection import init_database, close_database
from kol_api.graphql.schema import schema
from kol_api.middleware.auth import AuthMiddleware
from kol_api.middleware.logging import LoggingMiddleware
from kol_api.middleware.rate_limit import RateLimitMiddleware
from kol_api.routers import (
    auth_router,
    campaigns_router,
    kols_router,
    budget_optimizer_router,
    scoring_router,
    upload_router,
    health_router,
)

# AIDEV-NOTE: Configure structured logging for production observability
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager for startup and shutdown events."""
    # AIDEV-NOTE: Startup - Initialize database connections and caching
    logger.info("Starting KOL Platform API", environment=settings.environment)
    
    try:
        await init_database()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error("Failed to initialize database", error=str(e))
        raise
    
    yield
    
    # AIDEV-NOTE: Shutdown - Clean up connections and resources
    logger.info("Shutting down KOL Platform API")
    try:
        await close_database()
        logger.info("Database connections closed successfully")
    except Exception as e:
        logger.error("Error during shutdown", error=str(e))


def create_app() -> FastAPI:
    """Create FastAPI application with all configurations and middleware."""
    
    app = FastAPI(
        title=settings.api_title,
        description=settings.api_description,
        version=settings.api_version,
        docs_url=settings.docs_url,
        redoc_url=settings.redoc_url,
        openapi_url=settings.openapi_url,
        default_response_class=ORJSONResponse,
        lifespan=lifespan,
    )
    
    # AIDEV-NOTE: Security middleware - Must be added first
    app.add_middleware(
        TrustedHostMiddleware, 
        allowed_hosts=settings.allowed_hosts
    )
    
    # AIDEV-NOTE: CORS middleware for frontend integration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=settings.cors_methods,
        allow_headers=settings.cors_headers,
    )
    
    # AIDEV-NOTE: Custom middleware stack
    app.add_middleware(LoggingMiddleware)
    app.add_middleware(RateLimitMiddleware)
    app.add_middleware(AuthMiddleware)
    
    # AIDEV-NOTE: GraphQL endpoint - Primary API interface
    graphql_app = GraphQLRouter(
        schema,
        context_getter=get_graphql_context,
        path="/graphql",
    )
    app.include_router(graphql_app, prefix="/api")
    
    # AIDEV-NOTE: REST API routers for specific use cases
    app.include_router(health_router, prefix="/api")
    app.include_router(auth_router, prefix=settings.api_prefix)
    app.include_router(kols_router, prefix=settings.api_prefix)
    app.include_router(campaigns_router, prefix=settings.api_prefix)
    app.include_router(budget_optimizer_router, prefix=settings.api_prefix)
    app.include_router(scoring_router, prefix=settings.api_prefix)
    app.include_router(upload_router, prefix=settings.api_prefix)
    
    # AIDEV-NOTE: Metrics endpoint for observability (POC monitoring)
    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)
    
    return app


async def get_graphql_context(request, response):
    """GraphQL context factory with user authentication and request info."""
    # AIDEV-NOTE: Extract user from auth middleware for GraphQL resolvers
    user = getattr(request.state, "user", None)
    
    return {
        "request": request,
        "response": response,
        "user": user,
        "is_authenticated": user is not None,
    }


# AIDEV-NOTE: Create app instance for deployment
app = create_app()


@app.get("/")
async def root():
    """API root endpoint with basic information."""
    return {
        "name": settings.api_title,
        "version": settings.api_version,
        "environment": settings.environment,
        "status": "healthy",
        "endpoints": {
            "graphql": "/api/graphql",
            "docs": settings.docs_url,
            "redoc": settings.redoc_url,
            "health": "/api/health",
            "metrics": "/metrics"
        }
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "kol_api.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        workers=1 if settings.debug else settings.workers,
        log_level="debug" if settings.debug else "info",
    )