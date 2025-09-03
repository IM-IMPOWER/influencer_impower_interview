"""Application configuration settings using Pydantic Settings."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, PostgresDsn, validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration settings."""
    
    # AIDEV-NOTE: Environment and debug settings
    environment: Literal["development", "staging", "production"] = "development"
    debug: bool = Field(default=False, description="Enable debug mode")
    
    # AIDEV-NOTE: API configuration
    api_title: str = "KOL Platform API"
    api_description: str = "FastAPI backend for KOL platform with GraphQL gateway and REST endpoints"
    api_version: str = "0.1.0"
    api_prefix: str = "/api/v1"
    docs_url: str | None = "/docs"
    redoc_url: str | None = "/redoc"
    openapi_url: str | None = "/openapi.json"
    
    # AIDEV-NOTE: Server configuration
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    
    # AIDEV-NOTE: Database configuration
    database_url: PostgresDsn = Field(
        default="postgresql://postgres:password@localhost:5432/kol_platform",
        description="PostgreSQL database URL with pgvector support"
    )
    database_pool_size: int = 10
    database_max_overflow: int = 20
    database_echo: bool = False
    
    # AIDEV-NOTE: Redis configuration for caching
    redis_url: str = "redis://localhost:6379/0"
    redis_password: str | None = None
    redis_max_connections: int = 10
    
    # AIDEV-NOTE: Authentication settings
    secret_key: str = Field(
        default="your-secret-key-change-in-production",
        description="Secret key for JWT token signing"
    )
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7
    
    # AIDEV-NOTE: Security settings
    allowed_hosts: list[str] = Field(default=["*"])
    cors_origins: list[str] = Field(default=["http://localhost:3000", "http://localhost:3001"])
    cors_methods: list[str] = Field(default=["GET", "POST", "PUT", "DELETE", "PATCH"])
    cors_headers: list[str] = Field(default=["*"])
    
    # AIDEV-NOTE: File upload settings
    max_file_size: int = Field(default=10 * 1024 * 1024, description="Max file size in bytes (10MB)")
    upload_path: Path = Field(default=Path("uploads"), description="Upload directory path")
    allowed_file_types: list[str] = Field(default=["image/jpeg", "image/png", "image/webp", "video/mp4"])
    
    # AIDEV-NOTE: Background task settings (Celery)
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"
    celery_include: list[str] = Field(default=["kol_api.tasks"])
    
    # AIDEV-NOTE: External service configurations
    go_scraper_service_url: str = "http://localhost:8080"
    go_scraper_api_key: str | None = None
    
    # AIDEV-NOTE: AI/ML settings
    openai_api_key: str | None = None
    huggingface_api_key: str | None = None
    sentence_transformer_model: str = "all-MiniLM-L6-v2"
    
    # AIDEV-NOTE: KOL scoring weights (POC2 - multi-factor scoring)
    scoring_weights: dict[str, float] = Field(default={
        "engagement_rate": 0.3,
        "follower_quality": 0.25,
        "content_relevance": 0.2,
        "brand_safety": 0.15,
        "posting_consistency": 0.1
    })
    
    # AIDEV-NOTE: Budget optimization settings (POC4)
    budget_tiers: dict[str, dict[str, int]] = Field(default={
        "nano": {"min_followers": 1000, "max_followers": 10000, "avg_cost_per_post": 50},
        "micro": {"min_followers": 10000, "max_followers": 100000, "avg_cost_per_post": 500},
        "mid": {"min_followers": 100000, "max_followers": 1000000, "avg_cost_per_post": 5000},
        "macro": {"min_followers": 1000000, "max_followers": 10000000, "avg_cost_per_post": 50000},
    })
    
    # AIDEV-NOTE: Rate limiting settings
    rate_limit_requests: int = 1000
    rate_limit_window: int = 3600  # 1 hour in seconds
    
    @validator("database_echo")
    @classmethod
    def set_database_echo(cls, v: bool, values: dict) -> bool:
        """Enable database query logging only in development."""
        return v and values.get("environment") == "development"
    
    @validator("docs_url", "redoc_url", "openapi_url")
    @classmethod
    def disable_docs_in_production(cls, v: str | None, values: dict) -> str | None:
        """Disable API docs in production for security."""
        if values.get("environment") == "production":
            return None
        return v
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )


@lru_cache
def get_settings() -> Settings:
    """Get cached application settings."""
    return Settings()


# AIDEV-NOTE: Export settings instance for easy import
settings = get_settings()