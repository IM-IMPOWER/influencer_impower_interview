"""KOL (Key Opinion Leader) related models."""

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import List, Optional, Dict, Any

from sqlalchemy import (
    Boolean, String, Text, Integer, Numeric, DateTime, JSON,
    Index, ForeignKey, UniqueConstraint
)
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import Mapped, mapped_column, relationship
from pgvector.sqlalchemy import Vector

from kol_api.database.models.base import BaseModel


class PlatformType(str, Enum):
    """Social media platform enumeration."""
    TIKTOK = "tiktok"
    INSTAGRAM = "instagram"
    YOUTUBE = "youtube"
    FACEBOOK = "facebook"
    TWITTER = "twitter"
    LINKEDIN = "linkedin"


class KOLTier(str, Enum):
    """KOL tier classification based on follower count."""
    NANO = "nano"          # 1K-10K followers
    MICRO = "micro"        # 10K-100K followers
    MID = "mid"           # 100K-1M followers
    MACRO = "macro"       # 1M-10M followers
    MEGA = "mega"         # 10M+ followers


class ContentCategory(str, Enum):
    """Content category classification."""
    LIFESTYLE = "lifestyle"
    FASHION = "fashion"
    BEAUTY = "beauty"
    FITNESS = "fitness"
    FOOD = "food"
    TRAVEL = "travel"
    TECH = "tech"
    GAMING = "gaming"
    EDUCATION = "education"
    ENTERTAINMENT = "entertainment"
    BUSINESS = "business"
    HEALTH = "health"
    PARENTING = "parenting"
    AUTOMOTIVE = "automotive"
    HOME_DECOR = "home_decor"


class KOL(BaseModel):
    """Main KOL (Key Opinion Leader) model."""
    
    __tablename__ = "kols"
    
    # AIDEV-NOTE: Basic KOL information
    username: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
    )
    display_name: Mapped[str] = mapped_column(
        String(200),
        nullable=False,
    )
    platform: Mapped[PlatformType] = mapped_column(
        String(50),
        nullable=False,
        index=True,
    )
    platform_id: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )
    profile_url: Mapped[str] = mapped_column(
        String(500),
        nullable=False,
    )
    
    # AIDEV-NOTE: Profile information
    avatar_url: Mapped[Optional[str]] = mapped_column(
        String(500),
        nullable=True,
    )
    bio: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )
    location: Mapped[Optional[str]] = mapped_column(
        String(200),
        nullable=True,
    )
    
    # AIDEV-NOTE: Classification and categorization
    tier: Mapped[KOLTier] = mapped_column(
        String(50),
        nullable=False,
        index=True,
    )
    primary_category: Mapped[ContentCategory] = mapped_column(
        String(100),
        nullable=False,
        index=True,
    )
    secondary_categories: Mapped[List[str]] = mapped_column(
        ARRAY(String(100)),
        default=list,
        nullable=False,
    )
    
    # AIDEV-NOTE: Demographics
    age_range: Mapped[Optional[str]] = mapped_column(
        String(20),
        nullable=True,
    )
    gender: Mapped[Optional[str]] = mapped_column(
        String(20),
        nullable=True,
    )
    languages: Mapped[List[str]] = mapped_column(
        ARRAY(String(10)),
        default=list,
        nullable=False,
    )
    
    # AIDEV-NOTE: Status and safety flags
    is_verified: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
    )
    is_brand_safe: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
        index=True,
    )
    safety_notes: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )
    
    # AIDEV-NOTE: Data source and quality tracking
    data_source: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        default="scraper",
    )
    last_scraped: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
    )
    scrape_quality_score: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(5, 4),
        nullable=True,
    )
    
    # AIDEV-NOTE: Vector embedding for semantic search (POC2)
    content_embedding: Mapped[Optional[List[float]]] = mapped_column(
        Vector(384),  # Sentence transformer dimension
        nullable=True,
    )
    
    # AIDEV-NOTE: Relationships
    profiles = relationship(
        "KOLProfile",
        back_populates="kol",
        cascade="all, delete-orphan"
    )
    metrics = relationship(
        "KOLMetrics",
        back_populates="kol",
        cascade="all, delete-orphan"
    )
    content = relationship(
        "KOLContent",
        back_populates="kol",
        cascade="all, delete-orphan"
    )
    scores = relationship(
        "KOLScore",
        back_populates="kol",
        cascade="all, delete-orphan"
    )
    campaign_kols = relationship(
        "CampaignKOL",
        back_populates="kol"
    )
    
    # AIDEV-NOTE: Unique constraint on platform and platform_id
    __table_args__ = (
        UniqueConstraint("platform", "platform_id", name="uq_kol_platform_id"),
        Index("ix_kol_tier_category", "tier", "primary_category"),
        Index("ix_kol_brand_safe_active", "is_brand_safe", "is_active"),
        Index("ix_kol_content_embedding_gin", "content_embedding", postgresql_using="gin"),
    )
    
    def __repr__(self) -> str:
        """String representation of KOL."""
        return f"<KOL(username={self.username}, platform={self.platform}, tier={self.tier})>"


class KOLProfile(BaseModel):
    """Extended KOL profile information from different data sources."""
    
    __tablename__ = "kol_profiles"
    
    kol_id: Mapped[str] = mapped_column(
        ForeignKey("kols.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    
    # AIDEV-NOTE: Source information
    source: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )
    source_url: Mapped[Optional[str]] = mapped_column(
        String(500),
        nullable=True,
    )
    
    # AIDEV-NOTE: Extended profile data as JSON
    profile_data: Mapped[Dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )
    
    # AIDEV-NOTE: Contact information
    email: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )
    phone: Mapped[Optional[str]] = mapped_column(
        String(20),
        nullable=True,
    )
    management_contact: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )
    
    # AIDEV-NOTE: Pricing information
    rate_per_post: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(10, 2),
        nullable=True,
    )
    rate_per_story: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(10, 2),
        nullable=True,
    )
    rate_per_video: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(10, 2),
        nullable=True,
    )
    min_budget: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(10, 2),
        nullable=True,
    )
    currency: Mapped[str] = mapped_column(
        String(3),
        default="THB",
        nullable=False,
    )
    
    # AIDEV-NOTE: Relationships
    kol = relationship("KOL", back_populates="profiles")
    
    # AIDEV-NOTE: Unique constraint per source
    __table_args__ = (
        UniqueConstraint("kol_id", "source", name="uq_kol_profile_source"),
        Index("ix_kol_profile_rates", "rate_per_post", "rate_per_video"),
    )


class KOLMetrics(BaseModel):
    """KOL performance metrics and statistics."""
    
    __tablename__ = "kol_metrics"
    
    kol_id: Mapped[str] = mapped_column(
        ForeignKey("kols.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    
    # AIDEV-NOTE: Follower metrics
    follower_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        index=True,
    )
    following_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )
    
    # AIDEV-NOTE: Content metrics
    total_posts: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )
    total_videos: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )
    
    # AIDEV-NOTE: Engagement metrics
    avg_likes: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(12, 2),
        nullable=True,
    )
    avg_comments: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(12, 2),
        nullable=True,
    )
    avg_shares: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(12, 2),
        nullable=True,
    )
    avg_views: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(15, 2),
        nullable=True,
    )
    
    # AIDEV-NOTE: Calculated engagement rates (POC2 scoring factors)
    engagement_rate: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(5, 4),
        nullable=True,
        index=True,
    )
    like_rate: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(5, 4),
        nullable=True,
    )
    comment_rate: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(5, 4),
        nullable=True,
    )
    
    # AIDEV-NOTE: Audience quality metrics
    audience_quality_score: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(5, 4),
        nullable=True,
    )
    fake_follower_percentage: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(5, 4),
        nullable=True,
    )
    
    # AIDEV-NOTE: Posting consistency metrics
    posts_last_30_days: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )
    avg_posting_frequency: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(5, 2),
        nullable=True,
    )
    
    # AIDEV-NOTE: Growth metrics
    follower_growth_rate: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(5, 4),
        nullable=True,
    )
    engagement_trend: Mapped[Optional[str]] = mapped_column(
        String(20),
        nullable=True,
    )
    
    # AIDEV-NOTE: Data freshness
    metrics_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
    )
    
    # AIDEV-NOTE: Relationships
    kol = relationship("KOL", back_populates="metrics")
    
    # AIDEV-NOTE: Indexes for performance
    __table_args__ = (
        Index("ix_kol_metrics_engagement", "engagement_rate", "follower_count"),
        Index("ix_kol_metrics_date", "metrics_date", postgresql_using="btree"),
    )


class KOLContent(BaseModel):
    """Sample content from KOL for analysis and scoring."""
    
    __tablename__ = "kol_content"
    
    kol_id: Mapped[str] = mapped_column(
        ForeignKey("kols.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    
    # AIDEV-NOTE: Content identification
    platform_content_id: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )
    content_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )
    content_url: Mapped[str] = mapped_column(
        String(500),
        nullable=False,
    )
    
    # AIDEV-NOTE: Content metadata
    caption: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )
    hashtags: Mapped[List[str]] = mapped_column(
        ARRAY(String(100)),
        default=list,
        nullable=False,
    )
    mentions: Mapped[List[str]] = mapped_column(
        ARRAY(String(100)),
        default=list,
        nullable=False,
    )
    
    # AIDEV-NOTE: Content performance
    likes_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )
    comments_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )
    shares_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )
    views_count: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )
    
    # AIDEV-NOTE: Content analysis
    content_categories: Mapped[List[str]] = mapped_column(
        ARRAY(String(100)),
        default=list,
        nullable=False,
    )
    brand_mentions: Mapped[List[str]] = mapped_column(
        ARRAY(String(100)),
        default=list,
        nullable=False,
    )
    sentiment_score: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(3, 2),
        nullable=True,
    )
    
    # AIDEV-NOTE: Content embedding for similarity search
    content_embedding: Mapped[Optional[List[float]]] = mapped_column(
        Vector(384),
        nullable=True,
    )
    
    # AIDEV-NOTE: Content timing
    posted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
    )
    
    # AIDEV-NOTE: Relationships
    kol = relationship("KOL", back_populates="content")
    
    # AIDEV-NOTE: Unique constraint and indexes
    __table_args__ = (
        UniqueConstraint("kol_id", "platform_content_id", name="uq_kol_content"),
        Index("ix_kol_content_performance", "likes_count", "comments_count"),
        Index("ix_kol_content_embedding", "content_embedding", postgresql_using="gin"),
        Index("ix_kol_content_posted_at", "posted_at", postgresql_using="btree"),
    )