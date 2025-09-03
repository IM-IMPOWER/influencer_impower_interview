"""Multi-factor KOL scoring system models for POC2."""

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Optional, Dict, Any

from sqlalchemy import (
    String, Text, Numeric, DateTime, JSON,
    ForeignKey, Index, UniqueConstraint, CheckConstraint
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from kol_api.database.models.base import BaseModel


class ScoreType(str, Enum):
    """Score type enumeration."""
    OVERALL = "overall"
    ENGAGEMENT_RATE = "engagement_rate"
    FOLLOWER_QUALITY = "follower_quality"
    CONTENT_RELEVANCE = "content_relevance"
    BRAND_SAFETY = "brand_safety"
    POSTING_CONSISTENCY = "posting_consistency"
    AUDIENCE_MATCH = "audience_match"
    COST_EFFICIENCY = "cost_efficiency"


class ScoreSource(str, Enum):
    """Score calculation source."""
    ALGORITHMIC = "algorithmic"
    MANUAL = "manual"
    HYBRID = "hybrid"
    THIRD_PARTY = "third_party"


class KOLScore(BaseModel):
    """Multi-factor KOL scoring model for POC2 matching algorithm."""
    
    __tablename__ = "kol_scores"
    
    # AIDEV-NOTE: KOL association
    kol_id: Mapped[str] = mapped_column(
        ForeignKey("kols.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    
    # AIDEV-NOTE: Campaign context (optional for campaign-specific scoring)
    campaign_id: Mapped[Optional[str]] = mapped_column(
        ForeignKey("campaigns.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    
    # AIDEV-NOTE: Overall scoring
    overall_score: Mapped[Decimal] = mapped_column(
        Numeric(5, 4),
        nullable=False,
        index=True,
    )
    overall_percentile: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(5, 4),
        nullable=True,
    )
    
    # AIDEV-NOTE: Individual factor scores (POC2 multi-factor scoring)
    engagement_rate_score: Mapped[Decimal] = mapped_column(
        Numeric(5, 4),
        nullable=False,
        index=True,
    )
    follower_quality_score: Mapped[Decimal] = mapped_column(
        Numeric(5, 4),
        nullable=False,
        index=True,
    )
    content_relevance_score: Mapped[Decimal] = mapped_column(
        Numeric(5, 4),
        nullable=False,
    )
    brand_safety_score: Mapped[Decimal] = mapped_column(
        Numeric(5, 4),
        nullable=False,
        index=True,
    )
    posting_consistency_score: Mapped[Decimal] = mapped_column(
        Numeric(5, 4),
        nullable=False,
    )
    
    # AIDEV-NOTE: Campaign-specific scores (when campaign_id is provided)
    audience_match_score: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(5, 4),
        nullable=True,
    )
    cost_efficiency_score: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(5, 4),
        nullable=True,
    )
    brand_affinity_score: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(5, 4),
        nullable=True,
    )
    
    # AIDEV-NOTE: Scoring methodology
    scoring_model: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        default="weighted_average_v1",
    )
    model_version: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="1.0",
    )
    scoring_source: Mapped[ScoreSource] = mapped_column(
        String(50),
        default=ScoreSource.ALGORITHMIC,
        nullable=False,
    )
    
    # AIDEV-NOTE: Scoring weights used for calculation
    scoring_weights: Mapped[Dict[str, float]] = mapped_column(
        JSON,
        nullable=False,
    )
    
    # AIDEV-NOTE: Raw data used for scoring
    raw_metrics: Mapped[Dict[str, Any]] = mapped_column(
        JSON,
        default=dict,
        nullable=False,
    )
    
    # AIDEV-NOTE: Score confidence and reliability
    confidence_score: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(5, 4),
        nullable=True,
    )
    data_freshness_days: Mapped[Optional[int]] = mapped_column(
        nullable=True,
    )
    sample_size: Mapped[Optional[int]] = mapped_column(
        nullable=True,
    )
    
    # AIDEV-NOTE: Scoring context and notes
    scoring_context: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )
    exclusion_reasons: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSON,
        nullable=True,
    )
    
    # AIDEV-NOTE: Temporal information
    scored_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
    )
    expires_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    
    # AIDEV-NOTE: Relationships
    kol = relationship("KOL", back_populates="scores")
    campaign = relationship("Campaign", foreign_keys=[campaign_id])
    
    # AIDEV-NOTE: Constraints and indexes
    __table_args__ = (
        CheckConstraint("overall_score >= 0 AND overall_score <= 1", name="check_overall_score_range"),
        CheckConstraint("engagement_rate_score >= 0 AND engagement_rate_score <= 1", name="check_engagement_score_range"),
        CheckConstraint("follower_quality_score >= 0 AND follower_quality_score <= 1", name="check_follower_quality_range"),
        CheckConstraint("content_relevance_score >= 0 AND content_relevance_score <= 1", name="check_content_relevance_range"),
        CheckConstraint("brand_safety_score >= 0 AND brand_safety_score <= 1", name="check_brand_safety_range"),
        CheckConstraint("posting_consistency_score >= 0 AND posting_consistency_score <= 1", name="check_consistency_range"),
        UniqueConstraint("kol_id", "campaign_id", "scoring_model", "model_version", name="uq_kol_campaign_score"),
        Index("ix_kol_score_overall_campaign", "overall_score", "campaign_id"),
        Index("ix_kol_score_brand_safety", "brand_safety_score", "kol_id"),
        Index("ix_kol_score_freshness", "scored_at", "expires_at"),
        Index("ix_kol_score_composite", "overall_score", "engagement_rate_score", "follower_quality_score"),
    )
    
    @property
    def is_expired(self) -> bool:
        """Check if score has expired."""
        if not self.expires_at:
            return False
        return datetime.utcnow() > self.expires_at
    
    @property
    def score_breakdown(self) -> Dict[str, Decimal]:
        """Get breakdown of all scoring factors."""
        return {
            "overall": self.overall_score,
            "engagement_rate": self.engagement_rate_score,
            "follower_quality": self.follower_quality_score,
            "content_relevance": self.content_relevance_score,
            "brand_safety": self.brand_safety_score,
            "posting_consistency": self.posting_consistency_score,
            "audience_match": self.audience_match_score or Decimal("0"),
            "cost_efficiency": self.cost_efficiency_score or Decimal("0"),
            "brand_affinity": self.brand_affinity_score or Decimal("0"),
        }
    
    def __repr__(self) -> str:
        """String representation of KOL score."""
        campaign_info = f", campaign_id={self.campaign_id}" if self.campaign_id else ""
        return f"<KOLScore(kol_id={self.kol_id}{campaign_info}, overall_score={self.overall_score})>"


class ScoreHistory(BaseModel):
    """Historical tracking of KOL scores for trend analysis."""
    
    __tablename__ = "score_history"
    
    # AIDEV-NOTE: Score record reference
    score_id: Mapped[str] = mapped_column(
        ForeignKey("kol_scores.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    
    # AIDEV-NOTE: Quick reference fields (denormalized for performance)
    kol_id: Mapped[str] = mapped_column(
        ForeignKey("kols.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    campaign_id: Mapped[Optional[str]] = mapped_column(
        ForeignKey("campaigns.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    
    # AIDEV-NOTE: Historical score data
    historical_overall_score: Mapped[Decimal] = mapped_column(
        Numeric(5, 4),
        nullable=False,
    )
    score_change: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(6, 4),  # Can be negative for decreases
        nullable=True,
    )
    
    # AIDEV-NOTE: Change context
    change_reason: Mapped[Optional[str]] = mapped_column(
        String(200),
        nullable=True,
    )
    change_factors: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSON,
        nullable=True,
    )
    
    # AIDEV-NOTE: Snapshot metadata
    snapshot_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
    )
    model_version_at_time: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )
    
    # AIDEV-NOTE: Performance comparison
    previous_score: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(5, 4),
        nullable=True,
    )
    percentile_change: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(6, 4),
        nullable=True,
    )
    
    # AIDEV-NOTE: Relationships
    score = relationship("KOLScore", foreign_keys=[score_id])
    kol = relationship("KOL", foreign_keys=[kol_id])
    campaign = relationship("Campaign", foreign_keys=[campaign_id])
    
    # AIDEV-NOTE: Indexes for time-series queries
    __table_args__ = (
        Index("ix_score_history_kol_date", "kol_id", "snapshot_date"),
        Index("ix_score_history_campaign_date", "campaign_id", "snapshot_date"),
        Index("ix_score_history_score_change", "score_change", "snapshot_date"),
    )
    
    def __repr__(self) -> str:
        """String representation of score history."""
        return f"<ScoreHistory(kol_id={self.kol_id}, score={self.historical_overall_score}, date={self.snapshot_date})>"