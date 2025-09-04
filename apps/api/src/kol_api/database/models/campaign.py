"""Campaign management models."""

from datetime import datetime, date
from decimal import Decimal
from enum import Enum
from typing import List, Optional, Dict, Any

from sqlalchemy import (
    Boolean, String, Text, Integer, Numeric, DateTime, Date, JSON,
    ForeignKey, Index, UniqueConstraint
)
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import Mapped, mapped_column, relationship

from kol_api.database.models.base import BaseModel
from kol_api.database.models.kol import KOLTier, ContentCategory


class CampaignStatus(str, Enum):
    """Campaign status enumeration."""
    DRAFT = "draft"
    PLANNING = "planning"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class CampaignObjective(str, Enum):
    """Campaign objective enumeration."""
    BRAND_AWARENESS = "brand_awareness"
    ENGAGEMENT = "engagement" 
    LEAD_GENERATION = "lead_generation"
    SALES = "sales"
    APP_INSTALLS = "app_installs"
    WEBSITE_TRAFFIC = "website_traffic"


class CollaborationStatus(str, Enum):
    """KOL collaboration status."""
    INVITED = "invited"
    ACCEPTED = "accepted"
    DECLINED = "declined"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    PAID = "paid"


class Campaign(BaseModel):
    """Main campaign model."""
    
    __tablename__ = "campaigns"
    
    # AIDEV-NOTE: Basic campaign information
    name: Mapped[str] = mapped_column(
        String(200),
        nullable=False,
        index=True,
    )
    description: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )
    
    # AIDEV-NOTE: Campaign owner and team
    created_by_id: Mapped[str] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    client_name: Mapped[str] = mapped_column(
        String(200),
        nullable=False,
    )
    brand_name: Mapped[str] = mapped_column(
        String(200),
        nullable=False,
        index=True,
    )
    
    # AIDEV-NOTE: Campaign status and timeline
    status: Mapped[CampaignStatus] = mapped_column(
        String(50),
        default=CampaignStatus.DRAFT,
        nullable=False,
        index=True,
    )
    objective: Mapped[CampaignObjective] = mapped_column(
        String(50),
        nullable=False,
    )
    
    # AIDEV-NOTE: Campaign timeline
    start_date: Mapped[date] = mapped_column(
        Date,
        nullable=False,
        index=True,
    )
    end_date: Mapped[date] = mapped_column(
        Date,
        nullable=False,
    )
    application_deadline: Mapped[Optional[date]] = mapped_column(
        Date,
        nullable=True,
    )
    
    # AIDEV-NOTE: Budget information (POC4 - Budget Optimizer)
    total_budget: Mapped[Decimal] = mapped_column(
        Numeric(12, 2),
        nullable=False,
        index=True,
    )
    allocated_budget: Mapped[Decimal] = mapped_column(
        Numeric(12, 2),
        default=0,
        nullable=False,
    )
    spent_budget: Mapped[Decimal] = mapped_column(
        Numeric(12, 2),
        default=0,
        nullable=False,
    )
    currency: Mapped[str] = mapped_column(
        String(3),
        default="THB",
        nullable=False,
    )
    
    # AIDEV-NOTE: KOL requirements (POC2 - KOL Matching)
    target_kol_tiers: Mapped[List[str]] = mapped_column(
        ARRAY(String(50)),
        nullable=False,
    )
    target_categories: Mapped[List[str]] = mapped_column(
        ARRAY(String(100)),
        nullable=False,
    )
    target_demographics: Mapped[Dict[str, Any]] = mapped_column(
        JSON,
        default=dict,
        nullable=False,
    )
    min_follower_count: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )
    max_follower_count: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )
    min_engagement_rate: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(5, 4),
        nullable=True,
    )
    
    # AIDEV-NOTE: Content requirements
    content_guidelines: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )
    required_hashtags: Mapped[List[str]] = mapped_column(
        ARRAY(String(100)),
        default=list,
        nullable=False,
    )
    prohibited_content: Mapped[List[str]] = mapped_column(
        ARRAY(String(200)),
        default=list,
        nullable=False,
    )
    
    # AIDEV-NOTE: Performance tracking
    expected_reach: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )
    expected_engagement: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )
    expected_conversions: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )
    
    # AIDEV-NOTE: Relationships
    created_by = relationship("User", back_populates="campaigns")
    brief = relationship(
        "CampaignBrief",
        back_populates="campaign",
        uselist=False,
        cascade="all, delete-orphan"
    )
    campaign_kols = relationship(
        "CampaignKOL",
        back_populates="campaign",
        cascade="all, delete-orphan"
    )
    budget_plans = relationship(
        "BudgetPlan",
        back_populates="campaign",
        cascade="all, delete-orphan"
    )
    
    # AIDEV-NOTE: Indexes for performance
    __table_args__ = (
        Index("ix_campaign_status_dates", "status", "start_date", "end_date"),
        Index("ix_campaign_budget", "total_budget", "currency"),
        Index("ix_campaign_brand", "brand_name", "status"),
    )
    
    @property
    def remaining_budget(self) -> Decimal:
        """Calculate remaining budget."""
        return self.total_budget - self.allocated_budget
    
    @property
    def budget_utilization(self) -> Decimal:
        """Calculate budget utilization percentage."""
        if self.total_budget == 0:
            return Decimal("0.00")
        return (self.allocated_budget / self.total_budget) * 100
    
    def __repr__(self) -> str:
        """String representation of campaign."""
        return f"<Campaign(name={self.name}, status={self.status}, budget={self.total_budget})>"


class CampaignBrief(BaseModel):
    """Detailed campaign brief and requirements."""
    
    __tablename__ = "campaign_briefs"
    
    campaign_id: Mapped[str] = mapped_column(
        ForeignKey("campaigns.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )
    
    # AIDEV-NOTE: Detailed brief information
    executive_summary: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )
    target_audience: Mapped[Dict[str, Any]] = mapped_column(
        JSON,
        default=dict,
        nullable=False,
    )
    key_messages: Mapped[List[str]] = mapped_column(
        ARRAY(Text),
        default=list,
        nullable=False,
    )
    
    # AIDEV-NOTE: Creative requirements
    creative_direction: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )
    brand_guidelines: Mapped[Dict[str, Any]] = mapped_column(
        JSON,
        default=dict,
        nullable=False,
    )
    deliverables: Mapped[Dict[str, Any]] = mapped_column(
        JSON,
        default=dict,
        nullable=False,
    )
    
    # AIDEV-NOTE: Success metrics
    success_metrics: Mapped[Dict[str, Any]] = mapped_column(
        JSON,
        default=dict,
        nullable=False,
    )
    tracking_requirements: Mapped[Dict[str, Any]] = mapped_column(
        JSON,
        default=dict,
        nullable=False,
    )
    
    # AIDEV-NOTE: Legal and compliance
    legal_requirements: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )
    disclosure_requirements: Mapped[List[str]] = mapped_column(
        ARRAY(String(200)),
        default=list,
        nullable=False,
    )
    
    # AIDEV-NOTE: Additional assets
    reference_content: Mapped[List[str]] = mapped_column(
        ARRAY(String(500)),
        default=list,
        nullable=False,
    )
    brand_assets: Mapped[List[str]] = mapped_column(
        ARRAY(String(500)),
        default=list,
        nullable=False,
    )
    
    # AIDEV-NOTE: Relationships
    campaign = relationship("Campaign", back_populates="brief")


class CampaignKOL(BaseModel):
    """Many-to-many relationship between campaigns and KOLs."""
    
    __tablename__ = "campaign_kols"
    
    campaign_id: Mapped[str] = mapped_column(
        ForeignKey("campaigns.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    kol_id: Mapped[str] = mapped_column(
        ForeignKey("kols.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    
    # AIDEV-NOTE: Collaboration details
    status: Mapped[CollaborationStatus] = mapped_column(
        String(50),
        default=CollaborationStatus.INVITED,
        nullable=False,
        index=True,
    )
    
    # AIDEV-NOTE: Financial terms
    agreed_rate: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(10, 2),
        nullable=True,
    )
    bonus_rate: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(10, 2),
        nullable=True,
    )
    total_cost: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(10, 2),
        nullable=True,
        index=True,
    )
    payment_terms: Mapped[Optional[str]] = mapped_column(
        String(200),
        nullable=True,
    )
    
    # AIDEV-NOTE: Deliverables and timeline
    deliverables: Mapped[Dict[str, Any]] = mapped_column(
        JSON,
        default=dict,
        nullable=False,
    )
    content_deadline: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    
    # AIDEV-NOTE: Performance tracking
    actual_reach: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )
    actual_engagement: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )
    actual_conversions: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )
    
    # AIDEV-NOTE: Content tracking
    submitted_content: Mapped[List[str]] = mapped_column(
        ARRAY(String(500)),
        default=list,
        nullable=False,
    )
    approved_content: Mapped[List[str]] = mapped_column(
        ARRAY(String(500)),
        default=list,
        nullable=False,
    )
    published_content: Mapped[List[str]] = mapped_column(
        ARRAY(String(500)),
        default=list,
        nullable=False,
    )
    
    # AIDEV-NOTE: Communication and notes
    notes: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )
    invitation_sent_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    response_received_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    
    # AIDEV-NOTE: Relationships
    campaign = relationship("Campaign", back_populates="campaign_kols")
    kol = relationship("KOL", back_populates="campaign_kols")
    
    # AIDEV-NOTE: Unique constraint and indexes
    __table_args__ = (
        UniqueConstraint("campaign_id", "kol_id", name="uq_campaign_kol"),
        Index("ix_campaign_kol_status", "status", "campaign_id"),
        Index("ix_campaign_kol_cost", "total_cost", "status"),
    )
    
    @property
    def is_active(self) -> bool:
        """Check if collaboration is active."""
        return self.status in [CollaborationStatus.ACCEPTED, CollaborationStatus.IN_PROGRESS]
    
    def __repr__(self) -> str:
        """String representation of campaign KOL relationship."""
        return f"<CampaignKOL(campaign_id={self.campaign_id}, kol_id={self.kol_id}, status={self.status})>"