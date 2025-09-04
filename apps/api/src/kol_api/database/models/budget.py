"""Budget planning and optimization models for POC4."""

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import List, Optional, Dict, Any

from sqlalchemy import (
    Boolean, String, Text, Integer, Numeric, DateTime, JSON,
    ForeignKey, Index, UniqueConstraint, CheckConstraint
)
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import Mapped, mapped_column, relationship

from kol_api.database.models.base import BaseModel
from kol_api.database.models.kol import KOLTier


class BudgetStatus(str, Enum):
    """Budget plan status enumeration."""
    DRAFT = "draft"
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    ACTIVE = "active"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class OptimizationObjective(str, Enum):
    """Budget optimization objective."""
    MAXIMIZE_REACH = "maximize_reach"
    MAXIMIZE_ENGAGEMENT = "maximize_engagement"
    MAXIMIZE_CONVERSIONS = "maximize_conversions"
    MINIMIZE_COST = "minimize_cost"
    BALANCED = "balanced"


class AllocationStrategy(str, Enum):
    """Budget allocation strategy."""
    EQUAL_DISTRIBUTION = "equal_distribution"
    PERFORMANCE_WEIGHTED = "performance_weighted"
    TIER_BASED = "tier_based"
    CATEGORY_BASED = "category_based"
    CUSTOM = "custom"


class BudgetPlan(BaseModel):
    """Main budget plan model for campaign optimization."""
    
    __tablename__ = "budget_plans"
    
    # AIDEV-NOTE: Campaign association
    campaign_id: Mapped[str] = mapped_column(
        ForeignKey("campaigns.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    
    # AIDEV-NOTE: Plan metadata
    name: Mapped[str] = mapped_column(
        String(200),
        nullable=False,
    )
    description: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )
    version: Mapped[int] = mapped_column(
        Integer,
        default=1,
        nullable=False,
    )
    
    # AIDEV-NOTE: Budget plan status
    status: Mapped[BudgetStatus] = mapped_column(
        String(50),
        default=BudgetStatus.DRAFT,
        nullable=False,
        index=True,
    )
    
    # AIDEV-NOTE: Optimization parameters (POC4 core logic)
    optimization_objective: Mapped[OptimizationObjective] = mapped_column(
        String(50),
        nullable=False,
    )
    allocation_strategy: Mapped[AllocationStrategy] = mapped_column(
        String(50),
        nullable=False,
    )
    
    # AIDEV-NOTE: Budget constraints
    total_budget: Mapped[Decimal] = mapped_column(
        Numeric(12, 2),
        nullable=False,
        index=True,
    )
    reserved_buffer: Mapped[Decimal] = mapped_column(
        Numeric(12, 2),
        default=0,
        nullable=False,
    )
    available_budget: Mapped[Decimal] = mapped_column(
        Numeric(12, 2),
        nullable=False,
    )
    currency: Mapped[str] = mapped_column(
        String(3),
        default="THB",
        nullable=False,
    )
    
    # AIDEV-NOTE: KOL distribution requirements
    min_kols_required: Mapped[int] = mapped_column(
        Integer,
        default=1,
        nullable=False,
    )
    max_kols_allowed: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )
    tier_requirements: Mapped[Dict[str, int]] = mapped_column(
        JSON,
        default=dict,
        nullable=False,
    )
    category_requirements: Mapped[Dict[str, int]] = mapped_column(
        JSON,
        default=dict,
        nullable=False,
    )
    
    # AIDEV-NOTE: Optimization weights and parameters
    objective_weights: Mapped[Dict[str, float]] = mapped_column(
        JSON,
        default=dict,
        nullable=False,
    )
    constraint_parameters: Mapped[Dict[str, Any]] = mapped_column(
        JSON,
        default=dict,
        nullable=False,
    )
    
    # AIDEV-NOTE: Predicted performance metrics
    predicted_reach: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )
    predicted_engagement: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )
    predicted_conversions: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )
    predicted_roi: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(5, 4),
        nullable=True,
    )
    
    # AIDEV-NOTE: Cost efficiency metrics
    cost_per_reach: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(8, 4),
        nullable=True,
    )
    cost_per_engagement: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(8, 4),
        nullable=True,
    )
    cost_per_conversion: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(8, 4),
        nullable=True,
    )
    
    # AIDEV-NOTE: Algorithm metadata
    optimization_algorithm: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        default="linear_programming",
    )
    algorithm_version: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="v1.0",
    )
    optimization_score: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(5, 4),
        nullable=True,
    )
    
    # AIDEV-NOTE: Approval and execution tracking
    approved_by_id: Mapped[Optional[str]] = mapped_column(
        ForeignKey("users.id"),
        nullable=True,
    )
    approved_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    executed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    
    # AIDEV-NOTE: Additional metadata
    notes: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )
    tags: Mapped[List[str]] = mapped_column(
        ARRAY(String(50)),
        default=list,
        nullable=False,
    )
    
    # AIDEV-NOTE: Relationships
    campaign = relationship("Campaign", back_populates="budget_plans")
    approved_by = relationship("User", foreign_keys=[approved_by_id])
    allocations = relationship(
        "BudgetAllocation",
        back_populates="budget_plan",
        cascade="all, delete-orphan"
    )
    
    # AIDEV-NOTE: Constraints and indexes
    __table_args__ = (
        CheckConstraint("total_budget > 0", name="check_positive_total_budget"),
        CheckConstraint("available_budget >= 0", name="check_non_negative_available_budget"),
        CheckConstraint("reserved_buffer >= 0", name="check_non_negative_buffer"),
        CheckConstraint("min_kols_required > 0", name="check_positive_min_kols"),
        Index("ix_budget_plan_campaign_version", "campaign_id", "version"),
        Index("ix_budget_plan_status_budget", "status", "total_budget"),
        Index("ix_budget_plan_objective", "optimization_objective", "allocation_strategy"),
    )
    
    @property
    def allocated_budget(self) -> Decimal:
        """Calculate total allocated budget from allocations."""
        return sum(allocation.allocated_amount for allocation in self.allocations)
    
    @property
    def remaining_budget(self) -> Decimal:
        """Calculate remaining budget after allocations."""
        return self.available_budget - self.allocated_budget
    
    @property
    def budget_utilization(self) -> Decimal:
        """Calculate budget utilization percentage."""
        if self.available_budget == 0:
            return Decimal("0.00")
        return (self.allocated_budget / self.available_budget) * 100
    
    def __repr__(self) -> str:
        """String representation of budget plan."""
        return f"<BudgetPlan(campaign_id={self.campaign_id}, total_budget={self.total_budget}, status={self.status})>"


class BudgetAllocation(BaseModel):
    """Individual KOL budget allocation within a budget plan."""
    
    __tablename__ = "budget_allocations"
    
    # AIDEV-NOTE: Parent budget plan
    budget_plan_id: Mapped[str] = mapped_column(
        ForeignKey("budget_plans.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    
    # AIDEV-NOTE: KOL assignment (optional for general allocations)
    kol_id: Mapped[Optional[str]] = mapped_column(
        ForeignKey("kols.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    
    # AIDEV-NOTE: Allocation details
    allocation_name: Mapped[str] = mapped_column(
        String(200),
        nullable=False,
    )
    allocation_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True,
    )  # 'kol_payment', 'production_cost', 'platform_fee', 'contingency'
    
    # AIDEV-NOTE: Financial allocation
    allocated_amount: Mapped[Decimal] = mapped_column(
        Numeric(10, 2),
        nullable=False,
        index=True,
    )
    spent_amount: Mapped[Decimal] = mapped_column(
        Numeric(10, 2),
        default=0,
        nullable=False,
    )
    currency: Mapped[str] = mapped_column(
        String(3),
        default="THB",
        nullable=False,
    )
    
    # AIDEV-NOTE: KOL tier and category (for non-KOL specific allocations)
    target_tier: Mapped[Optional[str]] = mapped_column(
        String(50),
        nullable=True,
    )
    target_category: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
    )
    
    # AIDEV-NOTE: Expected performance from this allocation
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
    
    # AIDEV-NOTE: Optimization metrics
    efficiency_score: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(5, 4),
        nullable=True,
    )
    priority_score: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(5, 4),
        nullable=True,
    )
    risk_score: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(5, 4),
        nullable=True,
    )
    
    # AIDEV-NOTE: Execution tracking
    is_committed: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )
    committed_at: Mapped[Optional[datetime]] = mapped_column(
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
    actual_roi: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(5, 4),
        nullable=True,
    )
    
    # AIDEV-NOTE: Additional metadata
    notes: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )
    allocation_metadata: Mapped[Dict[str, Any]] = mapped_column(
        JSON,
        default=dict,
        nullable=False,
    )
    
    # AIDEV-NOTE: Relationships
    budget_plan = relationship("BudgetPlan", back_populates="allocations")
    kol = relationship("KOL", foreign_keys=[kol_id])
    
    # AIDEV-NOTE: Constraints and indexes
    __table_args__ = (
        CheckConstraint("allocated_amount > 0", name="check_positive_allocated_amount"),
        CheckConstraint("spent_amount >= 0", name="check_non_negative_spent_amount"),
        CheckConstraint("spent_amount <= allocated_amount", name="check_spent_not_exceeds_allocated"),
        Index("ix_budget_allocation_type_tier", "allocation_type", "target_tier"),
        Index("ix_budget_allocation_committed", "is_committed", "committed_at"),
        Index("ix_budget_allocation_efficiency", "efficiency_score", "priority_score"),
    )
    
    @property
    def remaining_amount(self) -> Decimal:
        """Calculate remaining amount in allocation."""
        return self.allocated_amount - self.spent_amount
    
    @property
    def utilization_rate(self) -> Decimal:
        """Calculate allocation utilization rate."""
        if self.allocated_amount == 0:
            return Decimal("0.00")
        return (self.spent_amount / self.allocated_amount) * 100
    
    @property
    def performance_efficiency(self) -> Optional[Decimal]:
        """Calculate performance efficiency based on actual vs expected."""
        if not self.expected_reach or not self.actual_reach:
            return None
        return Decimal(str(self.actual_reach / self.expected_reach))
    
    def __repr__(self) -> str:
        """String representation of budget allocation."""
        return f"<BudgetAllocation(name={self.allocation_name}, amount={self.allocated_amount}, type={self.allocation_type})>"