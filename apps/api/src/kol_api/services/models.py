"""Pydantic models for KOL services and algorithms."""

from datetime import datetime
from decimal import Decimal
from typing import List, Dict, Any, Optional, Union, Set
from enum import Enum

from pydantic import BaseModel, Field, validator, root_validator
import numpy as np


class OptimizationObjective(str, Enum):
    """Optimization objectives for budget allocation."""
    MAXIMIZE_REACH = "maximize_reach"
    MAXIMIZE_ENGAGEMENT = "maximize_engagement"
    MAXIMIZE_CONVERSIONS = "maximize_conversions"
    MINIMIZE_COST = "minimize_cost"
    MAXIMIZE_ROI = "maximize_roi"
    BALANCED = "balanced"


class KOLTier(str, Enum):
    """KOL tier classification."""
    NANO = "nano"
    MICRO = "micro"
    MID = "mid"
    MACRO = "macro"
    MEGA = "mega"


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


class ScoreComponents(BaseModel):
    """POC2 multi-factor scoring components with weights."""
    
    # Core scoring factors with default weights
    roi_score: Decimal = Field(..., ge=0, le=1, description="ROI Score (25%)")
    audience_quality_score: Decimal = Field(..., ge=0, le=1, description="Audience Quality Score (25%)")
    brand_safety_score: Decimal = Field(..., ge=0, le=1, description="Brand Safety Score (20%)")
    content_relevance_score: Decimal = Field(..., ge=0, le=1, description="Content Relevance Score (15%)")
    demographic_fit_score: Decimal = Field(..., ge=0, le=1, description="Demographic Fit Score (10%)")
    reliability_score: Decimal = Field(..., ge=0, le=1, description="Reliability Score (5%)")
    
    # Confidence scores for uncertain data handling
    roi_confidence: Decimal = Field(default=Decimal("1.0"), ge=0, le=1)
    audience_confidence: Decimal = Field(default=Decimal("1.0"), ge=0, le=1)
    brand_safety_confidence: Decimal = Field(default=Decimal("1.0"), ge=0, le=1)
    content_relevance_confidence: Decimal = Field(default=Decimal("1.0"), ge=0, le=1)
    demographic_confidence: Decimal = Field(default=Decimal("1.0"), ge=0, le=1)
    reliability_confidence: Decimal = Field(default=Decimal("1.0"), ge=0, le=1)
    
    # Metadata
    scoring_timestamp: datetime = Field(default_factory=datetime.utcnow)
    data_freshness_days: int = Field(default=0, ge=0)
    sample_size: Optional[int] = Field(default=None, ge=1)
    
    @property
    def overall_confidence(self) -> Decimal:
        """Calculate overall confidence based on individual component confidences."""
        confidences = [
            self.roi_confidence,
            self.audience_confidence,
            self.brand_safety_confidence,
            self.content_relevance_confidence,
            self.demographic_confidence,
            self.reliability_confidence
        ]
        return sum(confidences) / len(confidences)


class KOLMetricsData(BaseModel):
    """KOL metrics for performance calculation."""
    
    # Follower metrics
    follower_count: int = Field(..., ge=0)
    following_count: int = Field(..., ge=0)
    
    # Engagement metrics
    avg_likes: Optional[Decimal] = Field(None, ge=0)
    avg_comments: Optional[Decimal] = Field(None, ge=0)
    avg_shares: Optional[Decimal] = Field(None, ge=0)
    avg_views: Optional[Decimal] = Field(None, ge=0)
    engagement_rate: Optional[Decimal] = Field(None, ge=0, le=1)
    
    # Quality metrics
    fake_follower_percentage: Optional[Decimal] = Field(None, ge=0, le=1)
    audience_quality_score: Optional[Decimal] = Field(None, ge=0, le=1)
    
    # Consistency metrics
    posts_last_30_days: int = Field(default=0, ge=0)
    avg_posting_frequency: Optional[Decimal] = Field(None, ge=0)
    
    # Growth metrics
    follower_growth_rate: Optional[Decimal] = Field(None)
    engagement_trend: Optional[str] = Field(None)
    
    # Historical performance
    campaign_success_rate: Optional[Decimal] = Field(None, ge=0, le=1)
    response_rate: Optional[Decimal] = Field(None, ge=0, le=1)
    
    # Cost data
    rate_per_post: Optional[Decimal] = Field(None, ge=0)
    rate_per_video: Optional[Decimal] = Field(None, ge=0)
    min_budget: Optional[Decimal] = Field(None, ge=0)
    
    @property
    def has_sufficient_data(self) -> bool:
        """Check if we have sufficient data for reliable scoring."""
        required_fields = [
            self.follower_count > 0,
            self.engagement_rate is not None,
            self.posts_last_30_days > 0
        ]
        return all(required_fields)


class CampaignRequirements(BaseModel):
    """Campaign requirements for KOL matching."""
    
    # Basic requirements
    campaign_id: str
    target_kol_tiers: List[KOLTier]
    target_categories: List[ContentCategory]
    total_budget: Decimal = Field(..., gt=0)
    
    # Follower constraints
    min_follower_count: Optional[int] = Field(None, ge=1)
    max_follower_count: Optional[int] = Field(None, ge=1)
    
    # Performance constraints
    min_engagement_rate: Optional[Decimal] = Field(None, ge=0, le=1)
    min_avg_views: Optional[int] = Field(None, ge=0)
    
    # Demographic targeting
    target_demographics: Dict[str, Any] = Field(default_factory=dict)
    target_locations: List[str] = Field(default_factory=list)
    target_languages: List[str] = Field(default_factory=list)
    target_age_ranges: List[str] = Field(default_factory=list)
    
    # Brand safety requirements
    require_brand_safe: bool = Field(default=True)
    require_verified: bool = Field(default=False)
    exclude_controversial: bool = Field(default=True)
    
    # Content requirements
    required_hashtags: List[str] = Field(default_factory=list)
    excluded_hashtags: List[str] = Field(default_factory=list)
    content_sentiment_requirements: Optional[str] = Field(None)
    
    # Campaign specific
    campaign_objective: OptimizationObjective = Field(default=OptimizationObjective.BALANCED)
    expected_conversion_rate: Optional[Decimal] = Field(None, ge=0, le=1)
    
    @validator('max_follower_count')
    def validate_follower_range(cls, v, values):
        """Ensure max followers > min followers."""
        min_followers = values.get('min_follower_count')
        if min_followers and v and v <= min_followers:
            raise ValueError('max_follower_count must be greater than min_follower_count')
        return v


class BriefParsingResult(BaseModel):
    """Result of markdown brief parsing."""
    
    # Extracted requirements
    campaign_requirements: CampaignRequirements
    
    # Parsing metadata
    parsing_confidence: Decimal = Field(..., ge=0, le=1)
    ambiguous_requirements: List[str] = Field(default_factory=list)
    missing_requirements: List[str] = Field(default_factory=list)
    
    # Raw text analysis
    original_brief: str
    extracted_entities: Dict[str, List[str]] = Field(default_factory=dict)
    sentiment_analysis: Optional[Dict[str, Decimal]] = Field(None)
    
    @property
    def is_actionable(self) -> bool:
        """Check if brief parsing produced actionable requirements."""
        return (
            self.parsing_confidence >= Decimal("0.7") and
            len(self.missing_requirements) <= 2
        )


class KOLCandidate(BaseModel):
    """Enhanced KOL candidate with predictions and scoring."""
    
    # Basic info
    kol_id: str
    username: str
    display_name: str
    platform: str
    tier: KOLTier
    primary_category: ContentCategory
    
    # Metrics
    metrics: KOLMetricsData
    
    # POC2 Scoring components
    score_components: ScoreComponents
    overall_score: Decimal = Field(..., ge=0, le=1)
    
    # Performance predictions
    predicted_reach: int = Field(..., ge=0)
    predicted_engagement: int = Field(..., ge=0)
    predicted_conversions: int = Field(..., ge=0)
    predicted_roi: Optional[Decimal] = Field(None)
    
    # Cost estimates
    estimated_cost_per_post: Decimal = Field(..., ge=0)
    estimated_total_cost: Decimal = Field(..., ge=0)
    
    # Risk assessment
    risk_factors: List[str] = Field(default_factory=list)
    overall_risk_score: Decimal = Field(..., ge=0, le=1)
    
    # Vector similarity (for semantic matching)
    content_embedding: Optional[List[float]] = Field(None)
    semantic_similarity_score: Optional[Decimal] = Field(None, ge=0, le=1)
    
    @property
    def cost_per_engagement(self) -> Decimal:
        """Calculate cost per predicted engagement."""
        if self.predicted_engagement == 0:
            return Decimal("0")
        return self.estimated_total_cost / Decimal(str(self.predicted_engagement))
    
    @property
    def efficiency_ratio(self) -> Decimal:
        """Calculate efficiency ratio (engagement/cost)."""
        if self.estimated_total_cost == 0:
            return Decimal("0")
        return Decimal(str(self.predicted_engagement)) / self.estimated_total_cost


class ConstraintViolation(BaseModel):
    """Constraint violation details."""
    constraint_type: str
    constraint_value: Any
    actual_value: Any
    severity: str  # 'hard', 'soft', 'warning'
    description: str


class OptimizationConstraints(BaseModel):
    """Constraints for budget optimization algorithm."""
    
    # Budget constraints
    max_budget: Decimal = Field(..., gt=0)
    min_budget_utilization: Decimal = Field(default=Decimal("0.8"), ge=0, le=1)
    max_budget_utilization: Decimal = Field(default=Decimal("1.0"), ge=0, le=1)
    
    # KOL count constraints
    min_kols: int = Field(default=1, ge=1)
    max_kols: int = Field(default=50, ge=1)
    
    # Tier requirements (hard constraints)
    required_nano_count: int = Field(default=0, ge=0)
    required_micro_count: int = Field(default=0, ge=0)
    required_mid_count: int = Field(default=0, ge=0)
    required_macro_count: int = Field(default=0, ge=0)
    required_mega_count: int = Field(default=0, ge=0)
    
    # Performance requirements
    min_total_reach: Optional[int] = Field(None, ge=0)
    min_total_engagement: Optional[int] = Field(None, ge=0)
    min_avg_engagement_rate: Optional[Decimal] = Field(None, ge=0, le=1)
    
    # Risk constraints
    max_risk_per_kol: Decimal = Field(default=Decimal("0.8"), ge=0, le=1)
    max_portfolio_risk: Decimal = Field(default=Decimal("0.6"), ge=0, le=1)
    
    # Diversity constraints
    max_kols_per_category: Optional[int] = Field(None, ge=1)
    require_category_diversity: bool = Field(default=False)
    min_categories_covered: Optional[int] = Field(None, ge=1)
    
    @property
    def tier_requirements(self) -> Dict[str, int]:
        """Get tier requirements as dictionary."""
        return {
            "nano": self.required_nano_count,
            "micro": self.required_micro_count,
            "mid": self.required_mid_count,
            "macro": self.required_macro_count,
            "mega": self.required_mega_count
        }


class OptimizationResult(BaseModel):
    """Result of POC4 budget optimization."""
    
    # Selected KOLs
    selected_kols: List[KOLCandidate]
    
    # Cost breakdown
    total_cost: Decimal = Field(..., ge=0)
    cost_by_tier: Dict[str, Decimal] = Field(default_factory=dict)
    cost_by_category: Dict[str, Decimal] = Field(default_factory=dict)
    
    # Performance predictions
    predicted_total_reach: int = Field(..., ge=0)
    predicted_total_engagement: int = Field(..., ge=0)
    predicted_total_conversions: int = Field(..., ge=0)
    predicted_roi: Optional[Decimal] = Field(None)
    
    # Portfolio metrics
    portfolio_risk_score: Decimal = Field(..., ge=0, le=1)
    portfolio_diversity_score: Decimal = Field(..., ge=0, le=1)
    
    # Optimization quality
    optimization_score: Decimal = Field(..., ge=0, le=1)
    budget_utilization: Decimal = Field(..., ge=0, le=1)
    
    # Constraint validation
    constraints_satisfied: bool
    constraint_violations: List[ConstraintViolation] = Field(default_factory=list)
    
    # Alternative scenarios
    alternative_allocations: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Algorithm metadata
    algorithm_used: str
    optimization_time_seconds: float
    iterations_performed: int
    convergence_achieved: bool
    
    @property
    def cost_efficiency_score(self) -> Decimal:
        """Calculate overall cost efficiency."""
        if self.total_cost == 0:
            return Decimal("0")
        
        # Weight reach and engagement
        weighted_performance = (
            self.predicted_total_reach * Decimal("0.3") + 
            self.predicted_total_engagement * Decimal("0.7")
        )
        
        return weighted_performance / self.total_cost
    
    @property
    def tier_distribution(self) -> Dict[str, int]:
        """Get distribution of KOLs by tier."""
        distribution = {}
        for kol in self.selected_kols:
            tier = kol.tier.value
            distribution[tier] = distribution.get(tier, 0) + 1
        return distribution


class SemanticMatchingRequest(BaseModel):
    """Request for semantic KOL matching."""
    
    # Base requirements
    campaign_requirements: CampaignRequirements
    
    # Semantic matching parameters
    reference_content: Optional[str] = Field(None)
    reference_kol_ids: List[str] = Field(default_factory=list)
    similarity_threshold: Decimal = Field(default=Decimal("0.7"), ge=0, le=1)
    
    # Vector search parameters
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    max_candidates: int = Field(default=100, ge=1, le=1000)
    
    # Weighting for semantic vs other factors
    semantic_weight: Decimal = Field(default=Decimal("0.3"), ge=0, le=1)
    performance_weight: Decimal = Field(default=Decimal("0.4"), ge=0, le=1)
    cost_weight: Decimal = Field(default=Decimal("0.3"), ge=0, le=1)
    
    @root_validator
    def validate_weights(cls, values):
        """Ensure weights sum to 1.0."""
        semantic_w = values.get('semantic_weight', Decimal('0'))
        performance_w = values.get('performance_weight', Decimal('0'))
        cost_w = values.get('cost_weight', Decimal('0'))
        
        total = semantic_w + performance_w + cost_w
        if abs(total - Decimal('1.0')) > Decimal('0.001'):
            raise ValueError('Weights must sum to 1.0')
        
        return values


class CampaignPlanExport(BaseModel):
    """Data structure for CSV campaign plan export."""
    
    # Campaign info
    campaign_id: str
    campaign_name: str
    optimization_objective: str
    total_budget: Decimal
    
    # KOL details
    kol_selections: List[Dict[str, Any]]
    
    # Performance summary
    performance_summary: Dict[str, Any]
    
    # Export metadata
    export_timestamp: datetime = Field(default_factory=datetime.utcnow)
    export_format: str = Field(default="csv")
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            Decimal: lambda v: float(v)
        }