"""GraphQL types and scalars for KOL platform."""

import strawberry
from datetime import datetime, date
from decimal import Decimal
from typing import List, Optional, Dict, Any
from enum import Enum

from kol_api.database.models.kol import PlatformType, KOLTier, ContentCategory
from kol_api.database.models.campaign import CampaignStatus, CampaignObjective, CollaborationStatus
from kol_api.database.models.budget import BudgetStatus, OptimizationObjective, AllocationStrategy
from kol_api.database.models.scoring import ScoreType, ScoreSource
from kol_api.database.models.auth import UserRole

# AIDEV-NOTE: Custom scalar types
@strawberry.scalar(
    serializer=lambda v: float(v),
    parser=lambda v: Decimal(str(v))
)
class DecimalType:
    """Decimal scalar for precise numeric calculations."""
    pass

@strawberry.scalar(
    serializer=lambda v: v.isoformat(),
    parser=lambda v: datetime.fromisoformat(v)
)
class DateTimeType:
    """DateTime scalar with timezone support."""
    pass

@strawberry.scalar(
    serializer=lambda v: v.isoformat(),
    parser=lambda v: date.fromisoformat(v)
)
class DateType:
    """Date scalar."""
    pass

# AIDEV-NOTE: Enum types converted to GraphQL enums
@strawberry.enum
class PlatformTypeEnum(Enum):
    TIKTOK = "tiktok"
    INSTAGRAM = "instagram" 
    YOUTUBE = "youtube"
    FACEBOOK = "facebook"
    TWITTER = "twitter"
    LINKEDIN = "linkedin"

@strawberry.enum
class KOLTierEnum(Enum):
    NANO = "nano"
    MICRO = "micro"
    MID = "mid"
    MACRO = "macro"
    MEGA = "mega"

@strawberry.enum
class ContentCategoryEnum(Enum):
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

@strawberry.enum
class CampaignStatusEnum(Enum):
    DRAFT = "draft"
    PLANNING = "planning"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"

@strawberry.enum
class CampaignObjectiveEnum(Enum):
    BRAND_AWARENESS = "brand_awareness"
    ENGAGEMENT = "engagement"
    LEAD_GENERATION = "lead_generation"
    SALES = "sales"
    APP_INSTALLS = "app_installs"
    WEBSITE_TRAFFIC = "website_traffic"

@strawberry.enum
class CollaborationStatusEnum(Enum):
    INVITED = "invited"
    ACCEPTED = "accepted"
    DECLINED = "declined"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    PAID = "paid"

@strawberry.enum
class UserRoleEnum(Enum):
    ADMIN = "admin"
    MANAGER = "manager"
    ANALYST = "analyst"
    VIEWER = "viewer"

# AIDEV-NOTE: Core GraphQL types
@strawberry.type
class User:
    id: str
    email: str
    first_name: str
    last_name: str
    role: UserRoleEnum
    is_active: bool
    created_at: DateTimeType
    last_login: Optional[DateTimeType] = None
    avatar_url: Optional[str] = None
    bio: Optional[str] = None

# AIDEV-NOTE: Enhanced scoring components for POC2 sophisticated algorithms
@strawberry.type
class ScoreComponents:
    """Multi-factor KOL scoring components with confidence levels."""
    
    # Core scoring factors (POC2 enhanced algorithm)
    roi_score: DecimalType
    audience_quality_score: DecimalType
    brand_safety_score: DecimalType
    content_relevance_score: DecimalType
    demographic_fit_score: DecimalType
    reliability_score: DecimalType
    
    # Confidence scores for missing data handling
    roi_confidence: DecimalType
    audience_confidence: DecimalType
    brand_safety_confidence: DecimalType
    content_relevance_confidence: DecimalType
    demographic_confidence: DecimalType
    reliability_confidence: DecimalType
    
    # Overall metrics
    overall_confidence: DecimalType
    scoring_timestamp: DateTimeType
    data_freshness_days: int
    sample_size: Optional[int] = None

@strawberry.type
class SemanticMatchingData:
    """Semantic similarity and vector search results."""
    similarity_score: DecimalType
    content_match_score: DecimalType
    brand_affinity_score: DecimalType
    embedding_confidence: DecimalType
    matched_content_categories: List[str]
    semantic_keywords: List[str]

@strawberry.type
class PerformancePrediction:
    """KOL performance predictions based on historical data."""
    predicted_reach: int
    predicted_engagement: int
    predicted_conversions: int
    predicted_roi: DecimalType
    prediction_confidence: DecimalType
    risk_factors: List[str]
    historical_campaign_count: int

@strawberry.type
class KOLMetrics:
    """Enhanced KOL metrics with comprehensive data points."""
    
    # Basic metrics
    follower_count: int
    following_count: int
    total_posts: int
    
    # Engagement metrics
    avg_likes: Optional[DecimalType] = None
    avg_comments: Optional[DecimalType] = None
    avg_shares: Optional[DecimalType] = None
    avg_views: Optional[DecimalType] = None
    engagement_rate: Optional[DecimalType] = None
    
    # Quality metrics
    audience_quality_score: Optional[DecimalType] = None
    fake_follower_percentage: Optional[DecimalType] = None
    
    # Activity metrics
    posts_last_30_days: int
    avg_posting_frequency: Optional[DecimalType] = None
    
    # Growth metrics
    follower_growth_rate: Optional[DecimalType] = None
    engagement_trend: Optional[str] = None
    
    # Performance history
    campaign_success_rate: Optional[DecimalType] = None
    response_rate: Optional[DecimalType] = None
    
    # Cost data
    rate_per_post: Optional[DecimalType] = None
    rate_per_video: Optional[DecimalType] = None
    min_budget: Optional[DecimalType] = None
    
    # Metadata
    metrics_date: DateTimeType
    data_completeness_score: DecimalType

@strawberry.type
class KOLScore:
    """Enhanced KOL scoring with sophisticated multi-factor algorithm."""
    
    # Legacy scores (maintained for backward compatibility)
    overall_score: DecimalType
    engagement_rate_score: DecimalType
    follower_quality_score: DecimalType
    content_relevance_score: DecimalType
    brand_safety_score: DecimalType
    posting_consistency_score: DecimalType
    audience_match_score: Optional[DecimalType] = None
    cost_efficiency_score: Optional[DecimalType] = None
    confidence_score: Optional[DecimalType] = None
    scored_at: DateTimeType
    
    # Enhanced scoring components (POC2)
    score_components: Optional[ScoreComponents] = None
    semantic_matching: Optional[SemanticMatchingData] = None
    performance_prediction: Optional[PerformancePrediction] = None
    
    # Algorithm metadata
    scoring_algorithm: str
    algorithm_version: str
    weights_used: Dict[str, float] = strawberry.field(default_factory=dict)

@strawberry.type 
class KOL:
    """Enhanced KOL type with comprehensive data support."""
    
    # Basic information
    id: str
    username: str
    display_name: str
    platform: PlatformTypeEnum
    platform_id: str
    profile_url: str
    avatar_url: Optional[str] = None
    bio: Optional[str] = None
    location: Optional[str] = None
    
    # Classification
    tier: KOLTierEnum
    primary_category: ContentCategoryEnum
    secondary_categories: List[str]
    
    # Status flags
    is_verified: bool
    is_active: bool
    is_brand_safe: bool
    
    # Demographic data
    languages: List[str]
    target_demographics: Dict[str, Any] = strawberry.field(default_factory=dict)
    
    # Timestamps
    created_at: DateTimeType
    last_scraped: Optional[DateTimeType] = None
    last_scored: Optional[DateTimeType] = None
    
    # AIDEV-NOTE: Related data loaded separately to avoid N+1 queries
    metrics: Optional[KOLMetrics] = None
    score: Optional[KOLScore] = None
    
    # Data quality indicators
    data_completeness: DecimalType
    has_sufficient_data: bool
    missing_data_fields: List[str] = strawberry.field(default_factory=list)
    
    # Brand safety details
    brand_safety_notes: Optional[str] = None
    brand_safety_last_updated: Optional[DateTimeType] = None

@strawberry.type
class Campaign:
    id: str
    name: str
    description: Optional[str] = None
    client_name: str
    brand_name: str
    status: CampaignStatusEnum
    objective: CampaignObjectiveEnum
    start_date: DateType
    end_date: DateType
    total_budget: DecimalType
    allocated_budget: DecimalType
    spent_budget: DecimalType
    currency: str
    target_kol_tiers: List[str]
    target_categories: List[str]
    min_follower_count: Optional[int] = None
    max_follower_count: Optional[int] = None
    min_engagement_rate: Optional[DecimalType] = None
    created_at: DateTimeType
    
    # AIDEV-NOTE: Computed fields
    remaining_budget: DecimalType
    budget_utilization: DecimalType

@strawberry.type
class CampaignKOL:
    id: str
    campaign: Campaign
    kol: KOL
    status: CollaborationStatusEnum
    agreed_rate: Optional[DecimalType] = None
    total_cost: Optional[DecimalType] = None
    actual_reach: Optional[int] = None
    actual_engagement: Optional[int] = None
    content_deadline: Optional[DateTimeType] = None
    created_at: DateTimeType

@strawberry.type
class BudgetAllocation:
    """Enhanced budget allocation with POC4 optimization data."""
    
    id: str
    allocation_name: str
    allocation_type: str
    allocated_amount: DecimalType
    spent_amount: DecimalType
    
    # Targeting criteria
    target_tier: Optional[str] = None
    target_category: Optional[str] = None
    target_kol_ids: List[str] = strawberry.field(default_factory=list)
    
    # Performance predictions
    expected_reach: Optional[int] = None
    expected_engagement: Optional[int] = None
    expected_conversions: Optional[int] = None
    expected_roi: Optional[DecimalType] = None
    
    # Optimization metrics
    efficiency_score: Optional[DecimalType] = None
    risk_score: Optional[DecimalType] = None
    optimization_confidence: Optional[DecimalType] = None
    
    # Status and constraints
    is_committed: bool
    allocation_constraints: Dict[str, Any] = strawberry.field(default_factory=dict)
    
    # Alternative scenarios
    alternative_allocations: List[Dict[str, Any]] = strawberry.field(default_factory=list)

@strawberry.type
class BudgetPlan:
    """Enhanced budget plan with sophisticated optimization results."""
    
    # Basic information
    id: str
    name: str
    description: Optional[str] = None
    status: str  # BudgetStatus enum
    
    # Optimization configuration
    optimization_objective: str  # OptimizationObjective enum
    allocation_strategy: str  # AllocationStrategy enum
    optimization_algorithm: str
    algorithm_version: str
    
    # Budget breakdown
    total_budget: DecimalType
    available_budget: DecimalType
    reserved_budget: DecimalType
    
    # Performance predictions
    predicted_reach: Optional[int] = None
    predicted_engagement: Optional[int] = None
    predicted_conversions: Optional[int] = None
    predicted_roi: Optional[DecimalType] = None
    
    # Optimization results
    optimization_score: Optional[DecimalType] = None
    constraints_satisfaction_score: Optional[DecimalType] = None
    risk_assessment: Optional[Dict[str, Any]] = strawberry.field(default_factory=dict)
    
    # Objective weights and constraints
    objective_weights: Dict[str, float] = strawberry.field(default_factory=dict)
    tier_requirements: Dict[str, Any] = strawberry.field(default_factory=dict)
    
    # Metadata
    created_at: DateTimeType
    optimization_completed_at: Optional[DateTimeType] = None
    
    # Related data
    allocations: List[BudgetAllocation]
    
    # Alternative scenarios
    has_alternatives: bool
    alternative_count: int

# AIDEV-NOTE: Input types for mutations and queries
@strawberry.input
class KOLFilterInput:
    """Enhanced KOL filtering with comprehensive criteria."""
    
    # Basic filters
    platform: Optional[PlatformTypeEnum] = None
    tier: Optional[KOLTierEnum] = None
    category: Optional[ContentCategoryEnum] = None
    
    # Audience size filters
    min_followers: Optional[int] = None
    max_followers: Optional[int] = None
    
    # Performance filters
    min_engagement_rate: Optional[float] = None
    min_audience_quality: Optional[float] = None
    min_posting_frequency: Optional[int] = None
    
    # Geographic and language filters
    location: Optional[str] = None
    languages: Optional[List[str]] = None
    
    # Quality filters
    is_brand_safe: Optional[bool] = None
    is_verified: Optional[bool] = None
    max_fake_followers: Optional[float] = None
    
    # Performance history filters
    min_campaign_success_rate: Optional[float] = None
    min_response_rate: Optional[float] = None
    
    # Budget constraints
    max_rate_per_post: Optional[float] = None
    min_cost_efficiency: Optional[float] = None
    
    # Data quality filters
    require_complete_data: Optional[bool] = None
    min_data_freshness_days: Optional[int] = None

@strawberry.input
class CampaignRequirementsInput:
    """Comprehensive campaign requirements for POC2 matching."""
    
    # Target criteria
    target_kol_tiers: List[str]
    target_categories: List[str]
    total_budget: float
    
    # Follower constraints
    min_follower_count: Optional[int] = None
    max_follower_count: Optional[int] = None
    
    # Performance constraints
    min_engagement_rate: Optional[float] = None
    min_avg_views: Optional[int] = None
    
    # Demographic targeting
    target_demographics: Optional[str] = None  # JSON string
    target_locations: List[str] = strawberry.field(default_factory=list)
    target_languages: List[str] = strawberry.field(default_factory=list)
    target_age_ranges: List[str] = strawberry.field(default_factory=list)
    
    # Brand safety requirements
    require_brand_safe: bool = True
    require_verified: bool = False
    exclude_controversial: bool = True
    
    # Content requirements
    required_hashtags: List[str] = strawberry.field(default_factory=list)
    excluded_hashtags: List[str] = strawberry.field(default_factory=list)
    content_sentiment_requirements: Optional[str] = None
    
    # Campaign specific
    campaign_objective: str = "balanced"
    expected_conversion_rate: Optional[float] = None

@strawberry.input
class BudgetOptimizationConstraintsInput:
    """POC4 budget optimization constraints."""
    
    # Tier distribution requirements
    min_nano_percentage: Optional[float] = None
    min_micro_percentage: Optional[float] = None
    min_mid_percentage: Optional[float] = None
    min_macro_percentage: Optional[float] = None
    min_mega_percentage: Optional[float] = None
    
    # KOL quantity constraints
    min_kols_required: int = 1
    max_kols_allowed: Optional[int] = None
    
    # Budget allocation constraints
    max_single_kol_percentage: Optional[float] = None
    min_reserved_buffer: float = 0.0
    
    # Performance requirements
    min_total_reach: Optional[int] = None
    min_total_engagement: Optional[int] = None
    target_roi: Optional[float] = None
    
    # Risk management
    max_risk_score: Optional[float] = None
    diversification_requirement: bool = False
    
    # Category distribution
    category_distribution: Optional[str] = None  # JSON string
    
    # Objective weights
    reach_weight: float = 0.3
    engagement_weight: float = 0.3
    cost_efficiency_weight: float = 0.2
    risk_mitigation_weight: float = 0.2

@strawberry.input
class BriefUploadInput:
    """Input for uploading and parsing markdown briefs."""
    
    # File content
    brief_content: str
    filename: str
    
    # Parsing preferences
    auto_extract_requirements: bool = True
    confidence_threshold: float = 0.7
    
    # Override settings
    manual_overrides: Optional[str] = None  # JSON string of manual requirement overrides

@strawberry.input
class CampaignCreateInput:
    name: str
    description: Optional[str] = None
    client_name: str
    brand_name: str
    objective: CampaignObjectiveEnum
    start_date: DateType
    end_date: DateType
    total_budget: float
    currency: str = "THB"
    target_kol_tiers: List[str]
    target_categories: List[str]
    min_follower_count: Optional[int] = None
    max_follower_count: Optional[int] = None
    min_engagement_rate: Optional[float] = None

@strawberry.input
class BudgetPlanCreateInput:
    campaign_id: str
    name: str
    description: Optional[str] = None
    optimization_objective: str
    allocation_strategy: str
    total_budget: float
    reserved_buffer: Optional[float] = 0
    min_kols_required: int = 1
    max_kols_allowed: Optional[int] = None
    tier_requirements: Optional[str] = None  # JSON string
    objective_weights: Optional[str] = None  # JSON string

# AIDEV-NOTE: Response types for sophisticated operations
@strawberry.type
class KOLMatchingResult:
    """Enhanced KOL matching results with detailed metadata."""
    
    # Core results
    matched_kols: List[KOL]
    total_count: int
    
    # Matching criteria and scoring
    match_criteria: Dict[str, Any] = strawberry.field(default_factory=dict)
    similarity_scores: Dict[str, float] = strawberry.field(default_factory=dict)
    confidence_scores: Dict[str, float] = strawberry.field(default_factory=dict)
    
    # Algorithm metadata
    scoring_method: str
    algorithm_version: str
    weights_used: Dict[str, float] = strawberry.field(default_factory=dict)
    
    # Processing statistics
    total_candidates_evaluated: int
    candidates_passed_scoring: int
    processing_time_seconds: float
    
    # Data quality insights
    data_quality_summary: Dict[str, Any] = strawberry.field(default_factory=dict)
    missing_data_warnings: List[str] = strawberry.field(default_factory=list)
    
    # Semantic matching (if enabled)
    semantic_matching_enabled: bool
    embedding_quality_score: Optional[float] = None

@strawberry.type
class BudgetOptimizationResult:
    """Enhanced budget optimization with comprehensive analysis."""
    
    # Primary result
    optimized_plan: BudgetPlan
    
    # Alternative scenarios
    alternative_plans: List[BudgetPlan] = strawberry.field(default_factory=list)
    scenario_comparisons: Dict[str, Any] = strawberry.field(default_factory=dict)
    
    # Optimization analysis
    optimization_metadata: Dict[str, Any] = strawberry.field(default_factory=dict)
    constraint_satisfaction: Dict[str, float] = strawberry.field(default_factory=dict)
    efficiency_metrics: Dict[str, float] = strawberry.field(default_factory=dict)
    
    # Risk analysis
    risk_assessment: Dict[str, Any] = strawberry.field(default_factory=dict)
    sensitivity_analysis: Dict[str, float] = strawberry.field(default_factory=dict)
    
    # Performance projections
    performance_forecasts: Dict[str, Any] = strawberry.field(default_factory=dict)
    confidence_intervals: Dict[str, Any] = strawberry.field(default_factory=dict)
    
    # Processing metadata
    optimization_algorithm: str
    algorithm_version: str
    processing_time_seconds: float
    iterations_performed: int

@strawberry.type
class BriefParsingResult:
    """Result of markdown brief parsing and requirement extraction."""
    
    # Parsing success
    success: bool
    message: str
    
    # Extracted requirements
    campaign_requirements: Optional[str] = None  # JSON serialized CampaignRequirements
    
    # Parsing metadata
    parsing_confidence: float
    ambiguous_requirements: List[str] = strawberry.field(default_factory=list)
    missing_requirements: List[str] = strawberry.field(default_factory=list)
    
    # Raw analysis
    extracted_entities: Dict[str, Any] = strawberry.field(default_factory=dict)
    sentiment_analysis: Optional[Dict[str, float]] = None
    
    # Actionability
    is_actionable: bool
    required_manual_inputs: List[str] = strawberry.field(default_factory=list)

@strawberry.type
class DataRefreshResult:
    """Result of KOL data refresh operations."""
    
    success: bool
    message: str
    
    # Refresh statistics
    kols_refreshed: int
    platforms_updated: List[str] = strawberry.field(default_factory=list)
    
    # Data quality improvements
    data_completeness_improvements: Dict[str, float] = strawberry.field(default_factory=dict)
    new_metrics_collected: int
    
    # Errors and warnings
    refresh_errors: List[str] = strawberry.field(default_factory=list)
    rate_limit_warnings: List[str] = strawberry.field(default_factory=list)
    
    # Processing time
    processing_time_seconds: float
    next_refresh_recommended: DateTimeType

@strawberry.type
class OperationResult:
    """Enhanced operation result with detailed feedback."""
    
    success: bool
    message: str
    data: Optional[str] = None  # JSON serialized data
    
    # Enhanced feedback
    warnings: List[str] = strawberry.field(default_factory=list)
    processing_time_seconds: Optional[float] = None
    affected_records: Optional[int] = None
    
    # Operation metadata
    operation_id: Optional[str] = None
    timestamp: DateTimeType = strawberry.field(default_factory=lambda: datetime.utcnow())

@strawberry.type
class ExportResult:
    """Result of data export operations."""
    
    success: bool
    message: str
    
    # Export details
    export_url: Optional[str] = None
    file_format: str
    record_count: int
    file_size_bytes: Optional[int] = None
    
    # Expiration
    expires_at: Optional[DateTimeType] = None
    
    # Processing metadata
    export_id: str
    processing_time_seconds: float