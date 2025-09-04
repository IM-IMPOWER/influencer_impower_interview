"""Data conversion utilities between service objects and GraphQL types."""

from typing import List, Dict, Any, Optional
from decimal import Decimal
from datetime import datetime

from kol_api.graphql.types import (
    KOL as GraphQLKOL,
    KOLMetrics as GraphQLKOLMetrics,
    KOLScore as GraphQLKOLScore,
    BudgetPlan as GraphQLBudgetPlan,
    BudgetAllocation as GraphQLBudgetAllocation,
    BudgetOptimizationResult as GraphQLBudgetOptimizationResult,
    KOLMatchingResult as GraphQLKOLMatchingResult,
    PlatformTypeEnum,
    KOLTierEnum,
    ContentCategoryEnum,
    DecimalType,
    DateTimeType
)
from kol_api.database.models.kol import KOL, KOLMetrics
from kol_api.database.models.budget import BudgetPlan, BudgetAllocation
from kol_api.database.models.scoring import KOLScore
from kol_api.services.models import KOLCandidate, OptimizationResult
import structlog

logger = structlog.get_logger()


def convert_kol_to_graphql(
    kol: KOL,
    metrics: Optional[KOLMetrics] = None,
    score: Optional[KOLScore] = None
) -> GraphQLKOL:
    """
    Convert database KOL model to GraphQL KOL type.
    
    Args:
        kol: Database KOL model
        metrics: Optional KOL metrics
        score: Optional KOL score
        
    Returns:
        GraphQL KOL object
    """
    # AIDEV-NOTE: Convert basic KOL fields
    graphql_kol = GraphQLKOL(
        id=kol.id,
        username=kol.username,
        display_name=kol.display_name,
        platform=PlatformTypeEnum(kol.platform.value),
        platform_id=kol.platform_id,
        profile_url=kol.profile_url,
        avatar_url=kol.avatar_url,
        bio=kol.bio,
        location=kol.location,
        tier=KOLTierEnum(kol.tier.value),
        primary_category=ContentCategoryEnum(kol.primary_category.value),
        secondary_categories=kol.secondary_categories or [],
        is_verified=kol.is_verified,
        is_active=kol.is_active,
        is_brand_safe=kol.is_brand_safe,
        languages=kol.languages or [],
        created_at=DateTimeType(kol.created_at),
        last_scraped=DateTimeType(kol.last_scraped) if kol.last_scraped else None
    )
    
    # AIDEV-NOTE: Add metrics if provided
    if metrics:
        graphql_kol.metrics = convert_metrics_to_graphql(metrics)
    
    # AIDEV-NOTE: Add score if provided
    if score:
        graphql_kol.score = convert_score_to_graphql(score)
    
    return graphql_kol


def convert_metrics_to_graphql(metrics: KOLMetrics) -> GraphQLKOLMetrics:
    """
    Convert database KOLMetrics to GraphQL KOLMetrics.
    
    Args:
        metrics: Database KOLMetrics model
        
    Returns:
        GraphQL KOLMetrics object
    """
    return GraphQLKOLMetrics(
        follower_count=metrics.follower_count,
        following_count=metrics.following_count,
        total_posts=metrics.total_posts,
        avg_likes=DecimalType(metrics.avg_likes) if metrics.avg_likes else None,
        avg_comments=DecimalType(metrics.avg_comments) if metrics.avg_comments else None,
        avg_views=DecimalType(metrics.avg_views) if metrics.avg_views else None,
        engagement_rate=DecimalType(metrics.engagement_rate) if metrics.engagement_rate else None,
        audience_quality_score=DecimalType(metrics.audience_quality_score) if metrics.audience_quality_score else None,
        posts_last_30_days=metrics.posts_last_30_days,
        metrics_date=DateTimeType(metrics.metrics_date)
    )


def convert_score_to_graphql(score: KOLScore) -> GraphQLKOLScore:
    """
    Convert database KOLScore to GraphQL KOLScore.
    
    Args:
        score: Database KOLScore model
        
    Returns:
        GraphQL KOLScore object
    """
    return GraphQLKOLScore(
        overall_score=DecimalType(score.overall_score),
        engagement_rate_score=DecimalType(score.engagement_rate_score),
        follower_quality_score=DecimalType(score.follower_quality_score),
        content_relevance_score=DecimalType(score.content_relevance_score),
        brand_safety_score=DecimalType(score.brand_safety_score),
        posting_consistency_score=DecimalType(score.posting_consistency_score),
        audience_match_score=DecimalType(score.audience_match_score) if score.audience_match_score else None,
        cost_efficiency_score=DecimalType(score.cost_efficiency_score) if score.cost_efficiency_score else None,
        confidence_score=DecimalType(score.confidence_score) if score.confidence_score else None,
        scored_at=DateTimeType(score.created_at)
    )


def convert_kol_candidate_to_graphql(candidate: KOLCandidate) -> GraphQLKOL:
    """
    Convert KOL candidate service model to GraphQL KOL.
    
    Args:
        candidate: KOL candidate from service
        
    Returns:
        GraphQL KOL object with enriched data
    """
    # AIDEV-NOTE: Create basic GraphQL KOL structure
    graphql_kol = GraphQLKOL(
        id=candidate.kol_id,
        username=candidate.username,
        display_name=candidate.display_name,
        platform=PlatformTypeEnum(candidate.platform),
        platform_id=candidate.kol_id,  # Using kol_id as platform_id fallback
        profile_url=f"https://{candidate.platform}.com/{candidate.username}",
        tier=KOLTierEnum(candidate.tier.value),
        primary_category=ContentCategoryEnum(candidate.primary_category.value),
        secondary_categories=[],
        is_verified=True,  # Assume verified for candidates
        is_active=True,
        is_brand_safe=True,
        languages=["th", "en"],  # Default languages
        created_at=DateTimeType(datetime.utcnow()),
    )
    
    # AIDEV-NOTE: Add metrics from candidate data
    graphql_kol.metrics = GraphQLKOLMetrics(
        follower_count=candidate.metrics.follower_count,
        following_count=candidate.metrics.following_count,
        total_posts=candidate.metrics.posts_last_30_days * 12,  # Estimate
        avg_likes=DecimalType(candidate.metrics.avg_likes) if candidate.metrics.avg_likes else None,
        avg_comments=DecimalType(candidate.metrics.avg_comments) if candidate.metrics.avg_comments else None,
        avg_views=DecimalType(candidate.metrics.avg_views) if candidate.metrics.avg_views else None,
        engagement_rate=DecimalType(candidate.metrics.engagement_rate) if candidate.metrics.engagement_rate else None,
        audience_quality_score=DecimalType(candidate.metrics.audience_quality_score) if candidate.metrics.audience_quality_score else None,
        posts_last_30_days=candidate.metrics.posts_last_30_days,
        metrics_date=DateTimeType(datetime.utcnow())
    )
    
    # AIDEV-NOTE: Add score from candidate scoring components
    graphql_kol.score = GraphQLKOLScore(
        overall_score=DecimalType(candidate.overall_score),
        engagement_rate_score=DecimalType(candidate.score_components.roi_score),
        follower_quality_score=DecimalType(candidate.score_components.audience_quality_score),
        content_relevance_score=DecimalType(candidate.score_components.content_relevance_score),
        brand_safety_score=DecimalType(candidate.score_components.brand_safety_score),
        posting_consistency_score=DecimalType(candidate.score_components.reliability_score),
        audience_match_score=DecimalType(candidate.score_components.demographic_fit_score),
        confidence_score=DecimalType(candidate.score_components.overall_confidence),
        scored_at=DateTimeType(datetime.utcnow())
    )
    
    return graphql_kol


def convert_budget_plan_to_graphql(
    budget_plan: BudgetPlan,
    allocations: List[BudgetAllocation] = None
) -> GraphQLBudgetPlan:
    """
    Convert database BudgetPlan to GraphQL BudgetPlan.
    
    Args:
        budget_plan: Database BudgetPlan model
        allocations: Optional list of budget allocations
        
    Returns:
        GraphQL BudgetPlan object
    """
    graphql_allocations = []
    if allocations:
        graphql_allocations = [convert_budget_allocation_to_graphql(alloc) for alloc in allocations]
    
    return GraphQLBudgetPlan(
        id=budget_plan.id,
        name=budget_plan.name,
        description=budget_plan.description,
        status=budget_plan.status.value,
        optimization_objective=budget_plan.optimization_objective.value,
        allocation_strategy=budget_plan.allocation_strategy.value,
        total_budget=DecimalType(budget_plan.total_budget),
        available_budget=DecimalType(budget_plan.available_budget),
        predicted_reach=budget_plan.predicted_reach,
        predicted_engagement=budget_plan.predicted_engagement,
        predicted_roi=DecimalType(budget_plan.predicted_roi) if budget_plan.predicted_roi else None,
        optimization_score=DecimalType(budget_plan.optimization_score) if budget_plan.optimization_score else None,
        created_at=DateTimeType(budget_plan.created_at),
        allocations=graphql_allocations
    )


def convert_budget_allocation_to_graphql(allocation: BudgetAllocation) -> GraphQLBudgetAllocation:
    """
    Convert database BudgetAllocation to GraphQL BudgetAllocation.
    
    Args:
        allocation: Database BudgetAllocation model
        
    Returns:
        GraphQL BudgetAllocation object
    """
    return GraphQLBudgetAllocation(
        id=allocation.id,
        allocation_name=allocation.allocation_name,
        allocation_type=allocation.allocation_type,
        allocated_amount=DecimalType(allocation.allocated_amount),
        spent_amount=DecimalType(allocation.spent_amount),
        target_tier=allocation.target_tier,
        target_category=allocation.target_category,
        expected_reach=allocation.expected_reach,
        expected_engagement=allocation.expected_engagement,
        efficiency_score=DecimalType(allocation.efficiency_score) if allocation.efficiency_score else None,
        is_committed=allocation.is_committed
    )


def convert_optimization_result_to_graphql(
    result: OptimizationResult,
    campaign_id: str
) -> GraphQLBudgetOptimizationResult:
    """
    Convert service OptimizationResult to GraphQL BudgetOptimizationResult.
    
    Args:
        result: Service optimization result
        campaign_id: Campaign ID for context
        
    Returns:
        GraphQL BudgetOptimizationResult object
    """
    # AIDEV-NOTE: Convert selected KOLs to GraphQL format
    graphql_kols = [convert_kol_candidate_to_graphql(candidate) for candidate in result.selected_kols]
    
    # AIDEV-NOTE: Create a mock budget plan from optimization result
    mock_budget_plan = GraphQLBudgetPlan(
        id=f"opt_{campaign_id}_{int(datetime.utcnow().timestamp())}",
        name=f"Optimized Plan - {result.algorithm_used}",
        description=f"Generated optimization plan using {result.algorithm_used}",
        status="draft",
        optimization_objective="maximize_roi",
        allocation_strategy="performance_weighted",
        total_budget=DecimalType(result.total_cost),
        available_budget=DecimalType(result.total_cost),
        predicted_reach=result.predicted_total_reach,
        predicted_engagement=result.predicted_total_engagement,
        predicted_roi=DecimalType(result.predicted_roi) if result.predicted_roi else None,
        optimization_score=DecimalType(result.optimization_score),
        created_at=DateTimeType(datetime.utcnow()),
        allocations=[]  # Would be populated with actual allocations
    )
    
    # AIDEV-NOTE: Create optimization metadata
    optimization_metadata = {
        "algorithm_used": result.algorithm_used,
        "optimization_time": result.optimization_time_seconds,
        "iterations": result.iterations_performed,
        "convergence_achieved": result.convergence_achieved,
        "constraints_satisfied": result.constraints_satisfied,
        "budget_utilization": float(result.budget_utilization),
        "portfolio_risk_score": float(result.portfolio_risk_score),
        "diversity_score": float(result.portfolio_diversity_score),
        "tier_distribution": result.tier_distribution,
        "cost_by_tier": {k: float(v) for k, v in result.cost_by_tier.items()},
        "constraint_violations": [
            {
                "type": violation.constraint_type,
                "expected": violation.constraint_value,
                "actual": violation.actual_value,
                "severity": violation.severity,
                "description": violation.description
            }
            for violation in result.constraint_violations
        ]
    }
    
    return GraphQLBudgetOptimizationResult(
        optimized_plan=mock_budget_plan,
        optimization_metadata=optimization_metadata,
        alternative_plans=[]  # Could include alternative scenarios
    )


def convert_matching_result_to_graphql(
    matched_kols: List[KOLCandidate],
    metadata: Dict[str, Any]
) -> GraphQLKOLMatchingResult:
    """
    Convert KOL matching service result to GraphQL KOLMatchingResult.
    
    Args:
        matched_kols: List of matched KOL candidates
        metadata: Matching metadata
        
    Returns:
        GraphQL KOLMatchingResult object
    """
    # AIDEV-NOTE: Convert KOL candidates to GraphQL KOLs
    graphql_kols = [convert_kol_candidate_to_graphql(candidate) for candidate in matched_kols]
    
    # AIDEV-NOTE: Extract similarity scores if available
    similarity_scores = {}
    for candidate in matched_kols:
        if candidate.semantic_similarity_score:
            similarity_scores[candidate.kol_id] = float(candidate.semantic_similarity_score)
    
    return GraphQLKOLMatchingResult(
        matched_kols=graphql_kols,
        total_count=len(matched_kols),
        match_criteria=metadata,
        similarity_scores=similarity_scores
    )


def safe_decimal_convert(value: Any) -> Optional[DecimalType]:
    """
    Safely convert value to DecimalType for GraphQL.
    
    Args:
        value: Value to convert
        
    Returns:
        DecimalType or None if conversion fails
    """
    if value is None:
        return None
    
    try:
        if isinstance(value, Decimal):
            return DecimalType(value)
        elif isinstance(value, (int, float, str)):
            return DecimalType(Decimal(str(value)))
        else:
            return None
    except (ValueError, TypeError, Exception) as e:
        logger.warning("Failed to convert value to decimal", value=value, error=str(e))
        return None


def safe_datetime_convert(value: Any) -> Optional[DateTimeType]:
    """
    Safely convert value to DateTimeType for GraphQL.
    
    Args:
        value: Value to convert
        
    Returns:
        DateTimeType or None if conversion fails
    """
    if value is None:
        return None
    
    try:
        if isinstance(value, datetime):
            return DateTimeType(value)
        elif isinstance(value, str):
            return DateTimeType(datetime.fromisoformat(value))
        else:
            return None
    except (ValueError, TypeError, Exception) as e:
        logger.warning("Failed to convert value to datetime", value=value, error=str(e))
        return None


def extract_kol_summary_data(kol: KOL, metrics: Optional[KOLMetrics] = None) -> Dict[str, Any]:
    """
    Extract summary data from KOL for API responses.
    
    Args:
        kol: Database KOL model
        metrics: Optional KOL metrics
        
    Returns:
        Dictionary with KOL summary data
    """
    summary = {
        "id": kol.id,
        "username": kol.username,
        "display_name": kol.display_name,
        "platform": kol.platform.value,
        "tier": kol.tier.value,
        "category": kol.primary_category.value,
        "is_verified": kol.is_verified,
        "is_brand_safe": kol.is_brand_safe,
        "location": kol.location,
        "languages": kol.languages
    }
    
    if metrics:
        summary.update({
            "follower_count": metrics.follower_count,
            "engagement_rate": float(metrics.engagement_rate) if metrics.engagement_rate else None,
            "posts_last_30_days": metrics.posts_last_30_days,
            "audience_quality": float(metrics.audience_quality_score) if metrics.audience_quality_score else None
        })
    
    return summary