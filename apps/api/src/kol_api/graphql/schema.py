"""Main GraphQL schema with queries and mutations."""

import strawberry
from typing import List, Optional
from strawberry.types import Info

from kol_api.graphql.types import (
    KOL, Campaign, BudgetPlan, KOLMatchingResult, BudgetOptimizationResult,
    OperationResult, KOLFilterInput, CampaignCreateInput, BudgetPlanCreateInput,
    CampaignRequirementsInput, BudgetOptimizationConstraintsInput,
    BriefUploadInput, BriefParsingResult, DataRefreshResult, ExportResult
)
from kol_api.graphql.resolvers.kol_resolvers import KOLResolvers
from kol_api.graphql.resolvers.campaign_resolvers import CampaignResolvers
from kol_api.graphql.resolvers.budget_resolvers import BudgetResolvers
from kol_api.graphql.resolvers.scoring_resolvers import ScoringResolvers


@strawberry.type
class Query:
    """GraphQL Query root with all available queries."""
    
    # AIDEV-NOTE: KOL-related queries (POC1 & POC2)
    @strawberry.field
    async def kols(
        self,
        info: Info,
        filters: Optional[KOLFilterInput] = None,
        limit: int = 20,
        offset: int = 0,
        sort_by: Optional[str] = None,
        search: Optional[str] = None,
    ) -> List[KOL]:
        """
        Get KOLs with filtering, sorting, and pagination.
        POC1: Scale discovery and POC2: AI matching integration.
        """
        return await KOLResolvers.get_kols(
            info.context, filters, limit, offset, sort_by, search
        )
    
    @strawberry.field
    async def kol(self, info: Info, id: str) -> Optional[KOL]:
        """Get single KOL by ID with detailed information."""
        return await KOLResolvers.get_kol_by_id(info.context, id)
    
    @strawberry.field
    async def match_kols_for_campaign(
        self,
        info: Info,
        campaign_id: str,
        limit: int = 50,
        use_ai_scoring: bool = True,
        confidence_threshold: float = 0.7,
        enable_semantic_matching: bool = True,
    ) -> KOLMatchingResult:
        """
        AI-powered KOL matching for campaign (POC2).
        Uses sophisticated multi-factor scoring and semantic similarity.
        """
        return await KOLResolvers.match_kols_for_campaign(
            info.context, campaign_id, limit, use_ai_scoring, 
            confidence_threshold, enable_semantic_matching
        )
    
    @strawberry.field
    async def match_kols_with_requirements(
        self,
        info: Info,
        requirements: CampaignRequirementsInput,
        limit: int = 50,
        confidence_threshold: float = 0.7,
        enable_semantic_matching: bool = True,
    ) -> KOLMatchingResult:
        """
        Direct KOL matching using comprehensive requirements (POC2).
        Bypasses campaign lookup for ad-hoc matching scenarios.
        """
        return await KOLResolvers.match_kols_with_requirements(
            info.context, requirements, limit, confidence_threshold, enable_semantic_matching
        )
    
    @strawberry.field
    async def similar_kols(
        self,
        info: Info,
        kol_id: str,
        limit: int = 10,
        similarity_threshold: float = 0.7,
    ) -> List[KOL]:
        """Find similar KOLs using vector similarity search."""
        return await KOLResolvers.find_similar_kols(
            info.context, kol_id, limit, similarity_threshold
        )
    
    # AIDEV-NOTE: Campaign-related queries
    @strawberry.field
    async def campaigns(
        self,
        info: Info,
        status: Optional[str] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> List[Campaign]:
        """Get campaigns with optional status filtering."""
        return await CampaignResolvers.get_campaigns(
            info.context, status, limit, offset
        )
    
    @strawberry.field
    async def campaign(self, info: Info, id: str) -> Optional[Campaign]:
        """Get single campaign by ID."""
        return await CampaignResolvers.get_campaign_by_id(info.context, id)
    
    # AIDEV-NOTE: Budget optimization queries (POC4)
    @strawberry.field
    async def budget_plans(
        self,
        info: Info,
        campaign_id: Optional[str] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> List[BudgetPlan]:
        """Get budget plans, optionally filtered by campaign."""
        return await BudgetResolvers.get_budget_plans(
            info.context, campaign_id, limit, offset
        )
    
    @strawberry.field
    async def budget_plan(self, info: Info, id: str) -> Optional[BudgetPlan]:
        """Get single budget plan by ID."""
        return await BudgetResolvers.get_budget_plan_by_id(info.context, id)
    
    @strawberry.field
    async def optimize_budget(
        self,
        info: Info,
        campaign_id: str,
        optimization_objective: str,
        total_budget: float,
        constraints: Optional[BudgetOptimizationConstraintsInput] = None,
        generate_alternatives: bool = True,
        include_risk_analysis: bool = True,
    ) -> BudgetOptimizationResult:
        """
        Generate optimized budget allocation (POC4).
        Uses sophisticated algorithmic optimization with multi-constraint support.
        """
        return await BudgetResolvers.optimize_campaign_budget(
            info.context, campaign_id, optimization_objective, total_budget,
            constraints, generate_alternatives, include_risk_analysis
        )
    
    @strawberry.field
    async def budget_optimization_scenarios(
        self,
        info: Info,
        campaign_id: str,
        budget_ranges: List[float],
        objectives: List[str],
    ) -> List[BudgetOptimizationResult]:
        """
        Generate multiple budget optimization scenarios for comparison.
        Useful for budget planning and what-if analysis.
        """
        return await BudgetResolvers.generate_optimization_scenarios(
            info.context, campaign_id, budget_ranges, objectives
        )
    
    # AIDEV-NOTE: Analytics and reporting queries
    @strawberry.field
    async def kol_performance_analytics(
        self,
        info: Info,
        kol_ids: List[str],
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        include_predictions: bool = True,
        include_risk_factors: bool = True,
    ) -> dict:
        """Get enhanced performance analytics for specific KOLs with predictions."""
        return await ScoringResolvers.get_kol_performance_analytics(
            info.context, kol_ids, date_from, date_to, include_predictions, include_risk_factors
        )
    
    @strawberry.field
    async def campaign_performance_summary(
        self,
        info: Info,
        campaign_id: str,
        include_forecasts: bool = True,
        include_optimization_suggestions: bool = True,
    ) -> dict:
        """Get comprehensive campaign performance summary with optimization insights."""
        return await CampaignResolvers.get_campaign_performance_summary(
            info.context, campaign_id, include_forecasts, include_optimization_suggestions
        )
    
    @strawberry.field
    async def kol_market_analysis(
        self,
        info: Info,
        categories: List[str],
        budget_range: Optional[List[float]] = None,
        geographic_focus: Optional[List[str]] = None,
    ) -> dict:
        """
        Analyze KOL market conditions and availability.
        Useful for strategic campaign planning.
        """
        return await ScoringResolvers.get_market_analysis(
            info.context, categories, budget_range, geographic_focus
        )
    
    @strawberry.field
    async def data_quality_report(
        self,
        info: Info,
        platform: Optional[str] = None,
        category: Optional[str] = None,
    ) -> dict:
        """
        Get data quality report showing completeness and freshness metrics.
        Helps identify data refresh needs.
        """
        return await KOLResolvers.get_data_quality_report(
            info.context, platform, category
        )


@strawberry.type
class Mutation:
    """GraphQL Mutation root with all available mutations."""
    
    # AIDEV-NOTE: Campaign management mutations
    @strawberry.mutation
    async def create_campaign(
        self,
        info: Info,
        input: CampaignCreateInput,
    ) -> OperationResult:
        """Create a new campaign."""
        return await CampaignResolvers.create_campaign(info.context, input)
    
    @strawberry.mutation
    async def update_campaign_status(
        self,
        info: Info,
        campaign_id: str,
        status: str,
    ) -> OperationResult:
        """Update campaign status."""
        return await CampaignResolvers.update_campaign_status(
            info.context, campaign_id, status
        )
    
    # AIDEV-NOTE: KOL management mutations
    @strawberry.mutation
    async def invite_kol_to_campaign(
        self,
        info: Info,
        campaign_id: str,
        kol_id: str,
        proposed_rate: Optional[float] = None,
    ) -> OperationResult:
        """Invite KOL to participate in campaign."""
        return await CampaignResolvers.invite_kol_to_campaign(
            info.context, campaign_id, kol_id, proposed_rate
        )
    
    @strawberry.mutation
    async def update_kol_collaboration_status(
        self,
        info: Info,
        campaign_id: str,
        kol_id: str,
        status: str,
    ) -> OperationResult:
        """Update KOL collaboration status."""
        return await CampaignResolvers.update_kol_collaboration_status(
            info.context, campaign_id, kol_id, status
        )
    
    # AIDEV-NOTE: Budget planning mutations (POC4)
    @strawberry.mutation
    async def create_budget_plan(
        self,
        info: Info,
        input: BudgetPlanCreateInput,
        auto_optimize: bool = True,
        generate_alternatives: bool = True,
    ) -> OperationResult:
        """Create new budget plan with sophisticated optimization."""
        return await BudgetResolvers.create_budget_plan(
            info.context, input, auto_optimize, generate_alternatives
        )
    
    @strawberry.mutation
    async def create_optimized_budget_plan(
        self,
        info: Info,
        campaign_id: str,
        plan_name: str,
        optimization_result_id: str,
        scenario_index: int = 0,
    ) -> OperationResult:
        """
        Create budget plan from existing optimization result.
        Allows selection of specific scenarios from optimization runs.
        """
        return await BudgetResolvers.create_plan_from_optimization(
            info.context, campaign_id, plan_name, optimization_result_id, scenario_index
        )
    
    @strawberry.mutation
    async def approve_budget_plan(
        self,
        info: Info,
        plan_id: str,
    ) -> OperationResult:
        """Approve and activate budget plan."""
        return await BudgetResolvers.approve_budget_plan(info.context, plan_id)
    
    @strawberry.mutation
    async def execute_budget_plan(
        self,
        info: Info,
        plan_id: str,
    ) -> OperationResult:
        """Execute approved budget plan."""
        return await BudgetResolvers.execute_budget_plan(info.context, plan_id)
    
    # AIDEV-NOTE: KOL scoring mutations (POC2)
    @strawberry.mutation
    async def rescore_kol(
        self,
        info: Info,
        kol_id: str,
        campaign_id: Optional[str] = None,
        force_refresh: bool = False,
    ) -> OperationResult:
        """Trigger KOL rescoring with latest data."""
        return await ScoringResolvers.rescore_kol(
            info.context, kol_id, campaign_id, force_refresh
        )
    
    @strawberry.mutation
    async def bulk_rescore_kols(
        self,
        info: Info,
        kol_ids: List[str],
        campaign_id: Optional[str] = None,
    ) -> OperationResult:
        """Trigger bulk KOL rescoring for multiple KOLs."""
        return await ScoringResolvers.bulk_rescore_kols(
            info.context, kol_ids, campaign_id
        )
    
    # AIDEV-NOTE: Data management mutations (POC1 + enhancements)
    @strawberry.mutation
    async def trigger_kol_data_refresh(
        self,
        info: Info,
        kol_ids: Optional[List[str]] = None,
        platform: Optional[str] = None,
        priority: str = "normal",
        force_refresh: bool = False,
    ) -> DataRefreshResult:
        """Trigger enhanced data refresh for KOLs via Go scraper service."""
        return await KOLResolvers.trigger_data_refresh(
            info.context, kol_ids, platform, priority, force_refresh
        )
    
    @strawberry.mutation
    async def parse_campaign_brief(
        self,
        info: Info,
        brief: BriefUploadInput,
    ) -> BriefParsingResult:
        """
        Parse markdown campaign brief and extract requirements.
        Uses NLP to understand campaign goals and constraints.
        """
        return await CampaignResolvers.parse_campaign_brief(info.context, brief)
    
    @strawberry.mutation
    async def export_kol_data(
        self,
        info: Info,
        kol_ids: Optional[List[str]] = None,
        filters: Optional[KOLFilterInput] = None,
        format: str = "csv",
        include_scores: bool = True,
        include_predictions: bool = False,
    ) -> ExportResult:
        """Export KOL data in various formats with comprehensive information."""
        return await KOLResolvers.export_kol_data(
            info.context, kol_ids, filters, format, include_scores, include_predictions
        )
    
    @strawberry.mutation
    async def export_optimization_results(
        self,
        info: Info,
        optimization_result_id: str,
        format: str = "xlsx",
        include_alternatives: bool = True,
    ) -> ExportResult:
        """Export budget optimization results with detailed analysis."""
        return await BudgetResolvers.export_optimization_results(
            info.context, optimization_result_id, format, include_alternatives
        )
    
    @strawberry.mutation
    async def update_kol_brand_safety_status(
        self,
        info: Info,
        kol_id: str,
        is_brand_safe: bool,
        safety_notes: Optional[str] = None,
        risk_factors: Optional[List[str]] = None,
        reviewer_id: Optional[str] = None,
    ) -> OperationResult:
        """Update KOL brand safety status with detailed audit trail."""
        return await KOLResolvers.update_brand_safety_status(
            info.context, kol_id, is_brand_safe, safety_notes, risk_factors, reviewer_id
        )
    
    @strawberry.mutation
    async def bulk_update_brand_safety(
        self,
        info: Info,
        updates: List[str],  # JSON array of brand safety updates
        reviewer_id: str,
    ) -> OperationResult:
        """Bulk update brand safety status for multiple KOLs."""
        return await KOLResolvers.bulk_update_brand_safety(
            info.context, updates, reviewer_id
        )


# AIDEV-NOTE: Create the main GraphQL schema with enhanced capabilities
schema = strawberry.Schema(
    query=Query,
    mutation=Mutation,
    extensions=[
        # AIDEV-NOTE: Add query complexity analysis to prevent abuse of sophisticated operations
        # QueryComplexityExtension(max_complexity=2000),  # Increased for complex AI operations
        # AIDEV-NOTE: Add field-level authorization for sensitive operations
        # FieldAuthorizationExtension(),
        # AIDEV-NOTE: Add query depth limiting for nested operations
        # QueryDepthExtension(max_depth=15),
    ],
    types=[
        # AIDEV-NOTE: Explicitly include sophisticated types for proper introspection
        KOL, Campaign, BudgetPlan, KOLMatchingResult, BudgetOptimizationResult,
        BriefParsingResult, DataRefreshResult, ExportResult,
        CampaignRequirementsInput, BudgetOptimizationConstraintsInput, BriefUploadInput
    ]
)