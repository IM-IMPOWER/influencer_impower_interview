"""
GraphQL Integration and End-to-End Workflow Tests

AIDEV-NOTE: Comprehensive integration tests for GraphQL resolvers, end-to-end workflows,
and API contract validation for the KOL platform scoring and optimization systems.
"""
import pytest
import asyncio
import json
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import List, Dict, Any, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock, patch

import graphene
from graphql import build_schema, execute
from starlette.testclient import TestClient

from kol_api.graphql.schema import schema
from kol_api.graphql.resolvers.scoring_resolvers import ScoringResolver
from kol_api.graphql.resolvers.budget_resolvers import BudgetOptimizationResolver
from kol_api.graphql.resolvers.campaign_resolvers import CampaignResolver
from kol_api.graphql.resolvers.kol_resolvers import KOLResolver
from kol_api.graphql.types import (
    KOLScoreResult, OptimizationResult as GraphQLOptimizationResult,
    CampaignInput, OptimizationConstraintsInput
)
from kol_api.services.models import (
    OptimizationResult, KOLCandidate, OptimizationConstraints,
    OptimizationObjective, KOLTier, ContentCategory
)


# AIDEV-NOTE: GraphQL Test Utilities

class GraphQLTestClient:
    """Test client for GraphQL operations."""
    
    def __init__(self, schema):
        self.schema = schema
    
    async def execute(self, query: str, variables: Dict[str, Any] = None, context: Dict[str, Any] = None):
        """Execute GraphQL query."""
        result = await self.schema.execute_async(
            query,
            variable_values=variables or {},
            context_value=context or {}
        )
        
        return {
            "data": result.data,
            "errors": [str(error) for error in result.errors] if result.errors else None
        }


@pytest.fixture
def graphql_client():
    """GraphQL test client."""
    return GraphQLTestClient(schema)


@pytest.fixture
def mock_graphql_context():
    """Mock GraphQL context with database session and services."""
    context = MagicMock()
    
    # Mock database session
    context.db_session = AsyncMock()
    
    # Mock services
    context.kol_scorer = MagicMock()
    context.budget_optimizer = MagicMock()
    context.campaign_service = MagicMock()
    
    # Mock authentication
    context.current_user = MagicMock()
    context.current_user.id = "test_user_123"
    context.current_user.email = "test@example.com"
    
    return context


# AIDEV-NOTE: Scoring Resolver Integration Tests

@pytest.mark.integration
class TestScoringResolverIntegration:
    """Integration tests for scoring GraphQL resolvers."""
    
    @pytest.mark.asyncio
    async def test_score_kol_mutation(self, graphql_client, mock_graphql_context):
        """Test KOL scoring GraphQL mutation."""
        
        # Mock scoring service response
        mock_score_breakdown = MagicMock()
        mock_score_breakdown.roi_score = 0.85
        mock_score_breakdown.audience_quality_score = 0.90
        mock_score_breakdown.brand_safety_score = 0.95
        mock_score_breakdown.content_relevance_score = 0.80
        mock_score_breakdown.demographic_fit_score = 0.85
        mock_score_breakdown.reliability_score = 0.90
        mock_score_breakdown.composite_score = 0.87
        mock_score_breakdown.overall_confidence = 0.88
        mock_score_breakdown.missing_data_penalty = 0.05
        
        mock_graphql_context.kol_scorer.score_kol.return_value = mock_score_breakdown
        
        query = \"\"\"\n        mutation ScoreKOL($kolId: String!, $campaignId: String!) {
            scoreKOL(kolId: $kolId, campaignId: $campaignId) {
                kolId
                campaignId
                overallScore
                confidence
                scoreComponents {
                    roiScore
                    audienceQualityScore
                    brandSafetyScore
                    contentRelevanceScore
                    demographicFitScore
                    reliabilityScore
                }
                missingDataPenalty
                calculatedAt
            }
        }
        \"\"\"
        
        variables = {
            "kolId": "test_kol_123",
            "campaignId": "test_campaign_456"
        }
        
        result = await graphql_client.execute(
            query, 
            variables=variables, 
            context=mock_graphql_context
        )
        
        # Verify no errors
        assert result["errors"] is None, f"GraphQL errors: {result['errors']}"
        
        # Verify response structure and data
        score_result = result["data"]["scoreKOL"]
        assert score_result["kolId"] == "test_kol_123"
        assert score_result["campaignId"] == "test_campaign_456"
        assert score_result["overallScore"] == 0.87
        assert score_result["confidence"] == 0.88
        
        # Verify score components
        components = score_result["scoreComponents"]
        assert components["roiScore"] == 0.85
        assert components["audienceQualityScore"] == 0.90
        assert components["brandSafetyScore"] == 0.95
        
        # Verify service was called correctly
        mock_graphql_context.kol_scorer.score_kol.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_bulk_score_kols_mutation(self, graphql_client, mock_graphql_context):
        """Test bulk KOL scoring GraphQL mutation."""
        
        # Mock multiple scoring responses
        mock_scores = []
        for i in range(3):
            mock_score = MagicMock()
            mock_score.roi_score = 0.8 + (i * 0.02)
            mock_score.composite_score = 0.85 + (i * 0.01)
            mock_score.overall_confidence = 0.88
            mock_scores.append(mock_score)
        
        mock_graphql_context.kol_scorer.score_multiple_kols.return_value = mock_scores
        
        query = \"\"\"\n        mutation BulkScoreKOLs($kolIds: [String!]!, $campaignId: String!) {
            bulkScoreKOLs(kolIds: $kolIds, campaignId: $campaignId) {
                results {
                    kolId
                    overallScore
                    confidence
                }
                totalProcessed
                processingTimeSeconds
            }
        }
        \"\"\"
        
        variables = {
            "kolIds": ["kol_1", "kol_2", "kol_3"],
            "campaignId": "campaign_123"
        }
        
        result = await graphql_client.execute(
            query,
            variables=variables,
            context=mock_graphql_context
        )
        
        # Verify response
        assert result["errors"] is None
        bulk_result = result["data"]["bulkScoreKOLs"]
        assert bulk_result["totalProcessed"] == 3
        assert len(bulk_result["results"]) == 3
        assert bulk_result["processingTimeSeconds"] > 0
    
    @pytest.mark.asyncio
    async def test_score_kol_with_invalid_ids(self, graphql_client, mock_graphql_context):
        """Test scoring with invalid KOL or campaign IDs."""
        
        # Mock service to raise exception for invalid IDs
        mock_graphql_context.kol_scorer.score_kol.side_effect = ValueError("KOL not found")
        
        query = \"\"\"\n        mutation ScoreKOL($kolId: String!, $campaignId: String!) {
            scoreKOL(kolId: $kolId, campaignId: $campaignId) {
                kolId
                overallScore
            }
        }
        \"\"\"
        
        variables = {
            "kolId": "invalid_kol",
            "campaignId": "invalid_campaign"
        }
        
        result = await graphql_client.execute(
            query,
            variables=variables,
            context=mock_graphql_context
        )
        
        # Should have errors
        assert result["errors"] is not None
        assert "KOL not found" in result["errors"][0]
    
    @pytest.mark.asyncio
    async def test_get_kol_score_history_query(self, graphql_client, mock_graphql_context):
        """Test querying KOL score history."""
        
        # Mock historical score data
        mock_history = [
            {
                "scored_at": datetime.now(timezone.utc) - timedelta(days=1),
                "campaign_id": "campaign_1",
                "overall_score": 0.85,
                "confidence": 0.88
            },
            {
                "scored_at": datetime.now(timezone.utc) - timedelta(days=7),
                "campaign_id": "campaign_2",
                "overall_score": 0.82,
                "confidence": 0.86
            }
        ]
        
        mock_graphql_context.kol_scorer.get_score_history.return_value = mock_history
        
        query = \"\"\"\n        query GetKOLScoreHistory($kolId: String!, $limit: Int) {
            kolScoreHistory(kolId: $kolId, limit: $limit) {
                kolId
                scores {
                    campaignId
                    overallScore
                    confidence
                    scoredAt
                }
                totalCount
            }
        }
        \"\"\"
        
        variables = {
            "kolId": "test_kol_123",
            "limit": 10
        }
        
        result = await graphql_client.execute(
            query,
            variables=variables,
            context=mock_graphql_context
        )
        
        # Verify response
        assert result["errors"] is None
        history_result = result["data"]["kolScoreHistory"]
        assert history_result["kolId"] == "test_kol_123"
        assert len(history_result["scores"]) == 2
        assert history_result["totalCount"] == 2


# AIDEV-NOTE: Budget Optimization Resolver Integration Tests

@pytest.mark.integration
class TestBudgetOptimizationResolverIntegration:
    """Integration tests for budget optimization GraphQL resolvers."""
    
    @pytest.mark.asyncio
    async def test_optimize_campaign_budget_mutation(self, graphql_client, mock_graphql_context):
        \"\"\"Test campaign budget optimization GraphQL mutation.\"\"\"
        
        # Mock optimization result
        mock_kol_candidates = [
            MagicMock(
                kol_id=\"kol_1\",
                username=\"@influencer1\",
                overall_score=Decimal(\"0.87\"),
                estimated_total_cost=Decimal(\"2000.00\"),
                predicted_reach=7500
            ),
            MagicMock(
                kol_id=\"kol_2\",
                username=\"@influencer2\",
                overall_score=Decimal(\"0.84\"),
                estimated_total_cost=Decimal(\"1500.00\"),
                predicted_reach=5000
            )
        ]
        
        mock_optimization_result = OptimizationResult(
            selected_kols=mock_kol_candidates,
            total_cost=Decimal(\"3500.00\"),
            cost_by_tier={\"micro\": Decimal(\"3500.00\")},
            cost_by_category={\"lifestyle\": Decimal(\"3500.00\")},
            predicted_total_reach=12500,
            predicted_total_engagement=400,
            predicted_total_conversions=8,
            predicted_roi=Decimal(\"2.28\"),
            portfolio_risk_score=Decimal(\"0.25\"),
            portfolio_diversity_score=Decimal(\"0.85\"),
            optimization_score=Decimal(\"0.855\"),
            budget_utilization=Decimal(\"0.70\"),
            constraints_satisfied=True,
            constraint_violations=[],
            alternative_allocations=[],
            algorithm_used=\"constraint_satisfaction\",
            optimization_time_seconds=2.5,
            iterations_performed=150,
            convergence_achieved=True,
            tier_distribution={\"micro\": 2}
        )
        
        mock_graphql_context.budget_optimizer.optimize_campaign_budget_advanced.return_value = mock_optimization_result
        
        query = \"\"\"
        mutation OptimizeCampaignBudget($campaignId: String!, $constraints: OptimizationConstraintsInput!) {
            optimizeCampaignBudget(campaignId: $campaignId, constraints: $constraints) {
                campaignId
                selectedKOLs {
                    kolId
                    username
                    overallScore
                    estimatedCost
                    predictedReach
                }
                totalCost
                budgetUtilization
                predictedTotalReach
                optimizationScore
                constraintsSatisfied
                algorithmUsed
                processingTimeSeconds
            }
        }
        \"\"\"
        
        variables = {
            \"campaignId\": \"campaign_123\",
            \"constraints\": {
                \"maxBudget\": 5000.00,
                \"minKols\": 2,
                \"maxKols\": 5,
                \"maxRiskPerKol\": 0.6,
                \"objective\": \"MAXIMIZE_REACH\"
            }
        }
        
        result = await graphql_client.execute(
            query,
            variables=variables,
            context=mock_graphql_context
        )
        
        # Verify response
        assert result[\"errors\"] is None, f\"GraphQL errors: {result['errors']}\"
        
        optimization = result[\"data\"][\"optimizeCampaignBudget\"]
        assert optimization[\"campaignId\"] == \"campaign_123\"
        assert len(optimization[\"selectedKOLs\"]) == 2
        assert optimization[\"totalCost\"] == 3500.00
        assert optimization[\"budgetUtilization\"] == 0.70
        assert optimization[\"constraintsSatisfied\"] == True
        assert optimization[\"algorithmUsed\"] == \"constraint_satisfaction\"
        
        # Verify selected KOLs
        selected_kols = optimization[\"selectedKOLs\"]
        assert selected_kols[0][\"kolId\"] == \"kol_1\"
        assert selected_kols[0][\"overallScore\"] == 0.87
        assert selected_kols[1][\"kolId\"] == \"kol_2\"
    
    @pytest.mark.asyncio
    async def test_optimization_with_constraint_violations(self, graphql_client, mock_graphql_context):
        \"\"\"Test optimization that results in constraint violations.\"\"\"
        
        from kol_api.services.models import ConstraintViolation
        
        # Mock optimization result with violations
        mock_violations = [
            ConstraintViolation(
                constraint_type=\"budget\",
                constraint_value=5000.0,
                actual_value=5500.0,
                severity=\"soft\",
                description=\"Budget slightly exceeded due to optimization requirements\"
            ),
            ConstraintViolation(
                constraint_type=\"min_reach\",
                constraint_value=10000,
                actual_value=8500,
                severity=\"soft\",
                description=\"Total reach below minimum requirement\"
            )
        ]
        
        mock_optimization_result = OptimizationResult(
            selected_kols=[],
            total_cost=Decimal(\"5500.00\"),
            cost_by_tier={},
            cost_by_category={},
            predicted_total_reach=8500,
            predicted_total_engagement=300,
            predicted_total_conversions=6,
            predicted_roi=Decimal(\"1.09\"),
            portfolio_risk_score=Decimal(\"0.30\"),
            portfolio_diversity_score=Decimal(\"0.60\"),
            optimization_score=Decimal(\"0.65\"),
            budget_utilization=Decimal(\"1.10\"),
            constraints_satisfied=False,
            constraint_violations=mock_violations,
            alternative_allocations=[],
            algorithm_used=\"genetic_algorithm\",
            optimization_time_seconds=5.2,
            iterations_performed=200,
            convergence_achieved=False,
            tier_distribution={}
        )
        
        mock_graphql_context.budget_optimizer.optimize_campaign_budget_advanced.return_value = mock_optimization_result
        
        query = \"\"\"
        mutation OptimizeCampaignBudget($campaignId: String!, $constraints: OptimizationConstraintsInput!) {
            optimizeCampaignBudget(campaignId: $campaignId, constraints: $constraints) {
                constraintsSatisfied
                constraintViolations {
                    constraintType
                    severity
                    description
                    constraintValue
                    actualValue
                }
                budgetUtilization
                optimizationScore
            }
        }
        \"\"\"
        
        variables = {
            \"campaignId\": \"constrained_campaign\",
            \"constraints\": {
                \"maxBudget\": 5000.00,
                \"minKols\": 5,
                \"maxKols\": 10,
                \"minTotalReach\": 10000
            }
        }
        
        result = await graphql_client.execute(
            query,
            variables=variables,
            context=mock_graphql_context
        )
        
        # Verify response includes violation information
        assert result[\"errors\"] is None
        
        optimization = result[\"data\"][\"optimizeCampaignBudget\"]
        assert optimization[\"constraintsSatisfied\"] == False
        assert len(optimization[\"constraintViolations\"]) == 2
        
        violations = optimization[\"constraintViolations\"]
        budget_violation = next(v for v in violations if v[\"constraintType\"] == \"budget\")
        assert budget_violation[\"severity\"] == \"soft\"
        assert budget_violation[\"constraintValue\"] == 5000.0
        assert budget_violation[\"actualValue\"] == 5500.0
    
    @pytest.mark.asyncio
    async def test_generate_alternative_scenarios(self, graphql_client, mock_graphql_context):
        \"\"\"Test generation of alternative budget allocation scenarios.\"\"\"
        
        # Mock alternative scenarios
        mock_alternatives = [
            {
                \"scenario_name\": \"Conservative Budget\",
                \"max_budget\": Decimal(\"3000.00\"),
                \"selected_kols\": 2,
                \"total_reach\": 8000,
                \"risk_level\": \"low\"
            },
            {
                \"scenario_name\": \"Aggressive Budget\",
                \"max_budget\": Decimal(\"7000.00\"),
                \"selected_kols\": 4,
                \"total_reach\": 20000,
                \"risk_level\": \"high\"
            }
        ]
        
        mock_graphql_context.budget_optimizer.generate_alternative_scenarios.return_value = mock_alternatives
        
        query = \"\"\"
        mutation GenerateAlternativeScenarios($campaignId: String!, $baseConstraints: OptimizationConstraintsInput!) {
            generateAlternativeScenarios(campaignId: $campaignId, baseConstraints: $baseConstraints) {
                scenarios {
                    scenarioName
                    maxBudget
                    selectedKols
                    totalReach
                    riskLevel
                }
                generationTimeSeconds
            }
        }
        \"\"\"
        
        variables = {
            \"campaignId\": \"campaign_123\",
            \"baseConstraints\": {
                \"maxBudget\": 5000.00,
                \"minKols\": 2,
                \"maxKols\": 5
            }
        }
        
        result = await graphql_client.execute(
            query,
            variables=variables,
            context=mock_graphql_context
        )
        
        # Verify alternative scenarios
        assert result[\"errors\"] is None
        
        scenarios = result[\"data\"][\"generateAlternativeScenarios\"]
        assert len(scenarios[\"scenarios\"]) == 2
        assert scenarios[\"scenarios\"][0][\"scenarioName\"] == \"Conservative Budget\"
        assert scenarios[\"scenarios\"][1][\"scenarioName\"] == \"Aggressive Budget\"


# AIDEV-NOTE: Campaign and KOL Resolver Integration Tests

@pytest.mark.integration
class TestCampaignKOLResolverIntegration:
    \"\"\"Integration tests for campaign and KOL GraphQL resolvers.\"\"\"
    
    @pytest.mark.asyncio
    async def test_create_campaign_with_requirements(self, graphql_client, mock_graphql_context):
        \"\"\"Test creating campaign with specific requirements.\"\"\"
        
        # Mock campaign creation
        mock_campaign = MagicMock()
        mock_campaign.id = \"new_campaign_123\"
        mock_campaign.name = \"Test Campaign\"
        mock_campaign.description = \"Test campaign description\"
        mock_campaign.total_budget = Decimal(\"25000.00\")
        mock_campaign.status = \"ACTIVE\"
        
        mock_graphql_context.campaign_service.create_campaign.return_value = mock_campaign
        
        query = \"\"\"
        mutation CreateCampaign($input: CampaignInput!) {
            createCampaign(input: $input) {
                id
                name
                description
                totalBudget
                status
                requirements {
                    targetCategories
                    minFollowerCount
                    maxFollowerCount
                    minEngagementRate
                    targetLocations
                }
                createdAt
            }
        }
        \"\"\"
        
        variables = {
            \"input\": {
                \"name\": \"Test Campaign\",
                \"description\": \"Test campaign description\",
                \"totalBudget\": 25000.00,
                \"requirements\": {
                    \"targetCategories\": [\"LIFESTYLE\", \"FASHION\"],
                    \"minFollowerCount\": 10000,
                    \"maxFollowerCount\": 500000,
                    \"minEngagementRate\": 0.02,
                    \"targetLocations\": [\"Bangkok\", \"Singapore\"]
                }
            }
        }
        
        result = await graphql_client.execute(
            query,
            variables=variables,
            context=mock_graphql_context
        )
        
        # Verify campaign creation
        assert result[\"errors\"] is None
        
        campaign = result[\"data\"][\"createCampaign\"]
        assert campaign[\"id\"] == \"new_campaign_123\"
        assert campaign[\"name\"] == \"Test Campaign\"
        assert campaign[\"totalBudget\"] == 25000.00
        
        # Verify service was called
        mock_graphql_context.campaign_service.create_campaign.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_search_kols_with_filters(self, graphql_client, mock_graphql_context):
        \"\"\"Test KOL search with various filters.\"\"\"
        
        # Mock KOL search results
        mock_kols = [
            MagicMock(
                id=\"kol_1\",
                username=\"@lifestyle_influencer\",
                display_name=\"Lifestyle Influencer\",
                follower_count=75000,
                engagement_rate=Decimal(\"0.045\"),
                tier=\"MICRO\",
                categories=[\"LIFESTYLE\", \"FASHION\"]
            ),
            MagicMock(
                id=\"kol_2\",
                username=\"@beauty_guru\",
                display_name=\"Beauty Guru\",
                follower_count=120000,
                engagement_rate=Decimal(\"0.038\"),
                tier=\"MID\",
                categories=[\"BEAUTY\", \"LIFESTYLE\"]
            )
        ]
        
        mock_search_result = {
            \"kols\": mock_kols,
            \"total_count\": 2,
            \"has_next_page\": False
        }
        
        mock_graphql_context.kol_service.search_kols.return_value = mock_search_result
        
        query = \"\"\"
        query SearchKOLs($filters: KOLFiltersInput!, $pagination: PaginationInput) {
            searchKOLs(filters: $filters, pagination: $pagination) {
                kols {
                    id
                    username
                    displayName
                    followerCount
                    engagementRate
                    tier
                    categories
                }
                totalCount
                hasNextPage
            }
        }
        \"\"\"
        
        variables = {
            \"filters\": {
                \"categories\": [\"LIFESTYLE\", \"FASHION\"],
                \"minFollowerCount\": 50000,
                \"maxFollowerCount\": 200000,
                \"minEngagementRate\": 0.03,
                \"locations\": [\"Bangkok\"],
                \"verified\": True
            },
            \"pagination\": {
                \"limit\": 10,
                \"offset\": 0
            }
        }
        
        result = await graphql_client.execute(
            query,
            variables=variables,
            context=mock_graphql_context
        )
        
        # Verify search results
        assert result[\"errors\"] is None
        
        search_result = result[\"data\"][\"searchKOLs\"]
        assert len(search_result[\"kols\"]) == 2
        assert search_result[\"totalCount\"] == 2
        assert search_result[\"hasNextPage\"] == False
        
        # Verify KOL data
        kol1 = search_result[\"kols\"][0]
        assert kol1[\"id\"] == \"kol_1\"
        assert kol1[\"username\"] == \"@lifestyle_influencer\"
        assert kol1[\"followerCount\"] == 75000
        assert kol1[\"tier\"] == \"MICRO\"


# AIDEV-NOTE: End-to-End Workflow Integration Tests

@pytest.mark.integration
class TestEndToEndWorkflowIntegration:
    \"\"\"End-to-end workflow integration tests.\"\"\"
    
    @pytest.mark.asyncio
    async def test_complete_campaign_optimization_workflow(self, graphql_client, mock_graphql_context):
        \"\"\"Test complete workflow from campaign creation to optimization.\"\"\"
        
        # Step 1: Create campaign
        mock_campaign = MagicMock()
        mock_campaign.id = \"workflow_campaign_123\"
        mock_campaign.name = \"Workflow Test Campaign\"
        mock_campaign.total_budget = Decimal(\"30000.00\")
        mock_campaign.status = \"ACTIVE\"
        
        mock_graphql_context.campaign_service.create_campaign.return_value = mock_campaign
        
        create_campaign_query = \"\"\"
        mutation CreateCampaign($input: CampaignInput!) {
            createCampaign(input: $input) {
                id
                name
                totalBudget
                status
            }
        }
        \"\"\"
        
        create_variables = {
            \"input\": {
                \"name\": \"Workflow Test Campaign\",
                \"description\": \"End-to-end workflow test\",
                \"totalBudget\": 30000.00,
                \"requirements\": {
                    \"targetCategories\": [\"LIFESTYLE\"],
                    \"minFollowerCount\": 10000,
                    \"maxFollowerCount\": 500000
                }
            }
        }
        
        create_result = await graphql_client.execute(
            create_campaign_query,
            variables=create_variables,
            context=mock_graphql_context
        )
        
        assert create_result[\"errors\"] is None
        campaign_id = create_result[\"data\"][\"createCampaign\"][\"id\"]
        
        # Step 2: Search for suitable KOLs
        mock_kols = [
            MagicMock(id=f\"kol_{i}\", username=f\"@user_{i}\", follower_count=50000 + i*10000)
            for i in range(10)
        ]
        
        mock_graphql_context.kol_service.search_kols.return_value = {
            \"kols\": mock_kols,
            \"total_count\": 10,
            \"has_next_page\": False
        }
        
        search_query = \"\"\"
        query SearchKOLs($filters: KOLFiltersInput!) {
            searchKOLs(filters: $filters) {
                kols {
                    id
                    username
                    followerCount
                }
                totalCount
            }
        }
        \"\"\"
        
        search_variables = {
            \"filters\": {
                \"categories\": [\"LIFESTYLE\"],
                \"minFollowerCount\": 10000,
                \"maxFollowerCount\": 500000
            }
        }
        
        search_result = await graphql_client.execute(
            search_query,
            variables=search_variables,
            context=mock_graphql_context
        )
        
        assert search_result[\"errors\"] is None
        found_kols = search_result[\"data\"][\"searchKOLs\"][\"kols\"]
        assert len(found_kols) == 10
        
        # Step 3: Score selected KOLs
        kol_ids = [kol[\"id\"] for kol in found_kols[:5]]  # Score top 5
        
        mock_scores = [
            MagicMock(composite_score=0.85 - i*0.02, overall_confidence=0.88)
            for i in range(5)
        ]
        
        mock_graphql_context.kol_scorer.score_multiple_kols.return_value = mock_scores
        
        scoring_query = \"\"\"
        mutation BulkScoreKOLs($kolIds: [String!]!, $campaignId: String!) {
            bulkScoreKOLs(kolIds: $kolIds, campaignId: $campaignId) {
                results {
                    kolId
                    overallScore
                    confidence
                }
                totalProcessed
            }
        }
        \"\"\"
        
        scoring_variables = {
            \"kolIds\": kol_ids,
            \"campaignId\": campaign_id
        }
        
        scoring_result = await graphql_client.execute(
            scoring_query,
            variables=scoring_variables,
            context=mock_graphql_context
        )
        
        assert scoring_result[\"errors\"] is None
        assert scoring_result[\"data\"][\"bulkScoreKOLs\"][\"totalProcessed\"] == 5
        
        # Step 4: Optimize budget allocation
        mock_optimization_result = OptimizationResult(
            selected_kols=[
                MagicMock(
                    kol_id=kol_ids[0],
                    overall_score=Decimal(\"0.85\"),
                    estimated_total_cost=Decimal(\"8000.00\"),
                    predicted_reach=15000
                ),
                MagicMock(
                    kol_id=kol_ids[1],
                    overall_score=Decimal(\"0.83\"),
                    estimated_total_cost=Decimal(\"7000.00\"),
                    predicted_reach=12000
                )
            ],
            total_cost=Decimal(\"15000.00\"),
            cost_by_tier={\"micro\": Decimal(\"15000.00\")},
            cost_by_category={\"lifestyle\": Decimal(\"15000.00\")},
            predicted_total_reach=27000,
            predicted_total_engagement=810,
            predicted_total_conversions=16,
            predicted_roi=Decimal(\"1.8\"),
            portfolio_risk_score=Decimal(\"0.25\"),
            portfolio_diversity_score=Decimal(\"0.80\"),
            optimization_score=Decimal(\"0.84\"),
            budget_utilization=Decimal(\"0.50\"),
            constraints_satisfied=True,
            constraint_violations=[],
            alternative_allocations=[],
            algorithm_used=\"constraint_satisfaction\",
            optimization_time_seconds=3.2,
            iterations_performed=120,
            convergence_achieved=True,
            tier_distribution={\"micro\": 2}
        )
        
        mock_graphql_context.budget_optimizer.optimize_campaign_budget_advanced.return_value = mock_optimization_result
        
        optimization_query = \"\"\"
        mutation OptimizeCampaignBudget($campaignId: String!, $constraints: OptimizationConstraintsInput!) {
            optimizeCampaignBudget(campaignId: $campaignId, constraints: $constraints) {
                campaignId
                selectedKOLs {
                    kolId
                    overallScore
                    estimatedCost
                    predictedReach
                }
                totalCost
                budgetUtilization
                predictedTotalReach
                optimizationScore
                constraintsSatisfied
                algorithmUsed
            }
        }
        \"\"\"
        
        optimization_variables = {
            \"campaignId\": campaign_id,
            \"constraints\": {
                \"maxBudget\": 30000.00,
                \"minKols\": 2,
                \"maxKols\": 5,
                \"objective\": \"MAXIMIZE_REACH\"
            }
        }
        
        optimization_result = await graphql_client.execute(
            optimization_query,
            variables=optimization_variables,
            context=mock_graphql_context
        )
        
        # Verify end-to-end workflow
        assert optimization_result[\"errors\"] is None
        
        final_optimization = optimization_result[\"data\"][\"optimizeCampaignBudget\"]
        assert final_optimization[\"campaignId\"] == campaign_id
        assert len(final_optimization[\"selectedKOLs\"]) == 2
        assert final_optimization[\"totalCost\"] == 15000.00
        assert final_optimization[\"constraintsSatisfied\"] == True
        assert final_optimization[\"predictedTotalReach\"] == 27000
    
    @pytest.mark.asyncio
    async def test_error_handling_in_workflow(self, graphql_client, mock_graphql_context):
        \"\"\"Test error handling throughout the workflow.\"\"\"
        
        # Test scoring with invalid campaign ID
        mock_graphql_context.kol_scorer.score_kol.side_effect = ValueError(\"Campaign not found\")
        
        query = \"\"\"
        mutation ScoreKOL($kolId: String!, $campaignId: String!) {
            scoreKOL(kolId: $kolId, campaignId: $campaignId) {
                overallScore
            }
        }
        \"\"\"
        
        variables = {
            \"kolId\": \"valid_kol_id\",
            \"campaignId\": \"invalid_campaign_id\"
        }
        
        result = await graphql_client.execute(
            query,
            variables=variables,
            context=mock_graphql_context
        )
        
        # Should handle error gracefully
        assert result[\"errors\"] is not None
        assert \"Campaign not found\" in result[\"errors\"][0]
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self, graphql_client, mock_graphql_context):
        \"\"\"Test handling of concurrent GraphQL operations.\"\"\"
        
        # Mock services for concurrent operations
        mock_graphql_context.kol_scorer.score_kol.return_value = MagicMock(composite_score=0.85)
        mock_graphql_context.budget_optimizer.optimize_campaign_budget_advanced.return_value = MagicMock()
        
        # Prepare concurrent queries
        scoring_query = \"\"\"
        mutation ScoreKOL($kolId: String!, $campaignId: String!) {
            scoreKOL(kolId: $kolId, campaignId: $campaignId) {
                overallScore
            }
        }
        \"\"\"
        
        optimization_query = \"\"\"
        mutation OptimizeCampaignBudget($campaignId: String!, $constraints: OptimizationConstraintsInput!) {
            optimizeCampaignBudget(campaignId: $campaignId, constraints: $constraints) {
                totalCost
            }
        }
        \"\"\"
        
        # Execute concurrent operations
        tasks = [
            graphql_client.execute(
                scoring_query,
                variables={\"kolId\": f\"kol_{i}\", \"campaignId\": \"campaign_123\"},
                context=mock_graphql_context
            )
            for i in range(3)
        ] + [
            graphql_client.execute(
                optimization_query,
                variables={
                    \"campaignId\": \"campaign_123\",
                    \"constraints\": {\"maxBudget\": 10000.00, \"minKols\": 1, \"maxKols\": 5}
                },
                context=mock_graphql_context
            )
            for _ in range(2)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # All operations should complete successfully
        for result in results:
            assert not isinstance(result, Exception), f\"Operation failed: {result}\"
            assert result[\"errors\"] is None or len(result[\"errors\"]) == 0


if __name__ == \"__main__\":
    pytest.main([__file__, \"-v\", \"-m\", \"integration\", \"--tb=short\"])