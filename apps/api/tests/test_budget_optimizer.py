"""
Comprehensive tests for the Budget Optimizer (POC4) - Constraint-Based Optimization System

AIDEV-NOTE: Production-ready tests for the sophisticated budget optimization algorithm with
comprehensive coverage of constraint satisfaction, multi-objective optimization, and performance predictions.
"""
import pytest
import asyncio
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import List, Dict, Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

from kol_api.services.budget_optimizer import (
    BudgetOptimizerService, 
    KOLCandidate, 
    OptimizationResult
)
from kol_api.database.models.campaign import Campaign
from kol_api.database.models.kol import KOL, KOLMetrics
from kol_api.database.models.budget import (
    BudgetPlan, 
    BudgetAllocation, 
    BudgetStatus,
    OptimizationObjective, 
    AllocationStrategy
)


# AIDEV-NOTE: Test fixtures for comprehensive optimization scenarios

@pytest.fixture
def mock_db_session():
    """Mock database session."""
    return AsyncMock()


@pytest.fixture
def sample_budget_tiers():
    """Sample budget tier configuration."""
    return {
        "nano": {
            "avg_cost_per_post": 500,
            "min_cost": 100,
            "max_cost": 1000,
            "typical_followers": (100, 10000),
            "typical_engagement": (0.03, 0.08)
        },
        "micro": {
            "avg_cost_per_post": 2500,
            "min_cost": 1000,
            "max_cost": 5000,
            "typical_followers": (10000, 100000),
            "typical_engagement": (0.02, 0.06)
        },
        "mid": {
            "avg_cost_per_post": 12500,
            "min_cost": 5000,
            "max_cost": 25000,
            "typical_followers": (100000, 1000000),
            "typical_engagement": (0.015, 0.04)
        },
        "macro": {
            "avg_cost_per_post": 62500,
            "min_cost": 25000,
            "max_cost": 125000,
            "typical_followers": (1000000, 10000000),
            "typical_engagement": (0.01, 0.025)
        }
    }


@pytest.fixture
def sample_campaign():
    """Sample campaign for optimization testing."""
    campaign = MagicMock(spec=Campaign)
    campaign.id = "optimization_campaign_123"
    campaign.name = "Budget Optimization Test Campaign"
    campaign.objective = MagicMock()
    campaign.objective.value = "maximize_engagement"
    campaign.total_budget = Decimal("100000")
    
    # Campaign constraints
    campaign.target_kol_tiers = ["micro", "mid"]
    campaign.target_categories = ["lifestyle", "fashion"]
    campaign.min_follower_count = 10000
    campaign.max_follower_count = 500000
    campaign.min_engagement_rate = Decimal("0.02")
    
    return campaign


@pytest.fixture
def diverse_kol_candidates(sample_budget_tiers):
    """Create diverse set of KOL candidates for optimization testing."""
    candidates = []
    
    # Nano influencers (high engagement, low cost)
    for i in range(5):
        kol = MagicMock(spec=KOL)
        kol.id = f"nano_kol_{i}"
        kol.display_name = f"Nano KOL {i}"
        kol.tier = MagicMock()
        kol.tier.value = "nano"
        kol.primary_category = MagicMock()
        kol.primary_category.value = "lifestyle"
        kol.is_active = True
        kol.is_brand_safe = True
        kol.is_verified = False
        
        metrics = MagicMock(spec=KOLMetrics)
        metrics.follower_count = 5000 + (i * 1000)
        metrics.engagement_rate = Decimal(str(0.05 + (i * 0.005)))  # 5-7.5%
        metrics.posts_last_30_days = 15 + i
        metrics.fake_follower_percentage = Decimal("0.03")
        
        candidate = KOLCandidate(
            kol=kol,
            metrics=metrics,
            estimated_cost=Decimal(str(300 + (i * 100))),
            predicted_reach=int(metrics.follower_count * 0.15),
            predicted_engagement=int(metrics.follower_count * float(metrics.engagement_rate)),
            predicted_conversions=int(metrics.follower_count * float(metrics.engagement_rate) * 0.02),
            efficiency_score=Decimal("0.8"),
            risk_score=Decimal("0.2")
        )
        candidates.append(candidate)
    
    # Micro influencers (good balance)
    for i in range(8):
        kol = MagicMock(spec=KOL)
        kol.id = f"micro_kol_{i}"
        kol.display_name = f"Micro KOL {i}"
        kol.tier = MagicMock()
        kol.tier.value = "micro"
        kol.primary_category = MagicMock()
        kol.primary_category.value = "lifestyle" if i % 2 == 0 else "fashion"
        kol.is_active = True
        kol.is_brand_safe = True
        kol.is_verified = i < 4  # Half verified
        
        metrics = MagicMock(spec=KOLMetrics)
        metrics.follower_count = 25000 + (i * 10000)
        metrics.engagement_rate = Decimal(str(0.03 + (i * 0.002)))  # 3-4.6%
        metrics.posts_last_30_days = 12 + i
        metrics.fake_follower_percentage = Decimal("0.05")
        
        candidate = KOLCandidate(
            kol=kol,
            metrics=metrics,
            estimated_cost=Decimal(str(2000 + (i * 300))),
            predicted_reach=int(metrics.follower_count * 0.18),
            predicted_engagement=int(metrics.follower_count * float(metrics.engagement_rate)),
            predicted_conversions=int(metrics.follower_count * float(metrics.engagement_rate) * 0.018),
            efficiency_score=Decimal("0.7"),
            risk_score=Decimal("0.15")
        )
        candidates.append(candidate)
    
    # Mid-tier influencers (high reach, higher cost)
    for i in range(4):
        kol = MagicMock(spec=KOL)
        kol.id = f"mid_kol_{i}"
        kol.display_name = f"Mid KOL {i}"
        kol.tier = MagicMock()
        kol.tier.value = "mid"
        kol.primary_category = MagicMock()
        kol.primary_category.value = "fashion"
        kol.is_active = True
        kol.is_brand_safe = True
        kol.is_verified = True
        
        metrics = MagicMock(spec=KOLMetrics)
        metrics.follower_count = 200000 + (i * 100000)
        metrics.engagement_rate = Decimal(str(0.02 + (i * 0.001)))  # 2-2.3%
        metrics.posts_last_30_days = 10 + i
        metrics.fake_follower_percentage = Decimal("0.08")
        
        candidate = KOLCandidate(
            kol=kol,
            metrics=metrics,
            estimated_cost=Decimal(str(10000 + (i * 2500))),
            predicted_reach=int(metrics.follower_count * 0.22),
            predicted_engagement=int(metrics.follower_count * float(metrics.engagement_rate)),
            predicted_conversions=int(metrics.follower_count * float(metrics.engagement_rate) * 0.015),
            efficiency_score=Decimal("0.6"),
            risk_score=Decimal("0.25")
        )
        candidates.append(candidate)
    
    # High-risk candidates for risk testing
    high_risk_kol = MagicMock(spec=KOL)
    high_risk_kol.id = "high_risk_kol"
    high_risk_kol.display_name = "High Risk KOL"
    high_risk_kol.tier = MagicMock()
    high_risk_kol.tier.value = "micro"
    high_risk_kol.primary_category = MagicMock()
    high_risk_kol.primary_category.value = "lifestyle"
    high_risk_kol.is_active = True
    high_risk_kol.is_brand_safe = False  # Risk factor
    high_risk_kol.is_verified = False
    
    high_risk_metrics = MagicMock(spec=KOLMetrics)
    high_risk_metrics.follower_count = 50000
    high_risk_metrics.engagement_rate = Decimal("0.001")  # Suspiciously low
    high_risk_metrics.posts_last_30_days = 1  # Inactive
    high_risk_metrics.fake_follower_percentage = Decimal("0.45")  # Very high
    
    high_risk_candidate = KOLCandidate(
        kol=high_risk_kol,
        metrics=high_risk_metrics,
        estimated_cost=Decimal("1500"),
        predicted_reach=2500,
        predicted_engagement=50,
        predicted_conversions=1,
        efficiency_score=Decimal("0.2"),
        risk_score=Decimal("0.9")
    )
    candidates.append(high_risk_candidate)
    
    return candidates


@pytest.fixture 
def budget_optimizer_service(mock_db_session, sample_budget_tiers):
    """Budget optimizer service with mocked dependencies."""
    with patch('kol_api.services.budget_optimizer.settings') as mock_settings:
        mock_settings.budget_tiers = sample_budget_tiers
        return BudgetOptimizerService(mock_db_session)


# AIDEV-NOTE: Core Optimization Algorithm Tests

class TestBudgetOptimizerCore:
    """Test core optimization algorithms and constraint satisfaction."""
    
    @pytest.mark.asyncio
    async def test_optimization_maximize_reach(
        self,
        budget_optimizer_service,
        sample_campaign,
        diverse_kol_candidates
    ):
        """Test optimization algorithm maximizing reach."""
        # Mock database queries
        with patch.object(budget_optimizer_service, '_get_campaign_requirements', return_value=sample_campaign), \
             patch.object(budget_optimizer_service, '_get_kol_candidates', return_value=diverse_kol_candidates):
            
            result = await budget_optimizer_service.optimize_campaign_budget(
                campaign_id=sample_campaign.id,
                total_budget=Decimal("50000"),
                optimization_objective=OptimizationObjective.MAXIMIZE_REACH,
                allocation_strategy=AllocationStrategy.PERFORMANCE_WEIGHTED
            )
            
            assert isinstance(result, OptimizationResult)
            assert len(result.selected_kols) > 0
            assert result.total_cost <= Decimal("50000")
            assert result.constraints_met is True
            
            # Should prioritize KOLs with high reach potential
            total_predicted_reach = sum(kol.predicted_reach for kol in result.selected_kols)
            assert total_predicted_reach == result.predicted_performance["reach"]
            
            # Optimization score should be positive
            assert result.optimization_score > Decimal("0.0")
    
    @pytest.mark.asyncio
    async def test_optimization_maximize_engagement(
        self,
        budget_optimizer_service,
        sample_campaign,
        diverse_kol_candidates
    ):
        """Test optimization algorithm maximizing engagement."""
        with patch.object(budget_optimizer_service, '_get_campaign_requirements', return_value=sample_campaign), \
             patch.object(budget_optimizer_service, '_get_kol_candidates', return_value=diverse_kol_candidates):
            
            result = await budget_optimizer_service.optimize_campaign_budget(
                campaign_id=sample_campaign.id,
                total_budget=Decimal("50000"),
                optimization_objective=OptimizationObjective.MAXIMIZE_ENGAGEMENT,
                allocation_strategy=AllocationStrategy.PERFORMANCE_WEIGHTED
            )
            
            assert isinstance(result, OptimizationResult)
            assert len(result.selected_kols) > 0
            
            # Should prioritize KOLs with high engagement potential
            total_predicted_engagement = sum(kol.predicted_engagement for kol in result.selected_kols)
            assert total_predicted_engagement == result.predicted_performance["engagement"]
            
            # Should favor nano/micro influencers with higher engagement rates
            nano_micro_count = sum(1 for kol in result.selected_kols if kol.kol.tier.value in ["nano", "micro"])
            assert nano_micro_count > 0
    
    @pytest.mark.asyncio
    async def test_optimization_maximize_conversions(
        self,
        budget_optimizer_service,
        sample_campaign,
        diverse_kol_candidates
    ):
        """Test optimization algorithm maximizing conversions."""
        with patch.object(budget_optimizer_service, '_get_campaign_requirements', return_value=sample_campaign), \
             patch.object(budget_optimizer_service, '_get_kol_candidates', return_value=diverse_kol_candidates):
            
            result = await budget_optimizer_service.optimize_campaign_budget(
                campaign_id=sample_campaign.id,
                total_budget=Decimal("75000"),
                optimization_objective=OptimizationObjective.MAXIMIZE_CONVERSIONS,
                allocation_strategy=AllocationStrategy.PERFORMANCE_WEIGHTED
            )
            
            assert isinstance(result, OptimizationResult)
            assert len(result.selected_kols) > 0
            
            # Should optimize for conversion potential
            total_predicted_conversions = sum(kol.predicted_conversions for kol in result.selected_kols)
            assert total_predicted_conversions == result.predicted_performance["conversions"]
            assert total_predicted_conversions > 0
    
    @pytest.mark.asyncio
    async def test_optimization_minimize_cost(
        self,
        budget_optimizer_service,
        sample_campaign,
        diverse_kol_candidates
    ):
        """Test optimization algorithm minimizing cost while meeting requirements."""
        # Set minimum performance constraints
        constraints = {
            "min_total_reach": 100000,
            "min_total_engagement": 5000,
            "min_kols": 3,
            "max_kols": 8
        }
        
        with patch.object(budget_optimizer_service, '_get_campaign_requirements', return_value=sample_campaign), \
             patch.object(budget_optimizer_service, '_get_kol_candidates', return_value=diverse_kol_candidates):
            
            result = await budget_optimizer_service.optimize_campaign_budget(
                campaign_id=sample_campaign.id,
                total_budget=Decimal("100000"),
                optimization_objective=OptimizationObjective.MINIMIZE_COST,
                allocation_strategy=AllocationStrategy.COST_EFFICIENT,
                constraints=constraints
            )
            
            assert isinstance(result, OptimizationResult)
            assert len(result.selected_kols) >= 3
            assert len(result.selected_kols) <= 8
            
            # Should meet minimum requirements
            assert result.predicted_performance["reach"] >= 100000
            assert result.predicted_performance["engagement"] >= 5000
            
            # Should prioritize cost-efficient options (nano/micro influencers)
            avg_cost_per_kol = result.total_cost / len(result.selected_kols)
            assert avg_cost_per_kol < Decimal("10000")  # Should be relatively low cost
    
    @pytest.mark.asyncio
    async def test_constraint_satisfaction_tier_requirements(
        self,
        budget_optimizer_service,
        sample_campaign,
        diverse_kol_candidates
    ):
        """Test constraint satisfaction with specific tier requirements."""
        # Require specific tier distribution
        constraints = {
            "tier_requirements": {
                "nano": 2,
                "micro": 3,
                "mid": 1
            },
            "max_kols": 10
        }
        
        with patch.object(budget_optimizer_service, '_get_campaign_requirements', return_value=sample_campaign), \
             patch.object(budget_optimizer_service, '_get_kol_candidates', return_value=diverse_kol_candidates):
            
            result = await budget_optimizer_service.optimize_campaign_budget(
                campaign_id=sample_campaign.id,
                total_budget=Decimal("80000"),
                optimization_objective=OptimizationObjective.BALANCED,
                allocation_strategy=AllocationStrategy.PERFORMANCE_WEIGHTED,
                constraints=constraints
            )
            
            # Count tier distribution
            tier_counts = {"nano": 0, "micro": 0, "mid": 0, "macro": 0}
            for kol in result.selected_kols:
                tier_counts[kol.kol.tier.value] += 1
            
            # Should meet tier requirements
            assert tier_counts["nano"] >= 2
            assert tier_counts["micro"] >= 3
            assert tier_counts["mid"] >= 1
            assert result.constraints_met is True
    
    @pytest.mark.asyncio
    async def test_constraint_satisfaction_risk_limits(
        self,
        budget_optimizer_service,
        sample_campaign,
        diverse_kol_candidates
    ):
        """Test risk constraint satisfaction."""
        constraints = {
            "max_risk_per_kol": Decimal("0.5"),
            "max_kols": 10
        }
        
        with patch.object(budget_optimizer_service, '_get_campaign_requirements', return_value=sample_campaign), \
             patch.object(budget_optimizer_service, '_get_kol_candidates', return_value=diverse_kol_candidates):
            
            result = await budget_optimizer_service.optimize_campaign_budget(
                campaign_id=sample_campaign.id,
                total_budget=Decimal("60000"),
                optimization_objective=OptimizationObjective.BALANCED,
                allocation_strategy=AllocationStrategy.RISK_BALANCED,
                constraints=constraints
            )
            
            # All selected KOLs should meet risk constraint
            for kol in result.selected_kols:
                assert kol.risk_score <= Decimal("0.5")
            
            # High-risk KOL should be excluded
            high_risk_kol_selected = any(
                kol.kol.id == "high_risk_kol" for kol in result.selected_kols
            )
            assert high_risk_kol_selected is False
    
    @pytest.mark.asyncio
    async def test_budget_constraint_satisfaction(
        self,
        budget_optimizer_service,
        sample_campaign,
        diverse_kol_candidates
    ):
        """Test strict budget constraint enforcement."""
        strict_budget = Decimal("25000")  # Tight budget
        
        with patch.object(budget_optimizer_service, '_get_campaign_requirements', return_value=sample_campaign), \
             patch.object(budget_optimizer_service, '_get_kol_candidates', return_value=diverse_kol_candidates):
            
            result = await budget_optimizer_service.optimize_campaign_budget(
                campaign_id=sample_campaign.id,
                total_budget=strict_budget,
                optimization_objective=OptimizationObjective.MAXIMIZE_ENGAGEMENT,
                allocation_strategy=AllocationStrategy.COST_EFFICIENT
            )
            
            # Should not exceed budget
            assert result.total_cost <= strict_budget
            
            # Should maximize utilization within budget
            utilization_rate = result.total_cost / strict_budget
            assert utilization_rate >= Decimal("0.8")  # At least 80% utilization
            
            # Should select cost-efficient KOLs (likely nano/micro)
            selected_tiers = {kol.kol.tier.value for kol in result.selected_kols}
            assert "mid" not in selected_tiers or len(result.selected_kols) == 1  # Can't afford many mid-tier


class TestBudgetOptimizerPerformanceEstimation:
    """Test performance prediction accuracy and methodology."""
    
    def test_kol_cost_estimation(self, budget_optimizer_service):
        """Test KOL cost estimation accuracy."""
        # Create test KOL with known characteristics
        test_kol = MagicMock(spec=KOL)
        test_kol.tier = MagicMock()
        test_kol.tier.value = "micro"
        test_kol.is_verified = True
        test_kol.location = "bangkok"
        
        test_metrics = MagicMock(spec=KOLMetrics)
        test_metrics.engagement_rate = Decimal("0.04")  # 4%
        
        estimated_cost = budget_optimizer_service._estimate_kol_cost(test_kol, test_metrics)
        
        # Should be within expected micro tier range
        assert Decimal("1000") <= estimated_cost <= Decimal("10000")
        
        # Verified account should have premium
        test_kol.is_verified = False
        unverified_cost = budget_optimizer_service._estimate_kol_cost(test_kol, test_metrics)
        assert unverified_cost < estimated_cost
    
    def test_reach_prediction(self, budget_optimizer_service):
        """Test reach prediction methodology."""
        test_kol = MagicMock(spec=KOL)
        test_kol.tier = MagicMock()
        test_kol.tier.value = "micro"
        
        test_metrics = MagicMock(spec=KOLMetrics)
        test_metrics.follower_count = 50000
        test_metrics.engagement_rate = Decimal("0.035")  # 3.5%
        
        predicted_reach = budget_optimizer_service._predict_reach(test_kol, test_metrics)
        
        # Should be realistic percentage of followers
        reach_rate = predicted_reach / test_metrics.follower_count
        assert 0.1 <= reach_rate <= 0.3  # 10-30% reach rate is realistic
        
        # Should not exceed follower count
        assert predicted_reach <= test_metrics.follower_count
    
    def test_engagement_prediction(self, budget_optimizer_service):
        """Test engagement prediction based on reach and rates."""
        test_kol = MagicMock(spec=KOL)
        test_kol.tier = MagicMock()
        test_kol.tier.value = "micro"
        
        test_metrics = MagicMock(spec=KOLMetrics)
        test_metrics.follower_count = 100000
        test_metrics.engagement_rate = Decimal("0.03")  # 3%
        
        predicted_engagement = budget_optimizer_service._predict_engagement(test_kol, test_metrics)
        predicted_reach = budget_optimizer_service._predict_reach(test_kol, test_metrics)
        
        # Engagement should be related to reach and engagement rate
        expected_engagement = int(predicted_reach * float(test_metrics.engagement_rate))
        
        # Should be close to expected (within 20% tolerance)
        assert abs(predicted_engagement - expected_engagement) / expected_engagement <= 0.2
    
    def test_conversion_prediction_by_campaign_type(self, budget_optimizer_service):
        """Test conversion prediction varies by campaign objective."""
        test_kol = MagicMock(spec=KOL)
        test_kol.tier = MagicMock()
        test_kol.tier.value = "micro"
        
        test_metrics = MagicMock(spec=KOLMetrics)
        test_metrics.follower_count = 50000
        test_metrics.engagement_rate = Decimal("0.04")
        
        # Brand awareness campaign
        awareness_campaign = MagicMock(spec=Campaign)
        awareness_campaign.objective = MagicMock()
        awareness_campaign.objective.value = "brand_awareness"
        
        awareness_conversions = budget_optimizer_service._predict_conversions(
            test_kol, test_metrics, awareness_campaign
        )
        
        # Sales campaign
        sales_campaign = MagicMock(spec=Campaign)
        sales_campaign.objective = MagicMock()
        sales_campaign.objective.value = "sales"
        
        sales_conversions = budget_optimizer_service._predict_conversions(
            test_kol, test_metrics, sales_campaign
        )
        
        # Sales should have higher conversion rate than brand awareness
        assert sales_conversions > awareness_conversions
    
    def test_efficiency_score_calculation(self, budget_optimizer_service):
        """Test cost efficiency score calculation."""
        # High efficiency scenario (low cost, high performance)
        high_efficiency = budget_optimizer_service._calculate_efficiency_score(
            cost=Decimal("1000"),
            reach=10000,
            engagement=500
        )
        
        # Low efficiency scenario (high cost, low performance)
        low_efficiency = budget_optimizer_service._calculate_efficiency_score(
            cost=Decimal("10000"),
            reach=5000,
            engagement=200
        )
        
        # High efficiency should score better
        assert high_efficiency > low_efficiency
        
        # Scores should be between 0 and 1
        assert Decimal("0.0") <= high_efficiency <= Decimal("1.0")
        assert Decimal("0.0") <= low_efficiency <= Decimal("1.0")
    
    def test_risk_score_calculation(self, budget_optimizer_service):
        """Test risk score calculation methodology."""
        # Low risk KOL
        low_risk_kol = MagicMock(spec=KOL)
        low_risk_kol.is_verified = True
        low_risk_kol.is_brand_safe = True
        
        low_risk_metrics = MagicMock(spec=KOLMetrics)
        low_risk_metrics.engagement_rate = Decimal("0.04")  # Good engagement
        low_risk_metrics.posts_last_30_days = 15  # Active
        low_risk_metrics.fake_follower_percentage = Decimal("0.05")  # Low fake followers
        low_risk_metrics.metrics_date = datetime.now(timezone.utc)
        low_risk_metrics.created_at = datetime.now(timezone.utc)
        
        low_risk_score = budget_optimizer_service._calculate_risk_score(low_risk_kol, low_risk_metrics)
        
        # High risk KOL
        high_risk_kol = MagicMock(spec=KOL)
        high_risk_kol.is_verified = False
        high_risk_kol.is_brand_safe = False
        
        high_risk_metrics = MagicMock(spec=KOLMetrics)
        high_risk_metrics.engagement_rate = Decimal("0.005")  # Very low engagement
        high_risk_metrics.posts_last_30_days = 1  # Inactive
        high_risk_metrics.fake_follower_percentage = Decimal("0.4")  # High fake followers
        high_risk_metrics.metrics_date = datetime.now(timezone.utc)
        high_risk_metrics.created_at = datetime.now(timezone.utc)
        
        high_risk_score = budget_optimizer_service._calculate_risk_score(high_risk_kol, high_risk_metrics)
        
        # High risk should score higher than low risk
        assert high_risk_score > low_risk_score
        assert low_risk_score < Decimal("0.5")
        assert high_risk_score > Decimal("0.5")


class TestBudgetOptimizerScenarios:
    """Test various real-world optimization scenarios."""
    
    @pytest.mark.asyncio
    async def test_small_budget_scenario(
        self,
        budget_optimizer_service,
        sample_campaign,
        diverse_kol_candidates
    ):
        """Test optimization with very small budget - should prioritize nano influencers."""
        small_budget = Decimal("5000")
        
        with patch.object(budget_optimizer_service, '_get_campaign_requirements', return_value=sample_campaign), \
             patch.object(budget_optimizer_service, '_get_kol_candidates', return_value=diverse_kol_candidates):
            
            result = await budget_optimizer_service.optimize_campaign_budget(
                campaign_id=sample_campaign.id,
                total_budget=small_budget,
                optimization_objective=OptimizationObjective.MAXIMIZE_ENGAGEMENT,
                allocation_strategy=AllocationStrategy.COST_EFFICIENT
            )
            
            # Should only select affordable KOLs
            assert result.total_cost <= small_budget
            
            # Should prioritize nano influencers for small budget
            selected_tiers = [kol.kol.tier.value for kol in result.selected_kols]
            nano_count = selected_tiers.count("nano")
            
            # Should have significant nano influencer representation
            assert nano_count >= 1
            assert len(result.selected_kols) >= 2  # Should be able to afford multiple nano
    
    @pytest.mark.asyncio
    async def test_large_budget_scenario(
        self,
        budget_optimizer_service,
        sample_campaign,
        diverse_kol_candidates
    ):
        """Test optimization with large budget - should optimize for performance."""
        large_budget = Decimal("200000")
        
        with patch.object(budget_optimizer_service, '_get_campaign_requirements', return_value=sample_campaign), \
             patch.object(budget_optimizer_service, '_get_kol_candidates', return_value=diverse_kol_candidates):
            
            result = await budget_optimizer_service.optimize_campaign_budget(
                campaign_id=sample_campaign.id,
                total_budget=large_budget,
                optimization_objective=OptimizationObjective.MAXIMIZE_REACH,
                allocation_strategy=AllocationStrategy.PERFORMANCE_WEIGHTED
            )
            
            # Should select diverse mix including higher-tier KOLs
            selected_tiers = {kol.kol.tier.value for kol in result.selected_kols}
            assert len(selected_tiers) >= 2  # Should have tier diversity
            
            # Should include mid-tier for reach maximization
            assert "mid" in selected_tiers
            
            # Should achieve high performance
            total_reach = result.predicted_performance["reach"]
            assert total_reach > 500000  # Should achieve significant reach
    
    @pytest.mark.asyncio
    async def test_balanced_portfolio_optimization(
        self,
        budget_optimizer_service,
        sample_campaign,
        diverse_kol_candidates
    ):
        """Test balanced optimization considering multiple factors."""
        constraints = {
            "max_risk_per_kol": Decimal("0.4"),
            "min_kols": 5,
            "max_kols": 12,
            "require_category_diversity": True
        }
        
        with patch.object(budget_optimizer_service, '_get_campaign_requirements', return_value=sample_campaign), \
             patch.object(budget_optimizer_service, '_get_kol_candidates', return_value=diverse_kol_candidates):
            
            result = await budget_optimizer_service.optimize_campaign_budget(
                campaign_id=sample_campaign.id,
                total_budget=Decimal("100000"),
                optimization_objective=OptimizationObjective.BALANCED,
                allocation_strategy=AllocationStrategy.RISK_BALANCED,
                constraints=constraints
            )
            
            # Should meet count constraints
            assert 5 <= len(result.selected_kols) <= 12
            
            # Should have category diversity
            selected_categories = {kol.kol.primary_category.value for kol in result.selected_kols}
            assert len(selected_categories) >= 2  # Lifestyle and fashion
            
            # Should balance risk
            avg_risk = sum(kol.risk_score for kol in result.selected_kols) / len(result.selected_kols)
            assert avg_risk <= Decimal("0.4")
            
            # Should have reasonable tier distribution
            selected_tiers = [kol.kol.tier.value for kol in result.selected_kols]
            tier_diversity = len(set(selected_tiers))
            assert tier_diversity >= 2  # At least 2 different tiers
    
    @pytest.mark.asyncio
    async def test_impossible_constraints_handling(
        self,
        budget_optimizer_service,
        sample_campaign,
        diverse_kol_candidates
    ):
        """Test handling of impossible constraint combinations."""
        impossible_constraints = {
            "tier_requirements": {
                "macro": 5  # Require 5 macro influencers
            },
            "max_budget": Decimal("10000"),  # But only $10k budget
            "max_risk_per_kol": Decimal("0.1")  # Very low risk tolerance
        }
        
        with patch.object(budget_optimizer_service, '_get_campaign_requirements', return_value=sample_campaign), \
             patch.object(budget_optimizer_service, '_get_kol_candidates', return_value=diverse_kol_candidates):
            
            result = await budget_optimizer_service.optimize_campaign_budget(
                campaign_id=sample_campaign.id,
                total_budget=Decimal("10000"),
                optimization_objective=OptimizationObjective.MAXIMIZE_ENGAGEMENT,
                allocation_strategy=AllocationStrategy.PERFORMANCE_WEIGHTED,
                constraints=impossible_constraints
            )
            
            # Should indicate constraints were not met
            assert result.constraints_met is False
            
            # Should still provide best possible solution
            assert result.total_cost <= Decimal("10000")
            assert len(result.selected_kols) > 0
            
            # Should have alternative allocations suggested
            assert len(result.alternative_allocations) >= 1
    
    @pytest.mark.asyncio
    async def test_generate_alternative_scenarios(
        self,
        budget_optimizer_service,
        sample_campaign,
        diverse_kol_candidates
    ):
        """Test generation of alternative budget scenarios."""
        base_budget = Decimal("50000")
        scenario_budgets = [
            Decimal("25000"),   # 50% budget
            Decimal("37500"),   # 75% budget
            Decimal("62500"),   # 125% budget
            Decimal("75000")    # 150% budget
        ]
        
        with patch.object(budget_optimizer_service, '_get_campaign_requirements', return_value=sample_campaign), \
             patch.object(budget_optimizer_service, '_get_kol_candidates', return_value=diverse_kol_candidates):
            
            scenarios = await budget_optimizer_service.generate_alternative_scenarios(
                campaign_id=sample_campaign.id,
                base_budget=base_budget,
                budget_scenarios=scenario_budgets,
                optimization_objective=OptimizationObjective.MAXIMIZE_ENGAGEMENT
            )
            
            # Should generate scenarios for each budget level
            assert len(scenarios) == len(scenario_budgets)
            
            # Performance should generally increase with budget
            performances = [s.predicted_performance["engagement"] for s in scenarios]
            
            # At least some scenarios should show increasing performance with budget
            increasing_performance = any(
                performances[i] < performances[i+1] 
                for i in range(len(performances)-1)
            )
            assert increasing_performance


class TestBudgetPlanCreation:
    """Test budget plan creation and management."""
    
    @pytest.mark.asyncio
    async def test_create_budget_plan_from_optimization(
        self,
        budget_optimizer_service,
        diverse_kol_candidates
    ):
        """Test creating formal budget plan from optimization results."""
        # Create mock optimization result
        selected_kols = diverse_kol_candidates[:5]  # Select first 5 KOLs
        total_cost = sum(kol.estimated_cost for kol in selected_kols)
        
        optimization_result = OptimizationResult(
            selected_kols=selected_kols,
            total_cost=total_cost,
            predicted_performance={
                "reach": sum(kol.predicted_reach for kol in selected_kols),
                "engagement": sum(kol.predicted_engagement for kol in selected_kols),
                "conversions": sum(kol.predicted_conversions for kol in selected_kols)
            },
            optimization_score=Decimal("0.85"),
            alternative_allocations=[],
            constraints_met=True,
            optimization_metadata={
                "objective": "maximize_engagement",
                "strategy": "performance_weighted",
                "algorithm": "greedy_selection"
            }
        )
        
        # Mock database operations
        budget_optimizer_service.db_session.add = MagicMock()
        budget_optimizer_service.db_session.flush = AsyncMock()
        budget_optimizer_service.db_session.commit = AsyncMock()
        
        budget_plan = await budget_optimizer_service.create_budget_plan_from_optimization(
            campaign_id="test_campaign",
            optimization_result=optimization_result,
            plan_name="Test Budget Plan",
            user_id="test_user"
        )
        
        # Should create budget plan with correct attributes
        assert isinstance(budget_plan, BudgetPlan)
        assert budget_plan.campaign_id == "test_campaign"
        assert budget_plan.name == "Test Budget Plan"
        assert budget_plan.status == BudgetStatus.DRAFT
        assert budget_plan.total_budget == total_cost
        assert budget_plan.optimization_score == Decimal("0.85")
        
        # Should have called database operations
        assert budget_optimizer_service.db_session.add.called
        assert budget_optimizer_service.db_session.flush.called
        assert budget_optimizer_service.db_session.commit.called


class TestBudgetOptimizerEdgeCases:
    """Test edge cases and error scenarios."""
    
    @pytest.mark.asyncio
    async def test_no_candidates_available(
        self,
        budget_optimizer_service,
        sample_campaign
    ):
        """Test optimization when no KOL candidates are available."""
        with patch.object(budget_optimizer_service, '_get_campaign_requirements', return_value=sample_campaign), \
             patch.object(budget_optimizer_service, '_get_kol_candidates', return_value=[]):
            
            with pytest.raises(ValueError, match="No suitable KOL candidates found"):
                await budget_optimizer_service.optimize_campaign_budget(
                    campaign_id=sample_campaign.id,
                    total_budget=Decimal("50000"),
                    optimization_objective=OptimizationObjective.MAXIMIZE_ENGAGEMENT,
                    allocation_strategy=AllocationStrategy.PERFORMANCE_WEIGHTED
                )
    
    @pytest.mark.asyncio
    async def test_campaign_not_found(self, budget_optimizer_service):
        """Test optimization when campaign is not found."""
        with patch.object(budget_optimizer_service, '_get_campaign_requirements', return_value=None):
            
            with pytest.raises(ValueError, match="Campaign .* not found"):
                await budget_optimizer_service.optimize_campaign_budget(
                    campaign_id="nonexistent_campaign",
                    total_budget=Decimal("50000"),
                    optimization_objective=OptimizationObjective.MAXIMIZE_ENGAGEMENT,
                    allocation_strategy=AllocationStrategy.PERFORMANCE_WEIGHTED
                )
    
    @pytest.mark.asyncio
    async def test_zero_budget_handling(
        self,
        budget_optimizer_service,
        sample_campaign,
        diverse_kol_candidates
    ):
        """Test handling of zero or negative budget."""
        with patch.object(budget_optimizer_service, '_get_campaign_requirements', return_value=sample_campaign), \
             patch.object(budget_optimizer_service, '_get_kol_candidates', return_value=diverse_kol_candidates):
            
            # Zero budget should result in no selections
            result = await budget_optimizer_service.optimize_campaign_budget(
                campaign_id=sample_campaign.id,
                total_budget=Decimal("0"),
                optimization_objective=OptimizationObjective.MAXIMIZE_ENGAGEMENT,
                allocation_strategy=AllocationStrategy.PERFORMANCE_WEIGHTED
            )
            
            assert len(result.selected_kols) == 0
            assert result.total_cost == Decimal("0")
            assert result.constraints_met is False
    
    def test_optimization_score_edge_cases(self, budget_optimizer_service):
        """Test optimization score calculation edge cases."""
        # Empty selection
        empty_score = budget_optimizer_service._calculate_optimization_score(
            [], OptimizationObjective.MAXIMIZE_REACH, Decimal("50000")
        )
        assert empty_score == Decimal("0.0")
        
        # Single very expensive KOL
        expensive_kol = MagicMock()
        expensive_kol.estimated_cost = Decimal("100000")
        expensive_kol.predicted_reach = 1000000
        expensive_kol.predicted_engagement = 50000
        expensive_kol.efficiency_score = Decimal("0.5")
        
        expensive_score = budget_optimizer_service._calculate_optimization_score(
            [expensive_kol], OptimizationObjective.MAXIMIZE_REACH, Decimal("100000")
        )
        
        # Should be normalized between 0 and 1
        assert Decimal("0.0") <= expensive_score <= Decimal("1.0")
    
    def test_constraint_validation_edge_cases(self, budget_optimizer_service):
        """Test constraint validation with edge cases."""
        # Empty selection with requirements
        empty_constraints_met = budget_optimizer_service._validate_constraints(
            [], {"min_kols": 1}, {"micro": 1}
        )
        assert empty_constraints_met is False
        
        # Selection that exactly meets requirements
        mock_kol = MagicMock()
        mock_kol.kol.tier.value = "micro"
        
        exact_constraints_met = budget_optimizer_service._validate_constraints(
            [mock_kol], {"min_kols": 1, "max_kols": 1}, {"micro": 1}
        )
        assert exact_constraints_met is True


class TestBudgetOptimizerPerformance:
    """Test performance characteristics of the optimization system."""
    
    @pytest.mark.asyncio
    async def test_optimization_performance_large_candidate_set(
        self,
        budget_optimizer_service,
        sample_campaign
    ):
        """Test optimization performance with large number of candidates."""
        # Create large candidate set (100 candidates)
        large_candidate_set = []
        
        for i in range(100):
            kol = MagicMock(spec=KOL)
            kol.id = f"perf_test_kol_{i}"
            kol.display_name = f"Performance Test KOL {i}"
            kol.tier = MagicMock()
            kol.tier.value = ["nano", "micro", "mid"][i % 3]
            kol.primary_category = MagicMock()
            kol.primary_category.value = ["lifestyle", "fashion", "beauty"][i % 3]
            kol.is_active = True
            kol.is_brand_safe = True
            kol.is_verified = i % 2 == 0
            
            metrics = MagicMock(spec=KOLMetrics)
            metrics.follower_count = 10000 + (i * 1000)
            metrics.engagement_rate = Decimal(str(0.02 + (i % 10) * 0.003))
            
            candidate = KOLCandidate(
                kol=kol,
                metrics=metrics,
                estimated_cost=Decimal(str(1000 + (i * 100))),
                predicted_reach=int(metrics.follower_count * 0.15),
                predicted_engagement=int(metrics.follower_count * float(metrics.engagement_rate)),
                predicted_conversions=int(metrics.follower_count * float(metrics.engagement_rate) * 0.02),
                efficiency_score=Decimal(str(0.5 + (i % 10) * 0.05)),
                risk_score=Decimal(str(0.1 + (i % 10) * 0.05))
            )
            large_candidate_set.append(candidate)
        
        import time
        start_time = time.time()
        
        with patch.object(budget_optimizer_service, '_get_campaign_requirements', return_value=sample_campaign), \
             patch.object(budget_optimizer_service, '_get_kol_candidates', return_value=large_candidate_set):
            
            result = await budget_optimizer_service.optimize_campaign_budget(
                campaign_id=sample_campaign.id,
                total_budget=Decimal("100000"),
                optimization_objective=OptimizationObjective.MAXIMIZE_ENGAGEMENT,
                allocation_strategy=AllocationStrategy.PERFORMANCE_WEIGHTED
            )
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Should complete within reasonable time
        assert execution_time < 10.0  # Should complete within 10 seconds
        
        # Should still produce valid results
        assert isinstance(result, OptimizationResult)
        assert len(result.selected_kols) > 0
        assert result.total_cost <= Decimal("100000")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])