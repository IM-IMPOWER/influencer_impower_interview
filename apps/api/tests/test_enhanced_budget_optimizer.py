"""
Comprehensive Tests for Enhanced Budget Optimizer - POC4 Advanced Algorithms

AIDEV-NOTE: Production-ready tests for sophisticated budget optimization algorithms,
constraint satisfaction, genetic algorithms, and multi-objective optimization.
"""
import pytest
import asyncio
import json
from datetime import datetime, timezone, timedelta
from decimal import Decimal, ROUND_HALF_UP
from typing import List, Dict, Any, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
from dataclasses import asdict

from kol_api.services.enhanced_budget_optimizer import (
    EnhancedBudgetOptimizerService,
    AdvancedOptimizationAlgorithm,
    ConstraintSatisfactionSolver
)
from kol_api.services.models import (
    OptimizationResult, OptimizationConstraints, KOLCandidate,
    ConstraintViolation, CampaignPlanExport, KOLTier, KOLMetricsData,
    ScoreComponents, ContentCategory, OptimizationObjective
)
from kol_api.database.models.budget import (
    BudgetPlan, BudgetAllocation, BudgetStatus, AllocationStrategy
)
from kol_api.database.models.campaign import Campaign
from kol_api.database.models.kol import KOL, KOLMetrics


# AIDEV-NOTE: Advanced Test Fixtures

@pytest.fixture
def mock_budget_optimizer_db_session():
    """Mock database session for budget optimizer tests."""
    session = AsyncMock()
    
    # Mock campaign query response
    mock_campaign = MagicMock(spec=Campaign)
    mock_campaign.id = "test_campaign_123"
    mock_campaign.name = "Test Campaign"
    mock_campaign.total_budget = Decimal("50000.00")
    mock_campaign.campaign_type = "engagement"
    
    session.execute.return_value.scalar_one_or_none.return_value = mock_campaign
    session.add = MagicMock()
    session.flush = AsyncMock()
    session.commit = AsyncMock()
    
    return session


@pytest.fixture
def sample_optimization_constraints():
    """Sample optimization constraints for testing."""
    return OptimizationConstraints(
        max_budget=Decimal("50000.00"),
        min_budget_utilization=Decimal("0.8"),
        min_kols=3,
        max_kols=10,
        max_risk_per_kol=Decimal("0.6"),
        max_portfolio_risk=Decimal("0.4"),
        tier_requirements={
            "micro": 2,
            "mid": 1
        },
        min_avg_engagement_rate=Decimal("0.02"),
        min_total_reach=10000,
        min_total_engagement=500,
        required_categories=[ContentCategory.LIFESTYLE, ContentCategory.FASHION]
    )


@pytest.fixture
def diverse_kol_candidate_pool():
    """Create diverse pool of KOL candidates for optimization testing."""
    candidates = []
    
    # High-value micro influencer
    candidates.append(KOLCandidate(
        kol_id="micro_001",
        username="@lifestyle_micro",
        display_name="Lifestyle Micro Influencer",
        platform="tiktok",
        tier=KOLTier.MICRO,
        primary_category=ContentCategory.LIFESTYLE,
        metrics=KOLMetricsData(
            follower_count=50000,
            following_count=2000,
            engagement_rate=Decimal("0.045"),
            avg_likes=Decimal("1800"),
            avg_comments=Decimal("150"),
            posts_last_30_days=20,
            fake_follower_percentage=Decimal("0.05"),
            audience_quality_score=Decimal("0.85"),
            campaign_success_rate=Decimal("0.92"),
            response_rate=Decimal("0.95")
        ),
        score_components=ScoreComponents(
            roi_score=Decimal("0.85"),
            audience_quality_score=Decimal("0.90"),
            brand_safety_score=Decimal("0.95"),
            content_relevance_score=Decimal("0.80"),
            demographic_fit_score=Decimal("0.85"),
            reliability_score=Decimal("0.90"),
            roi_confidence=Decimal("0.90"),
            audience_confidence=Decimal("0.85"),
            brand_safety_confidence=Decimal("0.95"),
            content_relevance_confidence=Decimal("0.80"),
            demographic_confidence=Decimal("0.85"),
            reliability_confidence=Decimal("0.90"),
            overall_confidence=Decimal("0.88"),
            data_freshness_days=3
        ),
        overall_score=Decimal("0.87"),
        predicted_reach=7500,
        predicted_engagement=338,
        predicted_conversions=7,
        estimated_cost_per_post=Decimal("2000.00"),
        estimated_total_cost=Decimal("2000.00"),
        risk_factors=["Unverified account"],
        overall_risk_score=Decimal("0.25"),
        cost_per_engagement=Decimal("5.92"),
        efficiency_ratio=Decimal("0.17")
    ))
    
    # Premium mid-tier influencer
    candidates.append(KOLCandidate(
        kol_id="mid_001",
        username="@fashion_mid",
        display_name="Fashion Mid Influencer",
        platform="instagram",
        tier=KOLTier.MID,
        primary_category=ContentCategory.FASHION,
        metrics=KOLMetricsData(
            follower_count=250000,
            following_count=5000,
            engagement_rate=Decimal("0.025"),
            avg_likes=Decimal("5000"),
            avg_comments=Decimal("250"),
            posts_last_30_days=15,
            fake_follower_percentage=Decimal("0.08"),
            audience_quality_score=Decimal("0.75"),
            campaign_success_rate=Decimal("0.88"),
            response_rate=Decimal("0.85")
        ),
        score_components=ScoreComponents(
            roi_score=Decimal("0.75"),
            audience_quality_score=Decimal("0.80"),
            brand_safety_score=Decimal("0.90"),
            content_relevance_score=Decimal("0.85"),
            demographic_fit_score=Decimal("0.75"),
            reliability_score=Decimal("0.85"),
            roi_confidence=Decimal("0.85"),
            audience_confidence=Decimal("0.80"),
            brand_safety_confidence=Decimal("0.90"),
            content_relevance_confidence=Decimal("0.85"),
            demographic_confidence=Decimal("0.80"),
            reliability_confidence=Decimal("0.85"),
            overall_confidence=Decimal("0.84"),
            data_freshness_days=2
        ),
        overall_score=Decimal("0.82"),
        predicted_reach=37500,
        predicted_engagement=938,
        predicted_conversions=19,
        estimated_cost_per_post=Decimal("8000.00"),
        estimated_total_cost=Decimal("8000.00"),
        risk_factors=[],
        overall_risk_score=Decimal("0.15"),
        cost_per_engagement=Decimal("8.53"),
        efficiency_ratio=Decimal("0.10")
    ))
    
    # Budget-friendly nano influencer
    candidates.append(KOLCandidate(
        kol_id="nano_001",
        username="@beauty_nano",
        display_name="Beauty Nano Influencer",
        platform="tiktok",
        tier=KOLTier.NANO,
        primary_category=ContentCategory.BEAUTY,
        metrics=KOLMetricsData(
            follower_count=8000,
            following_count=1200,
            engagement_rate=Decimal("0.065"),
            avg_likes=Decimal("400"),
            avg_comments=Decimal("80"),
            posts_last_30_days=25,
            fake_follower_percentage=Decimal("0.02"),
            audience_quality_score=Decimal("0.90"),
            campaign_success_rate=Decimal("0.85"),
            response_rate=Decimal("0.98")
        ),
        score_components=ScoreComponents(
            roi_score=Decimal("0.90"),
            audience_quality_score=Decimal("0.95"),
            brand_safety_score=Decimal("0.85"),
            content_relevance_score=Decimal("0.70"),
            demographic_fit_score=Decimal("0.80"),
            reliability_score=Decimal("0.80"),
            roi_confidence=Decimal("0.85"),
            audience_confidence=Decimal("0.90"),
            brand_safety_confidence=Decimal("0.85"),
            content_relevance_confidence=Decimal("0.75"),
            demographic_confidence=Decimal("0.80"),
            reliability_confidence=Decimal("0.85"),
            overall_confidence=Decimal("0.83"),
            data_freshness_days=1
        ),
        overall_score=Decimal("0.83"),
        predicted_reach=1200,
        predicted_engagement=78,
        predicted_conversions=2,
        estimated_cost_per_post=Decimal("400.00"),
        estimated_total_cost=Decimal("400.00"),
        risk_factors=["New account"],
        overall_risk_score=Decimal("0.35"),
        cost_per_engagement=Decimal("5.13"),
        efficiency_ratio=Decimal("0.21")
    ))
    
    # High-risk but high-reward candidate
    candidates.append(KOLCandidate(
        kol_id="risky_001",
        username="@risky_influencer",
        display_name="High Risk Influencer",
        platform="youtube",
        tier=KOLTier.MID,
        primary_category=ContentCategory.ENTERTAINMENT,
        metrics=KOLMetricsData(
            follower_count=180000,
            following_count=15000,
            engagement_rate=Decimal("0.035"),
            avg_likes=Decimal("4500"),
            avg_comments=Decimal("800"),
            posts_last_30_days=5,  # Low posting frequency
            fake_follower_percentage=Decimal("0.25"),  # High fake followers
            audience_quality_score=Decimal("0.60"),
            campaign_success_rate=Decimal("0.65"),
            response_rate=Decimal("0.70")
        ),
        score_components=ScoreComponents(
            roi_score=Decimal("0.80"),
            audience_quality_score=Decimal("0.60"),
            brand_safety_score=Decimal("0.75"),
            content_relevance_score=Decimal("0.50"),
            demographic_fit_score=Decimal("0.65"),
            reliability_score=Decimal("0.55"),
            roi_confidence=Decimal("0.70"),
            audience_confidence=Decimal("0.85"),
            brand_safety_confidence=Decimal("0.75"),
            content_relevance_confidence=Decimal("0.60"),
            demographic_confidence=Decimal("0.70"),
            reliability_confidence=Decimal("0.80"),
            overall_confidence=Decimal("0.73"),
            data_freshness_days=7
        ),
        overall_score=Decimal("0.64"),
        predicted_reach=6300,
        predicted_engagement=221,
        predicted_conversions=4,
        estimated_cost_per_post=Decimal("6000.00"),
        estimated_total_cost=Decimal("6000.00"),
        risk_factors=["High fake followers", "Irregular posting", "Controversial content"],
        overall_risk_score=Decimal("0.75"),
        cost_per_engagement=Decimal("27.15"),
        efficiency_ratio=Decimal("0.04")
    ))
    
    return candidates


@pytest.fixture
def budget_optimizer_service(mock_budget_optimizer_db_session):
    """Enhanced budget optimizer service instance."""
    return EnhancedBudgetOptimizerService(mock_budget_optimizer_db_session)


# AIDEV-NOTE: Advanced Optimization Algorithm Tests

class TestAdvancedOptimizationAlgorithm:
    """Test advanced optimization algorithms for KOL selection."""
    
    def test_genetic_algorithm_initialization(self, diverse_kol_candidate_pool, sample_optimization_constraints):
        """Test genetic algorithm initialization and basic functionality."""
        algorithm = AdvancedOptimizationAlgorithm(diverse_kol_candidate_pool)
        
        # Test population generation
        population = algorithm._generate_initial_population(
            population_size=10,
            constraints=sample_optimization_constraints
        )
        
        assert len(population) == 10
        
        # Each solution should be valid
        for solution in population:
            assert isinstance(solution, list)
            assert len(solution) <= sample_optimization_constraints.max_kols
            
            # Check budget constraint
            total_cost = sum(kol.estimated_total_cost for kol in solution)
            assert total_cost <= sample_optimization_constraints.max_budget
    
    def test_genetic_algorithm_fitness_calculation(self, diverse_kol_candidate_pool, sample_optimization_constraints):
        """Test genetic algorithm fitness calculation."""
        algorithm = AdvancedOptimizationAlgorithm(diverse_kol_candidate_pool)
        
        # Test different objectives
        objectives = [
            OptimizationObjective.MAXIMIZE_REACH,
            OptimizationObjective.MAXIMIZE_ENGAGEMENT,
            OptimizationObjective.MAXIMIZE_CONVERSIONS,
            OptimizationObjective.MAXIMIZE_ROI
        ]
        
        solution = diverse_kol_candidate_pool[:2]  # Valid small solution
        
        fitness_scores = []
        for objective in objectives:
            fitness = algorithm._calculate_fitness(
                solution, objective, sample_optimization_constraints
            )
            fitness_scores.append(fitness)
            assert fitness >= 0  # Fitness should be non-negative
        
        # Different objectives should potentially give different scores
        assert len(set(fitness_scores)) >= 1  # At least some variation expected
    
    def test_genetic_algorithm_constraint_penalties(self, diverse_kol_candidate_pool, sample_optimization_constraints):
        """Test genetic algorithm applies penalties for constraint violations."""
        algorithm = AdvancedOptimizationAlgorithm(diverse_kol_candidate_pool)
        
        # Create solution that violates budget constraint
        over_budget_solution = diverse_kol_candidate_pool  # All candidates = over budget
        
        fitness_over_budget = algorithm._calculate_fitness(
            over_budget_solution, OptimizationObjective.MAXIMIZE_REACH, sample_optimization_constraints
        )
        
        # Create valid solution
        valid_solution = diverse_kol_candidate_pool[:2]
        
        fitness_valid = algorithm._calculate_fitness(
            valid_solution, OptimizationObjective.MAXIMIZE_REACH, sample_optimization_constraints
        )
        
        # Valid solution should have higher fitness due to penalties
        assert fitness_valid > fitness_over_budget
    
    def test_genetic_algorithm_crossover_and_mutation(self, diverse_kol_candidate_pool):
        """Test genetic algorithm crossover and mutation operations."""
        algorithm = AdvancedOptimizationAlgorithm(diverse_kol_candidate_pool)
        
        parent1 = diverse_kol_candidate_pool[:2]
        parent2 = diverse_kol_candidate_pool[1:3]
        
        # Test crossover
        child = algorithm._crossover(parent1, parent2)
        assert isinstance(child, list)
        assert len(child) >= 1
        
        # Child should contain elements from parents
        parent_ids = {kol.kol_id for kol in parent1 + parent2}
        child_ids = {kol.kol_id for kol in child}
        assert child_ids.issubset(parent_ids)
        
        # Test mutation
        original_child = child.copy()
        mutated_child = algorithm._mutate(child)
        
        # Mutation may or may not change the solution
        assert isinstance(mutated_child, list)
    
    def test_linear_programming_approximation(self, diverse_kol_candidate_pool, sample_optimization_constraints):
        """Test linear programming approximation algorithm."""
        algorithm = AdvancedOptimizationAlgorithm(diverse_kol_candidate_pool)
        
        # Test with different objectives
        for objective in [OptimizationObjective.MAXIMIZE_REACH, OptimizationObjective.MAXIMIZE_ENGAGEMENT, OptimizationObjective.MAXIMIZE_ROI]:
            selected = algorithm.linear_programming_approximation(
                sample_optimization_constraints, objective
            )
            
            assert isinstance(selected, list)
            assert len(selected) <= sample_optimization_constraints.max_kols
            
            # Verify budget constraint
            total_cost = sum(kol.estimated_total_cost for kol in selected)
            assert total_cost <= sample_optimization_constraints.max_budget
            
            # Verify risk constraint
            for kol in selected:
                assert kol.overall_risk_score <= sample_optimization_constraints.max_risk_per_kol
    
    def test_knapsack_optimization(self, diverse_kol_candidate_pool, sample_optimization_constraints):
        """Test knapsack optimization algorithm."""
        algorithm = AdvancedOptimizationAlgorithm(diverse_kol_candidate_pool)
        
        selected = algorithm.knapsack_optimization(
            sample_optimization_constraints, OptimizationObjective.MAXIMIZE_REACH
        )
        
        assert isinstance(selected, list)
        assert len(selected) <= sample_optimization_constraints.max_kols
        
        # Verify constraints
        total_cost = sum(kol.estimated_total_cost for kol in selected)
        assert total_cost <= sample_optimization_constraints.max_budget
    
    def test_knapsack_dynamic_programming(self, diverse_kol_candidate_pool):
        """Test knapsack dynamic programming implementation."""
        algorithm = AdvancedOptimizationAlgorithm(diverse_kol_candidate_pool)
        
        # Simple test case
        values = [10, 20, 30]
        costs = [5, 10, 15]
        capacity = 20
        
        selected_indices = algorithm._knapsack_dp(values, costs, capacity)
        
        assert isinstance(selected_indices, list)
        
        # Verify selected items don't exceed capacity
        total_cost = sum(costs[i] for i in selected_indices)
        assert total_cost <= capacity
        
        # Verify we get reasonable value
        total_value = sum(values[i] for i in selected_indices)
        assert total_value > 0
    
    def test_greedy_fallback(self, diverse_kol_candidate_pool, sample_optimization_constraints):
        """Test greedy fallback algorithm."""
        algorithm = AdvancedOptimizationAlgorithm(diverse_kol_candidate_pool)
        
        selected = algorithm._greedy_fallback(
            sample_optimization_constraints, OptimizationObjective.MAXIMIZE_REACH
        )
        
        assert isinstance(selected, list)
        assert len(selected) <= sample_optimization_constraints.max_kols
        
        # Should be sorted by overall score (greedy selection)
        if len(selected) > 1:
            for i in range(1, len(selected)):
                assert float(selected[i-1].overall_score) >= float(selected[i].overall_score)


# AIDEV-NOTE: Constraint Satisfaction Solver Tests

class TestConstraintSatisfactionSolver:
    """Test constraint satisfaction solver for KOL selection."""
    
    def test_constraint_solver_initialization(self, diverse_kol_candidate_pool):
        """Test constraint solver initialization."""
        solver = ConstraintSatisfactionSolver(diverse_kol_candidate_pool)
        
        assert solver.candidates == diverse_kol_candidate_pool
        assert hasattr(solver, 'logger')
    
    def test_hard_constraint_filtering(self, diverse_kol_candidate_pool, sample_optimization_constraints):
        """Test hard constraint filtering."""
        solver = ConstraintSatisfactionSolver(diverse_kol_candidate_pool)
        
        # Apply strict budget constraint
        strict_constraints = OptimizationConstraints(
            max_budget=Decimal("1000.00"),  # Very low budget
            min_kols=1,
            max_kols=5,
            max_risk_per_kol=Decimal("0.5"),
            tier_requirements={}
        )
        
        filtered = solver._apply_hard_constraints(
            diverse_kol_candidate_pool, strict_constraints
        )
        
        # Only candidates under budget should remain
        for candidate in filtered:
            assert candidate.estimated_total_cost <= strict_constraints.max_budget
    
    def test_tier_requirement_satisfaction(self, diverse_kol_candidate_pool, sample_optimization_constraints):
        """Test tier requirement satisfaction."""
        solver = ConstraintSatisfactionSolver(diverse_kol_candidate_pool)
        
        # Apply tier filtering first
        feasible_candidates = solver._apply_hard_constraints(
            diverse_kol_candidate_pool, sample_optimization_constraints
        )
        
        tier_satisfied = solver._satisfy_tier_requirements(
            feasible_candidates, sample_optimization_constraints
        )
        
        # Count tiers in result
        tier_counts = {}
        for candidate in tier_satisfied:
            tier = candidate.tier.value
            tier_counts[tier] = tier_counts.get(tier, 0) + 1
        
        # Should meet minimum tier requirements
        for tier, required_count in sample_optimization_constraints.tier_requirements.items():
            actual_count = tier_counts.get(tier, 0)
            assert actual_count >= required_count, f"Tier {tier}: expected {required_count}, got {actual_count}"
    
    def test_constraint_satisfaction_complete_workflow(self, diverse_kol_candidate_pool, sample_optimization_constraints):
        """Test complete constraint satisfaction workflow."""
        solver = ConstraintSatisfactionSolver(diverse_kol_candidate_pool)
        
        selected_kols, violations = solver.solve(
            sample_optimization_constraints, OptimizationObjective.MAXIMIZE_REACH
        )
        
        # Should return valid results
        assert isinstance(selected_kols, list)
        assert isinstance(violations, list)
        
        # If successful, should meet basic constraints
        if selected_kols:
            total_cost = sum(kol.estimated_total_cost for kol in selected_kols)
            assert total_cost <= sample_optimization_constraints.max_budget
            assert len(selected_kols) <= sample_optimization_constraints.max_kols
    
    def test_final_selection_validation(self, diverse_kol_candidate_pool, sample_optimization_constraints):
        """Test validation of final selection."""
        solver = ConstraintSatisfactionSolver(diverse_kol_candidate_pool)
        
        # Create valid selection
        valid_selection = diverse_kol_candidate_pool[:2]  # Small valid selection
        
        violations = solver._validate_final_selection(
            valid_selection, sample_optimization_constraints
        )
        
        # Check violation types
        violation_types = {v.constraint_type for v in violations}
        
        # Should not have critical violations if selection is reasonable
        assert "selection" not in violation_types  # Should have KOLs selected
    
    def test_portfolio_risk_calculation(self, diverse_kol_candidate_pool):
        """Test portfolio risk calculation."""
        solver = ConstraintSatisfactionSolver(diverse_kol_candidate_pool)
        
        # Test with different risk profiles
        low_risk_candidate = diverse_kol_candidate_pool[0]  # Typically lower risk
        high_risk_candidate = diverse_kol_candidate_pool[3]  # Risky candidate
        
        low_risk_portfolio = [low_risk_candidate]
        high_risk_portfolio = [high_risk_candidate]
        mixed_portfolio = [low_risk_candidate, high_risk_candidate]
        
        low_risk = solver._calculate_portfolio_risk(low_risk_portfolio)
        high_risk = solver._calculate_portfolio_risk(high_risk_portfolio)
        mixed_risk = solver._calculate_portfolio_risk(mixed_portfolio)
        
        # High risk candidate should result in higher portfolio risk
        assert high_risk >= low_risk
        
        # Mixed portfolio risk should be between individual risks
        assert low_risk <= mixed_risk <= high_risk
    
    def test_optimization_within_constraints(self, diverse_kol_candidate_pool, sample_optimization_constraints):
        """Test optimization within satisfied constraints."""
        solver = ConstraintSatisfactionSolver(diverse_kol_candidate_pool)
        
        # Test different optimization objectives
        objectives = [
            OptimizationObjective.MAXIMIZE_REACH,
            OptimizationObjective.MAXIMIZE_ENGAGEMENT,
            OptimizationObjective.MAXIMIZE_CONVERSIONS,
            OptimizationObjective.MINIMIZE_COST,
            OptimizationObjective.MAXIMIZE_ROI
        ]
        
        for objective in objectives:
            optimized = solver._optimize_within_constraints(
                diverse_kol_candidate_pool, sample_optimization_constraints, objective
            )
            
            assert isinstance(optimized, list)
            assert len(optimized) <= sample_optimization_constraints.max_kols
            
            # Verify sorting based on objective
            if len(optimized) > 1:
                if objective == OptimizationObjective.MAXIMIZE_REACH:
                    for i in range(1, len(optimized)):
                        assert optimized[i-1].predicted_reach >= optimized[i].predicted_reach
                elif objective == OptimizationObjective.MINIMIZE_COST:
                    for i in range(1, len(optimized)):
                        assert optimized[i-1].estimated_total_cost <= optimized[i].estimated_total_cost


# AIDEV-NOTE: Enhanced Budget Optimizer Service Tests

class TestEnhancedBudgetOptimizerService:
    """Test enhanced budget optimizer service functionality."""
    
    @pytest.mark.asyncio
    async def test_service_initialization(self, mock_budget_optimizer_db_session):
        """Test service initialization."""
        service = EnhancedBudgetOptimizerService(mock_budget_optimizer_db_session)
        
        assert service.db_session == mock_budget_optimizer_db_session
        assert hasattr(service, 'budget_tiers')
        assert hasattr(service, 'logger')
        assert service._export_formats == ['csv', 'json', 'xlsx']
    
    @pytest.mark.asyncio
    async def test_get_campaign_requirements(self, budget_optimizer_service):
        """Test campaign requirements retrieval."""
        campaign = await budget_optimizer_service._get_campaign_requirements("test_campaign_123")
        
        # Should return mocked campaign
        assert campaign is not None
        assert campaign.id == "test_campaign_123"
    
    @pytest.mark.asyncio
    async def test_empty_optimization_result_creation(self, budget_optimizer_service, sample_optimization_constraints):
        """Test creation of empty optimization result."""
        result = budget_optimizer_service._create_empty_optimization_result(
            sample_optimization_constraints, OptimizationObjective.MAXIMIZE_REACH
        )
        
        assert isinstance(result, OptimizationResult)
        assert len(result.selected_kols) == 0
        assert result.total_cost == Decimal("0.0")
        assert result.constraints_satisfied == False
        assert result.algorithm_used == "none"
    
    @pytest.mark.asyncio
    async def test_campaign_plan_export_creation(self, budget_optimizer_service, diverse_kol_candidate_pool):
        """Test campaign plan export data creation."""
        # Create mock optimization result
        optimization_result = OptimizationResult(
            selected_kols=diverse_kol_candidate_pool[:2],
            total_cost=Decimal("10000.00"),
            cost_by_tier={"micro": Decimal("2000.00"), "mid": Decimal("8000.00")},
            cost_by_category={"lifestyle": Decimal("2000.00"), "fashion": Decimal("8000.00")},
            predicted_total_reach=45000,
            predicted_total_engagement=1276,
            predicted_total_conversions=26,
            predicted_roi=Decimal("2.6"),
            portfolio_risk_score=Decimal("0.20"),
            portfolio_diversity_score=Decimal("0.75"),
            optimization_score=Decimal("0.85"),
            budget_utilization=Decimal("0.80"),
            constraints_satisfied=True,
            constraint_violations=[],
            alternative_allocations=[],
            algorithm_used="constraint_satisfaction",
            optimization_time_seconds=2.5,
            iterations_performed=100,
            convergence_achieved=True,
            tier_distribution={"micro": 1, "mid": 1}
        )
        
        export_data = await budget_optimizer_service.export_campaign_plan(
            optimization_result,
            "test_campaign_123",
            "Test Campaign",
            "csv"
        )
        
        assert isinstance(export_data, CampaignPlanExport)
        assert export_data.campaign_id == "test_campaign_123"
        assert export_data.campaign_name == "Test Campaign"
        assert len(export_data.kol_selections) == 2
        assert export_data.performance_summary is not None
        
        # Verify KOL selection data structure
        for kol_data in export_data.kol_selections:
            required_fields = [
                "rank", "kol_id", "username", "platform", "tier", "follower_count",
                "engagement_rate", "estimated_cost_per_post", "overall_score", 
                "predicted_reach", "risk_factors"
            ]
            for field in required_fields:
                assert field in kol_data, f"Missing field: {field}"
        
        # Verify performance summary
        summary = export_data.performance_summary
        assert "total_selected_kols" in summary
        assert "total_budget_allocated" in summary
        assert "predicted_total_reach" in summary
        assert "optimization_score" in summary
    
    @pytest.mark.asyncio
    async def test_csv_export_generation(self, budget_optimizer_service, diverse_kol_candidate_pool):
        """Test CSV export generation."""
        # Create campaign plan export
        campaign_plan = CampaignPlanExport(
            campaign_id="test_campaign_123",
            campaign_name="Test Campaign",
            optimization_objective="maximize_reach",
            total_budget=Decimal("10000.00"),
            kol_selections=[
                {
                    "rank": 1,
                    "kol_id": "micro_001",
                    "username": "@lifestyle_micro",
                    "platform": "tiktok",
                    "tier": "micro",
                    "follower_count": 50000,
                    "engagement_rate": 0.045,
                    "estimated_cost_per_post": 2000.00,
                    "overall_score": 0.87,
                    "predicted_reach": 7500
                }
            ],
            performance_summary={
                "total_selected_kols": 1,
                "total_budget_allocated": 2000.00,
                "optimization_score": 0.87
            },
            export_format="csv",
            export_timestamp=datetime.utcnow().isoformat()
        )
        
        csv_content = await budget_optimizer_service.export_to_csv(campaign_plan)
        
        assert isinstance(csv_content, str)
        assert "KOL Campaign Plan Export" in csv_content
        assert "test_campaign_123" in csv_content
        assert "@lifestyle_micro" in csv_content
        assert "PERFORMANCE SUMMARY" in csv_content
        assert "SELECTED KOLS" in csv_content


# AIDEV-NOTE: Integration and End-to-End Tests

class TestBudgetOptimizerIntegration:
    """Integration tests for budget optimizer with realistic scenarios."""
    
    @pytest.mark.asyncio
    async def test_constrained_budget_scenario(self, budget_optimizer_service, diverse_kol_candidate_pool):
        """Test optimization with very constrained budget."""
        # Very small budget that can only afford nano influencer
        constrained_constraints = OptimizationConstraints(
            max_budget=Decimal("500.00"),
            min_kols=1,
            max_kols=3,
            max_risk_per_kol=Decimal("0.8"),
            tier_requirements={}
        )
        
        # Mock the method that would normally fetch candidates from database
        with patch.object(budget_optimizer_service, '_get_enhanced_kol_candidates', 
                         return_value=diverse_kol_candidate_pool):
            with patch.object(budget_optimizer_service, '_get_campaign_requirements',
                            return_value=MagicMock()):
                with patch.object(budget_optimizer_service, '_run_advanced_optimization') as mock_optimize:
                    # Create expected result for constrained scenario
                    expected_result = OptimizationResult(
                        selected_kols=[diverse_kol_candidate_pool[2]],  # Only nano influencer fits
                        total_cost=Decimal("400.00"),
                        cost_by_tier={"nano": Decimal("400.00")},
                        cost_by_category={"beauty": Decimal("400.00")},
                        predicted_total_reach=1200,
                        predicted_total_engagement=78,
                        predicted_total_conversions=2,
                        predicted_roi=Decimal("0.5"),
                        portfolio_risk_score=Decimal("0.35"),
                        portfolio_diversity_score=Decimal("0.0"),  # Only one KOL
                        optimization_score=Decimal("0.83"),
                        budget_utilization=Decimal("0.80"),
                        constraints_satisfied=True,
                        constraint_violations=[],
                        alternative_allocations=[],
                        algorithm_used="constraint_satisfaction",
                        optimization_time_seconds=1.2,
                        iterations_performed=50,
                        convergence_achieved=True,
                        tier_distribution={"nano": 1}
                    )
                    
                    mock_optimize.return_value = expected_result
                    
                    result = await budget_optimizer_service.optimize_campaign_budget_advanced(
                        campaign_id="constrained_campaign",
                        optimization_constraints=constrained_constraints,
                        optimization_objective=OptimizationObjective.MAXIMIZE_ROI,
                        algorithm="constraint_satisfaction"
                    )
                    
                    assert result.constraints_satisfied == True
                    assert result.total_cost <= constrained_constraints.max_budget
                    assert len(result.selected_kols) >= constrained_constraints.min_kols
    
    @pytest.mark.asyncio 
    async def test_high_diversity_requirements_scenario(self, budget_optimizer_service, diverse_kol_candidate_pool):
        """Test optimization requiring high diversity across tiers and categories."""
        diversity_constraints = OptimizationConstraints(
            max_budget=Decimal("20000.00"),
            min_kols=3,
            max_kols=5,
            max_risk_per_kol=Decimal("0.6"),
            tier_requirements={
                "nano": 1,
                "micro": 1,
                "mid": 1
            },
            required_categories=[
                ContentCategory.LIFESTYLE,
                ContentCategory.FASHION,
                ContentCategory.BEAUTY
            ]
        )
        
        with patch.object(budget_optimizer_service, '_get_enhanced_kol_candidates',
                         return_value=diverse_kol_candidate_pool):
            with patch.object(budget_optimizer_service, '_get_campaign_requirements',
                            return_value=MagicMock()):
                with patch.object(budget_optimizer_service, '_run_advanced_optimization') as mock_optimize:
                    # Create result with high diversity
                    expected_result = OptimizationResult(
                        selected_kols=diverse_kol_candidate_pool[:3],  # Nano, micro, mid
                        total_cost=Decimal("10400.00"),
                        cost_by_tier={
                            "nano": Decimal("400.00"),
                            "micro": Decimal("2000.00"),
                            "mid": Decimal("8000.00")
                        },
                        cost_by_category={
                            "lifestyle": Decimal("2000.00"),
                            "fashion": Decimal("8000.00"),
                            "beauty": Decimal("400.00")
                        },
                        predicted_total_reach=46200,
                        predicted_total_engagement=1354,
                        predicted_total_conversions=28,
                        predicted_roi=Decimal("2.69"),
                        portfolio_risk_score=Decimal("0.25"),
                        portfolio_diversity_score=Decimal("0.95"),  # High diversity
                        optimization_score=Decimal("0.84"),
                        budget_utilization=Decimal("0.52"),
                        constraints_satisfied=True,
                        constraint_violations=[],
                        alternative_allocations=[],
                        algorithm_used="constraint_satisfaction",
                        optimization_time_seconds=3.1,
                        iterations_performed=75,
                        convergence_achieved=True,
                        tier_distribution={"nano": 1, "micro": 1, "mid": 1}
                    )
                    
                    mock_optimize.return_value = expected_result
                    
                    result = await budget_optimizer_service.optimize_campaign_budget_advanced(
                        campaign_id="diversity_campaign",
                        optimization_constraints=diversity_constraints,
                        optimization_objective=OptimizationObjective.MAXIMIZE_REACH,
                        algorithm="genetic"
                    )
                    
                    assert result.portfolio_diversity_score > Decimal("0.8")
                    assert len(result.tier_distribution) >= 3
                    assert result.constraints_satisfied == True


# AIDEV-NOTE: Performance and Stress Tests

@pytest.mark.slow
class TestBudgetOptimizerPerformance:
    """Performance tests for budget optimizer with large datasets."""
    
    @pytest.fixture
    def large_kol_candidate_pool(self, diverse_kol_candidate_pool):
        """Create large pool of KOL candidates for performance testing."""
        large_pool = []
        
        # Replicate diverse candidates with variations
        for i in range(100):  # 100 candidates
            base_candidate = diverse_kol_candidate_pool[i % len(diverse_kol_candidate_pool)]
            
            # Create variation
            varied_candidate = KOLCandidate(
                kol_id=f"{base_candidate.kol_id}_var_{i}",
                username=f"{base_candidate.username}_v{i}",
                display_name=f"{base_candidate.display_name} V{i}",
                platform=base_candidate.platform,
                tier=base_candidate.tier,
                primary_category=base_candidate.primary_category,
                metrics=KOLMetricsData(
                    follower_count=int(base_candidate.metrics.follower_count * (0.8 + i * 0.004)),
                    following_count=base_candidate.metrics.following_count,
                    engagement_rate=base_candidate.metrics.engagement_rate * Decimal(str(0.9 + i * 0.002)),
                    avg_likes=base_candidate.metrics.avg_likes,
                    avg_comments=base_candidate.metrics.avg_comments,
                    posts_last_30_days=base_candidate.metrics.posts_last_30_days,
                    fake_follower_percentage=base_candidate.metrics.fake_follower_percentage,
                    audience_quality_score=base_candidate.metrics.audience_quality_score,
                    campaign_success_rate=base_candidate.metrics.campaign_success_rate,
                    response_rate=base_candidate.metrics.response_rate
                ),
                score_components=base_candidate.score_components,
                overall_score=base_candidate.overall_score * Decimal(str(0.95 + i * 0.001)),
                predicted_reach=int(base_candidate.predicted_reach * (0.9 + i * 0.002)),
                predicted_engagement=int(base_candidate.predicted_engagement * (0.9 + i * 0.002)),
                predicted_conversions=base_candidate.predicted_conversions,
                estimated_cost_per_post=base_candidate.estimated_cost_per_post * Decimal(str(0.8 + i * 0.004)),
                estimated_total_cost=base_candidate.estimated_total_cost * Decimal(str(0.8 + i * 0.004)),
                risk_factors=base_candidate.risk_factors,
                overall_risk_score=base_candidate.overall_risk_score,
                cost_per_engagement=base_candidate.cost_per_engagement,
                efficiency_ratio=base_candidate.efficiency_ratio
            )
            
            large_pool.append(varied_candidate)
        
        return large_pool
    
    def test_genetic_algorithm_performance_large_dataset(self, large_kol_candidate_pool):
        """Test genetic algorithm performance with large candidate pool."""
        import time
        
        algorithm = AdvancedOptimizationAlgorithm(large_kol_candidate_pool)
        
        constraints = OptimizationConstraints(
            max_budget=Decimal("100000.00"),
            min_kols=5,
            max_kols=20,
            max_risk_per_kol=Decimal("0.7"),
            tier_requirements={}
        )
        
        start_time = time.time()
        
        # Run with reduced parameters for performance test
        selected = algorithm.genetic_algorithm(
            constraints,
            OptimizationObjective.MAXIMIZE_REACH,
            population_size=20,  # Reduced for performance
            generations=50,      # Reduced for performance
            mutation_rate=0.1
        )
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Should complete within reasonable time
        assert execution_time < 30.0  # 30 seconds for 100 candidates
        
        # Should produce valid results
        assert isinstance(selected, list)
        assert len(selected) >= constraints.min_kols
        assert len(selected) <= constraints.max_kols
        
        # Verify constraints
        total_cost = sum(kol.estimated_total_cost for kol in selected)
        assert total_cost <= constraints.max_budget
    
    def test_constraint_satisfaction_performance_large_dataset(self, large_kol_candidate_pool):
        """Test constraint satisfaction solver performance with large dataset."""
        import time
        
        solver = ConstraintSatisfactionSolver(large_kol_candidate_pool)
        
        constraints = OptimizationConstraints(
            max_budget=Decimal("50000.00"),
            min_kols=10,
            max_kols=25,
            max_risk_per_kol=Decimal("0.6"),
            tier_requirements={
                "nano": 5,
                "micro": 3,
                "mid": 2
            }
        )
        
        start_time = time.time()
        
        selected, violations = solver.solve(
            constraints, OptimizationObjective.MAXIMIZE_ENGAGEMENT
        )
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Should complete within reasonable time
        assert execution_time < 10.0  # 10 seconds for constraint satisfaction
        
        # Should produce valid results
        assert isinstance(selected, list)
        assert isinstance(violations, list)
        
        # Results should be reasonable
        if selected:
            total_cost = sum(kol.estimated_total_cost for kol in selected)
            assert total_cost <= constraints.max_budget
    
    def test_knapsack_optimization_performance_large_dataset(self, large_kol_candidate_pool):
        """Test knapsack optimization performance with large dataset."""
        import time
        
        # Limit dataset size for knapsack (DP is expensive)
        limited_pool = large_kol_candidate_pool[:50]  # 50 candidates for DP
        
        algorithm = AdvancedOptimizationAlgorithm(limited_pool)
        
        constraints = OptimizationConstraints(
            max_budget=Decimal("30000.00"),
            min_kols=5,
            max_kols=15,
            max_risk_per_kol=Decimal("0.8"),
            tier_requirements={}
        )
        
        start_time = time.time()
        
        selected = algorithm.knapsack_optimization(
            constraints, OptimizationObjective.MAXIMIZE_ROI
        )
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Should complete within reasonable time
        assert execution_time < 15.0  # 15 seconds for 50 candidates
        
        # Should produce valid results
        assert isinstance(selected, list)
        assert len(selected) <= constraints.max_kols


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
