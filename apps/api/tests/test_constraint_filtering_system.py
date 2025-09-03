"""
Comprehensive Constraint Filtering System Tests

AIDEV-NOTE: Complete validation tests for hard/soft constraint filtering system,
ensuring budget optimization respects all constraint types correctly.
"""
import pytest
import asyncio
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import List, Dict, Any, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np

from kol_api.services.enhanced_budget_optimizer import (
    ConstraintSatisfactionSolver, AdvancedOptimizationAlgorithm
)
from kol_api.services.models import (
    KOLCandidate, KOLMetricsData, ScoreComponents, OptimizationConstraints,
    OptimizationObjective, KOLTier, ContentCategory, ConstraintViolation
)


# AIDEV-NOTE: Hard Constraint Validation Tests

class TestHardConstraintFiltering:
    """Test hard constraints that must never be violated."""
    
    @pytest.fixture
    def constraint_solver(self):
        """Constraint satisfaction solver for testing."""
        # Create sample candidates
        candidates = self._create_sample_candidates()
        return ConstraintSatisfactionSolver(candidates)
    
    @pytest.fixture
    def strict_constraints(self):
        """Strict constraints for testing hard limits."""
        return OptimizationConstraints(
            max_budget=Decimal("50000"),
            min_kols=3,
            max_kols=5,
            max_risk_per_kol=Decimal("0.4"),
            max_portfolio_risk=Decimal("0.3"),
            min_avg_engagement_rate=Decimal("0.025"),
            tier_requirements={
                "micro": 2,
                "mid": 1
            }
        )
    
    def test_budget_constraint_never_exceeded(self, constraint_solver, strict_constraints):
        """Test that budget constraint is never exceeded."""
        selected_kols, violations = constraint_solver.solve(
            strict_constraints, OptimizationObjective.MAXIMIZE_REACH
        )
        
        total_cost = sum(kol.estimated_total_cost for kol in selected_kols)
        
        # Hard constraint: never exceed budget
        assert total_cost <= strict_constraints.max_budget, \
            f"Budget exceeded: {total_cost} > {strict_constraints.max_budget}"
        
        # Check no hard budget violations reported
        budget_violations = [v for v in violations 
                           if v.constraint_type == "budget" and v.severity == "hard"]
        assert len(budget_violations) == 0, f"Hard budget violations found: {budget_violations}"
    
    def test_count_constraints_enforced(self, constraint_solver, strict_constraints):
        """Test that min/max KOL count constraints are enforced."""
        selected_kols, violations = constraint_solver.solve(
            strict_constraints, OptimizationObjective.MAXIMIZE_ENGAGEMENT
        )
        
        kol_count = len(selected_kols)
        
        # Hard constraints: respect count limits
        assert strict_constraints.min_kols <= kol_count <= strict_constraints.max_kols, \
            f"Count constraint violated: {kol_count} not in [{strict_constraints.min_kols}, {strict_constraints.max_kols}]"
        
        # Check no hard count violations
        count_violations = [v for v in violations 
                          if v.constraint_type in ["min_kols", "max_kols"] and v.severity == "hard"]
        assert len(count_violations) == 0, f"Hard count violations found: {count_violations}"
    
    def test_risk_constraints_respected(self, constraint_solver, strict_constraints):
        """Test that individual and portfolio risk constraints are respected."""
        selected_kols, violations = constraint_solver.solve(
            strict_constraints, OptimizationObjective.MINIMIZE_COST
        )
        
        if selected_kols:  # If we have any selection
            # Check individual risk constraints
            for kol in selected_kols:
                assert kol.overall_risk_score <= strict_constraints.max_risk_per_kol, \
                    f"Individual risk exceeded for {kol.kol_id}: {kol.overall_risk_score} > {strict_constraints.max_risk_per_kol}"
            
            # Check portfolio risk
            portfolio_risk = self._calculate_portfolio_risk(selected_kols)
            assert portfolio_risk <= strict_constraints.max_portfolio_risk, \
                f"Portfolio risk exceeded: {portfolio_risk} > {strict_constraints.max_portfolio_risk}"
    
    def test_engagement_rate_filtering(self, constraint_solver, strict_constraints):
        """Test minimum engagement rate constraint filtering."""
        selected_kols, violations = constraint_solver.solve(
            strict_constraints, OptimizationObjective.MAXIMIZE_CONVERSIONS
        )
        
        # All selected KOLs must meet minimum engagement rate
        for kol in selected_kols:
            if kol.metrics.engagement_rate:
                assert kol.metrics.engagement_rate >= strict_constraints.min_avg_engagement_rate, \
                    f"Engagement rate too low for {kol.kol_id}: {kol.metrics.engagement_rate} < {strict_constraints.min_avg_engagement_rate}"
    
    def test_tier_requirements_satisfaction(self, constraint_solver, strict_constraints):
        """Test that tier requirements are satisfied."""
        selected_kols, violations = constraint_solver.solve(
            strict_constraints, OptimizationObjective.BALANCED
        )
        
        # Count KOLs by tier
        tier_counts = {}
        for kol in selected_kols:
            tier = kol.tier.value
            tier_counts[tier] = tier_counts.get(tier, 0) + 1
        
        # Check tier requirements
        for tier, required_count in strict_constraints.tier_requirements.items():
            actual_count = tier_counts.get(tier, 0)
            assert actual_count >= required_count, \
                f"Insufficient {tier} KOLs: {actual_count} < {required_count}"
    
    def test_impossible_constraints_handling(self):
        """Test handling of impossible constraint combinations."""
        # Create impossible constraints
        impossible_constraints = OptimizationConstraints(
            max_budget=Decimal("1000"),  # Very small budget
            min_kols=10,                 # But require many KOLs
            max_kols=15,
            tier_requirements={"macro": 5}  # Expensive tier requirement
        )
        
        candidates = self._create_sample_candidates()
        solver = ConstraintSatisfactionSolver(candidates)
        
        selected_kols, violations = solver.solve(
            impossible_constraints, OptimizationObjective.MAXIMIZE_REACH
        )
        
        # Should return empty selection or minimal viable selection
        if not selected_kols:
            # Should have feasibility violation
            feasibility_violations = [v for v in violations 
                                    if v.constraint_type == "feasibility"]
            assert len(feasibility_violations) > 0, "Should report feasibility issues"
        else:
            # Should violate some constraints but report violations
            assert len(violations) > 0, "Should report constraint violations"
    
    def test_zero_budget_edge_case(self):
        """Test handling of zero or negative budget."""
        zero_budget_constraints = OptimizationConstraints(
            max_budget=Decimal("0"),
            min_kols=1,
            max_kols=5
        )
        
        candidates = self._create_sample_candidates()
        solver = ConstraintSatisfactionSolver(candidates)
        
        selected_kols, violations = solver.solve(
            zero_budget_constraints, OptimizationObjective.MINIMIZE_COST
        )
        
        # Should return empty selection
        assert len(selected_kols) == 0, "Should not select any KOLs with zero budget"
        
        # Should report budget constraint violation
        budget_violations = [v for v in violations if v.constraint_type == "budget"]
        assert len(budget_violations) > 0, "Should report budget constraint issues"
    
    def _create_sample_candidates(self) -> List[KOLCandidate]:
        """Create sample KOL candidates for testing."""
        from tests.fixtures.test_data_factory import KOLDataFactory
        
        candidates = []
        
        # Create diverse candidates across tiers
        tiers = ["nano", "micro", "mid", "macro"]
        for tier in tiers:
            for quality in ["low", "medium", "high"]:
                for _ in range(2):  # 2 of each combination
                    kol, metrics = KOLDataFactory.create_kol_profile(
                        tier=tier,
                        quality_level=quality,
                        data_completeness="complete"
                    )
                    
                    # Convert to KOLCandidate
                    candidate = self._convert_to_candidate(kol, metrics)
                    candidates.append(candidate)
        
        return candidates
    
    def _convert_to_candidate(self, kol, metrics) -> KOLCandidate:
        """Convert KOL profile and metrics to KOLCandidate."""
        metrics_data = KOLMetricsData(
            follower_count=metrics.follower_count,
            following_count=metrics.following_count,
            engagement_rate=metrics.engagement_rate,
            avg_likes=metrics.avg_likes or Decimal("0"),
            avg_comments=metrics.avg_comments or Decimal("0"),
            avg_views=metrics.avg_views or Decimal("0"),
            posts_last_30_days=metrics.posts_last_30_days,
            fake_follower_percentage=metrics.fake_follower_percentage,
            audience_quality_score=metrics.audience_quality_score,
            campaign_success_rate=metrics.campaign_success_rate,
            response_rate=metrics.response_rate
        )
        
        # Create sample score components based on tier and quality
        score_components = self._generate_score_components(kol.tier.value)
        
        # Calculate predicted metrics
        predicted_reach = int(metrics.follower_count * 0.15)
        predicted_engagement = int(predicted_reach * float(metrics.engagement_rate or 0.03))
        predicted_conversions = int(predicted_engagement * 0.02)
        
        return KOLCandidate(
            kol_id=kol.id,
            username=kol.username,
            display_name=kol.display_name,
            platform=kol.platform.value,
            tier=KOLTier(kol.tier.value.upper()),
            primary_category=ContentCategory(kol.primary_category.value.upper()),
            metrics=metrics_data,
            score_components=score_components,
            overall_score=Decimal(str(np.random.uniform(0.3, 0.9))),
            predicted_reach=predicted_reach,
            predicted_engagement=predicted_engagement,
            predicted_conversions=predicted_conversions,
            estimated_cost_per_post=metrics.rate_per_post or Decimal("1000"),
            estimated_total_cost=metrics.rate_per_post or Decimal("1000"),
            risk_factors=["Sample risk factor"],
            overall_risk_score=Decimal(str(np.random.uniform(0.1, 0.5)))
        )
    
    def _generate_score_components(self, tier: str) -> ScoreComponents:
        """Generate score components based on tier."""
        # Better tiers get better base scores
        tier_multipliers = {
            "nano": 0.7, "micro": 0.8, "mid": 0.85, "macro": 0.9
        }
        
        base_multiplier = tier_multipliers.get(tier, 0.75)
        
        return ScoreComponents(
            roi_score=Decimal(str(np.random.uniform(0.3, 0.9) * base_multiplier)),
            audience_quality_score=Decimal(str(np.random.uniform(0.5, 0.95) * base_multiplier)),
            brand_safety_score=Decimal(str(np.random.uniform(0.7, 1.0) * base_multiplier)),
            content_relevance_score=Decimal(str(np.random.uniform(0.4, 0.8) * base_multiplier)),
            demographic_fit_score=Decimal(str(np.random.uniform(0.5, 0.85) * base_multiplier)),
            reliability_score=Decimal(str(np.random.uniform(0.6, 0.9) * base_multiplier)),
            roi_confidence=Decimal("0.8"),
            audience_confidence=Decimal("0.85"),
            brand_safety_confidence=Decimal("0.9"),
            content_relevance_confidence=Decimal("0.75"),
            demographic_confidence=Decimal("0.8"),
            reliability_confidence=Decimal("0.85"),
            overall_confidence=Decimal("0.82"),
            data_freshness_days=np.random.randint(1, 30)
        )
    
    def _calculate_portfolio_risk(self, selected_kols: List[KOLCandidate]) -> Decimal:
        """Calculate portfolio risk for testing."""
        if not selected_kols:
            return Decimal("0.0")
        
        total_cost = sum(kol.estimated_total_cost for kol in selected_kols)
        
        if total_cost == 0:
            return Decimal("0.0")
        
        weighted_risk = sum(
            kol.overall_risk_score * (kol.estimated_total_cost / total_cost)
            for kol in selected_kols
        )
        
        return weighted_risk


# AIDEV-NOTE: Soft Constraint Validation Tests

class TestSoftConstraintHandling:
    """Test soft constraints that can be violated with penalties."""
    
    @pytest.fixture
    def soft_constraint_solver(self):
        """Constraint solver with soft constraints."""
        candidates = self._create_performance_candidates()
        return ConstraintSatisfactionSolver(candidates)
    
    @pytest.fixture
    def soft_constraints(self):
        """Constraints with soft performance requirements."""
        return OptimizationConstraints(
            max_budget=Decimal("100000"),
            min_kols=3,
            max_kols=10,
            max_risk_per_kol=Decimal("0.6"),
            # Soft constraints
            min_total_reach=100000,        # Desired reach
            min_total_engagement=5000,     # Desired engagement
            target_portfolio_diversity=Decimal("0.7")  # Desired diversity
        )
    
    def test_soft_constraint_optimization_tradeoffs(self, soft_constraint_solver, soft_constraints):
        """Test that soft constraints create optimization tradeoffs."""
        # Run optimization with different objectives
        objectives = [
            OptimizationObjective.MAXIMIZE_REACH,
            OptimizationObjective.MAXIMIZE_ENGAGEMENT,
            OptimizationObjective.MINIMIZE_COST
        ]
        
        results = {}
        for objective in objectives:
            selected_kols, violations = soft_constraint_solver.solve(
                soft_constraints, objective
            )
            
            results[objective] = {
                "kols": selected_kols,
                "violations": violations,
                "total_reach": sum(kol.predicted_reach for kol in selected_kols),
                "total_engagement": sum(kol.predicted_engagement for kol in selected_kols),
                "total_cost": sum(kol.estimated_total_cost for kol in selected_kols)
            }
        
        # MAXIMIZE_REACH should achieve highest reach
        reach_result = results[OptimizationObjective.MAXIMIZE_REACH]
        engagement_result = results[OptimizationObjective.MAXIMIZE_ENGAGEMENT]
        cost_result = results[OptimizationObjective.MINIMIZE_COST]
        
        # Verify optimization objectives work
        assert reach_result["total_reach"] >= engagement_result["total_reach"] * 0.8, \
            "MAXIMIZE_REACH should achieve competitive reach"
        
        assert engagement_result["total_engagement"] >= reach_result["total_engagement"] * 0.8, \
            "MAXIMIZE_ENGAGEMENT should achieve competitive engagement"
        
        assert cost_result["total_cost"] <= reach_result["total_cost"], \
            "MINIMIZE_COST should achieve lower cost"
    
    def test_soft_constraint_violation_reporting(self, soft_constraint_solver, soft_constraints):
        """Test that soft constraint violations are properly reported."""
        selected_kols, violations = soft_constraint_solver.solve(
            soft_constraints, OptimizationObjective.MINIMIZE_COST
        )
        
        # Check violation reporting
        soft_violations = [v for v in violations if v.severity == "soft"]
        
        # Calculate actual performance
        total_reach = sum(kol.predicted_reach for kol in selected_kols)
        total_engagement = sum(kol.predicted_engagement for kol in selected_kols)
        
        # If performance is below target, should report soft violations
        if total_reach < soft_constraints.min_total_reach:
            reach_violations = [v for v in soft_violations 
                             if v.constraint_type == "min_reach"]
            assert len(reach_violations) > 0, "Should report reach soft constraint violation"
        
        if total_engagement < soft_constraints.min_total_engagement:
            engagement_violations = [v for v in soft_violations 
                                   if v.constraint_type == "min_engagement"]
            assert len(engagement_violations) > 0, "Should report engagement soft constraint violation"
    
    def test_constraint_violation_severity_classification(self):
        """Test that constraint violations are correctly classified by severity."""
        constraints = OptimizationConstraints(
            max_budget=Decimal("10000"),     # Hard constraint (will be exceeded)
            min_kols=2,                      # Hard constraint
            max_kols=3,                      # Hard constraint
            min_total_reach=1000000,         # Soft constraint (unrealistic)
            min_total_engagement=50000       # Soft constraint (unrealistic)
        )
        
        candidates = self._create_expensive_candidates()
        solver = ConstraintSatisfactionSolver(candidates)
        
        selected_kols, violations = solver.solve(
            constraints, OptimizationObjective.MAXIMIZE_REACH
        )
        
        # Classify violations by severity
        hard_violations = [v for v in violations if v.severity == "hard"]
        soft_violations = [v for v in violations if v.severity == "soft"]
        
        # Should have both types of violations
        violation_types = {v.constraint_type for v in violations}
        
        # Budget violation should be hard
        budget_violations = [v for v in violations if v.constraint_type == "budget"]
        if budget_violations:
            assert all(v.severity == "hard" for v in budget_violations), \
                "Budget violations should be hard"
        
        # Performance violations should be soft
        performance_violations = [v for v in violations 
                                if v.constraint_type in ["min_reach", "min_engagement"]]
        if performance_violations:
            assert all(v.severity == "soft" for v in performance_violations), \
                "Performance violations should be soft"
    
    def test_constraint_penalty_scoring(self, soft_constraint_solver, soft_constraints):
        """Test that constraint violations affect optimization scoring."""
        # Create two optimization runs with different penalty weights
        
        # Run 1: Standard penalties
        selected_standard, violations_standard = soft_constraint_solver.solve(
            soft_constraints, OptimizationObjective.BALANCED
        )
        
        # Run 2: With stricter soft constraints (higher penalty)
        strict_soft_constraints = OptimizationConstraints(
            max_budget=soft_constraints.max_budget,
            min_kols=soft_constraints.min_kols,
            max_kols=soft_constraints.max_kols,
            max_risk_per_kol=soft_constraints.max_risk_per_kol,
            # Much higher soft targets (harder to satisfy)
            min_total_reach=soft_constraints.min_total_reach * 2,
            min_total_engagement=soft_constraints.min_total_engagement * 2,
            target_portfolio_diversity=Decimal("0.95")  # Very high diversity requirement
        )
        
        selected_strict, violations_strict = soft_constraint_solver.solve(
            strict_soft_constraints, OptimizationObjective.BALANCED
        )
        
        # Compare results
        standard_soft_violations = len([v for v in violations_standard if v.severity == "soft"])
        strict_soft_violations = len([v for v in violations_strict if v.severity == "soft"])
        
        # Stricter constraints should generally result in more violations
        assert strict_soft_violations >= standard_soft_violations, \
            "Stricter soft constraints should result in more violations"
    
    def test_pareto_optimality_approximation(self, soft_constraint_solver, soft_constraints):
        """Test that solutions approximate Pareto optimality across objectives."""
        # Run multiple optimizations with different objectives
        objectives = [
            OptimizationObjective.MAXIMIZE_REACH,
            OptimizationObjective.MAXIMIZE_ENGAGEMENT,
            OptimizationObjective.MAXIMIZE_CONVERSIONS,
            OptimizationObjective.MINIMIZE_COST
        ]
        
        solutions = []
        for objective in objectives:
            selected_kols, violations = soft_constraint_solver.solve(
                soft_constraints, objective
            )
            
            if selected_kols:  # Only consider valid solutions
                solution_metrics = {
                    "reach": sum(kol.predicted_reach for kol in selected_kols),
                    "engagement": sum(kol.predicted_engagement for kol in selected_kols),
                    "conversions": sum(kol.predicted_conversions for kol in selected_kols),
                    "cost": sum(kol.estimated_total_cost for kol in selected_kols),
                    "objective": objective
                }
                solutions.append(solution_metrics)
        
        # Check for Pareto optimality properties
        if len(solutions) >= 2:
            for i, solution_a in enumerate(solutions):
                for j, solution_b in enumerate(solutions):
                    if i != j:
                        # Check if solution A dominates solution B
                        dominates = (
                            solution_a["reach"] >= solution_b["reach"] and
                            solution_a["engagement"] >= solution_b["engagement"] and
                            solution_a["conversions"] >= solution_b["conversions"] and
                            solution_a["cost"] <= solution_b["cost"]
                        )
                        
                        # If A dominates B, at least one metric should be strictly better
                        if dominates:
                            strictly_better = (
                                solution_a["reach"] > solution_b["reach"] or
                                solution_a["engagement"] > solution_b["engagement"] or
                                solution_a["conversions"] > solution_b["conversions"] or
                                solution_a["cost"] < solution_b["cost"]
                            )
                            
                            assert strictly_better, \
                                f"Solution {solution_a['objective']} dominates {solution_b['objective']} but no metric is strictly better"
    
    def _create_performance_candidates(self) -> List[KOLCandidate]:
        """Create candidates optimized for performance testing."""
        from tests.fixtures.test_data_factory import KOLDataFactory
        
        candidates = []
        
        # Create varied performance candidates
        performance_profiles = [
            {"tier": "micro", "quality": "high", "reach_multiplier": 1.5},
            {"tier": "mid", "quality": "medium", "reach_multiplier": 1.2},
            {"tier": "macro", "quality": "low", "reach_multiplier": 0.8},
            {"tier": "nano", "quality": "high", "reach_multiplier": 2.0},
        ]
        
        for profile in performance_profiles:
            for _ in range(3):  # Create 3 of each profile
                kol, metrics = KOLDataFactory.create_kol_profile(
                    tier=profile["tier"],
                    quality_level=profile["quality"],
                    data_completeness="complete"
                )
                
                candidate = self._convert_to_candidate(kol, metrics, profile["reach_multiplier"])
                candidates.append(candidate)
        
        return candidates
    
    def _create_expensive_candidates(self) -> List[KOLCandidate]:
        """Create expensive candidates to test budget constraints."""
        from tests.fixtures.test_data_factory import KOLDataFactory
        
        candidates = []
        
        # Create high-cost candidates
        for _ in range(5):
            kol, metrics = KOLDataFactory.create_kol_profile(
                tier="macro",
                quality_level="high",
                data_completeness="complete"
            )
            
            # Make them expensive
            metrics.rate_per_post = Decimal("15000")  # High cost
            
            candidate = self._convert_to_candidate(kol, metrics)
            candidates.append(candidate)
        
        return candidates
    
    def _convert_to_candidate(self, kol, metrics, reach_multiplier: float = 1.0) -> KOLCandidate:
        """Convert KOL profile to candidate with performance adjustments."""
        # Reuse the conversion logic from hard constraint tests
        candidate = TestHardConstraintFiltering()._convert_to_candidate(kol, metrics)
        
        # Apply performance multipliers
        candidate.predicted_reach = int(candidate.predicted_reach * reach_multiplier)
        candidate.predicted_engagement = int(candidate.predicted_engagement * reach_multiplier * 0.8)
        candidate.predicted_conversions = int(candidate.predicted_conversions * reach_multiplier * 0.6)
        
        return candidate


# AIDEV-NOTE: Combined Constraint System Tests

class TestCombinedConstraintSystem:
    """Test combined hard and soft constraint systems."""
    
    def test_constraint_priority_enforcement(self):
        """Test that hard constraints take priority over soft constraints."""
        constraints = OptimizationConstraints(
            max_budget=Decimal("20000"),        # Hard constraint
            min_kols=2,                         # Hard constraint
            max_kols=3,                         # Hard constraint
            max_risk_per_kol=Decimal("0.5"),    # Hard constraint
            min_total_reach=1000000,            # Soft constraint (unrealistic)
            min_total_engagement=100000         # Soft constraint (unrealistic)
        )
        
        from tests.fixtures.test_data_factory import TestScenarioFactory
        kol_pool = TestScenarioFactory.create_mixed_quality_kol_pool(10)
        candidates = [self._convert_kol_to_candidate(kol, metrics) 
                     for kol, metrics in kol_pool]
        
        solver = ConstraintSatisfactionSolver(candidates)
        selected_kols, violations = solver.solve(
            constraints, OptimizationObjective.MAXIMIZE_REACH
        )
        
        # Hard constraints must be satisfied even if soft constraints are violated
        if selected_kols:
            total_cost = sum(kol.estimated_total_cost for kol in selected_kols)
            assert total_cost <= constraints.max_budget, "Hard budget constraint violated"
            
            assert constraints.min_kols <= len(selected_kols) <= constraints.max_kols, \
                "Hard count constraints violated"
            
            for kol in selected_kols:
                assert kol.overall_risk_score <= constraints.max_risk_per_kol, \
                    "Hard risk constraint violated"
        
        # Soft constraints can be violated
        soft_violations = [v for v in violations if v.severity == "soft"]
        # It's acceptable to have soft violations if hard constraints are satisfied
    
    def test_constraint_relaxation_strategies(self):
        """Test constraint relaxation when no feasible solution exists."""
        # Create impossible constraint combination
        impossible_constraints = OptimizationConstraints(
            max_budget=Decimal("5000"),         # Very small budget
            min_kols=5,                         # But need many KOLs
            tier_requirements={"macro": 3},     # Expensive tier requirement
            min_avg_engagement_rate=Decimal("0.1")  # Very high engagement requirement
        )
        
        from tests.fixtures.test_data_factory import TestScenarioFactory
        kol_pool = TestScenarioFactory.create_mixed_quality_kol_pool(15)
        candidates = [self._convert_kol_to_candidate(kol, metrics) 
                     for kol, metrics in kol_pool]
        
        solver = ConstraintSatisfactionSolver(candidates)
        selected_kols, violations = solver.solve(
            impossible_constraints, OptimizationObjective.MINIMIZE_COST
        )
        
        # Should either return empty selection or report many violations
        if not selected_kols:
            # Should report feasibility issues
            feasibility_violations = [v for v in violations 
                                    if "feasibility" in v.constraint_type.lower()]
            assert len(feasibility_violations) > 0, "Should report feasibility violations"
        else:
            # Should have significant violations
            hard_violations = [v for v in violations if v.severity == "hard"]
            assert len(hard_violations) > 0, "Should report hard constraint violations"
    
    def test_constraint_satisfaction_performance(self):
        """Test constraint satisfaction solver performance with large candidate pool."""
        from tests.fixtures.test_data_factory import TestScenarioFactory
        
        # Create large candidate pool
        large_kol_pool = TestScenarioFactory.create_mixed_quality_kol_pool(100)
        candidates = [self._convert_kol_to_candidate(kol, metrics) 
                     for kol, metrics in large_kol_pool[:50]]  # Limit for test performance
        
        constraints = OptimizationConstraints(
            max_budget=Decimal("100000"),
            min_kols=5,
            max_kols=15,
            tier_requirements={"micro": 3, "mid": 2}
        )
        
        solver = ConstraintSatisfactionSolver(candidates)
        
        # Time the constraint satisfaction
        import time
        start_time = time.time()
        
        selected_kols, violations = solver.solve(
            constraints, OptimizationObjective.BALANCED
        )
        
        end_time = time.time()
        solve_time = end_time - start_time
        
        # Performance requirement: should complete within reasonable time
        assert solve_time < 10.0, f"Constraint satisfaction too slow: {solve_time}s"
        
        # Should find a reasonable solution
        if len(candidates) >= constraints.min_kols:
            assert len(selected_kols) >= constraints.min_kols, \
                "Should find minimum required KOLs with large candidate pool"
    
    def _convert_kol_to_candidate(self, kol, metrics) -> KOLCandidate:
        """Convert KOL profile to candidate for testing."""
        return TestHardConstraintFiltering()._convert_to_candidate(kol, metrics)


# AIDEV-NOTE: Test Execution and Reporting

def run_constraint_filtering_tests():
    """Run all constraint filtering tests and generate report."""
    import subprocess
    import sys
    
    # Run pytest on this file
    result = subprocess.run([
        sys.executable, "-m", "pytest", 
        __file__, 
        "-v", 
        "--tb=short",
        "--durations=10"
    ], capture_output=True, text=True)
    
    print("Constraint Filtering Test Results:")
    print("=" * 50)
    print(result.stdout)
    
    if result.stderr:
        print("Errors:")
        print(result.stderr)
    
    return result.returncode == 0


if __name__ == "__main__":
    # Run tests when executed directly
    success = run_constraint_filtering_tests()
    
    if success:
        print("All constraint filtering tests passed!")
    else:
        print("Some constraint filtering tests failed.")
        exit(1)