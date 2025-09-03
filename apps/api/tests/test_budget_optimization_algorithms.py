"""
Comprehensive Budget Optimization Algorithm Tests

AIDEV-NOTE: Complete validation tests for budget optimization algorithms including
genetic algorithm, linear programming, knapsack optimization, and Pareto optimality.
"""
import pytest
import asyncio
import time
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import List, Dict, Any, Optional, Tuple, Set
from unittest.mock import AsyncMock, MagicMock, patch
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for testing
import matplotlib.pyplot as plt

from kol_api.services.enhanced_budget_optimizer import (
    AdvancedOptimizationAlgorithm, EnhancedBudgetOptimizerService
)
from kol_api.services.models import (
    KOLCandidate, KOLMetricsData, ScoreComponents, OptimizationConstraints,
    OptimizationObjective, KOLTier, ContentCategory, OptimizationResult
)


# AIDEV-NOTE: Genetic Algorithm Tests

class TestGeneticAlgorithmOptimization:
    """Test genetic algorithm for complex KOL selection optimization."""
    
    @pytest.fixture
    def diverse_candidates(self):
        """Create diverse candidate pool for genetic algorithm testing."""
        return self._create_diverse_candidate_pool(30)
    
    @pytest.fixture
    def ga_constraints(self):
        """Constraints suitable for genetic algorithm optimization."""
        return OptimizationConstraints(
            max_budget=Decimal("100000"),
            min_kols=5,
            max_kols=12,
            tier_requirements={
                "micro": 3,
                "mid": 2,
                "macro": 1
            },
            max_risk_per_kol=Decimal("0.5"),
            max_portfolio_risk=Decimal("0.4")
        )
    
    def test_genetic_algorithm_convergence(self, diverse_candidates, ga_constraints):
        """Test that genetic algorithm converges to better solutions over generations."""
        algorithm = AdvancedOptimizationAlgorithm(diverse_candidates)
        
        # Track fitness evolution across multiple runs
        fitness_histories = []
        
        for run in range(3):  # Multiple runs to verify consistency
            # Run with fitness tracking
            selected_kols = algorithm.genetic_algorithm(
                ga_constraints,
                OptimizationObjective.MAXIMIZE_REACH,
                population_size=20,
                generations=25,
                mutation_rate=0.1
            )
            
            # Calculate final fitness
            final_fitness = self._calculate_solution_fitness(
                selected_kols, OptimizationObjective.MAXIMIZE_REACH, ga_constraints
            )
            fitness_histories.append(final_fitness)
        
        # All runs should produce valid solutions
        for fitness in fitness_histories:
            assert fitness > 0, "Genetic algorithm should find valid solutions"
        
        # Average fitness should be reasonable
        avg_fitness = np.mean(fitness_histories)
        assert avg_fitness > 0.5, f"Average GA fitness too low: {avg_fitness}"
    
    def test_genetic_algorithm_constraint_satisfaction(self, diverse_candidates, ga_constraints):
        """Test that genetic algorithm respects constraints."""
        algorithm = AdvancedOptimizationAlgorithm(diverse_candidates)
        
        selected_kols = algorithm.genetic_algorithm(
            ga_constraints,
            OptimizationObjective.MAXIMIZE_ENGAGEMENT,
            population_size=30,
            generations=20
        )
        
        # Validate constraint satisfaction
        violations = self._check_constraint_violations(selected_kols, ga_constraints)
        
        # Should have minimal constraint violations
        hard_violations = [v for v in violations if "hard" in v.lower()]
        assert len(hard_violations) <= 2, f"Too many hard constraint violations: {hard_violations}"
        
        # Check specific constraints
        if selected_kols:
            total_cost = sum(kol.estimated_total_cost for kol in selected_kols)
            assert total_cost <= ga_constraints.max_budget * Decimal("1.1"), \
                f"Budget significantly exceeded: {total_cost} > {ga_constraints.max_budget}"
    
    def test_genetic_algorithm_population_diversity(self, diverse_candidates, ga_constraints):
        """Test that genetic algorithm maintains population diversity."""
        algorithm = AdvancedOptimizationAlgorithm(diverse_candidates)
        
        # Run multiple times to check solution diversity
        solutions = []
        for _ in range(5):
            selected_kols = algorithm.genetic_algorithm(
                ga_constraints,
                OptimizationObjective.BALANCED,
                population_size=25,
                generations=15,
                mutation_rate=0.15  # Higher mutation for diversity
            )
            
            # Convert solution to set of KOL IDs for comparison
            solution_ids = {kol.kol_id for kol in selected_kols}
            solutions.append(solution_ids)
        
        # Check solution diversity
        if len(solutions) > 1:
            diversity_scores = []
            for i, sol_a in enumerate(solutions):
                for j, sol_b in enumerate(solutions[i+1:], i+1):
                    # Calculate Jaccard similarity
                    intersection = len(sol_a.intersection(sol_b))
                    union = len(sol_a.union(sol_b))
                    similarity = intersection / union if union > 0 else 0
                    diversity_scores.append(1 - similarity)  # Diversity = 1 - similarity
            
            avg_diversity = np.mean(diversity_scores)
            assert avg_diversity > 0.3, f"Insufficient solution diversity: {avg_diversity}"
    
    def test_genetic_algorithm_parameter_sensitivity(self, diverse_candidates, ga_constraints):
        """Test genetic algorithm sensitivity to parameter changes."""
        algorithm = AdvancedOptimizationAlgorithm(diverse_candidates)
        
        # Test different parameter combinations
        parameter_sets = [
            {"population_size": 20, "generations": 20, "mutation_rate": 0.1},
            {"population_size": 40, "generations": 15, "mutation_rate": 0.05},
            {"population_size": 15, "generations": 30, "mutation_rate": 0.2}
        ]
        
        results = []
        for params in parameter_sets:
            start_time = time.time()
            
            selected_kols = algorithm.genetic_algorithm(
                ga_constraints,
                OptimizationObjective.MAXIMIZE_CONVERSIONS,
                **params
            )
            
            execution_time = time.time() - start_time
            
            # Calculate solution quality
            total_conversions = sum(kol.predicted_conversions for kol in selected_kols)
            total_cost = sum(kol.estimated_total_cost for kol in selected_kols)
            
            results.append({
                "params": params,
                "conversions": total_conversions,
                "cost": total_cost,
                "efficiency": total_conversions / float(total_cost) if total_cost > 0 else 0,
                "execution_time": execution_time
            })
        
        # All parameter sets should produce valid results
        for result in results:
            assert result["conversions"] > 0, "All parameter sets should produce valid solutions"
            assert result["execution_time"] < 60, "Execution time should be reasonable"
        
        # Different parameters should show some variation in results
        efficiencies = [r["efficiency"] for r in results]
        if len(efficiencies) > 1:
            efficiency_std = np.std(efficiencies)
            assert efficiency_std > 0.001, "Parameter changes should affect solution quality"


# AIDEV-NOTE: Linear Programming and Knapsack Algorithm Tests

class TestMathematicalOptimizationAlgorithms:
    """Test linear programming and knapsack optimization algorithms."""
    
    @pytest.fixture
    def optimization_candidates(self):
        """Create candidates optimized for mathematical algorithm testing."""
        return self._create_value_cost_candidates(25)
    
    @pytest.fixture
    def math_constraints(self):
        """Constraints for mathematical optimization testing."""
        return OptimizationConstraints(
            max_budget=Decimal("75000"),
            min_kols=4,
            max_kols=10,
            tier_requirements={"micro": 2, "mid": 1},
            max_risk_per_kol=Decimal("0.6")
        )
    
    def test_linear_programming_value_maximization(self, optimization_candidates, math_constraints):
        """Test linear programming approximation for value maximization."""
        algorithm = AdvancedOptimizationAlgorithm(optimization_candidates)
        
        # Test different optimization objectives
        objectives = [
            OptimizationObjective.MAXIMIZE_REACH,
            OptimizationObjective.MAXIMIZE_ENGAGEMENT,
            OptimizationObjective.MAXIMIZE_ROI
        ]
        
        results = {}
        for objective in objectives:
            selected_kols = algorithm.linear_programming_approximation(
                math_constraints, objective
            )
            
            results[objective] = {
                "kols": selected_kols,
                "total_reach": sum(kol.predicted_reach for kol in selected_kols),
                "total_engagement": sum(kol.predicted_engagement for kol in selected_kols),
                "total_roi": sum(float(kol.score_components.roi_score) for kol in selected_kols),
                "total_cost": sum(kol.estimated_total_cost for kol in selected_kols)
            }
        
        # Verify objective-specific optimization
        reach_result = results[OptimizationObjective.MAXIMIZE_REACH]
        engagement_result = results[OptimizationObjective.MAXIMIZE_ENGAGEMENT]
        roi_result = results[OptimizationObjective.MAXIMIZE_ROI]
        
        # MAXIMIZE_REACH should achieve highest reach per dollar
        if reach_result["total_cost"] > 0 and engagement_result["total_cost"] > 0:
            reach_efficiency = reach_result["total_reach"] / float(reach_result["total_cost"])
            engagement_efficiency = engagement_result["total_reach"] / float(engagement_result["total_cost"])
            
            assert reach_efficiency >= engagement_efficiency * 0.9, \
                "LP should optimize for reach efficiency when maximizing reach"
    
    def test_knapsack_optimization_efficiency(self, optimization_candidates, math_constraints):
        """Test knapsack optimization algorithm efficiency."""
        algorithm = AdvancedOptimizationAlgorithm(optimization_candidates)
        
        # Run knapsack optimization
        start_time = time.time()
        
        selected_kols = algorithm.knapsack_optimization(
            math_constraints, OptimizationObjective.MAXIMIZE_ENGAGEMENT
        )
        
        execution_time = time.time() - start_time
        
        # Performance requirements
        assert execution_time < 30, f"Knapsack optimization too slow: {execution_time}s"
        
        # Quality requirements
        if selected_kols:
            total_cost = sum(kol.estimated_total_cost for kol in selected_kols)
            total_engagement = sum(kol.predicted_engagement for kol in selected_kols)
            
            # Should utilize budget efficiently
            budget_utilization = float(total_cost) / float(math_constraints.max_budget)
            assert budget_utilization >= 0.7, f"Poor budget utilization: {budget_utilization}"
            
            # Should achieve reasonable engagement
            assert total_engagement > 0, "Should achieve positive engagement"
    
    def test_mathematical_algorithm_optimality_comparison(self, optimization_candidates, math_constraints):
        """Compare mathematical algorithms for optimality approximation."""
        algorithm = AdvancedOptimizationAlgorithm(optimization_candidates)
        
        # Run different algorithms
        algorithms = {
            "linear_programming": algorithm.linear_programming_approximation,
            "knapsack": algorithm.knapsack_optimization
        }
        
        results = {}
        for alg_name, alg_func in algorithms.items():
            start_time = time.time()
            
            selected_kols = alg_func(math_constraints, OptimizationObjective.MAXIMIZE_REACH)
            
            execution_time = time.time() - start_time
            
            results[alg_name] = {
                "selected_kols": selected_kols,
                "total_reach": sum(kol.predicted_reach for kol in selected_kols),
                "total_cost": sum(kol.estimated_total_cost for kol in selected_kols),
                "execution_time": execution_time,
                "reach_per_dollar": (sum(kol.predicted_reach for kol in selected_kols) / 
                                   float(sum(kol.estimated_total_cost for kol in selected_kols)))
                                   if selected_kols and sum(kol.estimated_total_cost for kol in selected_kols) > 0 else 0
            }
        
        # Both algorithms should produce valid results
        for alg_name, result in results.items():
            assert len(result["selected_kols"]) >= math_constraints.min_kols, \
                f"{alg_name} should satisfy minimum KOL requirement"
            
            assert result["total_cost"] <= math_constraints.max_budget, \
                f"{alg_name} should respect budget constraint"
        
        # Compare efficiency (reach per dollar)
        lp_efficiency = results["linear_programming"]["reach_per_dollar"]
        kp_efficiency = results["knapsack"]["reach_per_dollar"]
        
        # Both should be reasonably efficient (within 20% of each other)
        if lp_efficiency > 0 and kp_efficiency > 0:
            efficiency_ratio = min(lp_efficiency, kp_efficiency) / max(lp_efficiency, kp_efficiency)
            assert efficiency_ratio >= 0.8, \
                f"Algorithm efficiency gap too large: LP={lp_efficiency}, KP={kp_efficiency}"
    
    def test_algorithm_scalability_performance(self):
        """Test algorithm performance with varying candidate pool sizes."""
        candidate_sizes = [10, 25, 50, 75]
        performance_results = {}
        
        constraints = OptimizationConstraints(
            max_budget=Decimal("50000"),
            min_kols=3,
            max_kols=8
        )
        
        for size in candidate_sizes:
            candidates = self._create_value_cost_candidates(size)
            algorithm = AdvancedOptimizationAlgorithm(candidates)
            
            # Time linear programming
            start_time = time.time()
            lp_result = algorithm.linear_programming_approximation(
                constraints, OptimizationObjective.MAXIMIZE_REACH
            )
            lp_time = time.time() - start_time
            
            # Time knapsack
            start_time = time.time()
            kp_result = algorithm.knapsack_optimization(
                constraints, OptimizationObjective.MAXIMIZE_REACH
            )
            kp_time = time.time() - start_time
            
            performance_results[size] = {
                "lp_time": lp_time,
                "kp_time": kp_time,
                "lp_solution_size": len(lp_result),
                "kp_solution_size": len(kp_result)
            }
        
        # Verify reasonable scaling
        for size, result in performance_results.items():
            assert result["lp_time"] < 10.0, \
                f"LP too slow for {size} candidates: {result['lp_time']}s"
            assert result["kp_time"] < 15.0, \
                f"KP too slow for {size} candidates: {result['kp_time']}s"
        
        # Check that execution time scales reasonably
        times_lp = [performance_results[size]["lp_time"] for size in candidate_sizes]
        times_kp = [performance_results[size]["kp_time"] for size in candidate_sizes]
        
        # Should not have exponential scaling (reasonable for practical use)
        max_lp_time = max(times_lp)
        min_lp_time = min(times_lp)
        scaling_factor_lp = max_lp_time / min_lp_time if min_lp_time > 0 else float('inf')
        
        assert scaling_factor_lp < 50, \
            f"LP scaling too poor: {scaling_factor_lp}x between min and max problem sizes"


# AIDEV-NOTE: Pareto Optimality and Multi-Objective Tests

class TestParetoOptimalityValidation:
    """Test Pareto optimality and multi-objective optimization properties."""
    
    @pytest.fixture
    def pareto_candidates(self):
        """Create candidates with diverse trade-offs for Pareto testing."""
        return self._create_pareto_test_candidates(40)
    
    @pytest.fixture
    def pareto_constraints(self):
        """Flexible constraints for Pareto optimality testing."""
        return OptimizationConstraints(
            max_budget=Decimal("120000"),
            min_kols=3,
            max_kols=15,
            max_risk_per_kol=Decimal("0.7")
        )
    
    def test_pareto_frontier_approximation(self, pareto_candidates, pareto_constraints):
        """Test that algorithms approximate the Pareto frontier."""
        algorithm = AdvancedOptimizationAlgorithm(pareto_candidates)
        
        # Generate solutions for different objectives
        objectives = [
            OptimizationObjective.MAXIMIZE_REACH,
            OptimizationObjective.MAXIMIZE_ENGAGEMENT,
            OptimizationObjective.MAXIMIZE_CONVERSIONS,
            OptimizationObjective.MINIMIZE_COST,
            OptimizationObjective.MAXIMIZE_ROI
        ]
        
        pareto_solutions = []
        for objective in objectives:
            selected_kols = algorithm.linear_programming_approximation(
                pareto_constraints, objective
            )
            
            if selected_kols:
                solution_metrics = {
                    "reach": sum(kol.predicted_reach for kol in selected_kols),
                    "engagement": sum(kol.predicted_engagement for kol in selected_kols),
                    "conversions": sum(kol.predicted_conversions for kol in selected_kols),
                    "cost": float(sum(kol.estimated_total_cost for kol in selected_kols)),
                    "roi": sum(float(kol.score_components.roi_score) for kol in selected_kols),
                    "objective": objective.value
                }
                pareto_solutions.append(solution_metrics)
        
        # Verify Pareto properties
        assert len(pareto_solutions) >= 3, "Should generate multiple Pareto solutions"
        
        self._validate_pareto_optimality(pareto_solutions)
    
    def test_multi_objective_trade_offs(self, pareto_candidates, pareto_constraints):
        """Test that multi-objective optimization shows expected trade-offs."""
        algorithm = AdvancedOptimizationAlgorithm(pareto_candidates)
        
        # Compare extreme objectives
        max_reach_solution = algorithm.linear_programming_approximation(
            pareto_constraints, OptimizationObjective.MAXIMIZE_REACH
        )
        
        min_cost_solution = algorithm.linear_programming_approximation(
            pareto_constraints, OptimizationObjective.MINIMIZE_COST
        )
        
        if max_reach_solution and min_cost_solution:
            # Calculate metrics for both solutions
            reach_metrics = {
                "reach": sum(kol.predicted_reach for kol in max_reach_solution),
                "cost": sum(kol.estimated_total_cost for kol in max_reach_solution)
            }
            
            cost_metrics = {
                "reach": sum(kol.predicted_reach for kol in min_cost_solution),
                "cost": sum(kol.estimated_total_cost for kol in min_cost_solution)
            }
            
            # Trade-off validation: reach maximization should achieve higher reach but potentially higher cost
            assert reach_metrics["reach"] >= cost_metrics["reach"] * 0.8, \
                "Reach maximization should achieve competitive reach"
            
            # Cost minimization should achieve lower cost
            assert cost_metrics["cost"] <= reach_metrics["cost"], \
                "Cost minimization should achieve lower cost"
    
    def test_portfolio_diversification_optimization(self, pareto_candidates, pareto_constraints):
        """Test optimization for portfolio diversification."""
        algorithm = AdvancedOptimizationAlgorithm(pareto_candidates)
        
        # Run optimization for balanced objective (should promote diversification)
        balanced_solution = algorithm.linear_programming_approximation(
            pareto_constraints, OptimizationObjective.BALANCED
        )
        
        if balanced_solution:
            # Analyze portfolio diversification
            diversity_metrics = self._calculate_portfolio_diversity(balanced_solution)
            
            # Should have reasonable diversification
            assert diversity_metrics["tier_diversity"] >= 0.5, \
                f"Poor tier diversification: {diversity_metrics['tier_diversity']}"
            
            assert diversity_metrics["category_diversity"] >= 0.4, \
                f"Poor category diversification: {diversity_metrics['category_diversity']}"
            
            # Should not be overly concentrated in single tier/category
            assert diversity_metrics["max_tier_concentration"] <= 0.7, \
                f"Over-concentration in single tier: {diversity_metrics['max_tier_concentration']}"
    
    def test_risk_return_optimization_trade_off(self, pareto_candidates, pareto_constraints):
        """Test risk-return trade-off optimization."""
        # Create risk-sensitive constraints
        low_risk_constraints = OptimizationConstraints(
            max_budget=pareto_constraints.max_budget,
            min_kols=pareto_constraints.min_kols,
            max_kols=pareto_constraints.max_kols,
            max_risk_per_kol=Decimal("0.3"),  # Low risk tolerance
            max_portfolio_risk=Decimal("0.25")
        )
        
        high_risk_constraints = OptimizationConstraints(
            max_budget=pareto_constraints.max_budget,
            min_kols=pareto_constraints.min_kols,
            max_kols=pareto_constraints.max_kols,
            max_risk_per_kol=Decimal("0.8"),  # High risk tolerance
            max_portfolio_risk=Decimal("0.7")
        )
        
        algorithm = AdvancedOptimizationAlgorithm(pareto_candidates)
        
        # Optimize with different risk tolerances
        low_risk_solution = algorithm.linear_programming_approximation(
            low_risk_constraints, OptimizationObjective.MAXIMIZE_ROI
        )
        
        high_risk_solution = algorithm.linear_programming_approximation(
            high_risk_constraints, OptimizationObjective.MAXIMIZE_ROI
        )
        
        if low_risk_solution and high_risk_solution:
            # Calculate risk and return metrics
            low_risk_metrics = self._calculate_risk_return_metrics(low_risk_solution)
            high_risk_metrics = self._calculate_risk_return_metrics(high_risk_solution)
            
            # Validate risk-return relationship
            assert low_risk_metrics["avg_risk"] <= high_risk_metrics["avg_risk"] * 1.1, \
                "Low risk strategy should have lower risk"
            
            # High risk might achieve higher returns (but not guaranteed due to constraints)
            # This is a weaker assertion since risk-return trade-offs can be complex
            risk_return_ratio_low = low_risk_metrics["avg_return"] / max(low_risk_metrics["avg_risk"], 0.01)
            risk_return_ratio_high = high_risk_metrics["avg_return"] / max(high_risk_metrics["avg_risk"], 0.01)
            
            # Both strategies should achieve positive risk-adjusted returns
            assert risk_return_ratio_low > 0, "Low risk strategy should have positive returns"
            assert risk_return_ratio_high > 0, "High risk strategy should have positive returns"
    
    def _validate_pareto_optimality(self, solutions: List[Dict[str, Any]]):
        """Validate Pareto optimality properties of solutions."""
        # Check for Pareto dominance relationships
        dominated_count = 0
        
        for i, solution_a in enumerate(solutions):
            for j, solution_b in enumerate(solutions):
                if i != j:
                    # Check if solution A dominates solution B
                    dominates = (
                        solution_a["reach"] >= solution_b["reach"] and
                        solution_a["engagement"] >= solution_b["engagement"] and
                        solution_a["conversions"] >= solution_b["conversions"] and
                        solution_a["cost"] <= solution_b["cost"] and
                        solution_a["roi"] >= solution_b["roi"]
                    )
                    
                    # If A dominates B, at least one objective should be strictly better
                    if dominates:
                        strictly_better = (
                            solution_a["reach"] > solution_b["reach"] or
                            solution_a["engagement"] > solution_b["engagement"] or
                            solution_a["conversions"] > solution_b["conversions"] or
                            solution_a["cost"] < solution_b["cost"] or
                            solution_a["roi"] > solution_b["roi"]
                        )
                        
                        if strictly_better:
                            dominated_count += 1
        
        # Should have some dominated solutions (indicating proper optimization)
        total_comparisons = len(solutions) * (len(solutions) - 1)
        domination_rate = dominated_count / total_comparisons if total_comparisons > 0 else 0
        
        # Reasonable domination rate indicates Pareto-like behavior
        assert 0.1 <= domination_rate <= 0.5, \
            f"Unexpected domination rate: {domination_rate} (should be 0.1-0.5 for good Pareto approximation)"
    
    def _calculate_portfolio_diversity(self, selected_kols: List[KOLCandidate]) -> Dict[str, float]:
        """Calculate portfolio diversity metrics."""
        if not selected_kols:
            return {"tier_diversity": 0, "category_diversity": 0, "max_tier_concentration": 1}
        
        # Tier diversity
        tier_counts = defaultdict(int)
        category_counts = defaultdict(int)
        
        for kol in selected_kols:
            tier_counts[kol.tier.value] += 1
            category_counts[kol.primary_category.value] += 1
        
        total_kols = len(selected_kols)
        
        # Calculate Shannon diversity index
        tier_diversity = self._shannon_diversity([count/total_kols for count in tier_counts.values()])
        category_diversity = self._shannon_diversity([count/total_kols for count in category_counts.values()])
        
        # Calculate maximum concentration
        max_tier_concentration = max(tier_counts.values()) / total_kols
        
        return {
            "tier_diversity": tier_diversity,
            "category_diversity": category_diversity,
            "max_tier_concentration": max_tier_concentration
        }
    
    def _shannon_diversity(self, proportions: List[float]) -> float:
        """Calculate Shannon diversity index."""
        diversity = 0
        for p in proportions:
            if p > 0:
                diversity -= p * np.log(p)
        return diversity
    
    def _calculate_risk_return_metrics(self, selected_kols: List[KOLCandidate]) -> Dict[str, float]:
        """Calculate risk and return metrics for portfolio."""
        if not selected_kols:
            return {"avg_risk": 0, "avg_return": 0}
        
        total_cost = sum(kol.estimated_total_cost for kol in selected_kols)
        
        # Calculate weighted average risk and return
        avg_risk = sum(
            float(kol.overall_risk_score) * (float(kol.estimated_total_cost) / float(total_cost))
            for kol in selected_kols
        ) if total_cost > 0 else 0
        
        avg_return = sum(
            float(kol.score_components.roi_score) * (float(kol.estimated_total_cost) / float(total_cost))
            for kol in selected_kols
        ) if total_cost > 0 else 0
        
        return {"avg_risk": avg_risk, "avg_return": avg_return}
    
    # Helper methods for creating test data
    def _create_diverse_candidate_pool(self, size: int) -> List[KOLCandidate]:
        """Create diverse candidate pool for genetic algorithm testing."""
        from tests.fixtures.test_data_factory import TestScenarioFactory
        kol_pool = TestScenarioFactory.create_mixed_quality_kol_pool(size)
        
        candidates = []
        for kol, metrics in kol_pool:
            candidate = self._convert_to_candidate(kol, metrics)
            candidates.append(candidate)
        
        return candidates
    
    def _create_value_cost_candidates(self, size: int) -> List[KOLCandidate]:
        """Create candidates with clear value/cost relationships."""
        from tests.fixtures.test_data_factory import KOLDataFactory
        
        candidates = []
        
        # Create candidates with different value propositions
        value_profiles = [
            {"tier": "nano", "engagement_mult": 2.0, "cost_mult": 0.8},    # High value nano
            {"tier": "micro", "engagement_mult": 1.5, "cost_mult": 1.0},   # Standard micro
            {"tier": "mid", "engagement_mult": 1.0, "cost_mult": 1.2},     # Standard mid
            {"tier": "macro", "engagement_mult": 0.8, "cost_mult": 1.5}    # Lower efficiency macro
        ]
        
        for _ in range(size // len(value_profiles) + 1):
            for profile in value_profiles:
                if len(candidates) >= size:
                    break
                
                kol, metrics = KOLDataFactory.create_kol_profile(
                    tier=profile["tier"],
                    quality_level="medium",
                    data_completeness="complete"
                )
                
                candidate = self._convert_to_candidate(kol, metrics)
                
                # Apply value/cost adjustments
                candidate.predicted_engagement = int(
                    candidate.predicted_engagement * profile["engagement_mult"]
                )
                candidate.estimated_total_cost = Decimal(
                    str(int(float(candidate.estimated_total_cost) * profile["cost_mult"]))
                )
                
                candidates.append(candidate)
        
        return candidates[:size]
    
    def _create_pareto_test_candidates(self, size: int) -> List[KOLCandidate]:
        """Create candidates with diverse trade-offs for Pareto testing."""
        from tests.fixtures.test_data_factory import KOLDataFactory
        
        candidates = []
        
        # Create candidates with different trade-off profiles
        trade_off_profiles = [
            {"reach_mult": 2.0, "cost_mult": 1.8, "risk_mult": 1.2},  # High reach, high cost, higher risk
            {"reach_mult": 1.0, "cost_mult": 0.6, "risk_mult": 0.8},  # Low reach, low cost, lower risk
            {"reach_mult": 1.5, "cost_mult": 1.0, "risk_mult": 0.5},  # Medium reach, medium cost, low risk
            {"reach_mult": 1.2, "cost_mult": 1.5, "risk_mult": 1.5}   # Medium reach, high cost, high risk
        ]
        
        for _ in range(size // len(trade_off_profiles) + 1):
            for profile in trade_off_profiles:
                if len(candidates) >= size:
                    break
                
                kol, metrics = KOLDataFactory.create_kol_profile(
                    quality_level="medium",
                    data_completeness="complete"
                )
                
                candidate = self._convert_to_candidate(kol, metrics)
                
                # Apply trade-off adjustments
                candidate.predicted_reach = int(candidate.predicted_reach * profile["reach_mult"])
                candidate.estimated_total_cost = Decimal(
                    str(int(float(candidate.estimated_total_cost) * profile["cost_mult"]))
                )
                candidate.overall_risk_score = Decimal(
                    str(min(1.0, float(candidate.overall_risk_score) * profile["risk_mult"]))
                )
                
                candidates.append(candidate)
        
        return candidates[:size]
    
    def _convert_to_candidate(self, kol, metrics) -> KOLCandidate:
        """Convert KOL profile to candidate."""
        # Reuse conversion logic from constraint tests
        from .test_constraint_filtering_system import TestHardConstraintFiltering
        return TestHardConstraintFiltering()._convert_to_candidate(kol, metrics)
    
    def _calculate_solution_fitness(
        self, 
        selected_kols: List[KOLCandidate], 
        objective: OptimizationObjective,
        constraints: OptimizationConstraints
    ) -> float:
        """Calculate fitness score for a solution."""
        if not selected_kols:
            return 0.0
        
        # Base fitness from objective
        if objective == OptimizationObjective.MAXIMIZE_REACH:
            base_fitness = sum(kol.predicted_reach for kol in selected_kols)
        elif objective == OptimizationObjective.MAXIMIZE_ENGAGEMENT:
            base_fitness = sum(kol.predicted_engagement for kol in selected_kols)
        elif objective == OptimizationObjective.MAXIMIZE_CONVERSIONS:
            base_fitness = sum(kol.predicted_conversions for kol in selected_kols)
        elif objective == OptimizationObjective.MINIMIZE_COST:
            total_cost = sum(kol.estimated_total_cost for kol in selected_kols)
            base_fitness = 1.0 / (1.0 + float(total_cost) / 10000)  # Inverse cost fitness
        else:  # BALANCED or MAXIMIZE_ROI
            base_fitness = sum(float(kol.overall_score) for kol in selected_kols)
        
        # Apply constraint penalties
        penalty = 0.0
        
        # Budget constraint
        total_cost = sum(kol.estimated_total_cost for kol in selected_kols)
        if total_cost > constraints.max_budget:
            penalty += float(total_cost - constraints.max_budget) * 0.001
        
        # Risk constraint
        max_risk = max(kol.overall_risk_score for kol in selected_kols)
        if max_risk > constraints.max_risk_per_kol:
            penalty += (float(max_risk) - float(constraints.max_risk_per_kol)) * 1000
        
        return max(0.0, base_fitness - penalty)
    
    def _check_constraint_violations(
        self, 
        selected_kols: List[KOLCandidate], 
        constraints: OptimizationConstraints
    ) -> List[str]:
        """Check for constraint violations."""
        violations = []
        
        if not selected_kols:
            violations.append("No KOLs selected (hard violation)")
            return violations
        
        # Budget constraint
        total_cost = sum(kol.estimated_total_cost for kol in selected_kols)
        if total_cost > constraints.max_budget:
            violations.append(f"Budget exceeded (hard violation): {total_cost} > {constraints.max_budget}")
        
        # Count constraints
        if len(selected_kols) < constraints.min_kols:
            violations.append(f"Too few KOLs (hard violation): {len(selected_kols)} < {constraints.min_kols}")
        
        if len(selected_kols) > constraints.max_kols:
            violations.append(f"Too many KOLs (hard violation): {len(selected_kols)} > {constraints.max_kols}")
        
        # Risk constraints
        for kol in selected_kols:
            if kol.overall_risk_score > constraints.max_risk_per_kol:
                violations.append(f"KOL {kol.kol_id} risk too high (hard violation): {kol.overall_risk_score}")
        
        return violations


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "--durations=20"])