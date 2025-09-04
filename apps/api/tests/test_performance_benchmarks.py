"""
Performance and Benchmark Tests for KOL Platform Algorithms

AIDEV-NOTE: Comprehensive performance testing for scoring algorithms, constraint filtering,
budget optimization, and end-to-end workflows to ensure production scalability.
"""
import pytest
import asyncio
import time
import memory_profiler
import gc
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import List, Dict, Any, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock, patch
from decimal import Decimal
import statistics
from dataclasses import asdict

import numpy as np
import psutil
import threading

from kol_api.services.scoring.kol_scorer import KOLScorer
from kol_api.services.enhanced_budget_optimizer import (
    EnhancedBudgetOptimizerService, AdvancedOptimizationAlgorithm, ConstraintSatisfactionSolver
)
from kol_api.services.models import (
    OptimizationConstraints, OptimizationObjective, KOLCandidate,
    KOLTier, KOLMetricsData, ScoreComponents, ContentCategory
)
from tests.fixtures.test_data_factory import TestScenarioFactory, generate_performance_test_data


# AIDEV-NOTE: Performance Test Configuration

class PerformanceTestConfig:
    """Configuration for performance testing."""
    
    # Dataset sizes for different performance tiers
    SMALL_DATASET = 50
    MEDIUM_DATASET = 200
    LARGE_DATASET = 1000
    XLARGE_DATASET = 5000
    
    # Performance thresholds (in seconds)
    SCORING_THRESHOLD_PER_KOL = 0.1      # 100ms per KOL scoring
    OPTIMIZATION_THRESHOLD_SMALL = 5.0    # 5s for small dataset
    OPTIMIZATION_THRESHOLD_MEDIUM = 30.0  # 30s for medium dataset
    OPTIMIZATION_THRESHOLD_LARGE = 120.0  # 2min for large dataset
    
    # Memory thresholds (in MB)
    MEMORY_THRESHOLD_PER_KOL = 1.0       # 1MB per KOL
    MAX_MEMORY_USAGE = 1024              # 1GB max memory usage
    
    # Concurrency levels
    LOW_CONCURRENCY = 2
    MEDIUM_CONCURRENCY = 5
    HIGH_CONCURRENCY = 10


# AIDEV-NOTE: Performance Measurement Utilities

class PerformanceProfiler:
    """Utility class for performance profiling and measurement."""
    
    def __init__(self):
        self.measurements = {}
        self.start_times = {}
        self.memory_usage = {}
    
    def start_timing(self, operation_name: str):
        """Start timing an operation."""
        self.start_times[operation_name] = time.time()
        
        # Record initial memory usage
        process = psutil.Process()
        self.memory_usage[f"{operation_name}_start"] = process.memory_info().rss / 1024 / 1024  # MB
    
    def end_timing(self, operation_name: str) -> float:
        """End timing an operation and return elapsed time."""
        if operation_name not in self.start_times:
            raise ValueError(f"No start time recorded for {operation_name}")
        
        elapsed = time.time() - self.start_times[operation_name]
        self.measurements[operation_name] = elapsed
        
        # Record final memory usage
        process = psutil.Process()
        self.memory_usage[f"{operation_name}_end"] = process.memory_info().rss / 1024 / 1024  # MB
        
        return elapsed
    
    def get_memory_delta(self, operation_name: str) -> float:
        """Get memory usage delta for an operation."""
        start_key = f"{operation_name}_start"
        end_key = f"{operation_name}_end"
        
        if start_key in self.memory_usage and end_key in self.memory_usage:
            return self.memory_usage[end_key] - self.memory_usage[start_key]
        return 0.0
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        return {
            "timing": self.measurements.copy(),
            "memory_deltas": {
                op: self.get_memory_delta(op) 
                for op in self.measurements.keys()
            }
        }


@pytest.fixture
def performance_profiler():
    """Performance profiler fixture."""
    profiler = PerformanceProfiler()
    yield profiler
    
    # Clean up after test
    gc.collect()


# AIDEV-NOTE: Scoring Algorithm Performance Tests

@pytest.mark.performance
class TestScoringPerformance:
    """Performance tests for KOL scoring algorithms."""
    
    @pytest.fixture
    def kol_scorer_perf(self):
        """KOL scorer for performance testing."""
        with patch('kol_api.services.scoring.kol_scorer.SentimentAnalyzer'), \
             patch('kol_api.services.scoring.kol_scorer.DemographicMatcher'):
            return KOLScorer()
    
    @pytest.mark.asyncio
    async def test_single_kol_scoring_performance(
        self, 
        kol_scorer_perf, 
        performance_profiler,
        high_quality_kol_data,
        sample_campaign
    ):
        """Test performance of scoring a single KOL."""
        kol, metrics = high_quality_kol_data
        db_session = AsyncMock()
        
        # Warm up
        await kol_scorer_perf.score_kol(kol, sample_campaign, db_session)
        
        # Measure performance
        performance_profiler.start_timing("single_kol_scoring")
        
        score_breakdown = await kol_scorer_perf.score_kol(kol, sample_campaign, db_session)
        
        elapsed = performance_profiler.end_timing("single_kol_scoring")
        
        # Performance assertions
        assert elapsed <= PerformanceTestConfig.SCORING_THRESHOLD_PER_KOL, \
            f"Single KOL scoring took {elapsed:.3f}s, exceeds threshold of {PerformanceTestConfig.SCORING_THRESHOLD_PER_KOL}s"
        
        # Memory usage assertion
        memory_delta = performance_profiler.get_memory_delta("single_kol_scoring")
        assert memory_delta <= PerformanceTestConfig.MEMORY_THRESHOLD_PER_KOL, \
            f"Single KOL scoring used {memory_delta:.2f}MB memory, exceeds threshold of {PerformanceTestConfig.MEMORY_THRESHOLD_PER_KOL}MB"
        
        # Result validity
        assert score_breakdown is not None
        assert 0.0 <= score_breakdown.composite_score <= 1.0
    
    @pytest.mark.asyncio
    async def test_batch_kol_scoring_performance(
        self,
        kol_scorer_perf,
        performance_profiler,
        sample_campaign
    ):
        """Test performance of batch KOL scoring."""
        # Generate test dataset
        kol_pool = TestScenarioFactory.create_mixed_quality_kol_pool(
            size=PerformanceTestConfig.SMALL_DATASET
        )
        
        db_session = AsyncMock()
        
        # Sequential scoring performance test
        performance_profiler.start_timing("sequential_batch_scoring")
        
        sequential_results = []
        for kol, metrics in kol_pool:
            result = await kol_scorer_perf.score_kol(kol, sample_campaign, db_session)
            sequential_results.append(result)
        
        sequential_elapsed = performance_profiler.end_timing("sequential_batch_scoring")
        
        # Parallel scoring performance test
        performance_profiler.start_timing("parallel_batch_scoring")
        
        tasks = [
            kol_scorer_perf.score_kol(kol, sample_campaign, db_session)
            for kol, metrics in kol_pool
        ]\n        
        parallel_results = await asyncio.gather(*tasks)
        
        parallel_elapsed = performance_profiler.end_timing("parallel_batch_scoring")
        
        # Performance assertions
        avg_sequential_time = sequential_elapsed / len(kol_pool)
        avg_parallel_time = parallel_elapsed / len(kol_pool)\n        
        assert avg_sequential_time <= PerformanceTestConfig.SCORING_THRESHOLD_PER_KOL * 2, \
            f"Sequential batch scoring averaged {avg_sequential_time:.3f}s per KOL"
        \n        assert avg_parallel_time <= PerformanceTestConfig.SCORING_THRESHOLD_PER_KOL, \
            f"Parallel batch scoring averaged {avg_parallel_time:.3f}s per KOL"
        \n        # Parallel should be faster than sequential
        speedup_ratio = sequential_elapsed / parallel_elapsed
        assert speedup_ratio >= 2.0, \
            f"Parallel processing only achieved {speedup_ratio:.2f}x speedup, expected at least 2x"
        
        # Results should be identical
        assert len(sequential_results) == len(parallel_results)
        for seq_result, par_result in zip(sequential_results, parallel_results):
            assert abs(seq_result.composite_score - par_result.composite_score) < 0.001
    
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_large_dataset_scoring_performance(
        self,
        kol_scorer_perf,
        performance_profiler,
        sample_campaign
    ):
        """Test performance with large dataset (stress test)."""
        # Generate large dataset
        large_pool = TestScenarioFactory.create_mixed_quality_kol_pool(
            size=PerformanceTestConfig.MEDIUM_DATASET
        )
        
        db_session = AsyncMock()
        
        # Test batch processing with chunking
        chunk_size = 20
        performance_profiler.start_timing("large_dataset_scoring")
        
        all_results = []
        for i in range(0, len(large_pool), chunk_size):
            chunk = large_pool[i:i + chunk_size]
            
            # Process chunk in parallel
            chunk_tasks = [
                kol_scorer_perf.score_kol(kol, sample_campaign, db_session)
                for kol, metrics in chunk
            ]
            
            chunk_results = await asyncio.gather(*chunk_tasks)
            all_results.extend(chunk_results)
            
            # Allow other tasks to run
            await asyncio.sleep(0.01)
        
        elapsed = performance_profiler.end_timing("large_dataset_scoring")
        
        # Performance assertions
        avg_time_per_kol = elapsed / len(large_pool)
        assert avg_time_per_kol <= PerformanceTestConfig.SCORING_THRESHOLD_PER_KOL * 1.5, \
            f"Large dataset scoring averaged {avg_time_per_kol:.3f}s per KOL"
        
        # Memory usage should be reasonable
        memory_delta = performance_profiler.get_memory_delta("large_dataset_scoring")
        expected_max_memory = len(large_pool) * PerformanceTestConfig.MEMORY_THRESHOLD_PER_KOL * 2
        assert memory_delta <= expected_max_memory, \
            f"Large dataset scoring used {memory_delta:.2f}MB, expected max {expected_max_memory:.2f}MB"
        
        # Results should be valid
        assert len(all_results) == len(large_pool)
        for result in all_results:
            assert 0.0 <= result.composite_score <= 1.0
    
    @pytest.mark.asyncio
    async def test_concurrent_scoring_performance(
        self,
        kol_scorer_perf,
        performance_profiler,
        sample_campaign
    ):
        \"\"\"Test performance under concurrent load from multiple \"clients\".\"\"\"
        
        # Generate test data for concurrent access
        kol_pool = TestScenarioFactory.create_mixed_quality_kol_pool(size=20)
        db_session = AsyncMock()
        
        async def simulate_client_requests(client_id: int, request_count: int):
            \"\"\"Simulate requests from a single client.\"\"\"
            results = []
            for i in range(request_count):
                # Select random KOL
                kol, metrics = kol_pool[i % len(kol_pool)]
                result = await kol_scorer_perf.score_kol(kol, sample_campaign, db_session)
                results.append(result)
            return results
        
        # Test with different concurrency levels
        concurrency_levels = [
            PerformanceTestConfig.LOW_CONCURRENCY,
            PerformanceTestConfig.MEDIUM_CONCURRENCY,
            PerformanceTestConfig.HIGH_CONCURRENCY
        ]
        
        for concurrency in concurrency_levels:
            performance_profiler.start_timing(f"concurrent_scoring_{concurrency}")
            
            # Create concurrent client tasks
            client_tasks = [
                simulate_client_requests(client_id, 5)  # 5 requests per client
                for client_id in range(concurrency)
            ]
            
            all_client_results = await asyncio.gather(*client_tasks)
            
            elapsed = performance_profiler.end_timing(f"concurrent_scoring_{concurrency}")
            
            # Calculate metrics
            total_requests = concurrency * 5
            avg_time_per_request = elapsed / total_requests
            requests_per_second = total_requests / elapsed
            
            # Performance assertions
            assert avg_time_per_request <= PerformanceTestConfig.SCORING_THRESHOLD_PER_KOL * 2, \
                f"Concurrent scoring (level {concurrency}) averaged {avg_time_per_request:.3f}s per request"
            
            assert requests_per_second >= 5.0, \
                f"Concurrent scoring (level {concurrency}) achieved {requests_per_second:.2f} req/s, expected >= 5 req/s"
            
            # All results should be valid
            for client_results in all_client_results:
                for result in client_results:
                    assert 0.0 <= result.composite_score <= 1.0


# AIDEV-NOTE: Budget Optimization Performance Tests

@pytest.mark.performance
class TestBudgetOptimizationPerformance:
    \"\"\"Performance tests for budget optimization algorithms.\"\"\"
    
    @pytest.fixture
    def optimization_candidates_small(self):
        \"\"\"Small candidate pool for optimization testing.\"\"\"
        return generate_performance_test_data(PerformanceTestConfig.SMALL_DATASET)
    
    @pytest.fixture
    def optimization_candidates_medium(self):
        \"\"\"Medium candidate pool for optimization testing.\"\"\"
        return generate_performance_test_data(PerformanceTestConfig.MEDIUM_DATASET)
    
    @pytest.fixture
    def optimization_candidates_large(self):
        \"\"\"Large candidate pool for optimization testing.\"\"\"
        return generate_performance_test_data(PerformanceTestConfig.LARGE_DATASET)
    
    @pytest.fixture
    def standard_constraints(self):
        \"\"\"Standard optimization constraints for testing.\"\"\"
        return OptimizationConstraints(
            max_budget=Decimal(\"50000.00\"),
            min_kols=5,
            max_kols=15,
            max_risk_per_kol=Decimal(\"0.6\"),
            tier_requirements={\"micro\": 3, \"mid\": 2}
        )
    
    def test_genetic_algorithm_performance_scaling(
        self,
        optimization_candidates_small,
        optimization_candidates_medium,
        standard_constraints,
        performance_profiler
    ):
        \"\"\"Test genetic algorithm performance scaling with dataset size.\"\"\"
        
        datasets = [
            (\"small\", optimization_candidates_small[:PerformanceTestConfig.SMALL_DATASET//2]),
            (\"medium\", optimization_candidates_medium[:PerformanceTestConfig.MEDIUM_DATASET//2])
        ]
        
        results = {}
        
        for dataset_name, candidates in datasets:
            # Convert test data to KOLCandidate objects if needed
            if isinstance(candidates[0], tuple):
                kol_candidates = []
                for kol, metrics in candidates:
                    candidate = self._convert_to_kol_candidate(kol, metrics)
                    kol_candidates.append(candidate)
            else:
                kol_candidates = candidates
            
            algorithm = AdvancedOptimizationAlgorithm(kol_candidates)
            
            # Test with reduced parameters for performance
            performance_profiler.start_timing(f\"genetic_algorithm_{dataset_name}\")
            
            selected = algorithm.genetic_algorithm(
                standard_constraints,
                OptimizationObjective.MAXIMIZE_REACH,
                population_size=20,  # Reduced for performance testing
                generations=30,      # Reduced for performance testing
                mutation_rate=0.1
            )
            
            elapsed = performance_profiler.end_timing(f\"genetic_algorithm_{dataset_name}\")
            
            results[dataset_name] = {
                \"elapsed\": elapsed,
                \"candidates_count\": len(kol_candidates),
                \"selected_count\": len(selected),
                \"time_per_candidate\": elapsed / len(kol_candidates)
            }
        
        # Performance scaling assertions
        small_result = results[\"small\"]
        medium_result = results[\"medium\"]
        
        # Should complete within reasonable time
        assert small_result[\"elapsed\"] <= PerformanceTestConfig.OPTIMIZATION_THRESHOLD_SMALL
        assert medium_result[\"elapsed\"] <= PerformanceTestConfig.OPTIMIZATION_THRESHOLD_MEDIUM
        
        # Scaling should be sub-quadratic (better than O(n²))
        candidate_ratio = medium_result[\"candidates_count\"] / small_result[\"candidates_count\"]
        time_ratio = medium_result[\"elapsed\"] / small_result[\"elapsed\"]
        
        # Time ratio should be less than candidate_ratio² (sub-quadratic scaling)
        assert time_ratio <= candidate_ratio ** 1.5, \
            f\"Poor scaling: {candidate_ratio:.2f}x candidates took {time_ratio:.2f}x time\"
    
    def test_constraint_satisfaction_performance(
        self,
        optimization_candidates_small,
        standard_constraints,
        performance_profiler
    ):
        \"\"\"Test constraint satisfaction solver performance.\"\"\"
        
        # Convert test data to KOLCandidate objects
        kol_candidates = []
        for kol, metrics in optimization_candidates_small:
            candidate = self._convert_to_kol_candidate(kol, metrics)
            kol_candidates.append(candidate)
        
        solver = ConstraintSatisfactionSolver(kol_candidates)
        
        # Test different optimization objectives for performance variation
        objectives = [
            OptimizationObjective.MAXIMIZE_REACH,
            OptimizationObjective.MAXIMIZE_ENGAGEMENT,
            OptimizationObjective.MAXIMIZE_ROI,
            OptimizationObjective.MINIMIZE_COST
        ]
        
        for objective in objectives:
            performance_profiler.start_timing(f\"constraint_satisfaction_{objective.value}\")
            
            selected, violations = solver.solve(standard_constraints, objective)
            
            elapsed = performance_profiler.end_timing(f\"constraint_satisfaction_{objective.value}\")
            
            # Performance assertions
            assert elapsed <= 5.0, \
                f\"Constraint satisfaction for {objective.value} took {elapsed:.3f}s, expected <= 5s\"
            
            # Results should be valid
            assert isinstance(selected, list)
            assert isinstance(violations, list)
            
            if selected:
                total_cost = sum(kol.estimated_total_cost for kol in selected)
                assert total_cost <= standard_constraints.max_budget
    
    @pytest.mark.slow
    def test_knapsack_optimization_performance(
        self,
        optimization_candidates_small,
        standard_constraints,
        performance_profiler
    ):
        \"\"\"Test knapsack optimization performance with different sizes.\"\"\"
        
        # Test with different dataset sizes for knapsack (DP is expensive)
        sizes_to_test = [20, 50, 100]  # Reasonable sizes for DP algorithm
        
        for size in sizes_to_test:
            # Convert and limit test data
            kol_candidates = []
            for i, (kol, metrics) in enumerate(optimization_candidates_small[:size]):
                if i >= size:
                    break
                candidate = self._convert_to_kol_candidate(kol, metrics)
                kol_candidates.append(candidate)
            
            algorithm = AdvancedOptimizationAlgorithm(kol_candidates)
            
            performance_profiler.start_timing(f\"knapsack_optimization_{size}\")
            
            selected = algorithm.knapsack_optimization(
                standard_constraints,
                OptimizationObjective.MAXIMIZE_REACH
            )
            
            elapsed = performance_profiler.end_timing(f\"knapsack_optimization_{size}\")
            
            # Performance assertions (DP is inherently slower)
            expected_max_time = (size / 20) * 5.0  # Scale expected time with size
            assert elapsed <= expected_max_time, \
                f\"Knapsack optimization for {size} candidates took {elapsed:.3f}s, expected <= {expected_max_time:.3f}s\"
            
            # Results should be valid
            assert isinstance(selected, list)
            assert len(selected) <= standard_constraints.max_kols
    
    def test_optimization_algorithm_comparison(
        self,
        optimization_candidates_small,
        standard_constraints,
        performance_profiler
    ):
        \"\"\"Compare performance of different optimization algorithms.\"\"\"
        
        # Convert test data
        kol_candidates = []
        for kol, metrics in optimization_candidates_small[:50]:  # Limit for fair comparison
            candidate = self._convert_to_kol_candidate(kol, metrics)
            kol_candidates.append(candidate)
        
        algorithm = AdvancedOptimizationAlgorithm(kol_candidates)
        
        algorithms_to_test = [
            (\"linear_programming\", algorithm.linear_programming_approximation),
            (\"knapsack\", algorithm.knapsack_optimization),
            (\"genetic_algorithm_fast\", lambda c, o: algorithm.genetic_algorithm(
                c, o, population_size=10, generations=20, mutation_rate=0.1
            ))
        ]
        
        results = {}
        
        for algorithm_name, algorithm_func in algorithms_to_test:
            performance_profiler.start_timing(f\"algorithm_{algorithm_name}\")
            
            if algorithm_name == \"genetic_algorithm_fast\":
                selected = algorithm_func(standard_constraints, OptimizationObjective.MAXIMIZE_REACH)
            else:
                selected = algorithm_func(standard_constraints, OptimizationObjective.MAXIMIZE_REACH)
            
            elapsed = performance_profiler.end_timing(f\"algorithm_{algorithm_name}\")
            
            # Calculate quality metrics
            if selected:
                total_reach = sum(kol.predicted_reach for kol in selected)
                total_cost = sum(kol.estimated_total_cost for kol in selected)
                efficiency = total_reach / float(total_cost) if total_cost > 0 else 0
            else:
                total_reach = 0
                total_cost = 0
                efficiency = 0
            
            results[algorithm_name] = {
                \"elapsed\": elapsed,
                \"selected_count\": len(selected),
                \"total_reach\": total_reach,
                \"total_cost\": float(total_cost),
                \"efficiency\": efficiency
            }
        
        # Performance comparison
        fastest_algorithm = min(results.keys(), key=lambda k: results[k][\"elapsed\"])
        slowest_algorithm = max(results.keys(), key=lambda k: results[k][\"elapsed\"])
        
        fastest_time = results[fastest_algorithm][\"elapsed\"]
        slowest_time = results[slowest_algorithm][\"elapsed\"]
        
        # Ensure reasonable performance spread
        assert slowest_time <= fastest_time * 10, \
            f\"Performance spread too large: {slowest_algorithm} ({slowest_time:.2f}s) vs {fastest_algorithm} ({fastest_time:.2f}s)\"
        
        # All algorithms should produce valid results
        for algorithm_name, result in results.items():
            assert result[\"selected_count\"] <= standard_constraints.max_kols
            assert result[\"total_cost\"] <= float(standard_constraints.max_budget)
    
    def _convert_to_kol_candidate(self, kol_mock, metrics_mock) -> KOLCandidate:
        \"\"\"Convert test data to KOLCandidate object.\"\"\"
        
        # Extract metrics data
        metrics_data = KOLMetricsData(
            follower_count=getattr(metrics_mock, 'follower_count', 10000),
            following_count=getattr(metrics_mock, 'following_count', 500),
            engagement_rate=getattr(metrics_mock, 'engagement_rate', Decimal('0.03')),
            avg_likes=getattr(metrics_mock, 'avg_likes', Decimal('300')),
            avg_comments=getattr(metrics_mock, 'avg_comments', Decimal('30')),
            posts_last_30_days=getattr(metrics_mock, 'posts_last_30_days', 15),
            fake_follower_percentage=getattr(metrics_mock, 'fake_follower_percentage', Decimal('0.1')),
            audience_quality_score=getattr(metrics_mock, 'audience_quality_score', Decimal('0.8')),
            campaign_success_rate=getattr(metrics_mock, 'campaign_success_rate', Decimal('0.85')),
            response_rate=getattr(metrics_mock, 'response_rate', Decimal('0.9'))
        )
        
        # Create score components
        score_components = ScoreComponents(
            roi_score=Decimal('0.8'),
            audience_quality_score=Decimal('0.8'),
            brand_safety_score=Decimal('0.9'),
            content_relevance_score=Decimal('0.7'),
            demographic_fit_score=Decimal('0.8'),
            reliability_score=Decimal('0.85'),
            roi_confidence=Decimal('0.85'),
            audience_confidence=Decimal('0.85'),
            brand_safety_confidence=Decimal('0.9'),
            content_relevance_confidence=Decimal('0.8'),
            demographic_confidence=Decimal('0.8'),
            reliability_confidence=Decimal('0.85'),
            overall_confidence=Decimal('0.83'),
            data_freshness_days=2
        )
        
        # Determine tier based on followers
        followers = metrics_data.follower_count
        if followers < 10000:
            tier = KOLTier.NANO
        elif followers < 100000:
            tier = KOLTier.MICRO
        elif followers < 1000000:
            tier = KOLTier.MID
        else:
            tier = KOLTier.MACRO
        
        # Estimate costs
        base_costs = {
            KOLTier.NANO: 500,
            KOLTier.MICRO: 2000,
            KOLTier.MID: 8000,
            KOLTier.MACRO: 30000
        }
        
        estimated_cost = Decimal(str(base_costs[tier]))
        
        return KOLCandidate(
            kol_id=getattr(kol_mock, 'id', f'test_{hash(str(kol_mock))}'),
            username=getattr(kol_mock, 'username', f'@test_user_{hash(str(kol_mock))}'),
            display_name=getattr(kol_mock, 'display_name', 'Test User'),
            platform=getattr(kol_mock, 'platform', 'tiktok'),
            tier=tier,
            primary_category=ContentCategory.LIFESTYLE,
            metrics=metrics_data,
            score_components=score_components,
            overall_score=Decimal('0.82'),
            predicted_reach=int(followers * 0.15),
            predicted_engagement=int(followers * float(metrics_data.engagement_rate)),
            predicted_conversions=int(followers * float(metrics_data.engagement_rate) * 0.02),
            estimated_cost_per_post=estimated_cost,
            estimated_total_cost=estimated_cost,
            risk_factors=[],
            overall_risk_score=Decimal('0.2'),
            cost_per_engagement=estimated_cost / max(1, int(followers * float(metrics_data.engagement_rate))),
            efficiency_ratio=Decimal('0.15')
        )


# AIDEV-NOTE: Memory Usage and Resource Management Tests

@pytest.mark.performance
class TestMemoryAndResourceManagement:
    \"\"\"Test memory usage and resource management under load.\"\"\"
    
    def test_memory_usage_scaling(self, performance_profiler):
        \"\"\"Test memory usage scaling with dataset size.\"\"\"
        
        sizes_to_test = [50, 100, 200, 500]
        memory_measurements = []
        
        for size in sizes_to_test:
            # Force garbage collection before measurement
            gc.collect()
            
            performance_profiler.start_timing(f\"memory_test_{size}\")
            
            # Generate test data
            kol_pool = TestScenarioFactory.create_mixed_quality_kol_pool(size=size)
            
            # Measure memory after data generation
            process = psutil.Process()
            memory_used = process.memory_info().rss / 1024 / 1024  # MB
            
            performance_profiler.end_timing(f\"memory_test_{size}\")
            
            memory_per_kol = memory_used / size
            memory_measurements.append((size, memory_used, memory_per_kol))
            
            # Clean up
            del kol_pool
            gc.collect()
        
        # Analyze memory scaling
        for size, total_memory, memory_per_kol in memory_measurements:
            # Memory per KOL should be reasonable
            assert memory_per_kol <= PerformanceTestConfig.MEMORY_THRESHOLD_PER_KOL * 5, \
                f\"Memory usage of {memory_per_kol:.2f}MB per KOL exceeds threshold for size {size}\"
            
            # Total memory should not exceed limits
            assert total_memory <= PerformanceTestConfig.MAX_MEMORY_USAGE, \
                f\"Total memory usage of {total_memory:.2f}MB exceeds limit for size {size}\"
    
    def test_memory_leak_detection(self):
        \"\"\"Test for memory leaks in repeated operations.\"\"\"
        
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # Perform repeated operations
        for iteration in range(10):
            # Generate and process data
            kol_pool = TestScenarioFactory.create_mixed_quality_kol_pool(size=50)
            
            # Simulate scoring operations
            for kol, metrics in kol_pool:
                # Simulate some processing
                _ = hash(str(kol)) + hash(str(metrics))
            
            # Clean up
            del kol_pool
            
            # Force garbage collection every few iterations
            if iteration % 3 == 0:
                gc.collect()
        
        # Final garbage collection
        gc.collect()
        
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be minimal (no significant leaks)
        assert memory_increase <= 50, \
            f\"Potential memory leak detected: {memory_increase:.2f}MB increase after repeated operations\"
    
    def test_concurrent_memory_usage(self):
        \"\"\"Test memory usage under concurrent load.\"\"\"
        
        def worker_function(worker_id: int):
            \"\"\"Worker function for concurrent testing.\"\"\"
            # Each worker processes a small dataset
            kol_pool = TestScenarioFactory.create_mixed_quality_kol_pool(size=20)
            
            # Simulate processing
            results = []
            for kol, metrics in kol_pool:
                result = {
                    'kol_id': kol.id,
                    'score': hash(str(metrics)) % 100 / 100.0
                }
                results.append(result)
            
            return results
        
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # Run concurrent workers
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(worker_function, worker_id)
                for worker_id in range(5)
            ]
            
            # Wait for completion
            results = [future.result() for future in futures]
        
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable for concurrent execution
        expected_max_increase = 5 * 20 * PerformanceTestConfig.MEMORY_THRESHOLD_PER_KOL
        assert memory_increase <= expected_max_increase * 2, \
            f\"Concurrent memory usage of {memory_increase:.2f}MB exceeds expected maximum\"
        
        # All workers should have completed successfully
        assert len(results) == 5
        for worker_results in results:
            assert len(worker_results) == 20


# AIDEV-NOTE: End-to-End Performance Tests

@pytest.mark.performance
@pytest.mark.slow
class TestEndToEndPerformance:
    \"\"\"End-to-end performance tests simulating real workflows.\"\"\"
    
    @pytest.mark.asyncio
    async def test_complete_optimization_workflow_performance(
        self,
        performance_profiler
    ):
        \"\"\"Test complete optimization workflow performance.\"\"\"
        
        # Simulate complete workflow: data generation -> scoring -> optimization -> export
        
        performance_profiler.start_timing(\"complete_workflow\")
        
        # Step 1: Generate candidate data (simulates database query)
        performance_profiler.start_timing(\"data_generation\")
        kol_pool = TestScenarioFactory.create_mixed_quality_kol_pool(size=100)
        performance_profiler.end_timing(\"data_generation\")
        
        # Step 2: Score all candidates (simulates scoring service)
        performance_profiler.start_timing(\"candidate_scoring\")
        
        with patch('kol_api.services.scoring.kol_scorer.SentimentAnalyzer'), \
             patch('kol_api.services.scoring.kol_scorer.DemographicMatcher'):
            
            scorer = KOLScorer()
            campaign = MagicMock()
            db_session = AsyncMock()
            
            # Score candidates in batches for performance
            batch_size = 10
            scored_candidates = []
            
            for i in range(0, len(kol_pool), batch_size):
                batch = kol_pool[i:i + batch_size]
                
                batch_tasks = [
                    scorer.score_kol(kol, campaign, db_session)
                    for kol, metrics in batch
                ]
                
                batch_results = await asyncio.gather(*batch_tasks)
                scored_candidates.extend(batch_results)
        
        performance_profiler.end_timing(\"candidate_scoring\")
        
        # Step 3: Optimize selection (simulates budget optimizer)
        performance_profiler.start_timing(\"optimization\")
        
        # Convert to KOL candidates for optimization
        kol_candidates = []
        for i, ((kol, metrics), score_breakdown) in enumerate(zip(kol_pool, scored_candidates)):
            candidate = KOLCandidate(
                kol_id=kol.id,
                username=kol.username,
                display_name=kol.display_name,
                platform=\"tiktok\",
                tier=KOLTier.MICRO,
                primary_category=ContentCategory.LIFESTYLE,
                metrics=KOLMetricsData(
                    follower_count=metrics.follower_count,
                    engagement_rate=metrics.engagement_rate or Decimal('0.03'),
                    follower_count=10000,
                    following_count=500,
                    avg_likes=Decimal('300'),
                    avg_comments=Decimal('30'),
                    posts_last_30_days=15,
                    fake_follower_percentage=Decimal('0.1'),
                    audience_quality_score=Decimal('0.8'),
                    campaign_success_rate=Decimal('0.85'),
                    response_rate=Decimal('0.9')
                ),
                score_components=ScoreComponents(
                    roi_score=Decimal(str(score_breakdown.roi_score)),
                    audience_quality_score=Decimal(str(score_breakdown.audience_quality_score)),
                    brand_safety_score=Decimal(str(score_breakdown.brand_safety_score)),
                    content_relevance_score=Decimal(str(score_breakdown.content_relevance_score)),
                    demographic_fit_score=Decimal(str(score_breakdown.demographic_fit_score)),
                    reliability_score=Decimal(str(score_breakdown.reliability_score)),
                    roi_confidence=Decimal(str(score_breakdown.roi_confidence)),
                    audience_confidence=Decimal(str(score_breakdown.audience_quality_confidence)),
                    brand_safety_confidence=Decimal(str(score_breakdown.brand_safety_confidence)),
                    content_relevance_confidence=Decimal(str(score_breakdown.content_relevance_confidence)),
                    demographic_confidence=Decimal(str(score_breakdown.demographic_fit_confidence)),
                    reliability_confidence=Decimal(str(score_breakdown.reliability_confidence)),
                    overall_confidence=Decimal(str(score_breakdown.overall_confidence)),
                    data_freshness_days=2
                ),
                overall_score=Decimal(str(score_breakdown.composite_score)),
                predicted_reach=5000,
                predicted_engagement=150,
                predicted_conversions=3,
                estimated_cost_per_post=Decimal('2000'),
                estimated_total_cost=Decimal('2000'),
                risk_factors=[],
                overall_risk_score=Decimal('0.2'),
                cost_per_engagement=Decimal('13.33'),
                efficiency_ratio=Decimal('0.075')
            )
            kol_candidates.append(candidate)
        
        # Run optimization
        constraints = OptimizationConstraints(
            max_budget=Decimal(\"30000.00\"),
            min_kols=5,
            max_kols=15,
            max_risk_per_kol=Decimal(\"0.6\"),
            tier_requirements={}
        )
        
        solver = ConstraintSatisfactionSolver(kol_candidates)
        selected_kols, violations = solver.solve(constraints, OptimizationObjective.MAXIMIZE_REACH)
        
        performance_profiler.end_timing(\"optimization\")
        
        # Step 4: Generate export data (simulates export service)
        performance_profiler.start_timing(\"export_generation\")
        
        export_data = {
            \"selected_kols\": len(selected_kols),
            \"total_cost\": sum(kol.estimated_total_cost for kol in selected_kols),
            \"violations\": len(violations),
            \"optimization_score\": statistics.mean([float(kol.overall_score) for kol in selected_kols]) if selected_kols else 0
        }
        
        performance_profiler.end_timing(\"export_generation\")
        
        total_elapsed = performance_profiler.end_timing(\"complete_workflow\")
        
        # Performance assertions for complete workflow
        assert total_elapsed <= 60.0, \
            f\"Complete workflow took {total_elapsed:.2f}s, expected <= 60s\"
        
        # Individual step performance assertions
        data_gen_time = performance_profiler.measurements[\"data_generation\"]
        scoring_time = performance_profiler.measurements[\"candidate_scoring\"]
        optimization_time = performance_profiler.measurements[\"optimization\"]
        export_time = performance_profiler.measurements[\"export_generation\"]
        
        assert data_gen_time <= 5.0, f\"Data generation took {data_gen_time:.2f}s\"
        assert scoring_time <= 30.0, f\"Candidate scoring took {scoring_time:.2f}s\"
        assert optimization_time <= 20.0, f\"Optimization took {optimization_time:.2f}s\"
        assert export_time <= 5.0, f\"Export generation took {export_time:.2f}s\"
        
        # Results should be valid
        assert isinstance(selected_kols, list)
        assert len(selected_kols) >= constraints.min_kols
        assert len(selected_kols) <= constraints.max_kols
        
        if selected_kols:
            total_cost = sum(kol.estimated_total_cost for kol in selected_kols)
            assert total_cost <= constraints.max_budget
    
    def test_system_resource_monitoring(self, performance_profiler):
        \"\"\"Test system resource usage during intensive operations.\"\"\"
        
        initial_cpu_percent = psutil.cpu_percent(interval=1)
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        performance_profiler.start_timing(\"resource_intensive_operation\")
        
        # Simulate resource-intensive operation
        kol_pools = []
        for i in range(5):
            # Generate multiple datasets
            pool = TestScenarioFactory.create_mixed_quality_kol_pool(size=200)
            kol_pools.append(pool)
            
            # Simulate processing
            for kol, metrics in pool:
                # CPU-intensive operation
                _ = hash(str(kol) + str(metrics) + str(i))
        
        performance_profiler.end_timing(\"resource_intensive_operation\")
        
        final_cpu_percent = psutil.cpu_percent(interval=1)
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        memory_increase = final_memory - initial_memory
        
        # Resource usage assertions
        assert memory_increase <= 500, \
            f\"Memory usage increased by {memory_increase:.2f}MB, expected <= 500MB\"
        
        # CPU usage should return to reasonable levels after operation
        assert final_cpu_percent <= 80, \
            f\"CPU usage at {final_cpu_percent}% after operation, expected <= 80%\"
        
        # Clean up
        del kol_pools
        gc.collect()


if __name__ == \"__main__\":
    pytest.main([__file__, \"-v\", \"-m\", \"performance\", \"--tb=short\"])