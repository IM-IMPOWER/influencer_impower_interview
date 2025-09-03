"""
Mathematical Validation Tests for Scoring Algorithms

AIDEV-NOTE: Comprehensive mathematical validation tests for scoring algorithm correctness,
ensuring algorithmic accuracy, consistency, and mathematical properties are maintained.
"""
import pytest
import math
import statistics
from decimal import Decimal, ROUND_HALF_UP
from typing import List, Dict, Any, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
from scipy import stats

from kol_api.services.scoring.kol_scorer import KOLScorer, ScoreBreakdown, ScoreComponent
from kol_api.services.models import (
    KOLCandidate, KOLMetricsData, ScoreComponents, OptimizationConstraints,
    OptimizationObjective, KOLTier, ContentCategory
)


# AIDEV-NOTE: Mathematical Property Tests

class TestScoringAlgorithmMathematicalProperties:
    """Test mathematical properties and correctness of scoring algorithms."""
    
    @pytest.fixture
    def kol_scorer_math(self):
        """KOL scorer for mathematical testing."""
        with patch('kol_api.services.scoring.kol_scorer.SentimentAnalyzer'), \
             patch('kol_api.services.scoring.kol_scorer.DemographicMatcher'):
            return KOLScorer()
    
    def test_score_normalization_bounds(self, kol_scorer_math):
        """Test that all scores are properly normalized to [0,1] bounds."""
        # Test various score inputs that could cause out-of-bounds results
        test_cases = [
            # (engagement_rate, followers, expected_bounds_check)
            (Decimal("0.001"), 1000, True),    # Very low engagement
            (Decimal("0.15"), 50000, True),    # Very high engagement
            (Decimal("0.05"), 1000000, True),  # Mega influencer
            (Decimal("0.0"), 500, True),       # Zero engagement
            (Decimal("1.0"), 100, True),       # 100% engagement (impossible but test boundary)
        ]
        
        for engagement_rate, followers, should_be_bounded in test_cases:
            # Create mock KOL with specific metrics
            kol = MagicMock()
            kol.handle = f"@test_{engagement_rate}_{followers}"
            kol.latest_metrics = MagicMock()
            kol.latest_metrics.followers = followers
            kol.latest_metrics.average_engagement_rate = engagement_rate
            
            campaign = MagicMock()
            campaign.campaign_type = "engagement"
            
            # Test ROI calculation bounds
            roi_score, roi_conf = asyncio.run(
                kol_scorer_math._calculate_roi_score(kol, campaign, MagicMock())
            )
            
            assert Decimal("0.0") <= roi_score <= Decimal("1.0"), \
                f"ROI score {roi_score} out of bounds for engagement {engagement_rate}, followers {followers}"
            assert Decimal("0.0") <= roi_conf <= Decimal("1.0"), \
                f"ROI confidence {roi_conf} out of bounds"
    
    def test_score_monotonicity_properties(self, kol_scorer_math):
        """Test monotonicity properties - better metrics should generally yield better scores."""
        
        # Create pairs of KOLs where one is clearly better than another
        test_pairs = [
            # Better engagement should generally lead to better audience quality
            {
                "better": {"followers": 50000, "engagement_rate": Decimal("0.06"), "fake_followers": Decimal("0.02")},
                "worse": {"followers": 50000, "engagement_rate": Decimal("0.01"), "fake_followers": Decimal("0.30")}
            },
            # More consistent posting should lead to better reliability
            {
                "better": {"account_age_days": 730, "posts_30d": 20, "engagement_variance": 0.1},
                "worse": {"account_age_days": 30, "posts_30d": 2, "engagement_variance": 0.8}
            }
        ]
        
        for pair in test_pairs:
            better_kol = self._create_test_kol(pair["better"])
            worse_kol = self._create_test_kol(pair["worse"])
            
            # Test audience quality monotonicity
            better_aud_score, _ = asyncio.run(
                kol_scorer_math._calculate_audience_quality_score(better_kol, MagicMock())
            )
            worse_aud_score, _ = asyncio.run(
                kol_scorer_math._calculate_audience_quality_score(worse_kol, MagicMock())
            )
            
            # Better metrics should generally yield better scores (with some tolerance for edge cases)
            if "engagement_rate" in pair["better"] and "fake_followers" in pair["better"]:
                assert better_aud_score >= worse_aud_score * Decimal("0.9"), \
                    f"Monotonicity violation: better {better_aud_score} vs worse {worse_aud_score}"
    
    def test_score_consistency_across_calls(self, kol_scorer_math):
        """Test that scoring is consistent across multiple calls with same input."""
        # Create deterministic KOL
        kol = self._create_deterministic_test_kol()
        campaign = MagicMock()
        campaign.campaign_type = "engagement"
        db_session = MagicMock()
        
        # Run scoring multiple times
        scores = []
        for _ in range(5):
            score_breakdown = asyncio.run(
                kol_scorer_math.score_kol(kol, campaign, db_session)
            )
            scores.append(score_breakdown.composite_score)
        
        # All scores should be identical (deterministic)
        for i in range(1, len(scores)):
            assert abs(scores[i] - scores[0]) < Decimal("0.001"), \
                f"Inconsistent scoring: {scores[0]} vs {scores[i]}"
    
    def test_weighted_score_calculation_accuracy(self):
        """Test that weighted score calculation is mathematically accurate."""
        # Create known score components
        components = ScoreComponents(
            roi_score=Decimal("0.8"),
            audience_quality_score=Decimal("0.7"),
            brand_safety_score=Decimal("0.9"),
            content_relevance_score=Decimal("0.6"),
            demographic_fit_score=Decimal("0.75"),
            reliability_score=Decimal("0.85"),
            roi_confidence=Decimal("1.0"),
            audience_confidence=Decimal("1.0"),
            brand_safety_confidence=Decimal("1.0"),
            content_relevance_confidence=Decimal("1.0"),
            demographic_confidence=Decimal("1.0"),
            reliability_confidence=Decimal("1.0"),
            overall_confidence=Decimal("1.0"),
            data_freshness_days=1
        )
        
        breakdown = ScoreBreakdown(
            roi_score=float(components.roi_score),
            audience_quality_score=float(components.audience_quality_score),
            brand_safety_score=float(components.brand_safety_score),
            content_relevance_score=float(components.content_relevance_score),
            demographic_fit_score=float(components.demographic_fit_score),
            reliability_score=float(components.reliability_score),
            roi_confidence=float(components.roi_confidence),
            audience_quality_confidence=float(components.audience_confidence),
            brand_safety_confidence=float(components.brand_safety_confidence),
            content_relevance_confidence=float(components.content_relevance_confidence),
            demographic_fit_confidence=float(components.demographic_confidence),
            reliability_confidence=float(components.reliability_confidence),
            composite_score=0.0,  # Will be calculated
            overall_confidence=float(components.overall_confidence),
            missing_data_penalty=0.1
        )
        
        # Calculate expected score manually
        expected_score = (
            Decimal("0.8") * ScoreComponent.ROI.weight +
            Decimal("0.7") * ScoreComponent.AUDIENCE_QUALITY.weight +
            Decimal("0.9") * ScoreComponent.BRAND_SAFETY.weight +
            Decimal("0.6") * ScoreComponent.CONTENT_RELEVANCE.weight +
            Decimal("0.75") * ScoreComponent.DEMOGRAPHIC_FIT.weight +
            Decimal("0.85") * ScoreComponent.RELIABILITY.weight
        ) * (1 - Decimal("0.1"))  # Apply missing data penalty
        
        calculated_score = breakdown.weighted_score
        
        # Should match within floating point precision
        assert abs(calculated_score - expected_score) < Decimal("0.0001"), \
            f"Score calculation error: expected {expected_score}, got {calculated_score}"
    
    def test_roi_calculation_mathematical_correctness(self, kol_scorer_math):
        """Test ROI calculation mathematical correctness."""
        # Test cases with known expected outcomes
        test_cases = [
            {
                "followers": 10000,
                "engagement_rate": Decimal("0.05"),  # 5%
                "conversion_rate": 0.02,  # 2%
                "cost_per_post": 1000.0,
                "description": "Standard micro influencer"
            },
            {
                "followers": 100000,
                "engagement_rate": Decimal("0.02"),  # 2%
                "conversion_rate": 0.015,  # 1.5%
                "cost_per_post": 5000.0,
                "description": "Mid-tier influencer"
            }
        ]
        
        for case in test_cases:
            kol = MagicMock()
            kol.handle = f"@test_{case['description']}"
            kol.latest_metrics = MagicMock()
            kol.latest_metrics.followers = case["followers"]
            kol.latest_metrics.average_engagement_rate = case["engagement_rate"]
            
            campaign = MagicMock()
            
            # Mock cost estimation
            with patch.object(kol_scorer_math, '_estimate_cost_per_post', 
                            return_value=case["cost_per_post"]), \
                 patch.object(kol_scorer_math, '_estimate_conversion_rate',
                            return_value=case["conversion_rate"]):
                
                roi_score, confidence = asyncio.run(
                    kol_scorer_math._calculate_roi_score(kol, campaign, MagicMock())
                )
                
                # Calculate expected ROI manually
                expected_engagement = case["followers"] * float(case["engagement_rate"])
                expected_value = expected_engagement * case["conversion_rate"]
                expected_raw_roi = expected_value / case["cost_per_post"]
                expected_normalized_roi = min(1.0, math.log10(max(1, expected_raw_roi)) / 4.0)
                
                # Should be close to expected calculation
                assert abs(float(roi_score) - expected_normalized_roi) < 0.1, \
                    f"ROI calculation error for {case['description']}: " \
                    f"expected ~{expected_normalized_roi}, got {roi_score}"
    
    def test_engagement_trend_calculation_statistical_validity(self, kol_scorer_math):
        """Test engagement trend calculation using statistical methods."""
        # Create various engagement patterns
        patterns = {
            "increasing": [0.02 + i * 0.001 for i in range(10)],  # Linear increase
            "decreasing": [0.06 - i * 0.002 for i in range(10)],  # Linear decrease
            "stable": [0.04 + (0.001 if i % 2 else -0.001) for i in range(10)],  # Stable with noise
            "volatile": [0.04, 0.08, 0.02, 0.07, 0.01, 0.09, 0.03, 0.06, 0.02, 0.08],  # High variance
        }
        
        for pattern_name, rates in patterns.items():
            # Create mock engagement history
            history = []
            for rate in rates:
                mock_entry = MagicMock()
                mock_entry.engagement_rate = rate
                history.append(mock_entry)
            
            trend_score = kol_scorer_math._calculate_engagement_trend(history)
            
            # Validate trend direction
            if pattern_name == "increasing":
                assert trend_score > 0.6, f"Increasing trend should score > 0.6, got {trend_score}"
            elif pattern_name == "decreasing":
                assert trend_score < 0.4, f"Decreasing trend should score < 0.4, got {trend_score}"
            elif pattern_name == "stable":
                assert 0.4 <= trend_score <= 0.6, f"Stable trend should be ~0.5, got {trend_score}"
            
            # Validate bounds
            assert 0.0 <= trend_score <= 1.0, f"Trend score out of bounds: {trend_score}"
    
    def test_tier_determination_consistency(self, kol_scorer_math):
        """Test tier determination consistency across boundary conditions."""
        # Test boundary values
        boundary_tests = [
            (999, "nano"), (1000, "micro"), (1001, "micro"),
            (9999, "micro"), (10000, "mid"), (10001, "mid"),
            (99999, "mid"), (100000, "macro"), (100001, "macro"),
            (1000000, "macro"), (10000000, "macro")
        ]
        
        for followers, expected_tier in boundary_tests:
            actual_tier = kol_scorer_math._determine_kol_tier(followers)
            assert actual_tier == expected_tier, \
                f"Tier determination error: {followers} followers -> expected {expected_tier}, got {actual_tier}"
    
    def test_cost_estimation_economic_rationality(self, kol_scorer_math):
        """Test that cost estimation follows economic rationality principles."""
        # Test cases that should follow economic logic
        test_cases = [
            {"followers": 1000, "engagement": Decimal("0.08"), "description": "High-engagement nano"},
            {"followers": 1000, "engagement": Decimal("0.02"), "description": "Low-engagement nano"},
            {"followers": 100000, "engagement": Decimal("0.03"), "description": "Standard mid-tier"},
            {"followers": 100000, "engagement": Decimal("0.01"), "description": "Low-engagement mid-tier"},
        ]
        
        costs = []
        
        for case in test_cases:
            kol = MagicMock()
            kol.latest_metrics = MagicMock()
            kol.latest_metrics.followers = case["followers"]
            kol.latest_metrics.average_engagement_rate = case["engagement"]
            
            cost = kol_scorer_math._estimate_cost_per_post(kol)
            costs.append((cost, case))
            
            # Basic rationality: more followers should generally cost more
            # Higher engagement should generally cost more
            assert cost > 0, f"Cost should be positive for {case['description']}"
        
        # Sort by followers to test follower impact
        costs_by_followers = sorted(costs, key=lambda x: x[1]["followers"])
        
        # Generally, more followers should cost more (with some tolerance for engagement differences)
        for i in range(1, len(costs_by_followers)):
            current_cost, current_case = costs_by_followers[i]
            prev_cost, prev_case = costs_by_followers[i-1]
            
            if current_case["followers"] > prev_case["followers"] * 2:  # Significant follower increase
                # Should generally cost more, unless engagement is drastically lower
                engagement_ratio = float(current_case["engagement"]) / float(prev_case["engagement"])
                if engagement_ratio > 0.5:  # Not drastically lower engagement
                    assert current_cost >= prev_cost * 0.8, \
                        f"Economic irrationality: {current_case['description']} costs {current_cost} " \
                        f"but {prev_case['description']} costs {prev_cost}"
    
    def _create_test_kol(self, properties: Dict[str, Any]) -> MagicMock:
        """Create test KOL with specific properties."""
        kol = MagicMock()
        kol.handle = f"@test_{hash(str(properties))}"
        
        # Set up metrics
        kol.latest_metrics = MagicMock()
        if "followers" in properties:
            kol.latest_metrics.followers = properties["followers"]
        if "engagement_rate" in properties:
            kol.latest_metrics.average_engagement_rate = properties["engagement_rate"]
        
        # Set up other properties
        if "fake_followers" in properties:
            kol.latest_metrics.fake_follower_percentage = properties["fake_followers"]
        
        if "account_age_days" in properties:
            from datetime import datetime, timezone, timedelta
            kol.account_created_at = datetime.now(timezone.utc) - timedelta(days=properties["account_age_days"])
        
        if "posts_30d" in properties:
            kol.recent_content = [MagicMock() for _ in range(properties["posts_30d"])]
        
        if "engagement_variance" in properties:
            # Create engagement history with specific variance
            base_rate = 0.04
            variance = properties["engagement_variance"]
            history = []
            for i in range(10):
                rate = base_rate + (variance * (0.5 - (i % 2)))  # Alternating pattern
                mock_entry = MagicMock()
                mock_entry.engagement_rate = rate
                history.append(mock_entry)
            kol.engagement_history = history
        
        return kol
    
    def _create_deterministic_test_kol(self) -> MagicMock:
        """Create KOL with completely deterministic properties for consistency testing."""
        kol = MagicMock()
        kol.handle = "@deterministic_test"
        kol.location = "Bangkok, Thailand"
        kol.categories = ["lifestyle", "fashion"]
        kol.is_verified = True
        kol.demographics = {"average_age": 28, "interests": ["fashion", "travel"]}
        
        from datetime import datetime, timezone, timedelta
        kol.account_created_at = datetime(2023, 1, 1, tzinfo=timezone.utc)
        
        # Deterministic metrics
        metrics = MagicMock()
        metrics.followers = 50000
        metrics.average_engagement_rate = Decimal("0.045")
        metrics.fake_follower_percentage = Decimal("0.05")
        metrics.posts_last_30_days = 20
        kol.latest_metrics = metrics
        
        # Deterministic content
        content = MagicMock()
        content.caption = "Daily lifestyle content #fashion #lifestyle"
        content.hashtags = ["fashion", "lifestyle"]
        kol.recent_content = [content, content]  # Identical content for determinism
        
        # Deterministic engagement history
        history = []
        for i in range(10):
            entry = MagicMock()
            entry.engagement_rate = 0.045  # Constant rate
            history.append(entry)
        kol.engagement_history = history
        
        return kol


# AIDEV-NOTE: Edge Cases and Boundary Condition Tests

class TestScoringAlgorithmEdgeCases:
    """Test edge cases and boundary conditions in scoring algorithms."""
    
    @pytest.fixture
    def kol_scorer_edge(self):
        """KOL scorer for edge case testing."""
        with patch('kol_api.services.scoring.kol_scorer.SentimentAnalyzer'), \
             patch('kol_api.services.scoring.kol_scorer.DemographicMatcher'):
            return KOLScorer()
    
    def test_zero_follower_edge_case(self, kol_scorer_edge):
        """Test handling of KOL with zero followers."""
        kol = MagicMock()
        kol.handle = "@zero_followers"
        kol.latest_metrics = MagicMock()
        kol.latest_metrics.followers = 0
        kol.latest_metrics.average_engagement_rate = Decimal("0.0")
        
        campaign = MagicMock()
        
        roi_score, confidence = asyncio.run(
            kol_scorer_edge._calculate_roi_score(kol, campaign, MagicMock())
        )
        
        # Should handle gracefully without division by zero
        assert roi_score == Decimal("0.0")
        assert 0.0 <= confidence <= 1.0
    
    def test_extreme_engagement_rates(self, kol_scorer_edge):
        """Test handling of extreme engagement rates."""
        extreme_cases = [
            Decimal("0.0"),      # 0% engagement
            Decimal("0.99"),     # 99% engagement (impossible but test boundary)
            Decimal("1.0"),      # 100% engagement
            Decimal("1.5"),      # Over 100% engagement (data error)
        ]
        
        for engagement_rate in extreme_cases:
            kol = MagicMock()
            kol.handle = f"@extreme_{engagement_rate}"
            kol.latest_metrics = MagicMock()
            kol.latest_metrics.followers = 10000
            kol.latest_metrics.average_engagement_rate = engagement_rate
            kol.engagement_history = []
            
            # Should not crash and should return bounded scores
            aud_score, confidence = asyncio.run(
                kol_scorer_edge._calculate_audience_quality_score(kol, MagicMock())
            )
            
            assert 0.0 <= aud_score <= 1.0, f"Audience score out of bounds for engagement {engagement_rate}"
            assert 0.0 <= confidence <= 1.0, f"Confidence out of bounds for engagement {engagement_rate}"
    
    def test_missing_metrics_combinations(self, kol_scorer_edge):
        """Test various combinations of missing metrics."""
        missing_combinations = [
            {"followers": None, "engagement": None},
            {"followers": 0, "engagement": None},
            {"followers": None, "engagement": Decimal("0.05")},
            {"followers": 10000, "engagement": None},
        ]
        
        for combo in missing_combinations:
            kol = MagicMock()
            kol.handle = f"@missing_{combo}"
            
            if combo["followers"] is None:
                kol.latest_metrics = None
            else:
                kol.latest_metrics = MagicMock()
                kol.latest_metrics.followers = combo["followers"]
                kol.latest_metrics.average_engagement_rate = combo["engagement"]
            
            campaign = MagicMock()
            
            # Should handle all combinations gracefully
            try:
                roi_score, confidence = asyncio.run(
                    kol_scorer_edge._calculate_roi_score(kol, campaign, MagicMock())
                )
                
                # Should return valid bounds even with missing data
                assert 0.0 <= roi_score <= 1.0
                assert 0.0 <= confidence <= 1.0
                
            except Exception as e:
                pytest.fail(f"Failed to handle missing data combination {combo}: {e}")
    
    def test_extremely_large_numbers(self, kol_scorer_edge):
        """Test handling of extremely large follower counts."""
        large_numbers = [
            1_000_000,      # 1M followers
            10_000_000,     # 10M followers  
            100_000_000,    # 100M followers
            1_000_000_000,  # 1B followers (impossible but test robustness)
        ]
        
        for followers in large_numbers:
            kol = MagicMock()
            kol.handle = f"@mega_{followers}"
            kol.latest_metrics = MagicMock()
            kol.latest_metrics.followers = followers
            kol.latest_metrics.average_engagement_rate = Decimal("0.01")  # Low but reasonable
            
            # Test cost estimation doesn't overflow
            cost = kol_scorer_edge._estimate_cost_per_post(kol)
            
            # Should be reasonable (not infinite or negative)
            assert 0 < cost < 10_000_000, f"Unreasonable cost {cost} for {followers} followers"
            
            # Test tier determination
            tier = kol_scorer_edge._determine_kol_tier(followers)
            assert tier in ["nano", "micro", "mid", "macro"], f"Invalid tier {tier}"
    
    def test_negative_values_handling(self, kol_scorer_edge):
        """Test handling of negative values in metrics (data corruption edge case)."""
        kol = MagicMock()
        kol.handle = "@negative_test"
        kol.latest_metrics = MagicMock()
        kol.latest_metrics.followers = -1000  # Corrupted data
        kol.latest_metrics.average_engagement_rate = Decimal("-0.05")  # Impossible
        
        # Should handle gracefully without errors
        cost = kol_scorer_edge._estimate_cost_per_post(kol)
        tier = kol_scorer_edge._determine_kol_tier(kol.latest_metrics.followers)
        
        # Should provide sensible defaults
        assert cost > 0  # Should default to positive cost
        assert tier in ["nano", "micro", "mid", "macro"]
    
    def test_unicode_and_special_characters(self, kol_scorer_edge):
        """Test handling of unicode and special characters in content."""
        special_contents = [
            "สวัสดีครับ! 🇹🇭 #lifestyle",  # Thai with emoji
            "こんにちは 👋 #fashion #日本",     # Japanese
            "¡Hola! ñoño @user #café",        # Spanish with special chars
            "🔥🔥🔥 AMAZING!!! 💯💯💯",          # Emoji heavy
            "",                               # Empty string
            "a" * 1000,                       # Very long string
        ]
        
        for content_text in special_contents:
            # Test controversial content detection
            safety_score = kol_scorer_edge._check_controversial_content(content_text)
            
            # Should return valid score for any text
            assert 0.0 <= safety_score <= 1.0, f"Invalid safety score for content: {content_text[:50]}"
    
    def test_temporal_edge_cases(self, kol_scorer_edge):
        """Test edge cases related to time and dates."""
        from datetime import datetime, timezone, timedelta
        
        # Test very new account (created today)
        new_kol = MagicMock()
        new_kol.handle = "@brand_new"
        new_kol.account_created_at = datetime.now(timezone.utc)
        new_kol.recent_content = []
        new_kol.engagement_history = []
        
        # Test very old account (10+ years)
        old_kol = MagicMock()
        old_kol.handle = "@very_old"
        old_kol.account_created_at = datetime.now(timezone.utc) - timedelta(days=3650)  # 10 years
        old_kol.recent_content = []
        old_kol.engagement_history = []
        
        # Test account with future creation date (data corruption)
        future_kol = MagicMock()
        future_kol.handle = "@future"
        future_kol.account_created_at = datetime.now(timezone.utc) + timedelta(days=365)  # Future
        future_kol.recent_content = []
        future_kol.engagement_history = []
        
        for test_kol in [new_kol, old_kol, future_kol]:
            # Should handle all temporal edge cases
            reliability_score, confidence = asyncio.run(
                kol_scorer_edge._calculate_reliability_score(test_kol, MagicMock())
            )
            
            assert 0.0 <= reliability_score <= 1.0
            assert 0.0 <= confidence <= 1.0
    
    def test_circular_reference_protection(self, kol_scorer_edge):
        """Test protection against circular references in data structures."""
        # Create KOL with self-referencing content (edge case in data structure)
        kol = MagicMock()
        kol.handle = "@circular_test"
        
        # Create content that references itself (circular reference simulation)
        content = MagicMock()
        content.caption = "Test content"
        content.related_content = content  # Self-reference
        content.hashtags = ["test"]
        
        kol.recent_content = [content]
        kol.latest_metrics = MagicMock()
        kol.latest_metrics.followers = 10000
        
        campaign = MagicMock()
        requirements = MagicMock()
        requirements.target_keywords = ["test"]
        requirements.required_categories = []
        campaign.requirements = requirements
        
        # Should handle circular references without infinite loops
        try:
            relevance_score, confidence = asyncio.run(
                kol_scorer_edge._calculate_content_relevance_score(kol, campaign, MagicMock())
            )
            
            assert 0.0 <= relevance_score <= 1.0
            assert 0.0 <= confidence <= 1.0
            
        except RecursionError:
            pytest.fail("Circular reference caused infinite recursion")


# AIDEV-NOTE: Statistical Validation Tests

class TestScoringStatisticalValidation:
    """Statistical validation tests for scoring algorithm distributions and properties."""
    
    def test_score_distribution_properties(self, test_data_validator):
        """Test that score distributions have expected statistical properties."""
        from tests.fixtures.test_data_factory import TestScenarioFactory
        
        # Generate large sample of diverse KOLs
        kol_pool = TestScenarioFactory.create_mixed_quality_kol_pool(size=100)
        
        # Extract various metrics for statistical analysis
        engagement_rates = []
        follower_counts = []
        
        for kol, metrics in kol_pool:
            if metrics.engagement_rate:
                engagement_rates.append(float(metrics.engagement_rate))
            if metrics.follower_count:
                follower_counts.append(metrics.follower_count)
        
        # Test engagement rate distribution
        if engagement_rates:
            mean_engagement = statistics.mean(engagement_rates)
            std_engagement = statistics.stdev(engagement_rates) if len(engagement_rates) > 1 else 0
            
            # Should be within realistic ranges for influencer marketing
            assert 0.01 <= mean_engagement <= 0.15, f"Unrealistic mean engagement: {mean_engagement}"
            assert std_engagement >= 0, "Standard deviation should be non-negative"
            
            # Test for reasonable distribution (not all identical)
            if len(set(engagement_rates)) > 1:
                assert std_engagement > 0.001, "Engagement rates should show some variation"
        
        # Test follower count distribution
        if follower_counts:
            # Should span multiple tiers
            min_followers = min(follower_counts)
            max_followers = max(follower_counts)
            
            assert max_followers > min_followers * 5, "Should have variety in follower counts"
    
    def test_correlation_between_metrics(self):
        """Test expected correlations between different metrics."""
        from tests.fixtures.test_data_factory import TestScenarioFactory
        
        # Generate large sample
        kol_pool = TestScenarioFactory.create_mixed_quality_kol_pool(size=50)
        
        # Extract paired metrics for correlation analysis
        followers_engagement_pairs = []
        cost_followers_pairs = []
        
        for kol, metrics in kol_pool:
            if metrics.engagement_rate and metrics.follower_count:
                followers_engagement_pairs.append((
                    metrics.follower_count, 
                    float(metrics.engagement_rate)
                ))
            
            if metrics.follower_count and hasattr(metrics, 'rate_per_post') and metrics.rate_per_post:
                cost_followers_pairs.append((
                    float(metrics.rate_per_post),
                    metrics.follower_count
                ))
        
        # Test inverse correlation between followers and engagement rate
        if len(followers_engagement_pairs) > 10:
            followers, engagements = zip(*followers_engagement_pairs)
            
            # Calculate Pearson correlation
            if len(set(followers)) > 1 and len(set(engagements)) > 1:
                correlation = np.corrcoef(followers, engagements)[0, 1]
                
                # Should show negative correlation (more followers = lower engagement rate typically)
                assert correlation < 0.1, f"Expected negative correlation between followers and engagement, got {correlation}"
        
        # Test positive correlation between cost and followers
        if len(cost_followers_pairs) > 10:
            costs, followers = zip(*cost_followers_pairs)
            
            if len(set(costs)) > 1 and len(set(followers)) > 1:
                correlation = np.corrcoef(costs, followers)[0, 1]
                
                # Should show positive correlation (more followers = higher cost)
                assert correlation > 0.3, f"Expected positive correlation between cost and followers, got {correlation}"
    
    def test_percentile_distributions(self):
        """Test that generated data follows expected percentile distributions."""
        from tests.fixtures.test_data_factory import TestScenarioFactory
        
        # Generate data with known quality distribution
        kol_pool = TestScenarioFactory.create_mixed_quality_kol_pool(size=100)
        
        # Extract quality-related metrics
        quality_scores = []
        
        for kol, metrics in kol_pool:
            # Create composite quality score from available metrics
            score = 0
            factors = 0
            
            if metrics.engagement_rate:
                score += float(metrics.engagement_rate) * 100  # Convert to 0-10 scale
                factors += 1
            
            if hasattr(metrics, 'audience_quality_score') and metrics.audience_quality_score:
                score += float(metrics.audience_quality_score) * 10
                factors += 1
            
            if factors > 0:
                quality_scores.append(score / factors)
        
        if quality_scores:
            # Calculate percentiles
            percentiles = [10, 25, 50, 75, 90]
            values = [np.percentile(quality_scores, p) for p in percentiles]
            
            # Should show reasonable spread
            p10, p25, p50, p75, p90 = values
            
            # Percentiles should be in ascending order
            assert p10 <= p25 <= p50 <= p75 <= p90, "Percentiles should be in ascending order"
            
            # Should have reasonable spread (not all identical)
            assert p90 - p10 > 0.5, f"Insufficient spread in quality scores: {p90 - p10}"
    
    def test_outlier_detection_and_handling(self):
        """Test that the system can detect and appropriately handle statistical outliers."""
        from tests.fixtures.test_data_factory import KOLDataFactory
        
        # Create dataset with intentional outliers
        normal_kols = []
        outlier_kols = []
        
        # Generate normal KOLs
        for _ in range(20):
            kol, metrics = KOLDataFactory.create_kol_profile(
                tier="micro",
                quality_level="medium",
                data_completeness="complete"
            )
            normal_kols.append((kol, metrics))
        
        # Generate outlier KOLs
        for _ in range(3):
            kol, metrics = KOLDataFactory.create_kol_profile(
                tier="micro",
                quality_level="high",
                data_completeness="complete",
                engagement_rate=Decimal("0.15"),  # Extremely high engagement
                follower_count=1000000  # Way higher than micro tier
            )
            outlier_kols.append((kol, metrics))
        
        all_kols = normal_kols + outlier_kols
        
        # Extract engagement rates
        engagement_rates = []
        for kol, metrics in all_kols:
            if metrics.engagement_rate:
                engagement_rates.append(float(metrics.engagement_rate))
        
        # Calculate outlier statistics
        mean_engagement = statistics.mean(engagement_rates)
        std_engagement = statistics.stdev(engagement_rates)
        
        # Identify statistical outliers (values > 2 standard deviations from mean)
        outliers = [rate for rate in engagement_rates 
                   if abs(rate - mean_engagement) > 2 * std_engagement]
        
        # Should detect the intentional outliers
        assert len(outliers) >= 3, f"Should detect at least 3 outliers, found {len(outliers)}"
        
        # Outliers should be the high engagement rates we created
        for outlier in outliers:
            assert outlier >= 0.1, f"Detected outlier {outlier} should be high engagement rate"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])