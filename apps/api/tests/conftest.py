"""
Comprehensive Test Configuration for POC2 and POC4 Algorithm Tests

AIDEV-NOTE: Production-ready pytest configuration with shared fixtures,
test utilities, and comprehensive setup for all algorithm testing scenarios.
"""
import pytest
import asyncio
import logging
from datetime import datetime, timezone
from decimal import Decimal
from typing import List, Dict, Any, Optional, Generator
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np

# Import test data factories
from tests.fixtures.test_data_factory import (
    KOLDataFactory, 
    CampaignDataFactory, 
    TestScenarioFactory
)

from kol_api.services.models import (
    KOLCandidate,
    KOLMetricsData,
    CampaignRequirements,
    ScoreComponents,
    OptimizationConstraints,
    OptimizationObjective,
    KOLTier,
    ContentCategory
)


# AIDEV-NOTE: Pytest Configuration

def pytest_configure(config):
    """Configure pytest with custom markers and settings."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance tests"
    )
    config.addinivalue_line(
        "markers", "missing_data: marks tests for missing data scenarios"
    )
    config.addinivalue_line(
        "markers", "edge_case: marks tests for edge cases"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names."""
    for item in items:
        # Add slow marker to performance tests
        if "performance" in item.name.lower() or "large" in item.name.lower():
            item.add_marker(pytest.mark.slow)
        
        # Add integration marker to integration tests
        if "integration" in item.name.lower() or "end_to_end" in item.name.lower():
            item.add_marker(pytest.mark.integration)
        
        # Add missing_data marker
        if "missing" in item.name.lower() or "incomplete" in item.name.lower():
            item.add_marker(pytest.mark.missing_data)
        
        # Add edge_case marker
        if "edge" in item.name.lower() or "extreme" in item.name.lower():
            item.add_marker(pytest.mark.edge_case)


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# AIDEV-NOTE: Database and Service Fixtures

@pytest.fixture
def mock_db_session():
    """Mock database session with common query responses."""
    session = AsyncMock()
    
    # Setup default return values
    session.execute.return_value.fetchall.return_value = []
    session.execute.return_value.scalars.return_value.all.return_value = []
    session.execute.return_value.scalar_one_or_none.return_value = None
    
    session.add = MagicMock()
    session.flush = AsyncMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    
    return session


@pytest.fixture
def mock_redis_client():
    """Mock Redis client for caching tests."""
    client = MagicMock()
    client.get.return_value = None
    client.set.return_value = True
    client.delete.return_value = 1
    client.exists.return_value = False
    return client


@pytest.fixture
def mock_settings():
    """Mock application settings."""
    settings = MagicMock()
    settings.budget_tiers = {
        "nano": {"avg_cost_per_post": 500},
        "micro": {"avg_cost_per_post": 2500},
        "mid": {"avg_cost_per_post": 12500},
        "macro": {"avg_cost_per_post": 62500}
    }
    settings.ml_model_config = {
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "sentiment_model": "vader"
    }
    return settings


# AIDEV-NOTE: KOL Data Fixtures

@pytest.fixture
def high_quality_kol_data():
    """High-quality KOL with complete data for testing."""
    return KOLDataFactory.create_kol_profile(
        tier="micro",
        category=ContentCategory.LIFESTYLE,
        quality_level="high",
        data_completeness="complete"
    )


@pytest.fixture
def medium_quality_kol_data():
    """Medium-quality KOL with partial data for testing."""
    return KOLDataFactory.create_kol_profile(
        tier="micro",
        category=ContentCategory.FASHION,
        quality_level="medium",
        data_completeness="partial"
    )


@pytest.fixture
def low_quality_kol_data():
    """Low-quality KOL with minimal data for testing."""
    return KOLDataFactory.create_kol_profile(
        tier="nano",
        category=ContentCategory.BEAUTY,
        quality_level="low",
        data_completeness="minimal"
    )


@pytest.fixture
def mixed_quality_kol_pool():
    """Diverse pool of KOLs with mixed quality levels."""
    return TestScenarioFactory.create_mixed_quality_kol_pool(size=15)


@pytest.fixture
def micro_tier_kol_pool():
    """Pool of micro-tier KOLs for tier-specific testing."""
    return TestScenarioFactory.create_tier_specific_kol_pool(tier="micro", size=10)


@pytest.fixture
def missing_data_scenarios():
    """Various missing data scenarios for testing."""
    return TestScenarioFactory.create_missing_data_scenarios()


# AIDEV-NOTE: Campaign and Requirements Fixtures

@pytest.fixture
def engagement_campaign_requirements():
    """Campaign requirements focused on engagement."""
    return CampaignDataFactory.create_campaign_requirements(
        campaign_type="engagement",
        budget_size="medium",
        complexity="medium"
    )


@pytest.fixture
def awareness_campaign_requirements():
    """Campaign requirements focused on awareness."""
    return CampaignDataFactory.create_campaign_requirements(
        campaign_type="awareness",
        budget_size="large",
        complexity="simple"
    )


@pytest.fixture
def conversion_campaign_requirements():
    """Campaign requirements focused on conversions."""
    return CampaignDataFactory.create_campaign_requirements(
        campaign_type="conversion",
        budget_size="medium",
        complexity="complex"
    )


@pytest.fixture
def small_budget_campaign():
    """Small budget campaign for constraint testing."""
    return CampaignDataFactory.create_campaign_requirements(
        campaign_type="engagement",
        budget_size="small",
        complexity="simple",
        total_budget=Decimal("15000")
    )


@pytest.fixture
def complex_campaign_requirements():
    """Complex campaign with many constraints."""
    return CampaignDataFactory.create_campaign_requirements(
        campaign_type="conversion",
        budget_size="large",
        complexity="complex",
        target_kol_tiers=[KOLTier.MICRO, KOLTier.MID, KOLTier.MACRO],
        target_categories=[ContentCategory.LIFESTYLE, ContentCategory.FASHION, ContentCategory.BEAUTY],
        min_follower_count=25000,
        max_follower_count=1000000,
        min_engagement_rate=Decimal("0.025"),
        require_verified=True,
        required_hashtags=["campaign", "brand", "sponsored"],
        excluded_hashtags=["competitor", "negative", "controversial"]
    )


# AIDEV-NOTE: Optimization Constraints Fixtures

@pytest.fixture
def loose_optimization_constraints():
    """Loose optimization constraints for flexibility testing."""
    return CampaignDataFactory.create_optimization_constraints(
        strictness="loose",
        risk_tolerance="high",
        max_budget=Decimal("100000")
    )


@pytest.fixture
def strict_optimization_constraints():
    """Strict optimization constraints for constraint testing."""
    return CampaignDataFactory.create_optimization_constraints(
        strictness="strict",
        risk_tolerance="low",
        max_budget=Decimal("50000"),
        required_micro_count=3,
        required_mid_count=1,
        max_risk_per_kol=Decimal("0.3")
    )


@pytest.fixture
def balanced_optimization_constraints():
    """Balanced optimization constraints for standard testing."""
    return CampaignDataFactory.create_optimization_constraints(
        strictness="medium",
        risk_tolerance="medium",
        max_budget=Decimal("75000")
    )


# AIDEV-NOTE: Scenario-Based Fixtures

@pytest.fixture
def budget_optimization_scenarios():
    """Various budget optimization scenarios."""
    return TestScenarioFactory.create_budget_optimization_scenarios()


@pytest.fixture
def performance_test_kol_pool():
    """Large KOL pool for performance testing."""
    return TestScenarioFactory.create_mixed_quality_kol_pool(size=100)


@pytest.fixture(scope="session")
def large_performance_dataset():
    """Very large dataset for performance testing (session-scoped)."""
    return TestScenarioFactory.create_mixed_quality_kol_pool(size=500)


# AIDEV-NOTE: Scoring and Algorithm Fixtures

@pytest.fixture
def sample_score_components():
    """Sample score components for testing."""
    return ScoreComponents(
        roi_score=Decimal("0.75"),
        audience_quality_score=Decimal("0.85"),
        brand_safety_score=Decimal("0.95"),
        content_relevance_score=Decimal("0.70"),
        demographic_fit_score=Decimal("0.80"),
        reliability_score=Decimal("0.90"),
        roi_confidence=Decimal("0.85"),
        audience_confidence=Decimal("0.90"),
        brand_safety_confidence=Decimal("1.0"),
        content_relevance_confidence=Decimal("0.75"),
        demographic_confidence=Decimal("0.80"),
        reliability_confidence=Decimal("0.85")
    )


@pytest.fixture
def sample_kol_candidate(high_quality_kol_data, sample_score_components):
    """Complete KOL candidate for testing."""
    kol, metrics = high_quality_kol_data
    
    metrics_data = KOLMetricsData(
        follower_count=metrics.follower_count,
        following_count=metrics.following_count,
        engagement_rate=metrics.engagement_rate,
        avg_likes=metrics.avg_likes,
        avg_comments=metrics.avg_comments,
        avg_views=metrics.avg_views,
        posts_last_30_days=metrics.posts_last_30_days,
        fake_follower_percentage=metrics.fake_follower_percentage,
        audience_quality_score=metrics.audience_quality_score,
        campaign_success_rate=metrics.campaign_success_rate,
        response_rate=metrics.response_rate
    )
    
    return KOLCandidate(
        kol_id=kol.id,
        username=kol.username,
        display_name=kol.display_name,
        platform=kol.platform.value,
        tier=KOLTier(kol.tier.value),
        primary_category=ContentCategory(kol.primary_category.value),
        metrics=metrics_data,
        score_components=sample_score_components,
        overall_score=Decimal("0.82"),
        predicted_reach=7500,
        predicted_engagement=337,
        predicted_conversions=8,
        estimated_cost_per_post=Decimal("2500.00"),
        estimated_total_cost=Decimal("2500.00"),
        risk_factors=["Unverified account"],
        overall_risk_score=Decimal("0.15")
    )


# AIDEV-NOTE: Test Utilities

@pytest.fixture
def test_data_validator():
    """Utility for validating test data quality."""
    
    class TestDataValidator:
        @staticmethod
        def validate_kol_data_realism(kol_data):
            """Validate that KOL data is realistic."""
            kol, metrics = kol_data
            
            validations = []
            
            # Check follower/following ratio
            if metrics.following_count and metrics.follower_count:
                ratio = metrics.following_count / metrics.follower_count
                validations.append(("follow_ratio", 0.001 <= ratio <= 1.0))
            
            # Check engagement rate
            if metrics.engagement_rate:
                validations.append(("engagement_rate", 
                                  0.001 <= float(metrics.engagement_rate) <= 0.2))
            
            # Check posting frequency
            validations.append(("posting_frequency", 
                              0 <= metrics.posts_last_30_days <= 60))
            
            return dict(validations)
        
        @staticmethod
        def validate_score_components(components: ScoreComponents):
            """Validate score components are within valid ranges."""
            validations = []
            
            # Check all scores are between 0 and 1
            for field_name in ["roi_score", "audience_quality_score", "brand_safety_score",
                              "content_relevance_score", "demographic_fit_score", "reliability_score"]:
                score = getattr(components, field_name)
                validations.append((field_name, 0 <= score <= 1))
            
            # Check confidences are between 0 and 1
            for field_name in ["roi_confidence", "audience_confidence", "brand_safety_confidence",
                              "content_relevance_confidence", "demographic_confidence", "reliability_confidence"]:
                confidence = getattr(components, field_name)
                validations.append((field_name, 0 <= confidence <= 1))
            
            return dict(validations)
    
    return TestDataValidator()


@pytest.fixture
def algorithm_test_helpers():
    """Helper functions for algorithm testing."""
    
    class AlgorithmTestHelpers:
        @staticmethod
        def calculate_expected_roi(metrics: KOLMetricsData, campaign_type: str = "engagement"):
            """Calculate expected ROI for validation."""
            if not metrics.engagement_rate or not metrics.follower_count:
                return Decimal("0.0")
            
            base_roi = float(metrics.engagement_rate) * 100
            
            # Adjust for campaign type
            if campaign_type == "conversion":
                base_roi *= 1.2
            elif campaign_type == "awareness":
                base_roi *= 0.8
            
            return min(Decimal("1.0"), Decimal(str(base_roi / 10)))
        
        @staticmethod
        def estimate_performance_metrics(kol_candidate: KOLCandidate):
            """Estimate performance metrics for validation."""
            metrics = kol_candidate.metrics
            
            estimated_reach = int(metrics.follower_count * 0.15)
            estimated_engagement = int(estimated_reach * float(metrics.engagement_rate or 0.03))
            estimated_conversions = int(estimated_engagement * 0.02)
            
            return {
                "reach": estimated_reach,
                "engagement": estimated_engagement,
                "conversions": estimated_conversions
            }
        
        @staticmethod
        def check_constraint_satisfaction(selected_kols: List[KOLCandidate], 
                                        constraints: OptimizationConstraints):
            """Check if selected KOLs satisfy optimization constraints."""
            violations = []
            
            # Check count constraints
            if len(selected_kols) < constraints.min_kols:
                violations.append(f"Too few KOLs: {len(selected_kols)} < {constraints.min_kols}")
            
            if len(selected_kols) > constraints.max_kols:
                violations.append(f"Too many KOLs: {len(selected_kols)} > {constraints.max_kols}")
            
            # Check budget constraint
            total_cost = sum(kol.estimated_total_cost for kol in selected_kols)
            if total_cost > constraints.max_budget:
                violations.append(f"Budget exceeded: {total_cost} > {constraints.max_budget}")
            
            # Check risk constraints
            for kol in selected_kols:
                if kol.overall_risk_score > constraints.max_risk_per_kol:
                    violations.append(f"KOL {kol.kol_id} risk too high: {kol.overall_risk_score}")
            
            return violations
    
    return AlgorithmTestHelpers()


# AIDEV-NOTE: Performance Testing Fixtures

@pytest.fixture
def performance_timer():
    """Timer utility for performance testing."""
    
    class PerformanceTimer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
        
        def start(self):
            import time
            self.start_time = time.time()
        
        def stop(self):
            import time
            self.end_time = time.time()
        
        @property
        def elapsed_seconds(self):
            if self.start_time and self.end_time:
                return self.end_time - self.start_time
            return None
    
    return PerformanceTimer()


# AIDEV-NOTE: Mock Service Fixtures

@pytest.fixture
def mock_kol_scoring_service():
    """Mock KOL scoring service with realistic responses."""
    
    service = MagicMock()
    
    def mock_score_kol(kol, campaign, db_session):
        """Mock scoring with realistic values."""
        
        async def score():
            from kol_api.services.scoring.kol_scorer import ScoreBreakdown
            
            # Generate realistic scores based on KOL quality
            if hasattr(kol, 'tier') and kol.tier.value == "micro":
                base_scores = {
                    "roi_score": Decimal("0.75"),
                    "audience_quality_score": Decimal("0.80"),
                    "brand_safety_score": Decimal("0.90"),
                    "content_relevance_score": Decimal("0.70"),
                    "demographic_fit_score": Decimal("0.75"),
                    "reliability_score": Decimal("0.85")
                }
            else:
                base_scores = {
                    "roi_score": Decimal("0.60"),
                    "audience_quality_score": Decimal("0.70"),
                    "brand_safety_score": Decimal("0.80"),
                    "content_relevance_score": Decimal("0.60"),
                    "demographic_fit_score": Decimal("0.65"),
                    "reliability_score": Decimal("0.75")
                }
            
            return ScoreBreakdown(
                **base_scores,
                roi_confidence=Decimal("0.8"),
                audience_quality_confidence=Decimal("0.85"),
                brand_safety_confidence=Decimal("0.9"),
                content_relevance_confidence=Decimal("0.75"),
                demographic_fit_confidence=Decimal("0.8"),
                reliability_confidence=Decimal("0.85"),
                composite_score=Decimal("0.75"),
                overall_confidence=Decimal("0.82"),
                missing_data_penalty=Decimal("0.1")
            )
        
        return score()
    
    service.score_kol = mock_score_kol
    return service


# AIDEV-NOTE: Cleanup Fixtures

@pytest.fixture(autouse=True)
def cleanup_test_artifacts():
    """Automatically cleanup test artifacts after each test."""
    yield
    
    # Clear any global caches or state
    # This ensures tests don't interfere with each other
    pass


@pytest.fixture(scope="session", autouse=True)
def setup_test_logging():
    """Setup logging configuration for tests."""
    logging.basicConfig(
        level=logging.WARNING,  # Reduce log noise during testing
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Silence specific noisy loggers
    logging.getLogger('asyncio').setLevel(logging.ERROR)
    logging.getLogger('urllib3').setLevel(logging.ERROR)
    
    yield
    
    # Cleanup logging
    logging.shutdown()


# AIDEV-NOTE: Parametrized Test Data

@pytest.fixture(params=["nano", "micro", "mid", "macro"])
def kol_tier(request):
    """Parametrized KOL tier for comprehensive tier testing."""
    return request.param


@pytest.fixture(params=["low", "medium", "high"])
def quality_level(request):
    """Parametrized quality level for comprehensive quality testing."""
    return request.param


@pytest.fixture(params=["minimal", "partial", "complete"])
def data_completeness(request):
    """Parametrized data completeness for comprehensive completeness testing."""
    return request.param


@pytest.fixture(params=[
    ContentCategory.LIFESTYLE, ContentCategory.FASHION, ContentCategory.BEAUTY,
    ContentCategory.FITNESS, ContentCategory.FOOD, ContentCategory.TRAVEL
])
def content_category(request):
    """Parametrized content category for comprehensive category testing."""
    return request.param


@pytest.fixture(params=[
    OptimizationObjective.MAXIMIZE_REACH,
    OptimizationObjective.MAXIMIZE_ENGAGEMENT, 
    OptimizationObjective.MAXIMIZE_CONVERSIONS,
    OptimizationObjective.MINIMIZE_COST
])
def optimization_objective(request):
    """Parametrized optimization objective for comprehensive objective testing."""
    return request.param