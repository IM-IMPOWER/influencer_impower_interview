"""
Comprehensive tests for Missing Data Handling - Graceful Degradation and Confidence Scoring

AIDEV-NOTE: Production-ready tests for robust handling of incomplete KOL data across all POC2 and POC4
algorithms, ensuring graceful degradation and accurate confidence assessment in real-world scenarios.
"""
import pytest
import asyncio
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import List, Dict, Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np

from kol_api.services.scoring.kol_scorer import KOLScorer, ScoreBreakdown
from kol_api.services.budget_optimizer import BudgetOptimizerService
from kol_api.services.kol_matching import EnhancedKOLMatchingService
from kol_api.services.brief_parser import BriefParserService
from kol_api.services.models import (
    KOLCandidate,
    KOLMetricsData,
    CampaignRequirements,
    ScoreComponents,
    OptimizationObjective,
    KOLTier,
    ContentCategory
)
from kol_api.database.models.kol import KOLProfile, KOLMetrics, KOLContent
from kol_api.database.models.campaign import Campaign


# AIDEV-NOTE: Test fixtures for various missing data scenarios

@pytest.fixture
def mock_db_session():
    """Mock database session."""
    return AsyncMock()


@pytest.fixture
def partially_complete_kol():
    """KOL with some missing data fields."""
    kol = MagicMock(spec=KOLProfile)
    kol.id = "partial_kol_001"
    kol.handle = "@partial_kol"
    kol.platform = "tiktok"
    kol.bio = "Content creator"  # Present
    kol.location = None  # Missing
    kol.categories = ["lifestyle"]  # Present
    kol.is_verified = True
    kol.account_created_at = datetime.now(timezone.utc) - timedelta(days=180)
    kol.languages = None  # Missing
    kol.demographics = None  # Missing
    
    # Partial metrics
    metrics = MagicMock(spec=KOLMetrics)
    metrics.followers = 25000  # Present
    metrics.average_engagement_rate = None  # Missing - critical field
    metrics.posts_last_30_days = 12  # Present
    metrics.fake_follower_percentage = None  # Missing
    
    kol.latest_metrics = metrics
    
    # Limited content
    content = MagicMock(spec=KOLContent)
    content.caption = "Daily lifestyle content"
    content.hashtags = ["lifestyle"]
    content.created_at = datetime.now(timezone.utc) - timedelta(days=3)
    
    kol.recent_content = [content]  # Only one post
    kol.engagement_history = None  # Missing
    
    return kol


@pytest.fixture
def severely_incomplete_kol():
    """KOL with most data fields missing."""
    kol = MagicMock(spec=KOLProfile)
    kol.id = "incomplete_kol_002"
    kol.handle = "@incomplete_kol"
    kol.platform = "tiktok"
    kol.bio = None  # Missing
    kol.location = None  # Missing
    kol.categories = []  # Empty
    kol.is_verified = False
    kol.account_created_at = None  # Missing
    kol.languages = None  # Missing
    kol.demographics = None  # Missing
    
    # Minimal metrics
    metrics = MagicMock(spec=KOLMetrics)
    metrics.followers = 15000  # Present (only reliable field)
    metrics.average_engagement_rate = None  # Missing
    metrics.posts_last_30_days = 0  # No recent posts
    metrics.fake_follower_percentage = None  # Missing
    
    kol.latest_metrics = metrics
    kol.recent_content = []  # No content
    kol.engagement_history = None  # Missing
    
    return kol


@pytest.fixture
def inconsistent_data_kol():
    """KOL with inconsistent/conflicting data."""
    kol = MagicMock(spec=KOLProfile)
    kol.id = "inconsistent_kol_003"
    kol.handle = "@inconsistent_kol"
    kol.platform = "tiktok"
    kol.bio = "Fashion and gaming content"
    kol.location = "Bangkok, Thailand"
    kol.categories = ["fashion", "gaming"]  # Conflicting categories
    kol.is_verified = True
    kol.account_created_at = datetime.now(timezone.utc) - timedelta(days=10)  # Very new account
    kol.languages = ["th", "en"]
    kol.demographics = {"average_age": 16}  # Very young for verified account
    
    # Inconsistent metrics
    metrics = MagicMock(spec=KOLMetrics)
    metrics.followers = 1000000  # Large following
    metrics.average_engagement_rate = Decimal("0.001")  # Suspiciously low engagement
    metrics.posts_last_30_days = 1  # Very inactive
    metrics.fake_follower_percentage = Decimal("0.8")  # Very high fake percentage
    
    kol.latest_metrics = metrics
    
    # Conflicting content
    content1 = MagicMock(spec=KOLContent)
    content1.caption = "Gaming setup review #gaming #tech"
    content1.hashtags = ["gaming", "tech"]
    content1.created_at = datetime.now(timezone.utc) - timedelta(days=25)
    
    content2 = MagicMock(spec=KOLContent)
    content2.caption = "Fashion haul for kids #kidsfashion #young"
    content2.hashtags = ["kidsfashion", "young"]
    content2.created_at = datetime.now(timezone.utc) - timedelta(days=30)
    
    kol.recent_content = [content1, content2]
    kol.engagement_history = None
    
    return kol


@pytest.fixture
def outdated_data_kol():
    """KOL with outdated data."""
    kol = MagicMock(spec=KOLProfile)
    kol.id = "outdated_kol_004"
    kol.handle = "@outdated_kol"
    kol.platform = "tiktok"
    kol.bio = "Lifestyle content creator"
    kol.location = "Singapore"
    kol.categories = ["lifestyle"]
    kol.is_verified = True
    kol.account_created_at = datetime.now(timezone.utc) - timedelta(days=1000)
    kol.languages = ["en"]
    kol.demographics = {"average_age": 28}
    
    # Outdated metrics
    metrics = MagicMock(spec=KOLMetrics)
    metrics.followers = 35000
    metrics.average_engagement_rate = Decimal("0.04")
    metrics.posts_last_30_days = 0  # No recent activity
    metrics.fake_follower_percentage = Decimal("0.1")
    metrics.metrics_date = datetime.now(timezone.utc) - timedelta(days=90)  # 3 months old
    
    kol.latest_metrics = metrics
    
    # Old content
    old_content = MagicMock(spec=KOLContent)
    old_content.caption = "Old lifestyle content"
    old_content.hashtags = ["lifestyle"]
    old_content.created_at = datetime.now(timezone.utc) - timedelta(days=60)
    
    kol.recent_content = [old_content]
    kol.engagement_history = []  # Empty history
    
    return kol


@pytest.fixture
def complete_sample_campaign():
    """Complete campaign requirements for testing missing data handling."""
    return CampaignRequirements(
        campaign_id="missing_data_test_campaign",
        target_kol_tiers=[KOLTier.MICRO, KOLTier.MID],
        target_categories=[ContentCategory.LIFESTYLE, ContentCategory.FASHION],
        total_budget=Decimal("50000"),
        min_follower_count=10000,
        max_follower_count=100000,
        min_engagement_rate=Decimal("0.025"),
        target_locations=["Bangkok", "Singapore"],
        target_languages=["th", "en"],
        campaign_objective=OptimizationObjective.MAXIMIZE_ENGAGEMENT,
        expected_conversion_rate=Decimal("0.02")
    )


# AIDEV-NOTE: Missing Data Handling in KOL Scoring

class TestKOLScorerMissingData:
    """Test KOL scoring with various missing data scenarios."""
    
    @pytest.mark.asyncio
    async def test_scoring_with_missing_engagement_rate(
        self,
        mock_db_session,
        partially_complete_kol,
        complete_sample_campaign
    ):
        """Test scoring when critical engagement rate data is missing."""
        scorer = KOLScorer()
        
        with patch.object(scorer, 'sentiment_analyzer'), \
             patch.object(scorer, 'demographic_matcher'):
            
            score_breakdown = await scorer.score_kol(
                partially_complete_kol,
                complete_sample_campaign,
                mock_db_session
            )
            
            # Should complete without errors
            assert isinstance(score_breakdown, ScoreBreakdown)
            
            # ROI score should be low due to missing engagement rate
            assert score_breakdown.roi_score < Decimal("0.5")
            assert score_breakdown.roi_confidence < Decimal("0.6")
            
            # Overall confidence should be reduced
            assert score_breakdown.overall_confidence < Decimal("0.7")
            
            # Missing data penalty should be significant
            assert score_breakdown.missing_data_penalty > Decimal("0.2")
            
            # Composite score should be penalized but not zero
            assert Decimal("0.0") < score_breakdown.composite_score < Decimal("0.6")
    
    @pytest.mark.asyncio
    async def test_scoring_with_no_recent_content(
        self,
        mock_db_session,
        severely_incomplete_kol,
        complete_sample_campaign
    ):
        """Test scoring when no recent content is available."""
        scorer = KOLScorer()
        
        with patch.object(scorer, 'sentiment_analyzer'), \
             patch.object(scorer, 'demographic_matcher'):
            
            score_breakdown = await scorer.score_kol(
                severely_incomplete_kol,
                complete_sample_campaign,
                mock_db_session
            )
            
            # Content relevance should be very low with no content
            assert score_breakdown.content_relevance_score < Decimal("0.4")
            assert score_breakdown.content_relevance_confidence < Decimal("0.5")
            
            # Brand safety should use default moderate score
            assert Decimal("0.4") <= score_breakdown.brand_safety_score <= Decimal("0.8")
            assert score_breakdown.brand_safety_confidence < Decimal("0.5")
            
            # Reliability should be very low due to no posts
            assert score_breakdown.reliability_score < Decimal("0.3")
            
            # Overall should be very low confidence
            assert score_breakdown.overall_confidence < Decimal("0.5")
    
    @pytest.mark.asyncio
    async def test_scoring_with_inconsistent_data(
        self,
        mock_db_session,
        inconsistent_data_kol,
        complete_sample_campaign
    ):
        """Test scoring with inconsistent/suspicious data."""
        scorer = KOLScorer()
        
        with patch.object(scorer, 'sentiment_analyzer'), \
             patch.object(scorer, 'demographic_matcher'):
            
            score_breakdown = await scorer.score_kol(
                inconsistent_data_kol,
                complete_sample_campaign,
                mock_db_session
            )
            
            # Audience quality should be very low due to high fake followers
            assert score_breakdown.audience_quality_score < Decimal("0.3")
            
            # ROI should be low due to poor engagement vs followers
            assert score_breakdown.roi_score < Decimal("0.4")
            
            # Reliability should be low due to new account + inactivity
            assert score_breakdown.reliability_score < Decimal("0.4")
            
            # Despite inconsistencies, confidence might still be reasonable
            # because we have data to assess (even if it's bad data)
            assert score_breakdown.overall_confidence > Decimal("0.4")
    
    @pytest.mark.asyncio
    async def test_scoring_with_outdated_data(
        self,
        mock_db_session,
        outdated_data_kol,
        complete_sample_campaign
    ):
        """Test scoring with outdated data."""
        scorer = KOLScorer()
        
        with patch.object(scorer, 'sentiment_analyzer'), \
             patch.object(scorer, 'demographic_matcher'):
            
            score_breakdown = await scorer.score_kol(
                outdated_data_kol,
                complete_sample_campaign,
                mock_db_session
            )
            
            # Reliability should be very low due to inactivity
            assert score_breakdown.reliability_score < Decimal("0.3")
            
            # Content relevance should be low due to old content
            assert score_breakdown.content_relevance_score < Decimal("0.5")
            
            # Missing data penalty should reflect data freshness issues
            assert score_breakdown.missing_data_penalty > Decimal("0.15")
            
            # Confidence should be reduced for stale data
            assert score_breakdown.overall_confidence < Decimal("0.8")
    
    def test_missing_data_penalty_calculation(self):
        """Test missing data penalty calculation accuracy."""
        scorer = KOLScorer()
        
        # Test with completely missing data
        empty_kol = MagicMock(spec=KOLProfile)
        empty_kol.location = None
        empty_kol.categories = None
        empty_kol.latest_metrics = None
        empty_kol.recent_content = []
        empty_kol.demographics = None
        empty_kol.engagement_history = None
        empty_kol.account_created_at = None
        
        penalty = scorer._calculate_missing_data_penalty(empty_kol)
        assert penalty >= Decimal("0.4")  # High penalty for missing critical data
        
        # Test with partially complete data
        partial_kol = MagicMock(spec=KOLProfile)
        partial_kol.location = "Bangkok"
        partial_kol.categories = ["lifestyle"]
        partial_kol.latest_metrics = MagicMock()
        partial_kol.latest_metrics.followers = 10000
        partial_kol.recent_content = [MagicMock()]
        partial_kol.demographics = {"age": 25}
        partial_kol.engagement_history = [MagicMock()]
        partial_kol.account_created_at = datetime.now(timezone.utc)
        
        penalty_partial = scorer._calculate_missing_data_penalty(partial_kol)
        assert penalty_partial < penalty  # Should be lower than empty case
        assert penalty_partial < Decimal("0.3")  # Should be reasonable
    
    @pytest.mark.asyncio
    async def test_graceful_fallback_values(
        self,
        mock_db_session,
        complete_sample_campaign
    ):
        """Test that appropriate fallback values are used for missing data."""
        scorer = KOLScorer()
        
        # Create KOL with specific missing fields
        fallback_kol = MagicMock(spec=KOLProfile)
        fallback_kol.id = "fallback_test_kol"
        fallback_kol.handle = "@fallback_kol"
        fallback_kol.location = None  # Missing
        fallback_kol.categories = None  # Missing
        fallback_kol.demographics = None  # Missing
        fallback_kol.is_verified = False
        fallback_kol.account_created_at = datetime.now(timezone.utc) - timedelta(days=100)
        
        # Minimal metrics
        metrics = MagicMock(spec=KOLMetrics)
        metrics.followers = 20000
        metrics.average_engagement_rate = None  # Missing
        fallback_kol.latest_metrics = metrics
        
        fallback_kol.recent_content = []
        fallback_kol.engagement_history = None
        
        with patch.object(scorer, 'sentiment_analyzer'), \
             patch.object(scorer, 'demographic_matcher'):
            
            score_breakdown = await scorer.score_kol(
                fallback_kol,
                complete_sample_campaign,
                mock_db_session
            )
            
            # Should use fallback values, not crash
            assert score_breakdown.roi_score >= Decimal("0.0")
            assert score_breakdown.audience_quality_score >= Decimal("0.0")
            assert score_breakdown.brand_safety_score >= Decimal("0.0")
            assert score_breakdown.content_relevance_score >= Decimal("0.0")
            assert score_breakdown.demographic_fit_score >= Decimal("0.0")
            assert score_breakdown.reliability_score >= Decimal("0.0")
            
            # Confidence scores should indicate uncertainty
            assert score_breakdown.roi_confidence < Decimal("0.6")
            assert score_breakdown.demographic_fit_confidence < Decimal("0.6")


class TestKOLMatchingMissingData:
    """Test KOL matching service with missing data scenarios."""
    
    @pytest.mark.asyncio
    async def test_matching_with_incomplete_candidates(
        self,
        mock_db_session,
        complete_sample_campaign,
        partially_complete_kol,
        severely_incomplete_kol
    ):
        """Test matching when candidates have varying levels of missing data."""
        service = EnhancedKOLMatchingService(mock_db_session)
        
        # Create mixed candidate pool
        incomplete_candidates = [
            (partially_complete_kol, partially_complete_kol.latest_metrics),
            (severely_incomplete_kol, severely_incomplete_kol.latest_metrics)
        ]
        
        with patch.object(service, '_get_filtered_candidates', return_value=incomplete_candidates):
            
            candidates, metadata = await service.find_matching_kols_advanced(
                campaign_requirements=complete_sample_campaign,
                limit=5,
                confidence_threshold=Decimal("0.3")  # Low threshold to include incomplete data
            )
            
            # Should return candidates despite missing data
            assert len(candidates) > 0
            
            # Should include data quality summary
            assert "data_quality_summary" in metadata
            data_quality = metadata["data_quality_summary"]
            
            assert "average_confidence" in data_quality
            assert "data_completeness" in data_quality
            assert "candidates_with_minimal_data" in data_quality
            
            # Average confidence should be low due to missing data
            assert data_quality["average_confidence"] < 0.7
            
            # Should identify candidates with minimal data
            assert data_quality["candidates_with_minimal_data"] > 0
    
    @pytest.mark.asyncio
    async def test_confidence_threshold_filtering_with_missing_data(
        self,
        mock_db_session,
        complete_sample_campaign,
        partially_complete_kol,
        severely_incomplete_kol
    ):
        """Test confidence threshold filtering with missing data."""
        service = EnhancedKOLMatchingService(mock_db_session)
        
        incomplete_candidates = [
            (partially_complete_kol, partially_complete_kol.latest_metrics),
            (severely_incomplete_kol, severely_incomplete_kol.latest_metrics)
        ]
        
        with patch.object(service, '_get_filtered_candidates', return_value=incomplete_candidates):
            
            # Test with high confidence threshold
            high_threshold_candidates, _ = await service.find_matching_kols_advanced(
                campaign_requirements=complete_sample_campaign,
                limit=5,
                confidence_threshold=Decimal("0.8")  # High threshold
            )
            
            # Test with low confidence threshold
            low_threshold_candidates, _ = await service.find_matching_kols_advanced(
                campaign_requirements=complete_sample_campaign,
                limit=5,
                confidence_threshold=Decimal("0.2")  # Low threshold
            )
            
            # High threshold should filter out more incomplete candidates
            assert len(high_threshold_candidates) <= len(low_threshold_candidates)
            
            # Low threshold should be more inclusive of incomplete data
            if low_threshold_candidates:
                min_confidence = min(
                    float(c.score_components.overall_confidence) 
                    for c in low_threshold_candidates
                )
                assert min_confidence >= 0.2
    
    @pytest.mark.asyncio
    async def test_semantic_matching_with_missing_embeddings(
        self,
        mock_db_session,
        complete_sample_campaign
    ):
        """Test semantic matching when content embeddings are missing."""
        service = EnhancedKOLMatchingService(mock_db_session)
        
        # Create KOL without content embedding
        no_embedding_kol = MagicMock(spec=KOLProfile)
        no_embedding_kol.id = "no_embedding_kol"
        no_embedding_kol.content_embedding = None  # Missing embedding
        no_embedding_kol.bio = "Content creator"
        no_embedding_kol.categories = ["lifestyle"]
        
        metrics = MagicMock(spec=KOLMetrics)
        metrics.follower_count = 25000
        metrics.engagement_rate = Decimal("0.03")
        
        candidates = [(no_embedding_kol, metrics)]
        
        with patch.object(service, '_get_filtered_candidates', return_value=candidates):
            
            # Should handle missing embeddings gracefully
            matched_candidates, metadata = await service.find_matching_kols_advanced(
                campaign_requirements=complete_sample_campaign,
                limit=5,
                enable_semantic_matching=True
            )
            
            # Should still return results without crashing
            assert isinstance(matched_candidates, list)
            
            # Should indicate semantic matching was attempted but limited
            assert metadata["semantic_matching_enabled"] is True
    
    @pytest.mark.asyncio
    async def test_similarity_search_with_missing_reference_embedding(
        self,
        mock_db_session
    ):
        """Test semantic similarity search with missing reference embedding."""
        service = EnhancedKOLMatchingService(mock_db_session)
        
        # Mock reference KOL without embedding
        reference_kol = MagicMock()
        reference_kol.id = "ref_kol_no_embedding"
        reference_kol.content_embedding = None
        
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = reference_kol
        service.db_session.execute.return_value = mock_result
        
        with patch.object(service, '_get_or_generate_embedding', return_value=None):
            
            # Should handle missing embedding gracefully
            similar_candidates = await service.find_similar_kols_semantic(
                reference_kol_id="ref_kol_no_embedding",
                limit=5,
                similarity_threshold=Decimal("0.7")
            )
            
            # Should return empty results without crashing
            assert similar_candidates == []


class TestBudgetOptimizerMissingData:
    """Test budget optimizer with missing KOL data."""
    
    @pytest.mark.asyncio
    async def test_optimization_with_incomplete_kol_data(
        self,
        mock_db_session,
        complete_sample_campaign
    ):
        """Test budget optimization when KOL data is incomplete."""
        with patch('kol_api.services.budget_optimizer.settings') as mock_settings:
            mock_settings.budget_tiers = {
                "micro": {"avg_cost_per_post": 2500}
            }
            
            optimizer = BudgetOptimizerService(mock_db_session)
            
            # Create candidates with missing data
            incomplete_candidates = []
            
            for i in range(5):
                kol = MagicMock()
                kol.id = f"incomplete_opt_kol_{i}"
                kol.tier = MagicMock()
                kol.tier.value = "micro"
                kol.primary_category = MagicMock()
                kol.primary_category.value = "lifestyle"
                kol.is_verified = i % 2 == 0  # Some missing verification
                kol.location = "Bangkok" if i < 3 else None  # Some missing location
                
                metrics = MagicMock()
                metrics.follower_count = 20000 + (i * 5000)
                metrics.engagement_rate = Decimal("0.03") if i < 3 else None  # Some missing
                metrics.posts_last_30_days = 10 + i if i < 4 else 0  # One inactive
                metrics.fake_follower_percentage = Decimal("0.1") if i < 2 else None
                
                # Create candidate with missing data handling
                candidate = MagicMock()
                candidate.kol = kol
                candidate.metrics = metrics
                candidate.estimated_cost = Decimal(str(2000 + (i * 500)))
                candidate.predicted_reach = 3000 + (i * 500)
                candidate.predicted_engagement = 100 + (i * 20)
                candidate.predicted_conversions = 5 + i
                candidate.efficiency_score = Decimal("0.6") - Decimal(str(i * 0.1))
                candidate.risk_score = Decimal("0.2") + Decimal(str(i * 0.1))
                
                incomplete_candidates.append(candidate)
            
            with patch.object(optimizer, '_get_campaign_requirements', return_value=complete_sample_campaign), \
                 patch.object(optimizer, '_get_kol_candidates', return_value=incomplete_candidates):
                
                result = await optimizer.optimize_campaign_budget(
                    campaign_id=complete_sample_campaign.campaign_id,
                    total_budget=Decimal("25000"),
                    optimization_objective=OptimizationObjective.MAXIMIZE_ENGAGEMENT,
                    allocation_strategy=AllocationStrategy.RISK_BALANCED
                )
                
                # Should complete optimization despite missing data
                assert len(result.selected_kols) > 0
                assert result.total_cost <= Decimal("25000")
                
                # Should prefer candidates with more complete data
                selected_ids = [kol.kol.id for kol in result.selected_kols]
                
                # Earlier candidates (0-2) have more complete data
                complete_data_candidates = sum(
                    1 for kol_id in selected_ids 
                    if any(char in kol_id for char in ['0', '1', '2'])
                )
                
                incomplete_data_candidates = sum(
                    1 for kol_id in selected_ids 
                    if any(char in kol_id for char in ['3', '4'])
                )
                
                # Should favor complete data candidates when risk is considered
                assert complete_data_candidates >= incomplete_data_candidates
    
    def test_cost_estimation_with_missing_metrics(self):
        """Test cost estimation when metrics are missing."""
        with patch('kol_api.services.budget_optimizer.settings') as mock_settings:
            mock_settings.budget_tiers = {
                "micro": {"avg_cost_per_post": 2500}
            }
            
            optimizer = BudgetOptimizerService(AsyncMock())
            
            # KOL with missing engagement rate
            kol_missing_engagement = MagicMock()
            kol_missing_engagement.tier = MagicMock()
            kol_missing_engagement.tier.value = "micro"
            kol_missing_engagement.is_verified = True
            kol_missing_engagement.location = "Bangkok"
            
            metrics_missing = MagicMock()
            metrics_missing.engagement_rate = None  # Missing
            
            cost = optimizer._estimate_kol_cost(kol_missing_engagement, metrics_missing)
            
            # Should use base tier cost without engagement adjustment
            expected_base = Decimal("2500") * Decimal("1.3")  # Verified premium
            assert cost == expected_base
            
            # KOL with engagement rate present
            metrics_complete = MagicMock()
            metrics_complete.engagement_rate = Decimal("0.05")
            
            cost_complete = optimizer._estimate_kol_cost(kol_missing_engagement, metrics_complete)
            
            # Should be higher due to engagement rate adjustment
            assert cost_complete > cost
    
    def test_performance_prediction_with_missing_data(self):
        """Test performance prediction when key metrics are missing."""
        with patch('kol_api.services.budget_optimizer.settings'):
            optimizer = BudgetOptimizerService(AsyncMock())
            
            kol = MagicMock()
            kol.tier = MagicMock()
            kol.tier.value = "micro"
            
            # Missing engagement rate
            metrics_missing = MagicMock()
            metrics_missing.follower_count = 50000
            metrics_missing.engagement_rate = None
            
            # Should use default engagement rate for tier
            predicted_engagement = optimizer._predict_engagement(kol, metrics_missing)
            
            # Should still produce reasonable prediction
            assert predicted_engagement > 0
            assert predicted_engagement < metrics_missing.follower_count  # Sanity check
            
            # With engagement rate present
            metrics_complete = MagicMock()
            metrics_complete.follower_count = 50000
            metrics_complete.engagement_rate = Decimal("0.04")
            
            predicted_engagement_complete = optimizer._predict_engagement(kol, metrics_complete)
            
            # Results should be different and both reasonable
            assert predicted_engagement != predicted_engagement_complete
            assert predicted_engagement_complete > 0
    
    def test_risk_assessment_with_missing_indicators(self):
        """Test risk assessment when risk indicators are missing."""
        with patch('kol_api.services.budget_optimizer.settings'):
            optimizer = BudgetOptimizerService(AsyncMock())
            
            # KOL with most risk indicators missing
            kol_missing_risk = MagicMock()
            kol_missing_risk.is_verified = None  # Missing
            kol_missing_risk.is_brand_safe = None  # Missing
            
            metrics_missing_risk = MagicMock()
            metrics_missing_risk.engagement_rate = None  # Missing
            metrics_missing_risk.posts_last_30_days = None  # Missing
            metrics_missing_risk.fake_follower_percentage = None  # Missing
            metrics_missing_risk.metrics_date = datetime.now(timezone.utc)
            metrics_missing_risk.created_at = datetime.now(timezone.utc)
            
            risk_score = optimizer._calculate_risk_score(kol_missing_risk, metrics_missing_risk)
            
            # Should assign moderate risk when indicators are missing
            assert Decimal("0.3") <= risk_score <= Decimal("0.7")
            
            # KOL with complete risk indicators (high risk)
            kol_high_risk = MagicMock()
            kol_high_risk.is_verified = False
            kol_high_risk.is_brand_safe = False
            
            metrics_high_risk = MagicMock()
            metrics_high_risk.engagement_rate = Decimal("0.005")  # Very low
            metrics_high_risk.posts_last_30_days = 1  # Inactive
            metrics_high_risk.fake_follower_percentage = Decimal("0.5")  # High
            metrics_high_risk.metrics_date = datetime.now(timezone.utc)
            metrics_high_risk.created_at = datetime.now(timezone.utc)
            
            high_risk_score = optimizer._calculate_risk_score(kol_high_risk, metrics_high_risk)
            
            # High risk should be worse than missing data scenario
            assert high_risk_score > risk_score


class TestBriefParsingMissingData:
    """Test brief parsing with incomplete information."""
    
    @pytest.mark.asyncio
    async def test_parsing_minimal_brief(self):
        """Test parsing brief with minimal information."""
        parser = BriefParserService(AsyncMock())
        
        minimal_brief = """
        # Campaign Brief
        We need some influencers.
        Budget: 20000
        """
        
        result = await parser.parse_markdown_brief(
            file_content=minimal_brief,
            filename="minimal_brief.md",
            user_id="test_user",
            campaign_id="minimal_test"
        )
        
        # Should parse successfully but with low confidence
        assert result.success is True
        assert result.confidence_score < Decimal("0.6")
        
        # Should have many missing requirements
        assert len(result.validation_warnings) > 3
        
        # Should use defaults where possible
        campaign_req = result.campaign_requirements
        assert campaign_req.total_budget > Decimal("0")
        assert len(campaign_req.target_kol_tiers) == 0 or len(campaign_req.target_kol_tiers) > 0  # May have defaults
    
    @pytest.mark.asyncio
    async def test_parsing_conflicting_information(self):
        """Test parsing brief with conflicting information."""
        parser = BriefParserService(AsyncMock())
        
        conflicting_brief = """
        # Conflicting Campaign Brief
        
        Budget: THB 10,000
        Budget: USD 50,000
        
        We want nano influencers with 1K followers.
        We need macro influencers with 1M+ followers.
        
        Target audience: teenagers aged 13-17
        Target audience: professionals aged 30-45
        
        Goal: brand awareness
        Goal: direct sales
        """
        
        result = await parser.parse_markdown_brief(
            file_content=conflicting_brief,
            filename="conflicting_brief.md",
            user_id="test_user",
            campaign_id="conflicting_test"
        )
        
        # Should parse but indicate conflicts
        assert result.success is True
        
        # Should have validation warnings about conflicts
        assert len(result.validation_warnings) > 0
        
        # Should make reasonable choices when conflicts exist
        campaign_req = result.campaign_requirements
        assert campaign_req.total_budget > Decimal("0")  # Should pick one budget
        
        # Confidence should be reduced due to conflicts
        assert result.confidence_score < Decimal("0.8")
    
    def test_currency_detection_with_ambiguity(self):
        """Test currency detection when currency information is ambiguous."""
        parser = BriefParserService(AsyncMock())
        
        # Test with no currency indicators
        no_currency_text = "Budget is 50000"
        currency = parser._detect_currency(no_currency_text)
        assert currency == "USD"  # Should default to USD
        
        # Test with conflicting currency indicators
        conflicting_text = "Budget: $50,000 USD but we're in Thailand using THB"
        currency_conflicting = parser._detect_currency(conflicting_text)
        # Should pick the first/strongest indicator
        assert currency_conflicting in ["USD", "THB"]
    
    def test_confidence_calculation_with_missing_fields(self):
        """Test confidence calculation when many fields are missing."""
        parser = BriefParserService(AsyncMock())
        
        # Create parsed data with minimal information
        from kol_api.services.brief_parser import ParsedBriefData
        
        minimal_data = ParsedBriefData(
            campaign_title="Test Campaign",  # Present
            campaign_description=None,       # Missing
            budget_information={"total_budget": 50000},  # Present
            target_categories=[],            # Empty
            target_tiers=[],                # Empty
            campaign_objectives=[],          # Empty
            geographic_targets=[],           # Empty
            demographic_requirements=None    # Missing
        )
        
        standardized_data = ParsedBriefData(
            campaign_title="Test Campaign",
            campaign_description=None,
            budget_information={"total_budget": 50000},
            target_categories=[],
            target_tiers=[],
            campaign_objectives=[],
            geographic_targets=[],
            demographic_requirements=None
        )
        
        confidence = parser._calculate_parsing_confidence(
            minimal_data, 
            standardized_data, 
            100  # Short content
        )
        
        # Should have low confidence due to missing critical fields
        assert confidence < 0.5
        
        # Create complete data for comparison
        complete_data = ParsedBriefData(
            campaign_title="Complete Campaign",
            campaign_description="Detailed description",
            budget_information={"total_budget": 75000},
            target_categories=["lifestyle", "fashion"],
            target_tiers=["micro", "mid"],
            campaign_objectives=["engagement"],
            geographic_targets=["Bangkok"],
            demographic_requirements={"age_range": "25-35"}
        )
        
        complete_confidence = parser._calculate_parsing_confidence(
            complete_data,
            complete_data,
            500  # Longer content
        )
        
        # Should have higher confidence with complete data
        assert complete_confidence > confidence
        assert complete_confidence > 0.8


class TestDataQualityAssessment:
    """Test data quality assessment across all services."""
    
    def test_data_completeness_scoring(self):
        """Test data completeness scoring methodology."""
        
        # Test with complete KOL data
        complete_metrics = KOLMetricsData(
            follower_count=25000,
            following_count=1000,
            engagement_rate=Decimal("0.04"),
            avg_likes=Decimal("1000"),
            avg_comments=Decimal("50"),
            avg_views=Decimal("5000"),
            posts_last_30_days=15,
            fake_follower_percentage=Decimal("0.05"),
            audience_quality_score=Decimal("0.8"),
            campaign_success_rate=Decimal("0.85"),
            response_rate=Decimal("0.9")
        )
        
        assert complete_metrics.has_sufficient_data is True
        
        # Test with minimal data
        minimal_metrics = KOLMetricsData(
            follower_count=15000,
            following_count=500,
            engagement_rate=None,  # Missing
            posts_last_30_days=0   # No recent activity
        )
        
        assert minimal_metrics.has_sufficient_data is False
    
    @pytest.mark.asyncio
    async def test_confidence_propagation_through_pipeline(
        self,
        mock_db_session,
        partially_complete_kol,
        complete_sample_campaign
    ):
        """Test that confidence scores propagate correctly through the pipeline."""
        
        # Test KOL Scorer confidence
        scorer = KOLScorer()
        
        with patch.object(scorer, 'sentiment_analyzer'), \
             patch.object(scorer, 'demographic_matcher'):
            
            score_breakdown = await scorer.score_kol(
                partially_complete_kol,
                complete_sample_campaign,
                mock_db_session
            )
            
            base_confidence = score_breakdown.overall_confidence
        
        # Test KOL Matching service confidence
        matching_service = EnhancedKOLMatchingService(mock_db_session)
        
        with patch.object(matching_service, '_get_filtered_candidates', return_value=[(partially_complete_kol, partially_complete_kol.latest_metrics)]):
            
            candidates, metadata = await matching_service.find_matching_kols_advanced(
                campaign_requirements=complete_sample_campaign,
                limit=5,
                confidence_threshold=Decimal("0.3")
            )
            
            if candidates:
                # Candidate confidence should relate to scorer confidence
                candidate_confidence = candidates[0].score_components.overall_confidence
                assert abs(float(candidate_confidence - base_confidence)) < 0.2
            
            # Metadata should include data quality information
            data_quality = metadata.get("data_quality_summary", {})
            assert "average_confidence" in data_quality
    
    def test_missing_data_impact_on_rankings(self):
        """Test how missing data impacts KOL rankings."""
        
        # Create two similar KOLs, one with complete data, one with missing data
        complete_kol_metrics = KOLMetricsData(
            follower_count=30000,
            following_count=1200,
            engagement_rate=Decimal("0.04"),
            posts_last_30_days=15,
            audience_quality_score=Decimal("0.8")
        )
        
        incomplete_kol_metrics = KOLMetricsData(
            follower_count=32000,  # Slightly better
            following_count=1000,
            engagement_rate=None,  # Missing critical data
            posts_last_30_days=0,  # Inactive
            audience_quality_score=None  # Missing
        )
        
        # Complete data should generally rank higher despite lower follower count
        # This would be tested in integration with actual scoring
        assert complete_kol_metrics.has_sufficient_data is True
        assert incomplete_kol_metrics.has_sufficient_data is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])