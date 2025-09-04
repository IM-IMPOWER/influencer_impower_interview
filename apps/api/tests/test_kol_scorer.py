"""
Comprehensive tests for the KOL Scorer (POC2) - Multi-Factor Scoring System

AIDEV-NOTE: Production-ready tests for the sophisticated KOL scoring algorithm with
comprehensive coverage of all scoring components, edge cases, and missing data scenarios.
"""
import pytest
import asyncio
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import List, Dict, Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np

from kol_api.services.scoring.kol_scorer import KOLScorer, ScoreBreakdown, ScoreComponent
from kol_api.database.models.kol import KOLProfile, KOLMetrics, KOLContent
from kol_api.database.models.campaign import Campaign, CampaignRequirements
from kol_api.utils.ml_models import SentimentAnalyzer, DemographicMatcher
from kol_api.utils.text_processing import extract_keywords, calculate_similarity


# AIDEV-NOTE: Test fixtures for comprehensive data scenarios

@pytest.fixture
def mock_db_session():
    """Mock database session."""
    return AsyncMock()


@pytest.fixture
def sample_sentiment_analyzer():
    """Mock sentiment analyzer."""
    analyzer = MagicMock(spec=SentimentAnalyzer)
    
    async def mock_analyze(text):
        sentiment = MagicMock()
        # Default positive sentiment for most test cases
        if "terrible" in text.lower() or "awful" in text.lower():
            sentiment.score = 0.2
        elif "amazing" in text.lower() or "great" in text.lower():
            sentiment.score = 0.9
        else:
            sentiment.score = 0.7
        return sentiment
    
    analyzer.analyze = mock_analyze
    return analyzer


@pytest.fixture
def sample_demographic_matcher():
    """Mock demographic matcher."""
    matcher = MagicMock(spec=DemographicMatcher)
    
    async def mock_match_interests(kol_interests, target_interests):
        # Simple overlap calculation
        if not kol_interests or not target_interests:
            return 0.5
        overlap = len(set(kol_interests) & set(target_interests))
        return min(1.0, overlap / len(target_interests))
    
    matcher.match_interests = mock_match_interests
    return matcher


@pytest.fixture
def high_quality_kol_profile():
    """High quality KOL profile with complete data."""
    kol = MagicMock(spec=KOLProfile)
    kol.id = "high_quality_kol_123"
    kol.handle = "@high_quality_kol"
    kol.platform = "tiktok"
    kol.bio = "Lifestyle content creator sharing daily tips and inspiration"
    kol.location = "Bangkok, Thailand"
    kol.categories = ["lifestyle", "fashion", "beauty"]
    kol.is_verified = True
    kol.account_created_at = datetime.now(timezone.utc) - timedelta(days=730)
    kol.languages = ["th", "en"]
    kol.demographics = {
        "average_age": 28,
        "interests": ["fashion", "travel", "food"]
    }
    
    # High quality metrics
    metrics = MagicMock(spec=KOLMetrics)
    metrics.followers = 75000
    metrics.average_engagement_rate = Decimal("0.055")  # 5.5% - excellent
    metrics.posts_last_30_days = 20
    metrics.fake_follower_percentage = Decimal("0.02")  # Very low
    
    kol.latest_metrics = metrics
    
    # Recent content with positive sentiment
    content1 = MagicMock(spec=KOLContent)
    content1.caption = "Amazing new product launch! So excited to share this with you all! #lifestyle #fashion"
    content1.hashtags = ["lifestyle", "fashion", "excited", "newproduct"]
    content1.created_at = datetime.now(timezone.utc) - timedelta(days=2)
    
    content2 = MagicMock(spec=KOLContent)
    content2.caption = "Great day at the beach! Living my best life! #travel #lifestyle"
    content2.hashtags = ["travel", "lifestyle", "beach", "bestlife"]
    content2.created_at = datetime.now(timezone.utc) - timedelta(days=5)
    
    kol.recent_content = [content1, content2]
    
    # Engagement history for consistency check
    engagement_history = []
    for i in range(10):
        history = MagicMock()
        history.engagement_rate = 0.052 + (i % 5) * 0.003  # Consistent around 5.2-5.5%
        engagement_history.append(history)
    
    kol.engagement_history = engagement_history
    
    return kol


@pytest.fixture
def low_quality_kol_profile():
    """Low quality KOL profile with missing data and red flags."""
    kol = MagicMock(spec=KOLProfile)
    kol.id = "low_quality_kol_456"
    kol.handle = "@low_quality_kol"
    kol.platform = "tiktok"
    kol.bio = None  # Missing bio
    kol.location = None  # Missing location
    kol.categories = []  # No categories
    kol.is_verified = False
    kol.account_created_at = datetime.now(timezone.utc) - timedelta(days=30)  # Very new
    kol.languages = None
    kol.demographics = None
    
    # Poor metrics
    metrics = MagicMock(spec=KOLMetrics)
    metrics.followers = 10000
    metrics.average_engagement_rate = Decimal("0.008")  # 0.8% - very low
    metrics.posts_last_30_days = 2  # Inactive
    metrics.fake_follower_percentage = Decimal("0.35")  # High fake percentage
    
    kol.latest_metrics = metrics
    
    # Limited content with negative sentiment
    content1 = MagicMock(spec=KOLContent)
    content1.caption = "Terrible day, hate everything"
    content1.hashtags = []
    content1.created_at = datetime.now(timezone.utc) - timedelta(days=15)
    
    kol.recent_content = [content1]
    kol.engagement_history = []  # No engagement history
    
    return kol


@pytest.fixture 
def missing_data_kol_profile():
    """KOL profile with various missing data points for confidence testing."""
    kol = MagicMock(spec=KOLProfile)
    kol.id = "missing_data_kol_789"
    kol.handle = "@missing_data_kol"
    kol.platform = "tiktok"
    kol.bio = "Content creator"
    kol.location = "Thailand"
    kol.categories = ["lifestyle"]
    kol.is_verified = True
    kol.account_created_at = datetime.now(timezone.utc) - timedelta(days=365)
    kol.languages = ["th"]
    kol.demographics = None  # Missing demographics
    
    # Metrics with missing values
    metrics = MagicMock(spec=KOLMetrics)
    metrics.followers = 25000
    metrics.average_engagement_rate = None  # Missing engagement rate
    metrics.posts_last_30_days = 8
    metrics.fake_follower_percentage = None  # Missing fake follower data
    
    kol.latest_metrics = metrics
    kol.recent_content = []  # No recent content
    kol.engagement_history = None  # No engagement history
    
    return kol


@pytest.fixture
def sample_campaign():
    """Sample campaign for testing."""
    campaign = MagicMock(spec=Campaign)
    campaign.id = "test_campaign_123"
    campaign.campaign_type = "engagement"
    campaign.objective = MagicMock()
    campaign.objective.value = "maximize_engagement"
    
    # Campaign requirements
    requirements = MagicMock(spec=CampaignRequirements)
    requirements.required_categories = ["lifestyle", "fashion"]
    requirements.target_keywords = ["fashion", "lifestyle", "style"]
    requirements.target_locations = ["Bangkok", "Thailand"]
    requirements.target_age_range = (25, 35)
    requirements.target_interests = ["fashion", "travel", "food"]
    
    campaign.requirements = requirements
    return campaign


@pytest.fixture
def kol_scorer(mock_db_session, sample_sentiment_analyzer, sample_demographic_matcher):
    """KOL Scorer instance with mocked dependencies."""
    with patch('kol_api.services.scoring.kol_scorer.SentimentAnalyzer', return_value=sample_sentiment_analyzer), \
         patch('kol_api.services.scoring.kol_scorer.DemographicMatcher', return_value=sample_demographic_matcher):
        return KOLScorer()


# AIDEV-NOTE: Core Scoring Component Tests

class TestKOLScorerComponents:
    """Test individual scoring components with comprehensive coverage."""
    
    @pytest.mark.asyncio
    async def test_roi_score_calculation_high_performance(
        self, 
        kol_scorer, 
        high_quality_kol_profile, 
        sample_campaign,
        mock_db_session
    ):
        """Test ROI score calculation for high-performance KOL."""
        roi_score, roi_confidence = await kol_scorer._calculate_roi_score(
            high_quality_kol_profile, 
            sample_campaign, 
            mock_db_session
        )
        
        # High-performance KOL should have good ROI
        assert roi_score > Decimal("0.5")
        assert roi_confidence >= Decimal("0.8")  # High confidence with good data
        
        # Test score normalization (should be between 0 and 1)
        assert Decimal("0.0") <= roi_score <= Decimal("1.0")
    
    @pytest.mark.asyncio
    async def test_roi_score_calculation_missing_metrics(
        self,
        kol_scorer,
        missing_data_kol_profile,
        sample_campaign,
        mock_db_session
    ):
        """Test ROI score with missing metrics data."""
        # Remove metrics to test handling
        missing_data_kol_profile.latest_metrics = None
        
        roi_score, roi_confidence = await kol_scorer._calculate_roi_score(
            missing_data_kol_profile,
            sample_campaign,
            mock_db_session
        )
        
        # Should gracefully handle missing data
        assert roi_score == Decimal("0.0")
        assert roi_confidence == Decimal("0.0")
    
    @pytest.mark.asyncio
    async def test_roi_score_different_campaign_types(
        self,
        kol_scorer,
        high_quality_kol_profile,
        sample_campaign,
        mock_db_session
    ):
        """Test ROI score calculation for different campaign types."""
        # Test conversion campaign (higher conversion rate)
        sample_campaign.campaign_type = "conversion"
        roi_conversion, conf_conversion = await kol_scorer._calculate_roi_score(
            high_quality_kol_profile, sample_campaign, mock_db_session
        )
        
        # Test awareness campaign (lower conversion rate)
        sample_campaign.campaign_type = "awareness"
        roi_awareness, conf_awareness = await kol_scorer._calculate_roi_score(
            high_quality_kol_profile, sample_campaign, mock_db_session
        )
        
        # Conversion campaigns should have higher ROI than awareness
        assert roi_conversion > roi_awareness
        assert conf_conversion == conf_awareness  # Same confidence with same data quality
    
    @pytest.mark.asyncio
    async def test_audience_quality_score_high_quality(
        self,
        kol_scorer,
        high_quality_kol_profile,
        mock_db_session
    ):
        """Test audience quality score for high-quality KOL."""
        aud_score, aud_confidence = await kol_scorer._calculate_audience_quality_score(
            high_quality_kol_profile,
            mock_db_session
        )
        
        # High quality KOL should have good audience score
        assert aud_score > Decimal("0.7")
        assert aud_confidence >= Decimal("0.8")
    
    @pytest.mark.asyncio
    async def test_audience_quality_score_fake_followers(
        self,
        kol_scorer,
        low_quality_kol_profile,
        mock_db_session
    ):
        """Test audience quality score with high fake follower percentage."""
        aud_score, aud_confidence = await kol_scorer._calculate_audience_quality_score(
            low_quality_kol_profile,
            mock_db_session
        )
        
        # High fake followers should result in low audience quality
        assert aud_score < Decimal("0.5")
        # Should still be confident in the assessment
        assert aud_confidence >= Decimal("0.5")
    
    @pytest.mark.asyncio
    async def test_audience_quality_engagement_rate_analysis(
        self,
        kol_scorer,
        mock_db_session
    ):
        """Test audience quality with different engagement rate scenarios."""
        # Create KOL with suspiciously high engagement rate
        kol_high_engagement = MagicMock(spec=KOLProfile)
        kol_high_engagement.handle = "@high_engagement"
        kol_high_engagement.engagement_history = []
        
        metrics_high = MagicMock(spec=KOLMetrics)
        metrics_high.followers = 10000
        metrics_high.average_engagement_rate = Decimal("0.15")  # 15% - suspicious
        kol_high_engagement.latest_metrics = metrics_high
        
        # Create KOL with very low engagement rate
        kol_low_engagement = MagicMock(spec=KOLProfile)
        kol_low_engagement.handle = "@low_engagement"
        kol_low_engagement.engagement_history = []
        
        metrics_low = MagicMock(spec=KOLMetrics)
        metrics_low.followers = 100000
        metrics_low.average_engagement_rate = Decimal("0.005")  # 0.5% - very low
        kol_low_engagement.latest_metrics = metrics_low
        
        # Test high engagement (suspicious)
        high_score, high_conf = await kol_scorer._calculate_audience_quality_score(
            kol_high_engagement, mock_db_session
        )
        
        # Test low engagement
        low_score, low_conf = await kol_scorer._calculate_audience_quality_score(
            kol_low_engagement, mock_db_session
        )
        
        # Very low engagement should be worse than suspicious high engagement
        assert low_score < high_score
        assert both scores reflect the authenticity concerns
    
    @pytest.mark.asyncio
    async def test_brand_safety_score_safe_content(
        self,
        kol_scorer,
        high_quality_kol_profile,
        mock_db_session
    ):
        """Test brand safety score with safe, positive content."""
        safety_score, safety_confidence = await kol_scorer._calculate_brand_safety_score(
            high_quality_kol_profile,
            mock_db_session
        )
        
        # Safe content should have high brand safety score
        assert safety_score > Decimal("0.8")
        assert safety_confidence >= Decimal("0.8")
    
    @pytest.mark.asyncio
    async def test_brand_safety_score_controversial_content(
        self,
        kol_scorer,
        mock_db_session
    ):
        """Test brand safety score with controversial content."""
        # Create KOL with controversial content
        controversial_kol = MagicMock(spec=KOLProfile)
        controversial_kol.id = "controversial_kol"
        controversial_kol.handle = "@controversial"
        controversial_kol.is_brand_safe = True
        controversial_kol.is_verified = False
        
        # Controversial content
        content1 = MagicMock(spec=KOLContent)
        content1.caption = "This political war situation is just terrible and full of hate"
        
        content2 = MagicMock(spec=KOLContent)
        content2.caption = "Another racist scandal in the industry, so much violence"
        
        controversial_kol.recent_content = [content1, content2]
        
        safety_score, safety_confidence = await kol_scorer._calculate_brand_safety_score(
            controversial_kol,
            mock_db_session
        )
        
        # Controversial content should result in lower safety score
        assert safety_score < Decimal("0.5")
        # Should still be confident in assessment
        assert safety_confidence >= Decimal("0.7")
    
    @pytest.mark.asyncio
    async def test_brand_safety_score_no_content(
        self,
        kol_scorer,
        mock_db_session
    ):
        """Test brand safety score with no recent content."""
        no_content_kol = MagicMock(spec=KOLProfile)
        no_content_kol.id = "no_content_kol"
        no_content_kol.handle = "@no_content"
        no_content_kol.is_brand_safe = True
        no_content_kol.is_verified = True
        no_content_kol.recent_content = []  # No content
        
        safety_score, safety_confidence = await kol_scorer._calculate_brand_safety_score(
            no_content_kol,
            mock_db_session
        )
        
        # Should provide default moderate score with low confidence
        assert Decimal("0.5") <= safety_score <= Decimal("0.8")
        assert safety_confidence <= Decimal("0.5")  # Low confidence without content
    
    def test_controversial_content_detection(self, kol_scorer):
        """Test controversial content keyword detection."""
        # Test safe content
        safe_score = kol_scorer._check_controversial_content("Amazing product launch today!")
        assert safe_score > 0.8
        
        # Test controversial content
        controversial_score = kol_scorer._check_controversial_content(
            "This war and violence is terrible, full of hate and racist behavior"
        )
        assert controversial_score < 0.2
        
        # Test mildly controversial
        mild_score = kol_scorer._check_controversial_content("Political discussion about alcohol")
        assert 0.2 <= mild_score <= 0.8
    
    @pytest.mark.asyncio
    async def test_content_relevance_score_perfect_match(
        self,
        kol_scorer,
        high_quality_kol_profile,
        sample_campaign,
        mock_db_session
    ):
        """Test content relevance with perfect category and hashtag match."""
        # Ensure perfect category match
        high_quality_kol_profile.categories = ["lifestyle", "fashion"]  # Matches campaign
        
        relevance_score, relevance_confidence = await kol_scorer._calculate_content_relevance_score(
            high_quality_kol_profile,
            sample_campaign,
            mock_db_session
        )
        
        # Perfect match should score highly
        assert relevance_score > Decimal("0.8")
        assert relevance_confidence >= Decimal("0.7")
    
    @pytest.mark.asyncio
    async def test_content_relevance_score_no_match(
        self,
        kol_scorer,
        mock_db_session
    ):
        """Test content relevance with no category or hashtag match."""
        # Create KOL with no matching categories
        no_match_kol = MagicMock(spec=KOLProfile)
        no_match_kol.id = "no_match_kol"
        no_match_kol.handle = "@no_match"
        no_match_kol.categories = ["gaming", "tech"]  # Different from lifestyle/fashion
        
        # Content with no matching hashtags
        content = MagicMock(spec=KOLContent)
        content.hashtags = ["gaming", "esports", "technology"]
        content.caption = "Gaming setup review and tech analysis"
        no_match_kol.recent_content = [content]
        
        # Sample campaign requiring lifestyle/fashion
        campaign = MagicMock(spec=Campaign)
        requirements = MagicMock()
        requirements.required_categories = ["lifestyle", "fashion"]
        requirements.target_keywords = ["fashion", "lifestyle", "style"]
        campaign.requirements = requirements
        
        relevance_score, relevance_confidence = await kol_scorer._calculate_content_relevance_score(
            no_match_kol,
            campaign,
            mock_db_session
        )
        
        # No match should result in low relevance
        assert relevance_score < Decimal("0.4")
    
    @pytest.mark.asyncio
    async def test_demographic_fit_score_perfect_match(
        self,
        kol_scorer,
        high_quality_kol_profile,
        sample_campaign,
        mock_db_session
    ):
        """Test demographic fit with perfect location and interest match."""
        demo_score, demo_confidence = await kol_scorer._calculate_demographic_fit_score(
            high_quality_kol_profile,
            sample_campaign,
            mock_db_session
        )
        
        # Should score well with matching location and interests
        assert demo_score > Decimal("0.6")
        assert demo_confidence >= Decimal("0.7")
    
    @pytest.mark.asyncio
    async def test_demographic_fit_score_age_range_match(
        self,
        kol_scorer,
        mock_db_session
    ):
        """Test demographic fit with different age range scenarios."""
        # Create campaign with age range 25-35
        campaign = MagicMock(spec=Campaign)
        requirements = MagicMock()
        requirements.target_locations = []
        requirements.target_age_range = (25, 35)
        requirements.target_interests = []
        campaign.requirements = requirements
        
        # KOL within age range
        kol_in_range = MagicMock(spec=KOLProfile)
        kol_in_range.id = "kol_in_range"
        kol_in_range.location = None
        kol_in_range.demographics = {"average_age": 28}  # Perfect match
        
        # KOL outside age range
        kol_out_range = MagicMock(spec=KOLProfile)
        kol_out_range.id = "kol_out_range"
        kol_out_range.location = None
        kol_out_range.demographics = {"average_age": 45}  # Way outside range
        
        # Test in-range KOL
        in_score, in_conf = await kol_scorer._calculate_demographic_fit_score(
            kol_in_range, campaign, mock_db_session
        )
        
        # Test out-of-range KOL
        out_score, out_conf = await kol_scorer._calculate_demographic_fit_score(
            kol_out_range, campaign, mock_db_session
        )
        
        # In-range should score significantly better
        assert in_score > out_score
        assert in_score > Decimal("0.8")
        assert out_score < Decimal("0.5")
    
    @pytest.mark.asyncio
    async def test_reliability_score_consistent_performer(
        self,
        kol_scorer,
        high_quality_kol_profile,
        mock_db_session
    ):
        """Test reliability score for consistent, active KOL."""
        reliability_score, reliability_confidence = await kol_scorer._calculate_reliability_score(
            high_quality_kol_profile,
            mock_db_session
        )
        
        # Consistent, active KOL should have high reliability
        assert reliability_score > Decimal("0.7")
        assert reliability_confidence >= Decimal("0.7")
    
    @pytest.mark.asyncio
    async def test_reliability_score_inconsistent_performer(
        self,
        kol_scorer,
        mock_db_session
    ):
        """Test reliability score for inconsistent KOL."""
        # Create KOL with inconsistent posting and engagement
        inconsistent_kol = MagicMock(spec=KOLProfile)
        inconsistent_kol.id = "inconsistent_kol"
        inconsistent_kol.handle = "@inconsistent"
        inconsistent_kol.account_created_at = datetime.now(timezone.utc) - timedelta(days=30)  # New account
        
        # Very irregular posting pattern
        content_dates = [
            datetime.now(timezone.utc) - timedelta(days=1),
            datetime.now(timezone.utc) - timedelta(days=15),
            datetime.now(timezone.utc) - timedelta(days=25),
        ]
        
        recent_content = []
        for date in content_dates:
            content = MagicMock()
            content.created_at = date
            recent_content.append(content)
        
        inconsistent_kol.recent_content = recent_content
        
        # Inconsistent engagement history
        engagement_history = []
        rates = [0.02, 0.08, 0.01, 0.09, 0.03, 0.07, 0.015]  # High variance
        for rate in rates:
            history = MagicMock()
            history.engagement_rate = rate
            engagement_history.append(history)
        
        inconsistent_kol.engagement_history = engagement_history
        
        reliability_score, reliability_confidence = await kol_scorer._calculate_reliability_score(
            inconsistent_kol,
            mock_db_session
        )
        
        # Inconsistent performance should result in lower reliability
        assert reliability_score < Decimal("0.6")
    
    def test_engagement_trend_calculation(self, kol_scorer):
        """Test engagement trend calculation for reliability assessment."""
        # Positive trend
        positive_history = []
        for i in range(10):
            history = MagicMock()
            history.engagement_rate = 0.02 + (i * 0.002)  # Increasing trend
            positive_history.append(history)
        
        positive_trend = kol_scorer._calculate_engagement_trend(positive_history)
        assert positive_trend > 0.5  # Should be positive
        
        # Negative trend
        negative_history = []
        for i in range(10):
            history = MagicMock()
            history.engagement_rate = 0.08 - (i * 0.003)  # Decreasing trend
            negative_history.append(history)
        
        negative_trend = kol_scorer._calculate_engagement_trend(negative_history)
        assert negative_trend < 0.5  # Should be negative
        
        # Stable trend
        stable_history = []
        for i in range(10):
            history = MagicMock()
            history.engagement_rate = 0.05 + (i % 2) * 0.001  # Very stable
            stable_history.append(history)
        
        stable_trend = kol_scorer._calculate_engagement_trend(stable_history)
        assert 0.4 <= stable_trend <= 0.6  # Should be near neutral


# AIDEV-NOTE: Comprehensive Integration Tests

class TestKOLScorerIntegration:
    """Test complete KOL scoring workflow with various scenarios."""
    
    @pytest.mark.asyncio
    async def test_complete_kol_scoring_high_quality(
        self,
        kol_scorer,
        high_quality_kol_profile,
        sample_campaign,
        mock_db_session
    ):
        """Test complete scoring workflow for high-quality KOL."""
        score_breakdown = await kol_scorer.score_kol(
            high_quality_kol_profile,
            sample_campaign,
            mock_db_session
        )
        
        # Verify score breakdown structure
        assert isinstance(score_breakdown, ScoreBreakdown)
        
        # Verify all component scores are present
        assert score_breakdown.roi_score is not None
        assert score_breakdown.audience_quality_score is not None
        assert score_breakdown.brand_safety_score is not None
        assert score_breakdown.content_relevance_score is not None
        assert score_breakdown.demographic_fit_score is not None
        assert score_breakdown.reliability_score is not None
        
        # Verify confidence scores
        assert score_breakdown.roi_confidence is not None
        assert score_breakdown.audience_quality_confidence is not None
        assert score_breakdown.brand_safety_confidence is not None
        assert score_breakdown.content_relevance_confidence is not None
        assert score_breakdown.demographic_fit_confidence is not None
        assert score_breakdown.reliability_confidence is not None
        
        # High quality KOL should have high overall score and confidence
        assert score_breakdown.composite_score > Decimal("0.7")
        assert score_breakdown.overall_confidence > Decimal("0.8")
        assert score_breakdown.missing_data_penalty < Decimal("0.2")
    
    @pytest.mark.asyncio
    async def test_complete_kol_scoring_low_quality(
        self,
        kol_scorer,
        low_quality_kol_profile,
        sample_campaign,
        mock_db_session
    ):
        """Test complete scoring workflow for low-quality KOL."""
        score_breakdown = await kol_scorer.score_kol(
            low_quality_kol_profile,
            sample_campaign,
            mock_db_session
        )
        
        # Low quality KOL should have lower overall score
        assert score_breakdown.composite_score < Decimal("0.5")
        
        # Should have significant missing data penalty
        assert score_breakdown.missing_data_penalty > Decimal("0.2")
        
        # Individual component scores should reflect quality issues
        assert score_breakdown.audience_quality_score < Decimal("0.5")  # High fake followers
        assert score_breakdown.brand_safety_score < Decimal("0.6")     # Negative content
        assert score_breakdown.reliability_score < Decimal("0.6")      # New account, low activity
    
    @pytest.mark.asyncio
    async def test_complete_kol_scoring_missing_data(
        self,
        kol_scorer,
        missing_data_kol_profile,
        sample_campaign,
        mock_db_session
    ):
        """Test complete scoring workflow with missing data."""
        score_breakdown = await kol_scorer.score_kol(
            missing_data_kol_profile,
            sample_campaign,
            mock_db_session
        )
        
        # Should handle missing data gracefully
        assert score_breakdown.composite_score >= Decimal("0.0")
        
        # Should have high missing data penalty
        assert score_breakdown.missing_data_penalty > Decimal("0.3")
        
        # Overall confidence should be lower
        assert score_breakdown.overall_confidence < Decimal("0.6")
        
        # Specific confidence scores should be low for missing components
        if missing_data_kol_profile.latest_metrics.average_engagement_rate is None:
            assert score_breakdown.roi_confidence < Decimal("0.5")
    
    @pytest.mark.asyncio
    async def test_scoring_parallel_execution(
        self,
        kol_scorer,
        high_quality_kol_profile,
        sample_campaign,
        mock_db_session
    ):
        """Test that scoring components run in parallel efficiently."""
        import time
        
        start_time = time.time()
        
        # Run scoring multiple times to test parallelization
        tasks = []
        for _ in range(3):
            task = kol_scorer.score_kol(
                high_quality_kol_profile,
                sample_campaign,
                mock_db_session
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # All results should be similar (same input)
        for result in results:
            assert isinstance(result, ScoreBreakdown)
            assert abs(result.composite_score - results[0].composite_score) < Decimal("0.001")
        
        # Should complete relatively quickly due to parallel execution
        # This is more of a performance indicator than strict test
        assert execution_time < 10.0  # Should complete within 10 seconds
    
    def test_weighted_score_calculation(self, kol_scorer):
        """Test weighted composite score calculation."""
        # Create score breakdown with known values
        breakdown = ScoreBreakdown(
            roi_score=Decimal("0.8"),
            audience_quality_score=Decimal("0.9"),
            brand_safety_score=Decimal("0.7"),
            content_relevance_score=Decimal("0.6"),
            demographic_fit_score=Decimal("0.8"),
            reliability_score=Decimal("0.9"),
            roi_confidence=Decimal("1.0"),
            audience_quality_confidence=Decimal("1.0"),
            brand_safety_confidence=Decimal("1.0"),
            content_relevance_confidence=Decimal("1.0"),
            demographic_fit_confidence=Decimal("1.0"),
            reliability_confidence=Decimal("1.0"),
            composite_score=Decimal("0.0"),  # Will be calculated
            overall_confidence=Decimal("1.0"),
            missing_data_penalty=Decimal("0.1")
        )
        
        # Calculate expected weighted score
        expected_score = (
            Decimal("0.8") * ScoreComponent.ROI.weight +
            Decimal("0.9") * ScoreComponent.AUDIENCE_QUALITY.weight +
            Decimal("0.7") * ScoreComponent.BRAND_SAFETY.weight +
            Decimal("0.6") * ScoreComponent.CONTENT_RELEVANCE.weight +
            Decimal("0.8") * ScoreComponent.DEMOGRAPHIC_FIT.weight +
            Decimal("0.9") * ScoreComponent.RELIABILITY.weight
        ) * (1 - Decimal("0.1"))  # Apply missing data penalty
        
        calculated_score = breakdown.weighted_score
        
        # Should match expected calculation
        assert abs(calculated_score - expected_score) < Decimal("0.001")


# AIDEV-NOTE: Edge Cases and Error Handling Tests

class TestKOLScorerEdgeCases:
    """Test edge cases and error handling scenarios."""
    
    @pytest.mark.asyncio
    async def test_scoring_with_exception_handling(
        self,
        kol_scorer,
        high_quality_kol_profile,
        sample_campaign,
        mock_db_session
    ):
        """Test scoring continues even when individual components fail."""
        
        # Mock one component to raise an exception
        with patch.object(kol_scorer, '_calculate_roi_score', side_effect=Exception("ROI calculation failed")):
            score_breakdown = await kol_scorer.score_kol(
                high_quality_kol_profile,
                sample_campaign,
                mock_db_session
            )
            
            # Should handle exception gracefully
            assert isinstance(score_breakdown, ScoreBreakdown)
            
            # ROI component should have default values due to exception
            assert score_breakdown.roi_score == Decimal("0.0")
            assert score_breakdown.roi_confidence == Decimal("0.0")
            
            # Other components should still work
            assert score_breakdown.audience_quality_score > Decimal("0.0")
    
    def test_missing_data_penalty_calculation(self, kol_scorer):
        """Test missing data penalty calculation with various scenarios."""
        # KOL with no missing data
        complete_kol = MagicMock(spec=KOLProfile)
        complete_kol.location = "Bangkok"
        complete_kol.categories = ["lifestyle"]
        complete_kol.demographics = {"age": 28}
        complete_kol.account_created_at = datetime.now(timezone.utc)
        complete_kol.engagement_history = [MagicMock()]
        
        complete_metrics = MagicMock()
        complete_metrics.followers = 10000
        complete_kol.latest_metrics = complete_metrics
        complete_kol.recent_content = [MagicMock()]
        
        complete_penalty = kol_scorer._calculate_missing_data_penalty(complete_kol)
        assert complete_penalty < Decimal("0.2")
        
        # KOL with many missing fields
        incomplete_kol = MagicMock(spec=KOLProfile)
        incomplete_kol.location = None
        incomplete_kol.categories = None
        incomplete_kol.demographics = None
        incomplete_kol.account_created_at = None
        incomplete_kol.engagement_history = None
        incomplete_kol.latest_metrics = None
        incomplete_kol.recent_content = []
        
        incomplete_penalty = kol_scorer._calculate_missing_data_penalty(incomplete_kol)
        assert incomplete_penalty > Decimal("0.4")  # High penalty for missing critical data
    
    def test_tier_determination_edge_cases(self, kol_scorer):
        """Test KOL tier determination with edge cases."""
        # Test boundary cases
        assert kol_scorer._determine_kol_tier(999) == "nano"
        assert kol_scorer._determine_kol_tier(1000) == "micro"
        assert kol_scorer._determine_kol_tier(9999) == "micro"
        assert kol_scorer._determine_kol_tier(10000) == "mid"
        assert kol_scorer._determine_kol_tier(99999) == "mid"
        assert kol_scorer._determine_kol_tier(100000) == "macro"
        assert kol_scorer._determine_kol_tier(1000000) == "macro"
        
        # Test extreme values
        assert kol_scorer._determine_kol_tier(0) == "nano"
        assert kol_scorer._determine_kol_tier(10000000) == "macro"
    
    def test_cost_estimation_edge_cases(self, kol_scorer):
        """Test cost estimation with various scenarios."""
        # High-performing nano influencer
        nano_kol = MagicMock()
        nano_kol.is_verified = False
        nano_kol.location = "Rural area"
        
        nano_metrics = MagicMock()
        nano_metrics.followers = 500
        nano_metrics.average_engagement_rate = Decimal("0.08")  # Very high engagement
        nano_kol.latest_metrics = nano_metrics
        
        nano_cost = kol_scorer._estimate_cost_per_post(nano_kol)
        
        # Should be relatively low but adjusted for high engagement
        assert Decimal("50") <= nano_cost <= Decimal("1000")
        
        # Mega influencer with low engagement
        mega_kol = MagicMock()
        mega_kol.is_verified = True
        mega_kol.location = "Bangkok, Thailand"
        
        mega_metrics = MagicMock()
        mega_metrics.followers = 5000000
        mega_metrics.average_engagement_rate = Decimal("0.01")  # Low engagement
        mega_kol.latest_metrics = mega_metrics
        
        mega_cost = kol_scorer._estimate_cost_per_post(mega_kol)
        
        # Should be high due to follower count
        assert mega_cost > Decimal("5000")
    
    def test_conversion_rate_estimation(self, kol_scorer):
        """Test conversion rate estimation for different scenarios."""
        # High-tier KOL
        macro_kol = MagicMock()
        macro_metrics = MagicMock()
        macro_metrics.followers = 500000
        macro_kol.latest_metrics = macro_metrics
        
        # Awareness campaign
        awareness_campaign = MagicMock()
        awareness_campaign.campaign_type = "awareness"
        
        awareness_rate = kol_scorer._estimate_conversion_rate(macro_kol, awareness_campaign)
        
        # Conversion campaign
        conversion_campaign = MagicMock()
        conversion_campaign.campaign_type = "conversion"
        
        conversion_rate = kol_scorer._estimate_conversion_rate(macro_kol, conversion_campaign)
        
        # Conversion campaigns should have higher rates
        assert conversion_rate > awareness_rate
        
        # Both should be reasonable percentages
        assert 0.001 <= awareness_rate <= 0.1
        assert 0.001 <= conversion_rate <= 0.1


# AIDEV-NOTE: Performance and Load Tests

class TestKOLScorerPerformance:
    """Test performance characteristics of the scoring system."""
    
    @pytest.mark.asyncio
    async def test_scoring_performance_with_large_engagement_history(
        self,
        kol_scorer,
        mock_db_session
    ):
        """Test scoring performance with large engagement history."""
        # Create KOL with extensive engagement history
        kol_large_history = MagicMock(spec=KOLProfile)
        kol_large_history.id = "large_history_kol"
        kol_large_history.handle = "@large_history"
        kol_large_history.location = "Bangkok"
        kol_large_history.categories = ["lifestyle"]
        kol_large_history.is_verified = True
        kol_large_history.account_created_at = datetime.now(timezone.utc) - timedelta(days=365)
        kol_large_history.demographics = {"average_age": 28}
        
        # Create large engagement history (simulate 1 year of daily posts)
        engagement_history = []
        for i in range(365):
            history = MagicMock()
            history.engagement_rate = 0.03 + (i % 10) * 0.001  # Varied but consistent
            engagement_history.append(history)
        
        kol_large_history.engagement_history = engagement_history
        
        # Large amount of recent content
        recent_content = []
        for i in range(50):
            content = MagicMock()
            content.caption = f"Daily content post {i} #lifestyle #content"
            content.hashtags = ["lifestyle", "content", "daily"]
            content.created_at = datetime.now(timezone.utc) - timedelta(days=i)
            recent_content.append(content)
        
        kol_large_history.recent_content = recent_content
        
        # Standard metrics
        metrics = MagicMock()
        metrics.followers = 50000
        metrics.average_engagement_rate = Decimal("0.035")
        metrics.posts_last_30_days = 30
        
        kol_large_history.latest_metrics = metrics
        
        # Create campaign
        campaign = MagicMock()
        requirements = MagicMock()
        requirements.required_categories = ["lifestyle"]
        requirements.target_keywords = ["lifestyle", "content"]
        requirements.target_locations = ["Bangkok"]
        requirements.target_age_range = (25, 35)
        requirements.target_interests = ["lifestyle"]
        campaign.requirements = requirements
        
        # Time the scoring operation
        import time
        start_time = time.time()
        
        score_breakdown = await kol_scorer.score_kol(
            kol_large_history,
            campaign,
            mock_db_session
        )
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Should complete within reasonable time even with large data
        assert execution_time < 5.0  # Should complete within 5 seconds
        
        # Should still produce valid results
        assert isinstance(score_breakdown, ScoreBreakdown)
        assert score_breakdown.composite_score > Decimal("0.5")  # Should score well
    
    @pytest.mark.asyncio
    async def test_concurrent_scoring_performance(
        self,
        kol_scorer,
        high_quality_kol_profile,
        sample_campaign,
        mock_db_session
    ):
        """Test performance when scoring multiple KOLs concurrently."""
        import time
        
        # Create multiple KOL profiles
        kol_profiles = []
        for i in range(10):
            kol = MagicMock(spec=KOLProfile)
            kol.id = f"concurrent_kol_{i}"
            kol.handle = f"@concurrent_{i}"
            
            # Copy attributes from high_quality_kol_profile
            for attr in ['platform', 'bio', 'location', 'categories', 'is_verified', 
                        'account_created_at', 'languages', 'demographics', 
                        'latest_metrics', 'recent_content', 'engagement_history']:
                if hasattr(high_quality_kol_profile, attr):
                    setattr(kol, attr, getattr(high_quality_kol_profile, attr))
            
            kol_profiles.append(kol)
        
        # Time concurrent scoring
        start_time = time.time()
        
        # Score all KOLs concurrently
        tasks = [
            kol_scorer.score_kol(kol, sample_campaign, mock_db_session)
            for kol in kol_profiles
        ]
        
        results = await asyncio.gather(*tasks)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Should complete within reasonable time
        assert execution_time < 30.0  # 10 KOLs within 30 seconds
        
        # All results should be valid
        assert len(results) == 10
        for result in results:
            assert isinstance(result, ScoreBreakdown)
            assert result.composite_score > Decimal("0.0")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])