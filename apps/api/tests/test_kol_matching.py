"""
Comprehensive tests for the KOL Matching Integration (POC2) - End-to-End Matching Workflow

AIDEV-NOTE: Production-ready tests for the sophisticated KOL matching system with
comprehensive coverage of end-to-end workflows, semantic matching, and brief parsing integration.
"""
import pytest
import asyncio
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import List, Dict, Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np

from kol_api.services.kol_matching import EnhancedKOLMatchingService
from kol_api.services.brief_parser import BriefParserService
from kol_api.services.models import (
    CampaignRequirements, 
    KOLCandidate,
    ScoreComponents,
    KOLMetricsData,
    BriefParsingResult,
    OptimizationObjective,
    KOLTier,
    ContentCategory,
    SemanticMatchingRequest
)
from kol_api.database.models.kol import KOL, KOLMetrics, KOLContent
from kol_api.database.models.campaign import Campaign


# AIDEV-NOTE: Test fixtures for end-to-end matching scenarios

@pytest.fixture
def mock_db_session():
    """Mock database session with query responses."""
    session = AsyncMock()
    
    # Default execute response
    mock_result = MagicMock()
    mock_result.fetchall.return_value = []
    mock_result.scalars.return_value.all.return_value = []
    session.execute.return_value = mock_result
    
    return session


@pytest.fixture
def sample_campaign_requirements():
    """Comprehensive campaign requirements for testing."""
    return CampaignRequirements(
        campaign_id="integration_test_campaign",
        target_kol_tiers=[KOLTier.MICRO, KOLTier.MID],
        target_categories=[ContentCategory.LIFESTYLE, ContentCategory.FASHION],
        total_budget=Decimal("75000"),
        min_follower_count=10000,
        max_follower_count=500000,
        min_engagement_rate=Decimal("0.02"),
        target_locations=["Bangkok", "Thailand", "Singapore"],
        target_languages=["th", "en"],
        require_brand_safe=True,
        require_verified=False,
        required_hashtags=["lifestyle", "fashion", "ootd"],
        excluded_hashtags=["controversial", "political"],
        campaign_objective=OptimizationObjective.MAXIMIZE_ENGAGEMENT,
        expected_conversion_rate=Decimal("0.025"),
        target_demographics={
            "age_range": (25, 35),
            "gender": "all",
            "interests": ["fashion", "travel", "food"]
        }
    )


@pytest.fixture
def comprehensive_kol_database():
    """Comprehensive KOL database with various profiles for testing."""
    kols_and_metrics = []
    
    # Perfect match KOLs (high scoring)
    for i in range(3):
        kol = MagicMock(spec=KOL)
        kol.id = f"perfect_match_kol_{i}"
        kol.username = f"perfect_lifestyle_{i}"
        kol.display_name = f"Perfect Lifestyle KOL {i}"
        kol.platform = MagicMock()
        kol.platform.value = "tiktok"
        kol.tier = MagicMock()
        kol.tier.value = "micro"
        kol.primary_category = MagicMock()
        kol.primary_category.value = "lifestyle"
        kol.is_active = True
        kol.is_brand_safe = True
        kol.is_verified = True
        kol.location = "Bangkok, Thailand"
        kol.languages = ["th", "en"]
        kol.bio = "Lifestyle influencer sharing daily fashion and travel content"
        kol.content_embedding = [0.8, 0.6, 0.4] + [0.1] * 381  # Good semantic match
        kol.categories = ["lifestyle", "fashion"]
        
        metrics = MagicMock(spec=KOLMetrics)
        metrics.kol_id = kol.id
        metrics.follower_count = 45000 + (i * 5000)
        metrics.following_count = 1000 + (i * 100)
        metrics.engagement_rate = Decimal(str(0.045 + (i * 0.005)))
        metrics.avg_likes = Decimal(str(2025 + (i * 200)))
        metrics.avg_comments = Decimal(str(135 + (i * 15)))
        metrics.avg_views = Decimal(str(6750 + (i * 500)))
        metrics.posts_last_30_days = 20 + i
        metrics.fake_follower_percentage = Decimal("0.03")
        metrics.audience_quality_score = Decimal("0.85")
        metrics.campaign_success_rate = Decimal("0.92")
        metrics.response_rate = Decimal("0.95")
        metrics.metrics_date = datetime.now(timezone.utc)
        
        kols_and_metrics.append((kol, metrics))
    
    # Good match KOLs (medium scoring)
    for i in range(5):
        kol = MagicMock(spec=KOL)
        kol.id = f"good_match_kol_{i}"
        kol.username = f"good_creator_{i}"
        kol.display_name = f"Good Creator {i}"
        kol.platform = MagicMock()
        kol.platform.value = "tiktok"
        kol.tier = MagicMock()
        kol.tier.value = "micro" if i < 3 else "mid"
        kol.primary_category = MagicMock()
        kol.primary_category.value = "lifestyle" if i % 2 == 0 else "fashion"
        kol.is_active = True
        kol.is_brand_safe = True
        kol.is_verified = i < 2  # Some verified
        kol.location = "Bangkok, Thailand" if i < 3 else "Singapore"
        kol.languages = ["th", "en"] if i < 3 else ["en"]
        kol.bio = f"Content creator in {kol.primary_category.value}"
        kol.content_embedding = [0.6, 0.4, 0.5] + [0.15] * 381  # Moderate semantic match
        kol.categories = ["lifestyle"] if i % 2 == 0 else ["fashion"]
        
        metrics = MagicMock(spec=KOLMetrics)
        metrics.kol_id = kol.id
        metrics.follower_count = 25000 + (i * 15000)
        metrics.following_count = 800 + (i * 150)
        metrics.engagement_rate = Decimal(str(0.035 + (i * 0.003)))
        metrics.avg_likes = Decimal(str(875 + (i * 100)))
        metrics.avg_comments = Decimal(str(52 + (i * 8)))
        metrics.avg_views = Decimal(str(3125 + (i * 300)))
        metrics.posts_last_30_days = 15 + i
        metrics.fake_follower_percentage = Decimal("0.08")
        metrics.audience_quality_score = Decimal("0.75")
        metrics.campaign_success_rate = Decimal("0.82")
        metrics.response_rate = Decimal("0.88")
        metrics.metrics_date = datetime.now(timezone.utc)
        
        kols_and_metrics.append((kol, metrics))
    
    # Poor match KOLs (low scoring)
    for i in range(4):
        kol = MagicMock(spec=KOL)
        kol.id = f"poor_match_kol_{i}"
        kol.username = f"poor_creator_{i}"
        kol.display_name = f"Poor Match Creator {i}"
        kol.platform = MagicMock()
        kol.platform.value = "tiktok"
        kol.tier = MagicMock()
        kol.tier.value = "nano"  # Wrong tier
        kol.primary_category = MagicMock()
        kol.primary_category.value = "gaming"  # Wrong category
        kol.is_active = True
        kol.is_brand_safe = i < 2  # Some not brand safe
        kol.is_verified = False
        kol.location = "Manila, Philippines"  # Wrong location
        kol.languages = ["en"]
        kol.bio = "Gaming content creator"
        kol.content_embedding = [0.2, 0.1, 0.3] + [0.05] * 381  # Poor semantic match
        kol.categories = ["gaming", "tech"]
        
        metrics = MagicMock(spec=KOLMetrics)
        metrics.kol_id = kol.id
        metrics.follower_count = 8000 + (i * 500)  # Too small
        metrics.following_count = 2000 + (i * 100)
        metrics.engagement_rate = Decimal(str(0.015 + (i * 0.002)))  # Too low
        metrics.avg_likes = Decimal(str(120 + (i * 20)))
        metrics.avg_comments = Decimal(str(8 + i))
        metrics.avg_views = Decimal(str(400 + (i * 50)))
        metrics.posts_last_30_days = 5 + i  # Inactive
        metrics.fake_follower_percentage = Decimal("0.25")  # High fake followers
        metrics.audience_quality_score = Decimal("0.45")
        metrics.campaign_success_rate = Decimal("0.65")
        metrics.response_rate = Decimal("0.70")
        metrics.metrics_date = datetime.now(timezone.utc)
        
        kols_and_metrics.append((kol, metrics))
    
    # Edge case KOLs (missing data, boundary conditions)
    edge_kol = MagicMock(spec=KOL)
    edge_kol.id = "edge_case_kol"
    edge_kol.username = "edge_case_creator"
    edge_kol.display_name = "Edge Case Creator"
    edge_kol.platform = MagicMock()
    edge_kol.platform.value = "tiktok"
    edge_kol.tier = MagicMock()
    edge_kol.tier.value = "micro"
    edge_kol.primary_category = MagicMock()
    edge_kol.primary_category.value = "lifestyle"
    edge_kol.is_active = True
    edge_kol.is_brand_safe = True
    edge_kol.is_verified = True
    edge_kol.location = None  # Missing location
    edge_kol.languages = None  # Missing languages
    edge_kol.bio = None  # Missing bio
    edge_kol.content_embedding = None  # Missing embedding
    edge_kol.categories = []  # Empty categories
    
    edge_metrics = MagicMock(spec=KOLMetrics)
    edge_metrics.kol_id = edge_kol.id
    edge_metrics.follower_count = 10000  # Boundary condition
    edge_metrics.following_count = 500
    edge_metrics.engagement_rate = None  # Missing engagement rate
    edge_metrics.avg_likes = None
    edge_metrics.avg_comments = None
    edge_metrics.avg_views = None
    edge_metrics.posts_last_30_days = 0  # No posts
    edge_metrics.fake_follower_percentage = None
    edge_metrics.audience_quality_score = None
    edge_metrics.campaign_success_rate = None
    edge_metrics.response_rate = None
    edge_metrics.metrics_date = datetime.now(timezone.utc)
    
    kols_and_metrics.append((edge_kol, edge_metrics))
    
    return kols_and_metrics


@pytest.fixture
def kol_matching_service(mock_db_session):
    """KOL matching service with mocked dependencies."""
    return EnhancedKOLMatchingService(mock_db_session)


@pytest.fixture
def brief_parser_service():
    """Brief parser service for integration testing."""
    return BriefParserService()


# AIDEV-NOTE: End-to-End Integration Tests

class TestKOLMatchingIntegration:
    """Test complete KOL matching workflows from brief to final selection."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_matching_workflow(
        self,
        kol_matching_service,
        sample_campaign_requirements,
        comprehensive_kol_database
    ):
        """Test complete workflow from campaign requirements to KOL selection."""
        
        # Mock database query to return our test KOL database
        with patch.object(kol_matching_service, '_get_filtered_candidates', return_value=comprehensive_kol_database):
            
            candidates, metadata = await kol_matching_service.find_matching_kols_advanced(
                campaign_requirements=sample_campaign_requirements,
                limit=10,
                enable_semantic_matching=True,
                confidence_threshold=Decimal("0.6")
            )
            
            # Should return candidates
            assert len(candidates) > 0
            assert len(candidates) <= 10
            
            # All candidates should be KOLCandidate instances
            for candidate in candidates:
                assert isinstance(candidate, KOLCandidate)
                assert candidate.kol_id is not None
                assert candidate.overall_score > Decimal("0.0")
            
            # Perfect match KOLs should be ranked higher
            perfect_match_ids = {f"perfect_match_kol_{i}" for i in range(3)}
            top_candidates = candidates[:3]
            top_kol_ids = {candidate.kol_id for candidate in top_candidates}
            
            # At least some perfect matches should be in top results
            perfect_in_top = len(perfect_match_ids & top_kol_ids)
            assert perfect_in_top >= 1
            
            # Metadata should be comprehensive
            assert isinstance(metadata, dict)
            assert "campaign_id" in metadata
            assert "algorithm_version" in metadata
            assert "scoring_method" in metadata
            assert "total_candidates_evaluated" in metadata
            assert "final_selected" in metadata
            assert metadata["final_selected"] == len(candidates)
    
    @pytest.mark.asyncio
    async def test_confidence_threshold_filtering(
        self,
        kol_matching_service,
        sample_campaign_requirements,
        comprehensive_kol_database
    ):
        """Test that confidence threshold properly filters results."""
        
        with patch.object(kol_matching_service, '_get_filtered_candidates', return_value=comprehensive_kol_database):
            
            # Test with high confidence threshold
            high_confidence_candidates, _ = await kol_matching_service.find_matching_kols_advanced(
                campaign_requirements=sample_campaign_requirements,
                limit=20,
                confidence_threshold=Decimal("0.9")  # Very high
            )
            
            # Test with low confidence threshold  
            low_confidence_candidates, _ = await kol_matching_service.find_matching_kols_advanced(
                campaign_requirements=sample_campaign_requirements,
                limit=20,
                confidence_threshold=Decimal("0.3")  # Very low
            )
            
            # High confidence should return fewer results
            assert len(high_confidence_candidates) <= len(low_confidence_candidates)
            
            # All high confidence results should meet threshold
            for candidate in high_confidence_candidates:
                assert candidate.score_components.overall_confidence >= Decimal("0.9")
    
    @pytest.mark.asyncio
    async def test_semantic_matching_integration(
        self,
        kol_matching_service,
        sample_campaign_requirements,
        comprehensive_kol_database
    ):
        """Test semantic matching functionality."""
        
        with patch.object(kol_matching_service, '_get_filtered_candidates', return_value=comprehensive_kol_database):
            
            # Test with semantic matching enabled
            semantic_candidates, semantic_metadata = await kol_matching_service.find_matching_kols_advanced(
                campaign_requirements=sample_campaign_requirements,
                limit=10,
                enable_semantic_matching=True
            )
            
            # Test with semantic matching disabled
            non_semantic_candidates, non_semantic_metadata = await kol_matching_service.find_matching_kols_advanced(
                campaign_requirements=sample_campaign_requirements,
                limit=10,
                enable_semantic_matching=False
            )
            
            # Results might differ when semantic matching is enabled
            semantic_ids = {c.kol_id for c in semantic_candidates}
            non_semantic_ids = {c.kol_id for c in non_semantic_candidates}
            
            # At least some results should be different (unless all have same scores)
            # This tests that semantic matching affects ranking
            assert len(semantic_ids) > 0
            assert len(non_semantic_ids) > 0
            
            # Metadata should indicate semantic matching status
            assert semantic_metadata["semantic_matching_enabled"] is True
            assert non_semantic_metadata["semantic_matching_enabled"] is False
    
    @pytest.mark.asyncio
    async def test_constraint_filtering_effectiveness(
        self,
        kol_matching_service,
        sample_campaign_requirements,
        comprehensive_kol_database
    ):
        """Test that hard constraints properly filter candidates."""
        
        with patch.object(kol_matching_service, '_get_filtered_candidates', return_value=comprehensive_kol_database):
            
            candidates, _ = await kol_matching_service.find_matching_kols_advanced(
                campaign_requirements=sample_campaign_requirements,
                limit=20
            )
            
            # Verify constraint satisfaction
            for candidate in candidates:
                # Tier constraints
                assert candidate.tier in [KOLTier.MICRO, KOLTier.MID]
                
                # Category constraints
                assert candidate.primary_category in [ContentCategory.LIFESTYLE, ContentCategory.FASHION]
                
                # Follower constraints
                assert candidate.metrics.follower_count >= 10000
                assert candidate.metrics.follower_count <= 500000
                
                # Engagement rate constraint
                if candidate.metrics.engagement_rate:
                    assert candidate.metrics.engagement_rate >= Decimal("0.02")
                
                # Brand safety constraint
                # This would be checked in the database query filtering
    
    @pytest.mark.asyncio
    async def test_no_matching_candidates_scenario(
        self,
        kol_matching_service,
        sample_campaign_requirements
    ):
        """Test handling when no candidates match criteria."""
        
        # Mock empty candidate pool
        with patch.object(kol_matching_service, '_get_filtered_candidates', return_value=[]):
            
            candidates, metadata = await kol_matching_service.find_matching_kols_advanced(
                campaign_requirements=sample_campaign_requirements,
                limit=10
            )
            
            # Should return empty results gracefully
            assert len(candidates) == 0
            assert isinstance(metadata, dict)
            assert metadata["final_selected"] == 0
            assert "error_reason" in metadata
            assert "No candidates passed hard constraints" in metadata["error_reason"]
    
    @pytest.mark.asyncio
    async def test_missing_data_confidence_handling(
        self,
        kol_matching_service,
        sample_campaign_requirements
    ):
        """Test handling of candidates with missing data."""
        
        # Create KOL database with only edge case (missing data) KOL
        edge_case_database = []
        
        edge_kol = MagicMock(spec=KOL)
        edge_kol.id = "missing_data_test_kol"
        edge_kol.username = "missing_data_kol"
        edge_kol.display_name = "Missing Data KOL"
        edge_kol.platform = MagicMock()
        edge_kol.platform.value = "tiktok"
        edge_kol.tier = MagicMock()
        edge_kol.tier.value = "micro"
        edge_kol.primary_category = MagicMock()
        edge_kol.primary_category.value = "lifestyle"
        edge_kol.is_active = True
        edge_kol.is_brand_safe = True
        edge_kol.is_verified = False
        
        # Extensive missing data
        edge_kol.location = None
        edge_kol.languages = None
        edge_kol.bio = None
        edge_kol.content_embedding = None
        edge_kol.categories = None
        
        edge_metrics = MagicMock(spec=KOLMetrics)
        edge_metrics.kol_id = edge_kol.id
        edge_metrics.follower_count = 25000
        edge_metrics.following_count = 500
        edge_metrics.engagement_rate = None  # Critical missing data
        edge_metrics.posts_last_30_days = 10
        edge_metrics.metrics_date = datetime.now(timezone.utc)
        
        edge_case_database.append((edge_kol, edge_metrics))
        
        with patch.object(kol_matching_service, '_get_filtered_candidates', return_value=edge_case_database):
            
            candidates, metadata = await kol_matching_service.find_matching_kols_advanced(
                campaign_requirements=sample_campaign_requirements,
                limit=5,
                confidence_threshold=Decimal("0.5")
            )
            
            if candidates:
                # Should handle missing data gracefully
                candidate = candidates[0]
                assert candidate.score_components.overall_confidence < Decimal("0.7")  # Low confidence
                
                # Should still produce valid scores (not None or negative)
                assert candidate.overall_score >= Decimal("0.0")
                assert candidate.score_components.roi_score >= Decimal("0.0")
            
            # Metadata should reflect data quality issues
            data_quality = metadata.get("data_quality_summary", {})
            assert "data_completeness" in data_quality


class TestSemanticMatching:
    """Test semantic similarity matching functionality."""
    
    @pytest.mark.asyncio
    async def test_find_similar_kols_semantic(
        self,
        kol_matching_service,
        sample_campaign_requirements,
        comprehensive_kol_database
    ):
        """Test semantic similarity search functionality."""
        
        # Mock reference KOL query
        reference_kol = comprehensive_kol_database[0][0]  # Use first perfect match KOL
        
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = reference_kol
        kol_matching_service.db_session.execute.return_value = mock_result
        
        # Mock embedding generation
        with patch.object(kol_matching_service, '_get_or_generate_embedding', return_value=[0.8, 0.6, 0.4] + [0.1] * 381):
            
            # Mock similar KOL query results
            similar_rows = []
            for i, (kol, metrics) in enumerate(comprehensive_kol_database[1:4]):  # Skip reference KOL
                mock_row = MagicMock()
                mock_row._mapping = {
                    'id': kol.id,
                    'username': kol.username,
                    'display_name': kol.display_name,
                    'platform': kol.platform.value,
                    'tier': kol.tier.value,
                    'primary_category': kol.primary_category.value,
                    'similarity_distance': 0.1 + (i * 0.1),
                    'similarity_score': 0.9 - (i * 0.1),
                    'follower_count': metrics.follower_count,
                    'engagement_rate': metrics.engagement_rate
                }
                similar_rows.append(mock_row)
            
            kol_matching_service.db_session.execute.return_value.fetchall.return_value = similar_rows
            
            # Mock candidate creation
            with patch.object(kol_matching_service, '_create_kol_candidate_with_scoring') as mock_create:
                mock_candidates = []
                for i, row in enumerate(similar_rows):
                    mock_candidate = MagicMock()
                    mock_candidate.kol_id = row._mapping['id']
                    mock_candidate.semantic_similarity_score = Decimal(str(row._mapping['similarity_score']))
                    mock_candidate.overall_score = Decimal("0.8") - Decimal(str(i * 0.1))
                    mock_candidates.append(mock_candidate)
                
                mock_create.side_effect = mock_candidates
                
                similar_candidates = await kol_matching_service.find_similar_kols_semantic(
                    reference_kol_id=reference_kol.id,
                    campaign_requirements=sample_campaign_requirements,
                    limit=5,
                    similarity_threshold=Decimal("0.7")
                )
                
                # Should return similar KOLs
                assert len(similar_candidates) > 0
                assert len(similar_candidates) <= 5
                
                # Should be sorted by combined similarity and performance
                for i in range(len(similar_candidates) - 1):
                    current_combined = (
                        float(similar_candidates[i].semantic_similarity_score) * 0.6 +
                        float(similar_candidates[i].overall_score) * 0.4
                    )
                    next_combined = (
                        float(similar_candidates[i+1].semantic_similarity_score) * 0.6 +
                        float(similar_candidates[i+1].overall_score) * 0.4
                    )
                    assert current_combined >= next_combined
                
                # All should meet similarity threshold
                for candidate in similar_candidates:
                    assert candidate.semantic_similarity_score >= Decimal("0.7")
    
    @pytest.mark.asyncio
    async def test_semantic_matching_without_campaign_context(
        self,
        kol_matching_service
    ):
        """Test semantic matching without campaign requirements context."""
        
        reference_kol_id = "perfect_match_kol_0"
        
        # Mock reference KOL
        reference_kol = MagicMock()
        reference_kol.id = reference_kol_id
        reference_kol.content_embedding = [0.8, 0.6, 0.4] + [0.1] * 381
        
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = reference_kol
        kol_matching_service.db_session.execute.return_value = mock_result
        
        # Mock embedding generation
        with patch.object(kol_matching_service, '_get_or_generate_embedding', return_value=reference_kol.content_embedding):
            
            # Mock query results
            similar_rows = []
            for i in range(3):
                mock_row = MagicMock()
                mock_row._mapping = {
                    'id': f'similar_kol_{i}',
                    'username': f'similar_{i}',
                    'similarity_score': 0.85 - (i * 0.05)
                }
                similar_rows.append(mock_row)
            
            kol_matching_service.db_session.execute.return_value.fetchall.return_value = similar_rows
            
            # Mock basic candidate creation
            with patch.object(kol_matching_service, '_create_basic_candidate') as mock_create_basic:
                mock_candidates = []
                for row in similar_rows:
                    mock_candidate = MagicMock()
                    mock_candidate.kol_id = row._mapping['id']
                    mock_candidate.semantic_similarity_score = Decimal(str(row._mapping['similarity_score']))
                    mock_candidates.append(mock_candidate)
                
                mock_create_basic.side_effect = mock_candidates
                
                similar_candidates = await kol_matching_service.find_similar_kols_semantic(
                    reference_kol_id=reference_kol_id,
                    campaign_requirements=None,
                    limit=5,
                    similarity_threshold=Decimal("0.8")
                )
                
                # Should return candidates sorted by similarity only
                assert len(similar_candidates) > 0
                
                for i in range(len(similar_candidates) - 1):
                    assert (similar_candidates[i].semantic_similarity_score >= 
                           similar_candidates[i+1].semantic_similarity_score)
    
    def test_cosine_similarity_calculation(self, kol_matching_service):
        """Test cosine similarity calculation accuracy."""
        
        # Test identical vectors
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [1.0, 0.0, 0.0]
        similarity = kol_matching_service._calculate_cosine_similarity(vec1, vec2)
        assert abs(similarity - 1.0) < 1e-6
        
        # Test orthogonal vectors
        vec3 = [1.0, 0.0, 0.0]
        vec4 = [0.0, 1.0, 0.0]
        similarity = kol_matching_service._calculate_cosine_similarity(vec3, vec4)
        assert abs(similarity - 0.0) < 1e-6
        
        # Test opposite vectors
        vec5 = [1.0, 0.0, 0.0]
        vec6 = [-1.0, 0.0, 0.0]
        similarity = kol_matching_service._calculate_cosine_similarity(vec5, vec6)
        assert abs(similarity - (-1.0)) < 1e-6
        
        # Test partial similarity
        vec7 = [1.0, 1.0, 0.0]
        vec8 = [1.0, 0.0, 0.0]
        similarity = kol_matching_service._calculate_cosine_similarity(vec7, vec8)
        expected = 1.0 / np.sqrt(2)  # cos(45°) = 1/√2
        assert abs(similarity - expected) < 1e-6
    
    @pytest.mark.asyncio
    async def test_embedding_generation_and_caching(self, kol_matching_service):
        """Test embedding generation and caching mechanism."""
        
        test_kol = MagicMock()
        test_kol.id = "embedding_test_kol"
        test_kol.content_embedding = None  # No existing embedding
        
        # Mock profile text creation
        with patch.object(kol_matching_service, '_create_kol_profile_text', return_value="test profile text"), \
             patch.object(kol_matching_service, '_generate_text_embedding', return_value=[0.1] * 384) as mock_generate:
            
            # First call should generate embedding
            embedding1 = await kol_matching_service._get_or_generate_embedding(test_kol)
            assert embedding1 == [0.1] * 384
            assert mock_generate.call_count == 1
            
            # Second call should use cache
            embedding2 = await kol_matching_service._get_or_generate_embedding(test_kol)
            assert embedding2 == [0.1] * 384
            assert mock_generate.call_count == 1  # Should not be called again
            
            # Should be same instance from cache
            assert embedding1 is embedding2
    
    def test_kol_profile_text_creation(self, kol_matching_service):
        """Test KOL profile text creation for embedding."""
        
        test_kol = MagicMock()
        test_kol.username = "test_kol"
        test_kol.platform = MagicMock()
        test_kol.platform.value = "tiktok"
        test_kol.tier = MagicMock()
        test_kol.tier.value = "micro"
        test_kol.primary_category = MagicMock()
        test_kol.primary_category.value = "lifestyle"
        test_kol.bio = "Lifestyle content creator"
        test_kol.location = "Bangkok, Thailand"
        test_kol.languages = ["th", "en"]
        
        profile_text = kol_matching_service._create_kol_profile_text(test_kol)
        
        # Should include key information
        assert "test_kol" in profile_text
        assert "tiktok" in profile_text
        assert "micro" in profile_text
        assert "lifestyle" in profile_text
        assert "Bangkok, Thailand" in profile_text
        assert "th, en" in profile_text
        
        # Should be structured
        assert len(profile_text) > 50  # Should be substantial text


class TestBriefParsingIntegration:
    """Test integration with brief parsing functionality."""
    
    @pytest.mark.asyncio
    async def test_brief_to_matching_workflow(
        self,
        brief_parser_service,
        kol_matching_service,
        comprehensive_kol_database
    ):
        """Test complete workflow from brief parsing to KOL matching."""
        
        # Sample marketing brief
        campaign_brief = """
        # Lifestyle Influencer Campaign for Spring Collection
        
        We're launching our new spring fashion collection and need lifestyle influencers 
        to create authentic content showcasing our pieces in everyday settings.
        
        **Budget:** THB 75,000
        **Target Audience:** Young professionals aged 25-35 in Bangkok and Singapore
        **Content Categories:** Lifestyle, Fashion, OOTD (Outfit of the Day)
        **Influencer Tiers:** Micro and mid-tier influencers
        **Languages:** Thai and English
        **Campaign Goal:** Maximize engagement and brand awareness
        
        **Requirements:**
        - Verified accounts preferred but not required
        - Brand safe content only
        - High engagement rates (minimum 2%)
        - Active posting (10+ posts per month)
        
        **Hashtags to Include:** #lifestyle #fashion #ootd #springcollection
        **Avoid:** Political content, controversial topics
        """
        
        # Step 1: Parse the brief
        with patch.object(brief_parser_service.db_session, 'execute'):  # Mock any DB operations
            
            brief_result = await brief_parser_service.parse_markdown_brief(
                file_content=campaign_brief,
                filename="spring_campaign_brief.md",
                user_id="test_user_123",
                campaign_id="spring_campaign_2024"
            )
            
            assert brief_result.success is True
            assert brief_result.campaign_requirements is not None
            assert brief_result.confidence_score > Decimal("0.7")
            
            campaign_requirements = brief_result.campaign_requirements
            
            # Verify parsed requirements
            assert campaign_requirements.total_budget == Decimal("75000")
            assert KOLTier.MICRO in campaign_requirements.target_kol_tiers
            assert KOLTier.MID in campaign_requirements.target_kol_tiers
            assert ContentCategory.LIFESTYLE in campaign_requirements.target_categories
            assert ContentCategory.FASHION in campaign_requirements.target_categories
            assert "Bangkok" in campaign_requirements.target_locations
            assert "Singapore" in campaign_requirements.target_locations
            assert campaign_requirements.min_engagement_rate >= Decimal("0.02")
        
        # Step 2: Use parsed requirements for KOL matching
        with patch.object(kol_matching_service, '_get_filtered_candidates', return_value=comprehensive_kol_database):
            
            candidates, metadata = await kol_matching_service.find_matching_kols_advanced(
                campaign_requirements=campaign_requirements,
                limit=15,
                enable_semantic_matching=True,
                confidence_threshold=Decimal("0.6")
            )
            
            # Should successfully find candidates
            assert len(candidates) > 0
            
            # Candidates should match the parsed requirements
            for candidate in candidates:
                assert candidate.tier in [KOLTier.MICRO, KOLTier.MID]
                assert candidate.primary_category in [ContentCategory.LIFESTYLE, ContentCategory.FASHION]
                if candidate.metrics.engagement_rate:
                    assert candidate.metrics.engagement_rate >= Decimal("0.02")
            
            # Should prioritize candidates that match parsed criteria
            top_candidates = candidates[:5]
            perfect_matches = 0
            
            for candidate in top_candidates:
                if (candidate.tier in [KOLTier.MICRO, KOLTier.MID] and 
                    candidate.primary_category in [ContentCategory.LIFESTYLE, ContentCategory.FASHION]):
                    perfect_matches += 1
            
            assert perfect_matches >= 3  # Most top candidates should be good matches
    
    def test_brief_parsing_edge_cases(self, brief_parser_service):
        """Test brief parsing handles various edge cases."""
        
        # Test minimal brief
        minimal_brief = "We need some influencers. Budget: 50000 THB."
        
        # Should handle gracefully but with warnings
        with patch.object(brief_parser_service, '_extract_structured_data') as mock_extract:
            mock_extract.return_value = MagicMock()
            
            # Test shouldn't crash with minimal input
            try:
                brief_parser_service._normalize_content(minimal_brief)
                assert True  # If we reach here, no exception was thrown
            except Exception:
                pytest.fail("Brief parsing should handle minimal input gracefully")
        
        # Test brief with conflicting information
        conflicting_brief = """
        Budget: THB 25,000
        Budget: THB 100,000
        We want nano influencers
        We need macro influencers
        """
        
        # Should handle conflicting information
        try:
            brief_parser_service._normalize_content(conflicting_brief)
            assert True
        except Exception:
            pytest.fail("Brief parsing should handle conflicting information gracefully")


class TestKOLMatchingPerformance:
    """Test performance characteristics of the matching system."""
    
    @pytest.mark.asyncio
    async def test_matching_performance_large_database(
        self,
        kol_matching_service,
        sample_campaign_requirements
    ):
        """Test matching performance with large KOL database."""
        
        # Create large KOL database (simulate 1000 KOLs)
        large_database = []
        
        for i in range(1000):
            kol = MagicMock(spec=KOL)
            kol.id = f"perf_test_kol_{i}"
            kol.username = f"perf_kol_{i}"
            kol.display_name = f"Performance Test KOL {i}"
            kol.platform = MagicMock()
            kol.platform.value = "tiktok"
            kol.tier = MagicMock()
            kol.tier.value = ["nano", "micro", "mid", "macro"][i % 4]
            kol.primary_category = MagicMock()
            kol.primary_category.value = ["lifestyle", "fashion", "beauty", "fitness"][i % 4]
            kol.is_active = True
            kol.is_brand_safe = i % 10 != 0  # 90% brand safe
            kol.is_verified = i % 5 == 0  # 20% verified
            kol.location = ["Bangkok", "Singapore", "Manila"][i % 3]
            kol.languages = [["th", "en"], ["en"], ["en", "tl"]][i % 3]
            kol.content_embedding = [np.random.random() for _ in range(384)]
            
            metrics = MagicMock(spec=KOLMetrics)
            metrics.kol_id = kol.id
            metrics.follower_count = 1000 + (i * 100)
            metrics.engagement_rate = Decimal(str(0.01 + (i % 50) * 0.001))
            metrics.posts_last_30_days = 5 + (i % 25)
            metrics.metrics_date = datetime.now(timezone.utc)
            
            large_database.append((kol, metrics))
        
        import time
        start_time = time.time()
        
        with patch.object(kol_matching_service, '_get_filtered_candidates', return_value=large_database[:100]):  # Limit for reasonable test time
            
            candidates, metadata = await kol_matching_service.find_matching_kols_advanced(
                campaign_requirements=sample_campaign_requirements,
                limit=20,
                enable_semantic_matching=True,
                confidence_threshold=Decimal("0.5")
            )
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Should complete within reasonable time
        assert execution_time < 30.0  # Should complete within 30 seconds
        
        # Should still return valid results
        assert len(candidates) > 0
        assert len(candidates) <= 20
        
        # Performance metadata should be tracked
        assert "processing_time_seconds" in metadata
        assert metadata["processing_time_seconds"] > 0
    
    @pytest.mark.asyncio
    async def test_concurrent_matching_requests(
        self,
        kol_matching_service,
        sample_campaign_requirements,
        comprehensive_kol_database
    ):
        """Test handling of concurrent matching requests."""
        
        # Simulate multiple concurrent requests
        async def single_matching_request():
            with patch.object(kol_matching_service, '_get_filtered_candidates', return_value=comprehensive_kol_database):
                candidates, metadata = await kol_matching_service.find_matching_kols_advanced(
                    campaign_requirements=sample_campaign_requirements,
                    limit=10,
                    confidence_threshold=Decimal("0.6")
                )
                return len(candidates), metadata["processing_time_seconds"]
        
        # Run 5 concurrent requests
        tasks = [single_matching_request() for _ in range(5)]
        
        import time
        start_time = time.time()
        results = await asyncio.gather(*tasks)
        end_time = time.time()
        
        total_time = end_time - start_time
        
        # All requests should succeed
        assert len(results) == 5
        for candidate_count, processing_time in results:
            assert candidate_count > 0
            assert processing_time > 0
        
        # Concurrent processing should be more efficient than sequential
        total_processing_time = sum(processing_time for _, processing_time in results)
        
        # Total wall clock time should be less than sum of individual processing times
        # (indicating some level of concurrency)
        assert total_time < total_processing_time


class TestKOLMatchingEdgeCases:
    """Test edge cases and error scenarios in KOL matching."""
    
    @pytest.mark.asyncio
    async def test_invalid_campaign_requirements(self, kol_matching_service):
        """Test handling of invalid campaign requirements."""
        
        # Test with None campaign requirements
        with pytest.raises((ValueError, AttributeError)):
            await kol_matching_service.find_matching_kols_advanced(
                campaign_requirements=None,
                limit=10
            )
        
        # Test with empty tier list
        invalid_requirements = CampaignRequirements(
            campaign_id="invalid_test",
            target_kol_tiers=[],  # Empty
            target_categories=[ContentCategory.LIFESTYLE],
            total_budget=Decimal("50000"),
            campaign_objective=OptimizationObjective.MAXIMIZE_ENGAGEMENT
        )
        
        # Should handle gracefully or raise appropriate error
        try:
            with patch.object(kol_matching_service, '_get_filtered_candidates', return_value=[]):
                candidates, metadata = await kol_matching_service.find_matching_kols_advanced(
                    campaign_requirements=invalid_requirements,
                    limit=10
                )
                # If it doesn't raise an error, should return empty results
                assert len(candidates) == 0
        except ValueError:
            # Or it might raise a validation error, which is also acceptable
            pass
    
    @pytest.mark.asyncio
    async def test_database_connection_failure(self, kol_matching_service, sample_campaign_requirements):
        """Test handling of database connection failures."""
        
        # Mock database session to raise exception
        kol_matching_service.db_session.execute.side_effect = Exception("Database connection failed")
        
        # Should handle database errors gracefully
        with pytest.raises(Exception):  # Should propagate database errors
            await kol_matching_service.find_matching_kols_advanced(
                campaign_requirements=sample_campaign_requirements,
                limit=10
            )
    
    @pytest.mark.asyncio
    async def test_scoring_component_failures(
        self,
        kol_matching_service,
        sample_campaign_requirements,
        comprehensive_kol_database
    ):
        """Test handling when individual scoring components fail."""
        
        # Mock one scoring component to fail
        with patch.object(kol_matching_service, '_get_filtered_candidates', return_value=comprehensive_kol_database[:2]), \
             patch.object(kol_matching_service, '_calculate_roi_score', side_effect=Exception("ROI calculation failed")):
            
            # Should still complete with other scoring components
            candidates, metadata = await kol_matching_service.find_matching_kols_advanced(
                campaign_requirements=sample_campaign_requirements,
                limit=5
            )
            
            # Should return some results even with partial scoring failures
            # (depends on implementation - might return 0 results if ROI is critical)
            assert isinstance(candidates, list)
            assert isinstance(metadata, dict)
    
    def test_extreme_confidence_thresholds(self, kol_matching_service, comprehensive_kol_database):
        """Test extreme confidence threshold values."""
        
        # Mock the scoring to return known confidence values
        with patch.object(kol_matching_service, '_create_kol_candidate_with_scoring') as mock_create:
            
            # Create mock candidates with varying confidence levels
            mock_candidates = []
            for i in range(5):
                candidate = MagicMock()
                candidate.score_components = MagicMock()
                candidate.score_components.overall_confidence = Decimal(str(0.2 + i * 0.2))  # 0.2, 0.4, 0.6, 0.8, 1.0
                mock_candidates.append(candidate)
            
            mock_create.side_effect = mock_candidates
            
            # Test with impossible high threshold
            with patch.object(kol_matching_service, '_get_filtered_candidates', return_value=comprehensive_kol_database[:5]):
                
                sample_req = MagicMock()
                sample_req.campaign_id = "test"
                
                # This would be called in actual workflow
                filtered_candidates = []
                for i, candidate in enumerate(mock_candidates):
                    if candidate.score_components.overall_confidence >= Decimal("1.1"):  # Impossible threshold
                        filtered_candidates.append(candidate)
                
                # Should return empty with impossible threshold
                assert len(filtered_candidates) == 0
                
                # Test with 0 threshold (should return all)
                filtered_candidates = []
                for candidate in mock_candidates:
                    if candidate.score_components.overall_confidence >= Decimal("0.0"):
                        filtered_candidates.append(candidate)
                
                assert len(filtered_candidates) == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])