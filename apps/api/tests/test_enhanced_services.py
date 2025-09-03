"""Comprehensive unit tests for enhanced POC2 and POC4 algorithms."""

import pytest
from decimal import Decimal
from datetime import datetime, timedelta
from typing import List, Dict, Any
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np

from kol_api.services.models import (
    ScoreComponents, KOLMetricsData, CampaignRequirements,
    KOLCandidate, OptimizationConstraints, OptimizationResult,
    BriefParsingResult, OptimizationObjective, KOLTier, ContentCategory,
    ConstraintViolation, CampaignPlanExport
)
from kol_api.services.kol_matching import EnhancedKOLMatchingService
from kol_api.services.enhanced_budget_optimizer import (
    EnhancedBudgetOptimizerService, AdvancedOptimizationAlgorithm,
    ConstraintSatisfactionSolver
)
from kol_api.services.brief_parser import BriefParserService
from kol_api.database.models.kol import KOL, KOLMetrics


# AIDEV-NOTE: Test fixtures

@pytest.fixture
def mock_db_session():
    """Mock database session."""
    return AsyncMock()


@pytest.fixture
def sample_kol():
    """Sample KOL for testing."""
    kol = MagicMock(spec=KOL)
    kol.id = "kol_123"
    kol.username = "test_kol"
    kol.display_name = "Test KOL"
    kol.platform = "tiktok"
    kol.tier = MagicMock()
    kol.tier.value = "micro"
    kol.primary_category = MagicMock()
    kol.primary_category.value = "lifestyle"
    kol.is_brand_safe = True
    kol.is_verified = True
    kol.is_active = True
    kol.location = "Bangkok, Thailand"
    kol.languages = ["th", "en"]
    kol.bio = "Lifestyle content creator"
    kol.content_embedding = None
    return kol


@pytest.fixture
def sample_kol_metrics():
    """Sample KOL metrics for testing."""
    metrics = MagicMock(spec=KOLMetrics)
    metrics.follower_count = 50000
    metrics.following_count = 1000
    metrics.engagement_rate = Decimal("0.045")
    metrics.avg_likes = Decimal("2250")
    metrics.avg_comments = Decimal("125")
    metrics.avg_views = Decimal("7500")
    metrics.posts_last_30_days = 15
    metrics.fake_follower_percentage = Decimal("0.05")
    metrics.audience_quality_score = Decimal("0.85")
    metrics.campaign_success_rate = Decimal("0.90")
    metrics.response_rate = Decimal("0.95")
    metrics.metrics_date = datetime.utcnow()
    return metrics


@pytest.fixture
def sample_kol_metrics_data(sample_kol_metrics):
    """Sample KOL metrics data."""
    return KOLMetricsData(
        follower_count=sample_kol_metrics.follower_count,
        following_count=sample_kol_metrics.following_count,
        engagement_rate=sample_kol_metrics.engagement_rate,
        avg_likes=sample_kol_metrics.avg_likes,
        avg_comments=sample_kol_metrics.avg_comments,
        avg_views=sample_kol_metrics.avg_views,
        posts_last_30_days=sample_kol_metrics.posts_last_30_days,
        fake_follower_percentage=sample_kol_metrics.fake_follower_percentage,
        audience_quality_score=sample_kol_metrics.audience_quality_score,
        campaign_success_rate=sample_kol_metrics.campaign_success_rate,
        response_rate=sample_kol_metrics.response_rate
    )


@pytest.fixture
def sample_campaign_requirements():
    """Sample campaign requirements."""
    return CampaignRequirements(
        campaign_id="campaign_123",
        target_kol_tiers=[KOLTier.MICRO, KOLTier.MID],
        target_categories=[ContentCategory.LIFESTYLE, ContentCategory.FASHION],
        total_budget=Decimal("100000"),
        min_follower_count=10000,
        max_follower_count=500000,
        target_locations=["Bangkok", "Thailand"],
        target_languages=["th", "en"],
        campaign_objective=OptimizationObjective.MAXIMIZE_ENGAGEMENT
    )


@pytest.fixture
def sample_kol_candidate(sample_kol, sample_kol_metrics_data):
    """Sample KOL candidate."""
    score_components = ScoreComponents(
        roi_score=Decimal("0.75"),
        audience_quality_score=Decimal("0.85"),
        brand_safety_score=Decimal("0.95"),
        content_relevance_score=Decimal("0.80"),
        demographic_fit_score=Decimal("0.70"),
        reliability_score=Decimal("0.90")
    )
    
    return KOLCandidate(
        kol_id=sample_kol.id,
        username=sample_kol.username,
        display_name=sample_kol.display_name,
        platform=sample_kol.platform,
        tier=KOLTier.MICRO,
        primary_category=ContentCategory.LIFESTYLE,
        metrics=sample_kol_metrics_data,
        score_components=score_components,
        overall_score=Decimal("0.81"),
        predicted_reach=7500,
        predicted_engagement=337,
        predicted_conversions=8,
        estimated_cost_per_post=Decimal("2500.00"),
        estimated_total_cost=Decimal("2500.00"),
        risk_factors=["Unverified account"],
        overall_risk_score=Decimal("0.15")
    )


@pytest.fixture
def sample_optimization_constraints():
    """Sample optimization constraints."""
    return OptimizationConstraints(
        max_budget=Decimal("50000"),
        min_kols=3,
        max_kols=10,
        required_micro_count=2,
        required_mid_count=1,
        max_risk_per_kol=Decimal("0.6"),
        max_portfolio_risk=Decimal("0.5")
    )


# AIDEV-NOTE: Tests for Enhanced KOL Matching Service (POC2)

@pytest.mark.asyncio
class TestEnhancedKOLMatchingService:
    """Test cases for Enhanced KOL Matching Service."""
    
    async def test_init(self, mock_db_session):
        """Test service initialization."""
        service = EnhancedKOLMatchingService(mock_db_session)
        
        assert service.db_session == mock_db_session
        assert "roi_score" in service.scoring_weights
        assert service.scoring_weights["roi_score"] == 0.25
        assert service.scoring_weights["audience_quality"] == 0.25
        assert service.scoring_weights["brand_safety"] == 0.20
        assert isinstance(service._embedding_cache, dict)
        assert isinstance(service._score_cache, dict)
    
    async def test_calculate_sophisticated_kol_score(
        self,
        mock_db_session,
        sample_kol,
        sample_kol_metrics_data,
        sample_campaign_requirements
    ):
        """Test sophisticated multi-factor KOL scoring."""
        service = EnhancedKOLMatchingService(mock_db_session)
        
        # Mock the individual scoring methods
        with patch.object(service, '_calculate_roi_score', return_value=(Decimal("0.8"), Decimal("0.9"))), \
             patch.object(service, '_calculate_audience_quality_score', return_value=(Decimal("0.85"), Decimal("0.95"))), \
             patch.object(service, '_calculate_brand_safety_score', return_value=(Decimal("0.9"), Decimal("1.0"))), \
             patch.object(service, '_calculate_content_relevance_score', return_value=(Decimal("0.75"), Decimal("0.8"))), \
             patch.object(service, '_calculate_demographic_fit_score', return_value=(Decimal("0.7"), Decimal("0.85"))), \
             patch.object(service, '_calculate_reliability_score', return_value=(Decimal("0.85"), Decimal("0.9"))):
            
            score_components = await service.calculate_sophisticated_kol_score(
                kol=sample_kol,
                metrics=sample_kol_metrics_data,
                campaign_requirements=sample_campaign_requirements,
                enable_semantic=True
            )
            
            assert isinstance(score_components, ScoreComponents)
            assert score_components.roi_score == Decimal("0.8")
            assert score_components.audience_quality_score == Decimal("0.85")
            assert score_components.brand_safety_score == Decimal("0.9")
            assert score_components.content_relevance_score == Decimal("0.75")
            assert score_components.demographic_fit_score == Decimal("0.7")
            assert score_components.reliability_score == Decimal("0.85")
            
            # Test overall confidence calculation
            assert score_components.overall_confidence > Decimal("0.8")
    
    async def test_find_matching_kols_advanced(
        self,
        mock_db_session,
        sample_campaign_requirements
    ):
        """Test advanced KOL matching with comprehensive scoring."""
        service = EnhancedKOLMatchingService(mock_db_session)
        
        # Mock the database query results
        mock_candidates = [(MagicMock(), MagicMock()) for _ in range(5)]
        
        with patch.object(service, '_get_filtered_candidates', return_value=mock_candidates), \
             patch.object(service, '_create_kol_candidate_with_scoring') as mock_create_candidate:
            
            # Mock candidate creation
            mock_candidate = MagicMock()
            mock_candidate.score_components.overall_confidence = Decimal("0.8")
            mock_create_candidate.return_value = mock_candidate
            
            with patch.object(service, '_rank_and_select_candidates', return_value=[mock_candidate]):
                
                candidates, metadata = await service.find_matching_kols_advanced(
                    campaign_requirements=sample_campaign_requirements,
                    limit=10,
                    enable_semantic_matching=True,
                    confidence_threshold=Decimal("0.7")
                )
                
                assert len(candidates) == 1
                assert isinstance(metadata, dict)
                assert "campaign_id" in metadata
                assert "algorithm_version" in metadata
                assert metadata["algorithm_version"] == "2.1"
    
    async def test_find_similar_kols_semantic(
        self,
        mock_db_session,
        sample_campaign_requirements
    ):
        """Test semantic similarity search."""
        service = EnhancedKOLMatchingService(mock_db_session)
        
        # Mock database query
        mock_reference_kol = MagicMock()
        mock_reference_kol.id = "ref_kol_123"
        
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = mock_reference_kol
        
        with patch.object(service, '_get_or_generate_embedding', return_value=[0.1] * 384), \
             patch.object(service, '_create_kol_candidate_with_scoring') as mock_create_candidate:
            
            # Mock similar rows result
            mock_row = MagicMock()
            mock_row._mapping = {
                'id': 'similar_kol_123',
                'username': 'similar_kol',
                'similarity_score': 0.85
            }
            mock_db_session.execute.return_value.fetchall.return_value = [mock_row]
            
            mock_candidate = MagicMock()
            mock_candidate.semantic_similarity_score = Decimal("0.85")
            mock_create_candidate.return_value = mock_candidate
            
            similar_candidates = await service.find_similar_kols_semantic(
                reference_kol_id="ref_kol_123",
                campaign_requirements=sample_campaign_requirements,
                limit=10,
                similarity_threshold=Decimal("0.7")
            )
            
            assert len(similar_candidates) == 1
            assert similar_candidates[0].semantic_similarity_score == Decimal("0.85")
    
    def test_calculate_cosine_similarity(self, mock_db_session):
        """Test cosine similarity calculation."""
        service = EnhancedKOLMatchingService(mock_db_session)
        
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [0.0, 1.0, 0.0]
        vec3 = [1.0, 0.0, 0.0]
        
        # Test orthogonal vectors (similarity = 0)
        similarity = service._calculate_cosine_similarity(vec1, vec2)
        assert abs(similarity - 0.0) < 1e-6
        
        # Test identical vectors (similarity = 1)
        similarity = service._calculate_cosine_similarity(vec1, vec3)
        assert abs(similarity - 1.0) < 1e-6
    
    def test_simple_sentiment_analysis(self, mock_db_session):
        """Test simple sentiment analysis."""
        service = EnhancedKOLMatchingService(mock_db_session)
        
        # Test positive text
        positive_sentiment = service._simple_sentiment_analysis("This is amazing and awesome!")
        assert positive_sentiment > 0.5
        
        # Test negative text
        negative_sentiment = service._simple_sentiment_analysis("This is terrible and awful!")
        assert negative_sentiment < 0.5
        
        # Test neutral text
        neutral_sentiment = service._simple_sentiment_analysis("This is a product.")
        assert abs(neutral_sentiment - 0.5) < 0.1
    
    async def test_missing_data_handling(
        self,
        mock_db_session,
        sample_kol,
        sample_campaign_requirements
    ):
        """Test graceful handling of missing KOL data."""
        service = EnhancedKOLMatchingService(mock_db_session)
        
        # Create metrics with missing data
        incomplete_metrics = KOLMetricsData(
            follower_count=10000,
            following_count=500,
            engagement_rate=None,  # Missing
            posts_last_30_days=0   # Zero posts
        )
        
        with patch.object(service, '_calculate_roi_score', return_value=(Decimal("0.3"), Decimal("0.3"))), \
             patch.object(service, '_calculate_audience_quality_score', return_value=(Decimal("0.5"), Decimal("0.5"))), \
             patch.object(service, '_calculate_brand_safety_score', return_value=(Decimal("0.8"), Decimal("1.0"))), \
             patch.object(service, '_calculate_content_relevance_score', return_value=(Decimal("0.5"), Decimal("0.6"))), \
             patch.object(service, '_calculate_demographic_fit_score', return_value=(Decimal("0.5"), Decimal("0.5"))), \
             patch.object(service, '_calculate_reliability_score', return_value=(Decimal("0.2"), Decimal("0.4"))):
            
            score_components = await service.calculate_sophisticated_kol_score(
                kol=sample_kol,
                metrics=incomplete_metrics,
                campaign_requirements=sample_campaign_requirements
            )
            
            # Should handle missing data gracefully with lower confidence
            assert score_components.overall_confidence < Decimal("0.7")
            assert score_components.roi_confidence == Decimal("0.3")  # Low confidence for missing engagement rate


# AIDEV-NOTE: Tests for Enhanced Budget Optimizer Service (POC4)

@pytest.mark.asyncio
class TestEnhancedBudgetOptimizerService:
    """Test cases for Enhanced Budget Optimizer Service."""
    
    async def test_init(self, mock_db_session):
        """Test service initialization."""
        service = EnhancedBudgetOptimizerService(mock_db_session)
        
        assert service.db_session == mock_db_session
        assert isinstance(service.budget_tiers, dict)
        assert "nano" in service.budget_tiers
        assert "micro" in service.budget_tiers
        assert isinstance(service._optimization_history, list)
    
    async def test_export_campaign_plan(
        self,
        mock_db_session,
        sample_kol_candidate,
        sample_optimization_constraints
    ):
        """Test campaign plan export functionality."""
        service = EnhancedBudgetOptimizerService(mock_db_session)
        
        # Create sample optimization result
        optimization_result = OptimizationResult(
            selected_kols=[sample_kol_candidate],
            total_cost=Decimal("2500.00"),
            cost_by_tier={"micro": Decimal("2500.00")},
            cost_by_category={"lifestyle": Decimal("2500.00")},
            predicted_total_reach=7500,
            predicted_total_engagement=337,
            predicted_total_conversions=8,
            predicted_roi=Decimal("15.0"),
            portfolio_risk_score=Decimal("0.15"),
            portfolio_diversity_score=Decimal("0.6"),
            optimization_score=Decimal("0.85"),
            budget_utilization=Decimal("0.05"),
            constraints_satisfied=True,
            constraint_violations=[],
            alternative_allocations=[],
            algorithm_used="constraint_satisfaction",
            optimization_time_seconds=1.5,
            iterations_performed=1,
            convergence_achieved=True
        )
        
        campaign_plan = await service.export_campaign_plan(
            optimization_result=optimization_result,
            campaign_id="test_campaign",
            campaign_name="Test Campaign",
            export_format="csv"
        )
        
        assert isinstance(campaign_plan, CampaignPlanExport)
        assert campaign_plan.campaign_id == "test_campaign"
        assert campaign_plan.campaign_name == "Test Campaign"
        assert len(campaign_plan.kol_selections) == 1
        assert "total_selected_kols" in campaign_plan.performance_summary
        assert campaign_plan.performance_summary["total_selected_kols"] == 1
    
    async def test_export_to_csv(
        self,
        mock_db_session,
        sample_kol_candidate
    ):
        """Test CSV export functionality."""
        service = EnhancedBudgetOptimizerService(mock_db_session)
        
        # Create sample campaign plan
        campaign_plan = CampaignPlanExport(
            campaign_id="test_campaign",
            campaign_name="Test Campaign",
            optimization_objective="maximize_engagement",
            total_budget=Decimal("2500.00"),
            kol_selections=[{
                "rank": 1,
                "username": "test_kol",
                "tier": "micro",
                "estimated_total_cost": 2500.00
            }],
            performance_summary={
                "total_selected_kols": 1,
                "total_budget_allocated": 2500.00
            }
        )
        
        csv_content = await service.export_to_csv(campaign_plan)
        
        assert isinstance(csv_content, str)
        assert "KOL Campaign Plan Export" in csv_content
        assert "test_campaign" in csv_content
        assert "Test Campaign" in csv_content
        assert "test_kol" in csv_content


class TestAdvancedOptimizationAlgorithm:
    """Test cases for Advanced Optimization Algorithm."""
    
    def test_init(self, sample_kol_candidate):
        """Test algorithm initialization."""
        candidates = [sample_kol_candidate]
        algorithm = AdvancedOptimizationAlgorithm(candidates)
        
        assert algorithm.candidates == candidates
        assert hasattr(algorithm, 'logger')
    
    def test_knapsack_dp(self, sample_kol_candidate):
        """Test knapsack dynamic programming solution."""
        candidates = [sample_kol_candidate]
        algorithm = AdvancedOptimizationAlgorithm(candidates)
        
        values = [100, 60, 40]
        costs = [20, 10, 15]
        capacity = 25
        
        selected = algorithm._knapsack_dp(values, costs, capacity)
        
        assert isinstance(selected, list)
        # Should select items that fit within capacity and maximize value
        total_cost = sum(costs[i] for i in selected)
        assert total_cost <= capacity
    
    def test_calculate_fitness(self, sample_kol_candidate, sample_optimization_constraints):
        """Test fitness calculation for genetic algorithm."""
        candidates = [sample_kol_candidate]
        algorithm = AdvancedOptimizationAlgorithm(candidates)
        
        solution = [sample_kol_candidate]
        
        fitness = algorithm._calculate_fitness(
            solution,
            OptimizationObjective.MAXIMIZE_ENGAGEMENT,
            sample_optimization_constraints
        )
        
        assert isinstance(fitness, float)
        assert fitness > 0  # Should be positive for valid solution
    
    def test_crossover(self, sample_kol_candidate):
        """Test genetic algorithm crossover operation."""
        # Create two different candidates
        candidate1 = sample_kol_candidate
        candidate2 = MagicMock()
        candidate2.kol_id = "different_kol"
        
        candidates = [candidate1, candidate2]
        algorithm = AdvancedOptimizationAlgorithm(candidates)
        
        parent1 = [candidate1]
        parent2 = [candidate2]
        
        child = algorithm._crossover(parent1, parent2)
        
        assert isinstance(child, list)
        assert len(child) <= len(parent1) + len(parent2)
    
    def test_mutate(self, sample_kol_candidate):
        """Test genetic algorithm mutation operation."""
        candidate1 = sample_kol_candidate
        candidate2 = MagicMock()
        candidate2.kol_id = "different_kol"
        
        candidates = [candidate1, candidate2]
        algorithm = AdvancedOptimizationAlgorithm(candidates)
        
        solution = [candidate1]
        
        # Test mutation doesn't crash
        mutated = algorithm._mutate(solution)
        assert isinstance(mutated, list)


class TestConstraintSatisfactionSolver:
    """Test cases for Constraint Satisfaction Solver."""
    
    def test_init(self, sample_kol_candidate):
        """Test solver initialization."""
        candidates = [sample_kol_candidate]
        solver = ConstraintSatisfactionSolver(candidates)
        
        assert solver.candidates == candidates
        assert hasattr(solver, 'logger')
    
    def test_apply_hard_constraints(self, sample_kol_candidate, sample_optimization_constraints):
        """Test hard constraint filtering."""
        candidates = [sample_kol_candidate]
        solver = ConstraintSatisfactionSolver(candidates)
        
        # Test with candidate that meets constraints
        filtered = solver._apply_hard_constraints(candidates, sample_optimization_constraints)
        assert len(filtered) == 1
        
        # Test with candidate that exceeds budget
        expensive_constraints = OptimizationConstraints(
            max_budget=Decimal("1000"),  # Less than candidate cost
            min_kols=1,
            max_kols=5
        )
        filtered = solver._apply_hard_constraints(candidates, expensive_constraints)
        assert len(filtered) == 0
    
    def test_calculate_portfolio_risk(self, sample_kol_candidate):
        """Test portfolio risk calculation."""
        candidates = [sample_kol_candidate]
        solver = ConstraintSatisfactionSolver(candidates)
        
        # Test with single candidate
        risk = solver._calculate_portfolio_risk([sample_kol_candidate])
        assert isinstance(risk, Decimal)
        assert risk >= Decimal("0.0")
        assert risk <= Decimal("1.0")
        
        # Test with empty selection
        risk_empty = solver._calculate_portfolio_risk([])
        assert risk_empty == Decimal("0.0")
    
    async def test_solve(self, sample_kol_candidate, sample_optimization_constraints):
        """Test constraint satisfaction solving."""
        candidates = [sample_kol_candidate]
        solver = ConstraintSatisfactionSolver(candidates)
        
        selected, violations = solver.solve(
            sample_optimization_constraints,
            OptimizationObjective.MAXIMIZE_ENGAGEMENT
        )
        
        assert isinstance(selected, list)
        assert isinstance(violations, list)
        
        # Should select at least one candidate if constraints allow
        if selected:
            assert len(selected) <= sample_optimization_constraints.max_kols
            assert all(isinstance(kol, KOLCandidate) for kol in selected)


# AIDEV-NOTE: Tests for Brief Parser Service

class TestBriefParserService:
    """Test cases for Brief Parser Service."""
    
    def test_init(self):
        """Test parser initialization."""
        parser = BriefParserService()
        
        assert hasattr(parser, 'patterns')
        assert 'budget' in parser.patterns
        assert 'tiers' in parser.patterns
        assert 'categories' in parser.patterns
        assert isinstance(parser.tier_mapping, dict)
        assert isinstance(parser.category_mapping, dict)
    
    def test_clean_brief_text(self):
        """Test text cleaning functionality."""
        parser = BriefParserService()
        
        markdown_text = """
        # Campaign Brief
        **Budget**: THB 100,000
        *Target*: Lifestyle influencers
        `Note`: This is important
        [Link](https://example.com)
        """
        
        cleaned = parser._clean_brief_text(markdown_text)
        
        assert "campaign brief" in cleaned
        assert "budget" in cleaned
        assert "thb 100000" in cleaned  # Comma removed, lowercase
        assert "#" not in cleaned  # Markdown removed
        assert "**" not in cleaned  # Bold formatting removed
        assert "[link]" not in cleaned  # Link text only
    
    async def test_parse_brief_success(self):
        """Test successful brief parsing."""
        parser = BriefParserService()
        
        brief_text = """
        Campaign Brief
        
        We need lifestyle influencers for our new product launch.
        Budget: THB 50,000
        We want 2 micro influencers and 1 mid-tier influencer.
        Target audience: Bangkok, Thailand
        Language: Thai and English
        Goal: Increase brand awareness
        Required hashtags: #lifestyle #fashion #newproduct
        """
        
        result = await parser.parse_brief(
            brief_text=brief_text,
            campaign_id="test_campaign"
        )
        
        assert isinstance(result, BriefParsingResult)
        assert result.campaign_requirements.campaign_id == "test_campaign"
        assert result.campaign_requirements.total_budget > Decimal("0")
        assert len(result.campaign_requirements.target_kol_tiers) > 0
        assert len(result.campaign_requirements.target_categories) > 0
        assert result.parsing_confidence > Decimal("0.5")
        assert isinstance(result.extracted_entities, dict)
    
    async def test_parse_brief_minimal_info(self):
        """Test brief parsing with minimal information."""
        parser = BriefParserService()
        
        brief_text = "We need some influencers."
        
        result = await parser.parse_brief(
            brief_text=brief_text,
            fallback_budget=Decimal("10000")
        )
        
        assert isinstance(result, BriefParsingResult)
        assert result.campaign_requirements.total_budget == Decimal("10000")
        assert len(result.missing_requirements) > 0
        assert result.parsing_confidence < Decimal("0.7")
    
    def test_extract_budget(self):
        """Test budget extraction from text entities."""
        parser = BriefParserService()
        
        # Mock budget entities
        budget_entities = [
            MagicMock(value="budget: THB 75,000", confidence=0.9),
            MagicMock(value="spend 50000 baht", confidence=0.7)
        ]
        
        budget = parser._extract_budget(budget_entities)
        
        assert isinstance(budget, Decimal)
        assert budget > Decimal("0")
    
    def test_extract_kol_tiers(self):
        """Test KOL tier extraction."""
        parser = BriefParserService()
        
        # Mock tier entities
        tier_entities = [
            MagicMock(value="2 micro influencers", confidence=0.8),
            MagicMock(value="1 mid-tier KOL", confidence=0.9)
        ]
        
        tiers = parser._extract_kol_tiers(tier_entities)
        
        assert isinstance(tiers, list)
        assert KOLTier.MICRO in tiers
        assert KOLTier.MID in tiers
    
    def test_extract_categories(self):
        """Test content category extraction."""
        parser = BriefParserService()
        
        # Mock category entities
        category_entities = [
            MagicMock(value="lifestyle content", confidence=0.8),
            MagicMock(value="fashion influencers", confidence=0.9)
        ]
        
        categories = parser._extract_categories(category_entities)
        
        assert isinstance(categories, list)
        assert ContentCategory.LIFESTYLE in categories
        assert ContentCategory.FASHION in categories
    
    def test_calculate_match_confidence(self):
        """Test match confidence calculation."""
        parser = BriefParserService()
        
        # Mock regex match
        match = MagicMock()
        match.group.return_value = "budget: THB 50,000"
        match.start.return_value = 10
        match.end.return_value = 30
        
        full_text = "our campaign budget: thb 50,000 for influencers"
        
        confidence = parser._calculate_match_confidence(match, "budget", full_text)
        
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0
    
    def test_analyze_brief_sentiment(self):
        """Test brief sentiment analysis."""
        parser = BriefParserService()
        
        # Test positive sentiment
        positive_text = "we are excited about this amazing campaign"
        positive_result = parser._analyze_brief_sentiment(positive_text)
        
        assert isinstance(positive_result, dict)
        assert "overall_sentiment" in positive_result
        assert positive_result["overall_sentiment"] > Decimal("0.5")
        
        # Test negative sentiment
        negative_text = "we are worried about this terrible situation"
        negative_result = parser._analyze_brief_sentiment(negative_text)
        
        assert negative_result["overall_sentiment"] < Decimal("0.5")
        
        # Test urgency detection
        urgent_text = "we need this asap urgent deadline"
        urgent_result = parser._analyze_brief_sentiment(urgent_text)
        
        assert urgent_result["urgency_level"] > Decimal("0.0")


# AIDEV-NOTE: Integration tests

@pytest.mark.asyncio
class TestIntegration:
    """Integration tests for the enhanced services."""
    
    async def test_end_to_end_workflow(
        self,
        mock_db_session,
        sample_campaign_requirements,
        sample_optimization_constraints
    ):
        """Test end-to-end workflow from brief parsing to budget optimization."""
        
        # Step 1: Parse brief
        brief_parser = BriefParserService()
        
        brief_text = """
        Campaign: Lifestyle Influencer Marketing
        Budget: THB 50,000
        We need 2 micro influencers and 1 mid-tier influencer.
        Focus on lifestyle and fashion content.
        Target: Bangkok, Thailand
        Goal: Maximize engagement
        """
        
        brief_result = await brief_parser.parse_brief(brief_text)
        
        assert brief_result.is_actionable
        
        # Step 2: Enhanced KOL matching (mocked)
        kol_matching = EnhancedKOLMatchingService(mock_db_session)
        
        with patch.object(kol_matching, '_get_filtered_candidates', return_value=[]), \
             patch.object(kol_matching, '_create_empty_metadata') as mock_metadata:
            
            mock_metadata.return_value = {"campaign_id": "test"}
            
            candidates, metadata = await kol_matching.find_matching_kols_advanced(
                campaign_requirements=brief_result.campaign_requirements
            )
            
            # Should handle empty results gracefully
            assert isinstance(candidates, list)
            assert isinstance(metadata, dict)
        
        # Step 3: Budget optimization (mocked)
        budget_optimizer = EnhancedBudgetOptimizerService(mock_db_session)
        
        with patch.object(budget_optimizer, '_get_campaign_requirements', return_value=MagicMock()), \
             patch.object(budget_optimizer, '_get_enhanced_kol_candidates', return_value=[]), \
             patch.object(budget_optimizer, '_create_empty_optimization_result') as mock_result:
            
            mock_optimization_result = MagicMock()
            mock_result.return_value = mock_optimization_result
            
            optimization_result = await budget_optimizer.optimize_campaign_budget_advanced(
                campaign_id=brief_result.campaign_requirements.campaign_id,
                optimization_constraints=sample_optimization_constraints,
                optimization_objective=brief_result.campaign_requirements.campaign_objective
            )
            
            assert optimization_result is not None
    
    def test_data_quality_and_confidence_tracking(self, sample_kol_candidate):
        """Test data quality and confidence tracking throughout the pipeline."""
        
        # Test confidence propagation
        assert sample_kol_candidate.score_components.overall_confidence > Decimal("0.0")
        
        # Test data freshness tracking
        assert sample_kol_candidate.score_components.data_freshness_days >= 0
        
        # Test risk assessment
        assert Decimal("0.0") <= sample_kol_candidate.overall_risk_score <= Decimal("1.0")
        assert isinstance(sample_kol_candidate.risk_factors, list)


# AIDEV-NOTE: Performance and edge case tests

class TestPerformanceAndEdgeCases:
    """Test performance and edge cases."""
    
    def test_large_candidate_set_performance(self, mock_db_session):
        """Test performance with large candidate sets."""
        
        # Create large number of mock candidates
        candidates = []
        for i in range(1000):
            candidate = MagicMock()
            candidate.kol_id = f"kol_{i}"
            candidate.estimated_total_cost = Decimal(str(1000 + i))
            candidate.overall_score = Decimal(str(0.5 + (i % 100) / 200))
            candidate.overall_risk_score = Decimal(str(0.1 + (i % 50) / 500))
            candidates.append(candidate)
        
        # Test algorithm initialization (should be fast)
        algorithm = AdvancedOptimizationAlgorithm(candidates)
        assert len(algorithm.candidates) == 1000
        
        # Test constraint solver initialization
        solver = ConstraintSatisfactionSolver(candidates)
        assert len(solver.candidates) == 1000
    
    def test_empty_input_handling(self, mock_db_session):
        """Test handling of empty inputs."""
        
        # Test empty candidate list
        algorithm = AdvancedOptimizationAlgorithm([])
        
        constraints = OptimizationConstraints(
            max_budget=Decimal("10000"),
            min_kols=1,
            max_kols=5
        )
        
        # Should handle empty input gracefully
        result = algorithm._greedy_fallback(constraints, OptimizationObjective.MAXIMIZE_ENGAGEMENT)
        assert result == []
        
        # Test constraint solver with empty input
        solver = ConstraintSatisfactionSolver([])
        selected, violations = solver.solve(constraints, OptimizationObjective.MAXIMIZE_ENGAGEMENT)
        
        assert selected == []
        assert len(violations) > 0  # Should report constraint violations
    
    def test_extreme_constraint_values(self, sample_kol_candidate):
        """Test handling of extreme constraint values."""
        
        candidates = [sample_kol_candidate]
        solver = ConstraintSatisfactionSolver(candidates)
        
        # Test with very low budget (impossible to satisfy)
        impossible_constraints = OptimizationConstraints(
            max_budget=Decimal("1"),  # Impossibly low
            min_kols=1,
            max_kols=5
        )
        
        selected, violations = solver.solve(
            impossible_constraints,
            OptimizationObjective.MAXIMIZE_ENGAGEMENT
        )
        
        assert len(selected) == 0
        assert len(violations) > 0
        assert any(v.constraint_type == "feasibility" for v in violations)
        
        # Test with very high budget (should work fine)
        generous_constraints = OptimizationConstraints(
            max_budget=Decimal("1000000"),  # Very high
            min_kols=1,
            max_kols=5
        )
        
        selected, violations = solver.solve(
            generous_constraints,
            OptimizationObjective.MAXIMIZE_ENGAGEMENT
        )
        
        # Should work with generous constraints
        assert len(violations) == 0 or all(v.severity != "hard" for v in violations)
    
    async def test_malformed_brief_handling(self):
        """Test handling of malformed or problematic briefs."""
        
        parser = BriefParserService()
        
        # Test empty brief
        empty_result = await parser.parse_brief("")
        assert empty_result.parsing_confidence < Decimal("0.3")
        
        # Test brief with conflicting information
        conflicting_brief = """
        Budget: THB 10,000
        Budget: THB 100,000  
        We want micro influencers
        We need macro influencers
        Goal: increase sales
        Goal: brand awareness
        """
        
        conflicting_result = await parser.parse_brief(conflicting_brief)
        assert len(conflicting_result.ambiguous_requirements) > 0
        
        # Test brief with special characters
        special_chars_brief = "Budget: $$$ 50,000 %%% need influencers @@@ #test"
        
        special_result = await parser.parse_brief(special_chars_brief)
        assert isinstance(special_result, BriefParsingResult)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])