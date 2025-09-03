"""
Integration Tests for End-to-End Scoring Workflows

AIDEV-NOTE: Comprehensive integration tests that validate complete scoring workflows
from KOL data input through final optimization results, including GraphQL integration.
"""
import pytest
import asyncio
import json
from datetime import datetime, timezone, timedelta
from decimal import Decimal, ROUND_HALF_UP
from typing import List, Dict, Any, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock, patch, Mock
from dataclasses import asdict

import numpy as np

from kol_api.services.scoring.kol_scorer import KOLScorer, ScoreBreakdown
from kol_api.services.enhanced_budget_optimizer import EnhancedBudgetOptimizerService
from kol_api.services.kol_matching import KOLMatchingService
from kol_api.services.models import (
    KOLCandidate, KOLMetricsData, ScoreComponents, OptimizationConstraints,
    OptimizationObjective, OptimizationResult, KOLTier, ContentCategory
)
from kol_api.database.models.kol import KOLProfile, KOLMetrics
from kol_api.database.models.campaign import Campaign, CampaignRequirements


# AIDEV-NOTE: End-to-End Workflow Integration Tests

class TestEndToEndScoringWorkflows:
    """Test complete scoring workflows from data input to optimization output."""
    
    @pytest.fixture
    async def integrated_services(self, mock_db_session, mock_settings):
        """Setup integrated services for end-to-end testing."""
        services = {
            "kol_scorer": None,
            "budget_optimizer": None,
            "kol_matcher": None
        }
        
        # Mock external dependencies
        with patch('kol_api.services.scoring.kol_scorer.SentimentAnalyzer') as mock_sentiment, \
             patch('kol_api.services.scoring.kol_scorer.DemographicMatcher') as mock_demographics:
            
            # Setup mock responses
            mock_sentiment.return_value.analyze.return_value = MagicMock(score=0.7)
            mock_demographics.return_value.match_interests.return_value = 0.8
            
            services["kol_scorer"] = KOLScorer()
            services["budget_optimizer"] = EnhancedBudgetOptimizerService(mock_db_session)
            services["kol_matcher"] = KOLMatchingService(mock_db_session)
        
        return services
    
    @pytest.fixture
    def complete_campaign_scenario(self):
        """Complete campaign scenario with requirements and KOL pool."""
        from tests.fixtures.test_data_factory import (
            CampaignDataFactory, TestScenarioFactory
        )
        
        # Create campaign requirements
        campaign_requirements = CampaignDataFactory.create_campaign_requirements(
            campaign_type="engagement",
            budget_size="medium",
            complexity="complex",
            total_budget=Decimal("75000")
        )
        
        # Create diverse KOL pool
        kol_pool = TestScenarioFactory.create_mixed_quality_kol_pool(25)
        
        # Create optimization constraints
        constraints = CampaignDataFactory.create_optimization_constraints(
            strictness="medium",
            risk_tolerance="medium",
            max_budget=campaign_requirements.total_budget,
            tier_requirements={"micro": 3, "mid": 1}
        )
        
        return {
            "campaign_requirements": campaign_requirements,
            "kol_pool": kol_pool,
            "optimization_constraints": constraints
        }
    
    async def test_complete_kol_scoring_workflow(self, integrated_services, complete_campaign_scenario):
        """Test complete KOL scoring workflow from raw data to final scores."""
        kol_scorer = integrated_services["kol_scorer"]
        scenario = complete_campaign_scenario
        
        # Create mock campaign and KOL
        campaign = self._create_mock_campaign(scenario["campaign_requirements"])
        test_kol, test_metrics = scenario["kol_pool"][0]  # Use first KOL from pool
        
        # Run complete scoring workflow
        score_breakdown = await kol_scorer.score_kol(test_kol, campaign, MagicMock())
        
        # Validate scoring completeness
        assert isinstance(score_breakdown, ScoreBreakdown), "Should return ScoreBreakdown object"
        assert score_breakdown.composite_score > 0, "Should calculate composite score"
        assert 0 <= score_breakdown.overall_confidence <= 1, "Confidence should be in [0,1] range"
        
        # Validate individual score components
        score_components = [
            score_breakdown.roi_score,
            score_breakdown.audience_quality_score,
            score_breakdown.brand_safety_score,
            score_breakdown.content_relevance_score,
            score_breakdown.demographic_fit_score,
            score_breakdown.reliability_score
        ]
        
        for score in score_components:
            assert 0 <= score <= 1, f"Score component {score} out of bounds [0,1]"
        
        # Validate confidence components
        confidence_components = [
            score_breakdown.roi_confidence,
            score_breakdown.audience_quality_confidence,
            score_breakdown.brand_safety_confidence,
            score_breakdown.content_relevance_confidence,
            score_breakdown.demographic_fit_confidence,
            score_breakdown.reliability_confidence
        ]
        
        for confidence in confidence_components:
            assert 0 <= confidence <= 1, f"Confidence component {confidence} out of bounds [0,1]"
        
        # Validate weighted score calculation
        expected_weighted_score = score_breakdown.weighted_score
        calculated_weighted_score = score_breakdown.composite_score
        
        assert abs(expected_weighted_score - calculated_weighted_score) < 0.001, \
            "Weighted score calculation mismatch"
    
    async def test_end_to_end_budget_optimization_workflow(
        self, 
        integrated_services, 
        complete_campaign_scenario
    ):
        """Test complete budget optimization workflow from KOL pool to final allocation."""
        budget_optimizer = integrated_services["budget_optimizer"]
        scenario = complete_campaign_scenario
        
        # Mock campaign data in database
        campaign_id = "test_campaign_123"
        with patch.object(budget_optimizer, '_get_campaign_requirements') as mock_campaign, \
             patch.object(budget_optimizer, '_get_enhanced_kol_candidates') as mock_candidates, \
             patch.object(budget_optimizer, '_run_advanced_optimization') as mock_optimize:
            
            # Setup mock responses
            mock_campaign.return_value = self._create_mock_campaign(scenario["campaign_requirements"])
            mock_candidates.return_value = self._convert_kol_pool_to_candidates(scenario["kol_pool"])
            
            # Mock optimization result
            mock_optimization_result = self._create_mock_optimization_result(
                scenario["kol_pool"][:5],  # Select first 5 KOLs
                scenario["optimization_constraints"]
            )
            mock_optimize.return_value = mock_optimization_result
            
            # Run complete optimization workflow
            optimization_result = await budget_optimizer.optimize_campaign_budget_advanced(
                campaign_id=campaign_id,
                optimization_constraints=scenario["optimization_constraints"],
                optimization_objective=OptimizationObjective.MAXIMIZE_ENGAGEMENT,
                algorithm="constraint_satisfaction",
                enable_alternative_scenarios=True
            )
            
            # Validate optimization result structure
            assert isinstance(optimization_result, OptimizationResult), \
                "Should return OptimizationResult object"
            
            # Validate optimization result content
            assert len(optimization_result.selected_kols) > 0, "Should select KOLs"
            assert optimization_result.total_cost <= scenario["optimization_constraints"].max_budget, \
                "Should respect budget constraint"
            assert optimization_result.optimization_score > 0, "Should have positive optimization score"
            
            # Validate tier distribution
            tier_counts = {}
            for kol in optimization_result.selected_kols:
                tier = kol.tier.value.lower()
                tier_counts[tier] = tier_counts.get(tier, 0) + 1
            
            # Check tier requirements satisfaction
            tier_requirements = scenario["optimization_constraints"].tier_requirements
            for tier, required_count in tier_requirements.items():
                actual_count = tier_counts.get(tier, 0)
                assert actual_count >= required_count, \
                    f"Tier requirement not met: {tier} requires {required_count}, got {actual_count}"
    
    async def test_kol_matching_integration_workflow(self, integrated_services, complete_campaign_scenario):
        """Test KOL matching service integration with scoring system."""
        kol_matcher = integrated_services["kol_matcher"]
        scenario = complete_campaign_scenario
        
        # Mock database queries for KOL matching
        with patch.object(kol_matcher.db_session, 'execute') as mock_execute:
            # Setup mock KOL query results
            mock_kol_profiles = self._create_mock_db_kol_profiles(scenario["kol_pool"])
            mock_execute.return_value.scalars.return_value.all.return_value = mock_kol_profiles
            
            # Create campaign brief for matching
            campaign_brief = {
                "title": "Summer Fashion Campaign",
                "description": "Looking for fashion and lifestyle influencers for summer collection",
                "target_audience": "Women 18-35",
                "campaign_goals": ["brand awareness", "engagement"],
                "hashtags": ["summer", "fashion", "style"],
                "budget_range": {"min": 50000, "max": 100000}
            }
            
            # Run KOL matching workflow
            matching_results = await kol_matcher.find_matching_kols(
                campaign_brief=campaign_brief,
                max_results=10,
                include_scores=True
            )
            
            # Validate matching results
            assert len(matching_results) > 0, "Should find matching KOLs"
            
            for result in matching_results:
                assert "kol_profile" in result, "Should include KOL profile"
                assert "match_score" in result, "Should include match score"
                assert "score_breakdown" in result, "Should include score breakdown"
                
                # Validate match score range
                assert 0 <= result["match_score"] <= 1, \
                    f"Match score {result['match_score']} out of bounds [0,1]"
                
                # Validate score breakdown structure
                breakdown = result["score_breakdown"]
                assert isinstance(breakdown, dict), "Score breakdown should be dict"
                assert "roi_score" in breakdown, "Should include ROI score"
                assert "audience_quality_score" in breakdown, "Should include audience quality score"
    
    async def test_multi_service_workflow_consistency(self, integrated_services, complete_campaign_scenario):
        """Test consistency across multiple services in integrated workflow."""
        kol_scorer = integrated_services["kol_scorer"]
        budget_optimizer = integrated_services["budget_optimizer"]
        scenario = complete_campaign_scenario
        
        # Test the same KOL through multiple services
        test_kol, test_metrics = scenario["kol_pool"][0]
        campaign = self._create_mock_campaign(scenario["campaign_requirements"])
        
        # Score KOL individually
        individual_score = await kol_scorer.score_kol(test_kol, campaign, MagicMock())
        
        # Convert KOL to candidate format
        kol_candidate = self._convert_kol_to_candidate(test_kol, test_metrics, individual_score)
        
        # Run through budget optimizer
        with patch.object(budget_optimizer, '_get_campaign_requirements') as mock_campaign, \
             patch.object(budget_optimizer, '_get_enhanced_kol_candidates') as mock_candidates:
            
            mock_campaign.return_value = campaign
            mock_candidates.return_value = [kol_candidate]
            
            # Create minimal constraints for single KOL
            single_kol_constraints = OptimizationConstraints(
                max_budget=Decimal("10000"),
                min_kols=1,
                max_kols=1,
                max_risk_per_kol=Decimal("1.0")
            )
            
            with patch.object(budget_optimizer, '_run_advanced_optimization') as mock_optimize:
                mock_result = OptimizationResult(
                    selected_kols=[kol_candidate],
                    total_cost=kol_candidate.estimated_total_cost,
                    cost_by_tier={kol_candidate.tier.value: kol_candidate.estimated_total_cost},
                    cost_by_category={kol_candidate.primary_category.value: kol_candidate.estimated_total_cost},
                    predicted_total_reach=kol_candidate.predicted_reach,
                    predicted_total_engagement=kol_candidate.predicted_engagement,
                    predicted_total_conversions=kol_candidate.predicted_conversions,
                    predicted_roi=Decimal("0.15"),
                    portfolio_risk_score=kol_candidate.overall_risk_score,
                    portfolio_diversity_score=Decimal("1.0"),
                    optimization_score=Decimal(str(individual_score.composite_score)),
                    budget_utilization=kol_candidate.estimated_total_cost / single_kol_constraints.max_budget,
                    constraints_satisfied=True,
                    constraint_violations=[],
                    tier_distribution={kol_candidate.tier.value: 1},
                    alternative_allocations=[],
                    algorithm_used="test",
                    optimization_time_seconds=0.1,
                    iterations_performed=1,
                    convergence_achieved=True
                )
                mock_optimize.return_value = mock_result
                
                optimization_result = await budget_optimizer.optimize_campaign_budget_advanced(
                    campaign_id="test_campaign",
                    optimization_constraints=single_kol_constraints,
                    optimization_objective=OptimizationObjective.MAXIMIZE_ENGAGEMENT
                )
            
            # Validate consistency between individual scoring and optimization
            optimized_kol = optimization_result.selected_kols[0]
            
            # Scores should be consistent (within reasonable tolerance)
            score_diff = abs(float(optimized_kol.overall_score) - individual_score.composite_score)
            assert score_diff < 0.1, \
                f"Score inconsistency between services: {optimized_kol.overall_score} vs {individual_score.composite_score}"
    
    async def test_error_handling_in_integrated_workflow(self, integrated_services, complete_campaign_scenario):
        """Test error handling and resilience in integrated workflows."""
        budget_optimizer = integrated_services["budget_optimizer"]
        scenario = complete_campaign_scenario
        
        # Test missing campaign scenario
        with pytest.raises((ValueError, Exception)):  # Should raise appropriate error
            await budget_optimizer.optimize_campaign_budget_advanced(
                campaign_id="nonexistent_campaign",
                optimization_constraints=scenario["optimization_constraints"],
                optimization_objective=OptimizationObjective.MAXIMIZE_REACH
            )
        
        # Test empty KOL pool scenario
        with patch.object(budget_optimizer, '_get_campaign_requirements') as mock_campaign, \
             patch.object(budget_optimizer, '_get_enhanced_kol_candidates') as mock_candidates:
            
            mock_campaign.return_value = self._create_mock_campaign(scenario["campaign_requirements"])
            mock_candidates.return_value = []  # Empty KOL pool
            
            optimization_result = await budget_optimizer.optimize_campaign_budget_advanced(
                campaign_id="empty_pool_campaign",
                optimization_constraints=scenario["optimization_constraints"],
                optimization_objective=OptimizationObjective.MAXIMIZE_ENGAGEMENT
            )
            
            # Should handle gracefully with empty result
            assert len(optimization_result.selected_kols) == 0, "Should handle empty KOL pool gracefully"
            assert not optimization_result.constraints_satisfied, "Should indicate constraints not satisfied"
    
    async def test_performance_benchmarking_integrated_workflow(
        self, 
        integrated_services, 
        complete_campaign_scenario
    ):
        """Benchmark performance of integrated workflows."""
        kol_scorer = integrated_services["kol_scorer"]
        budget_optimizer = integrated_services["budget_optimizer"]
        scenario = complete_campaign_scenario
        
        # Performance test: Score multiple KOLs
        kol_pool = scenario["kol_pool"][:10]  # Limit for test performance
        campaign = self._create_mock_campaign(scenario["campaign_requirements"])
        
        import time
        
        # Benchmark individual KOL scoring
        start_time = time.time()
        
        scoring_tasks = []
        for kol, metrics in kol_pool:
            scoring_tasks.append(kol_scorer.score_kol(kol, campaign, MagicMock()))
        
        score_results = await asyncio.gather(*scoring_tasks)
        
        scoring_time = time.time() - start_time
        
        # Performance requirements
        assert scoring_time < 30.0, f"KOL scoring too slow: {scoring_time}s for {len(kol_pool)} KOLs"
        assert len(score_results) == len(kol_pool), "Should score all KOLs"
        
        # Validate all scores are valid
        for score_breakdown in score_results:
            assert isinstance(score_breakdown, ScoreBreakdown), "All results should be ScoreBreakdown objects"
            assert score_breakdown.composite_score >= 0, "All scores should be non-negative"
        
        # Benchmark optimization workflow
        candidates = [
            self._convert_kol_to_candidate(kol, metrics, score) 
            for (kol, metrics), score in zip(kol_pool, score_results)
        ]
        
        with patch.object(budget_optimizer, '_get_campaign_requirements') as mock_campaign, \
             patch.object(budget_optimizer, '_get_enhanced_kol_candidates') as mock_candidates, \
             patch.object(budget_optimizer, '_run_advanced_optimization') as mock_optimize:
            
            mock_campaign.return_value = campaign
            mock_candidates.return_value = candidates
            
            # Create realistic optimization result
            selected_candidates = candidates[:5]  # Select top 5
            mock_result = self._create_mock_optimization_result(
                [(c, None) for c in selected_candidates],  # Convert format
                scenario["optimization_constraints"]
            )
            mock_optimize.return_value = mock_result
            
            start_time = time.time()
            
            optimization_result = await budget_optimizer.optimize_campaign_budget_advanced(
                campaign_id="performance_test",
                optimization_constraints=scenario["optimization_constraints"],
                optimization_objective=OptimizationObjective.BALANCED
            )
            
            optimization_time = time.time() - start_time
            
            # Performance requirements for optimization
            assert optimization_time < 20.0, f"Budget optimization too slow: {optimization_time}s"
            assert len(optimization_result.selected_kols) > 0, "Should select KOLs within time limit"


# AIDEV-NOTE: GraphQL Integration Tests

class TestGraphQLIntegrationWorkflows:
    """Test GraphQL resolver integration with scoring and optimization services."""
    
    @pytest.fixture
    def mock_graphql_context(self, mock_db_session):
        """Mock GraphQL context with database session."""
        context = MagicMock()
        context.db_session = mock_db_session
        return context
    
    async def test_kol_scoring_graphql_resolver_integration(self, mock_graphql_context):
        """Test KOL scoring through GraphQL resolver."""
        from kol_api.graphql.resolvers.scoring_resolvers import score_kol_for_campaign
        
        # Mock input data
        kol_id = "test_kol_123"
        campaign_id = "test_campaign_456"
        
        # Mock database queries
        with patch('kol_api.graphql.resolvers.scoring_resolvers.select') as mock_select:
            # Setup mock KOL and campaign data
            mock_kol = self._create_mock_db_kol()
            mock_campaign = self._create_mock_db_campaign()
            
            # Setup query results
            mock_context.db_session.execute.return_value.scalar_one_or_none.side_effect = [
                mock_kol, mock_campaign
            ]
            
            # Mock KOL scorer
            with patch('kol_api.graphql.resolvers.scoring_resolvers.KOLScorer') as mock_scorer_class:
                mock_scorer = mock_scorer_class.return_value
                mock_score_breakdown = ScoreBreakdown(
                    roi_score=0.8, audience_quality_score=0.75, brand_safety_score=0.9,
                    content_relevance_score=0.7, demographic_fit_score=0.85,
                    reliability_score=0.8, roi_confidence=0.85,
                    audience_quality_confidence=0.8, brand_safety_confidence=0.95,
                    content_relevance_confidence=0.75, demographic_fit_confidence=0.8,
                    reliability_confidence=0.85, composite_score=0.78,
                    overall_confidence=0.83, missing_data_penalty=0.05
                )
                mock_scorer.score_kol.return_value = mock_score_breakdown
                
                # Call GraphQL resolver
                result = await score_kol_for_campaign(
                    None, None, kol_id=kol_id, campaign_id=campaign_id, 
                    info=MagicMock(context=mock_graphql_context)
                )
                
                # Validate GraphQL response structure
                assert result is not None, "Should return scoring result"
                assert "composite_score" in result, "Should include composite score"
                assert "score_components" in result, "Should include score components"
                assert "confidence_metrics" in result, "Should include confidence metrics"
                
                # Validate score values
                assert 0 <= result["composite_score"] <= 1, "Composite score should be in [0,1] range"
    
    async def test_budget_optimization_graphql_resolver_integration(self, mock_graphql_context):
        """Test budget optimization through GraphQL resolver."""
        from kol_api.graphql.resolvers.budget_resolvers import optimize_campaign_budget
        
        # Mock optimization parameters
        optimization_params = {
            "campaign_id": "test_campaign_789",
            "max_budget": 100000,
            "optimization_objective": "MAXIMIZE_ENGAGEMENT",
            "constraints": {
                "min_kols": 3,
                "max_kols": 8,
                "tier_requirements": {"micro": 2, "mid": 1}
            }
        }
        
        # Mock budget optimizer service
        with patch('kol_api.graphql.resolvers.budget_resolvers.EnhancedBudgetOptimizerService') as mock_service_class:
            mock_service = mock_service_class.return_value
            
            # Create mock optimization result
            from tests.fixtures.test_data_factory import TestScenarioFactory
            kol_pool = TestScenarioFactory.create_mixed_quality_kol_pool(5)
            mock_result = self._create_mock_optimization_result(kol_pool, None)
            mock_service.optimize_campaign_budget_advanced.return_value = mock_result
            
            # Call GraphQL resolver
            result = await optimize_campaign_budget(
                None, None, **optimization_params, 
                info=MagicMock(context=mock_graphql_context)
            )
            
            # Validate GraphQL response structure
            assert result is not None, "Should return optimization result"
            assert "selected_kols" in result, "Should include selected KOLs"
            assert "total_cost" in result, "Should include total cost"
            assert "optimization_score" in result, "Should include optimization score"
            assert "constraints_satisfied" in result, "Should include constraint satisfaction status"
            
            # Validate result content
            assert len(result["selected_kols"]) > 0, "Should select KOLs"
            assert result["total_cost"] > 0, "Should have positive total cost"
            assert result["optimization_score"] >= 0, "Should have non-negative optimization score"
    
    async def test_kol_search_and_filtering_graphql_integration(self, mock_graphql_context):
        """Test KOL search and filtering through GraphQL."""
        from kol_api.graphql.resolvers.kol_resolvers import search_kols_with_scoring
        
        # Mock search parameters
        search_params = {
            "filters": {
                "min_followers": 10000,
                "max_followers": 500000,
                "categories": ["LIFESTYLE", "FASHION"],
                "min_engagement_rate": 0.02,
                "verified_only": True
            },
            "scoring_context": {
                "campaign_id": "search_context_campaign",
                "include_match_score": True
            },
            "limit": 20
        }
        
        # Mock database query and KOL matching
        with patch('kol_api.graphql.resolvers.kol_resolvers.select') as mock_select, \
             patch('kol_api.graphql.resolvers.kol_resolvers.KOLMatchingService') as mock_matching_service:
            
            # Setup mock KOL data
            from tests.fixtures.test_data_factory import TestScenarioFactory
            kol_pool = TestScenarioFactory.create_mixed_quality_kol_pool(10)
            mock_kols = [self._create_mock_db_kol_from_factory(kol, metrics) 
                        for kol, metrics in kol_pool]
            
            mock_context.db_session.execute.return_value.scalars.return_value.all.return_value = mock_kols
            
            # Setup mock matching service
            mock_matching = mock_matching_service.return_value
            mock_matching_results = [
                {
                    "kol_profile": mock_kol,
                    "match_score": np.random.uniform(0.6, 0.95),
                    "score_breakdown": {
                        "roi_score": np.random.uniform(0.5, 0.9),
                        "audience_quality_score": np.random.uniform(0.6, 0.95),
                        "brand_safety_score": np.random.uniform(0.7, 1.0)
                    }
                }
                for mock_kol in mock_kols
            ]
            mock_matching.find_matching_kols.return_value = mock_matching_results
            
            # Call GraphQL resolver
            result = await search_kols_with_scoring(
                None, None, **search_params,
                info=MagicMock(context=mock_graphql_context)
            )
            
            # Validate search result structure
            assert result is not None, "Should return search results"
            assert "kols" in result, "Should include KOL list"
            assert "total_count" in result, "Should include total count"
            assert "search_metadata" in result, "Should include search metadata"
            
            # Validate individual KOL results
            for kol_result in result["kols"]:
                assert "profile" in kol_result, "Should include profile data"
                assert "match_score" in kol_result, "Should include match score"
                assert "score_components" in kol_result, "Should include score components"
                
                # Validate score ranges
                assert 0 <= kol_result["match_score"] <= 1, \
                    f"Match score {kol_result['match_score']} out of bounds"


# AIDEV-NOTE: Data Pipeline Integration Tests

class TestDataPipelineIntegration:
    """Test data pipeline integration with real-time scoring updates."""
    
    async def test_realtime_score_update_pipeline(self, mock_db_session, mock_redis_client):
        """Test real-time score updates through data pipeline."""
        from kol_api.services.scoring.kol_scorer import KOLScorer
        
        # Mock data pipeline components
        with patch('kol_api.services.scoring.kol_scorer.SentimentAnalyzer'), \
             patch('kol_api.services.scoring.kol_scorer.DemographicMatcher'):
            
            kol_scorer = KOLScorer()
            
            # Simulate KOL data update
            from tests.fixtures.test_data_factory import KOLDataFactory
            original_kol, original_metrics = KOLDataFactory.create_kol_profile(
                tier="micro", quality_level="medium", data_completeness="complete"
            )
            
            # Create updated KOL with improved metrics
            updated_kol, updated_metrics = KOLDataFactory.create_kol_profile(
                tier="micro", quality_level="high", data_completeness="complete",
                kol_id=original_kol.id,
                username=original_kol.username
            )
            
            # Mock campaign for scoring
            campaign = self._create_mock_campaign({
                "campaign_type": "engagement",
                "target_categories": ["lifestyle"],
                "total_budget": Decimal("50000")
            })
            
            # Score original KOL
            original_score = await kol_scorer.score_kol(original_kol, campaign, mock_db_session)
            
            # Score updated KOL
            updated_score = await kol_scorer.score_kol(updated_kol, campaign, mock_db_session)
            
            # Validate score improvement
            assert updated_score.composite_score >= original_score.composite_score * 0.95, \
                "Updated metrics should maintain or improve score"
            
            # Specific improvements should be reflected
            if (updated_metrics.engagement_rate and original_metrics.engagement_rate and 
                updated_metrics.engagement_rate > original_metrics.engagement_rate):
                assert updated_score.audience_quality_score >= original_score.audience_quality_score, \
                    "Improved engagement should improve audience quality score"
    
    async def test_batch_score_processing_pipeline(self, mock_db_session):
        """Test batch processing of KOL scores for large datasets."""
        from kol_api.services.scoring.kol_scorer import KOLScorer
        
        with patch('kol_api.services.scoring.kol_scorer.SentimentAnalyzer'), \
             patch('kol_api.services.scoring.kol_scorer.DemographicMatcher'):
            
            kol_scorer = KOLScorer()
            
            # Create large batch of KOLs
            from tests.fixtures.test_data_factory import TestScenarioFactory
            large_kol_pool = TestScenarioFactory.create_mixed_quality_kol_pool(50)
            
            # Create campaign for batch scoring
            campaign = self._create_mock_campaign({
                "campaign_type": "awareness",
                "target_categories": ["lifestyle", "fashion"],
                "total_budget": Decimal("200000")
            })
            
            # Batch process scores
            import time
            start_time = time.time()
            
            batch_tasks = [
                kol_scorer.score_kol(kol, campaign, mock_db_session)
                for kol, metrics in large_kol_pool
            ]
            
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            processing_time = time.time() - start_time
            
            # Validate batch processing performance
            assert processing_time < 60.0, f"Batch processing too slow: {processing_time}s for {len(large_kol_pool)} KOLs"
            
            # Validate all results
            successful_scores = [
                result for result in batch_results 
                if isinstance(result, ScoreBreakdown)
            ]
            
            assert len(successful_scores) >= len(large_kol_pool) * 0.9, \
                f"Too many batch processing failures: {len(successful_scores)}/{len(large_kol_pool)}"
            
            # Validate score distribution
            composite_scores = [score.composite_score for score in successful_scores]
            
            # Should have reasonable distribution
            assert min(composite_scores) >= 0, "All scores should be non-negative"
            assert max(composite_scores) <= 1, "All scores should be <= 1"
            
            # Should have some variety in scores
            score_std = np.std(composite_scores)
            assert score_std > 0.05, f"Insufficient score variety: std={score_std}"
    
    # Helper methods for creating test data
    def _create_mock_campaign(self, campaign_data: Dict[str, Any]) -> MagicMock:
        """Create mock campaign object."""
        campaign = MagicMock(spec=Campaign)
        campaign.id = campaign_data.get("campaign_id", "test_campaign")
        campaign.campaign_type = campaign_data.get("campaign_type", "engagement")
        
        # Create mock requirements
        requirements = MagicMock(spec=CampaignRequirements)
        requirements.required_categories = campaign_data.get("target_categories", [])
        requirements.target_keywords = campaign_data.get("target_keywords", [])
        requirements.target_locations = campaign_data.get("target_locations", [])
        requirements.target_age_range = campaign_data.get("target_age_range", (18, 35))
        requirements.target_interests = campaign_data.get("target_interests", [])
        
        campaign.requirements = requirements
        return campaign
    
    def _convert_kol_pool_to_candidates(self, kol_pool: List[Tuple]) -> List[KOLCandidate]:
        """Convert KOL pool to candidate list."""
        candidates = []
        for kol, metrics in kol_pool:
            # Create mock score components
            score_components = ScoreComponents(
                roi_score=Decimal(str(np.random.uniform(0.4, 0.9))),
                audience_quality_score=Decimal(str(np.random.uniform(0.5, 0.95))),
                brand_safety_score=Decimal(str(np.random.uniform(0.7, 1.0))),
                content_relevance_score=Decimal(str(np.random.uniform(0.3, 0.8))),
                demographic_fit_score=Decimal(str(np.random.uniform(0.4, 0.85))),
                reliability_score=Decimal(str(np.random.uniform(0.5, 0.9))),
                roi_confidence=Decimal("0.8"),
                audience_confidence=Decimal("0.85"),
                brand_safety_confidence=Decimal("0.9"),
                content_relevance_confidence=Decimal("0.75"),
                demographic_confidence=Decimal("0.8"),
                reliability_confidence=Decimal("0.85"),
                overall_confidence=Decimal("0.82"),
                data_freshness_days=np.random.randint(1, 30)
            )
            
            candidate = KOLCandidate(
                kol_id=kol.id,
                username=kol.username,
                display_name=kol.display_name,
                platform=kol.platform.value,
                tier=KOLTier(kol.tier.value.upper()),
                primary_category=ContentCategory(kol.primary_category.value.upper()),
                metrics=self._convert_metrics_to_data(metrics),
                score_components=score_components,
                overall_score=Decimal(str(np.random.uniform(0.4, 0.9))),
                predicted_reach=int(metrics.follower_count * 0.15),
                predicted_engagement=int(metrics.follower_count * float(metrics.engagement_rate or 0.03)),
                predicted_conversions=int(metrics.follower_count * float(metrics.engagement_rate or 0.03) * 0.02),
                estimated_cost_per_post=metrics.rate_per_post or Decimal("1000"),
                estimated_total_cost=metrics.rate_per_post or Decimal("1000"),
                risk_factors=["Standard risk factors"],
                overall_risk_score=Decimal(str(np.random.uniform(0.1, 0.4)))
            )
            candidates.append(candidate)
        
        return candidates
    
    def _convert_metrics_to_data(self, metrics) -> KOLMetricsData:
        """Convert metrics to KOLMetricsData."""
        return KOLMetricsData(
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
    
    def _convert_kol_to_candidate(
        self, 
        kol, 
        metrics, 
        score_breakdown: ScoreBreakdown
    ) -> KOLCandidate:
        """Convert KOL and score breakdown to candidate."""
        score_components = ScoreComponents(
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
            data_freshness_days=1
        )
        
        return KOLCandidate(
            kol_id=kol.id,
            username=kol.username,
            display_name=kol.display_name,
            platform=kol.platform.value,
            tier=KOLTier(kol.tier.value.upper()),
            primary_category=ContentCategory(kol.primary_category.value.upper()),
            metrics=self._convert_metrics_to_data(metrics),
            score_components=score_components,
            overall_score=Decimal(str(score_breakdown.composite_score)),
            predicted_reach=int(metrics.follower_count * 0.15),
            predicted_engagement=int(metrics.follower_count * float(metrics.engagement_rate or 0.03)),
            predicted_conversions=int(metrics.follower_count * float(metrics.engagement_rate or 0.03) * 0.02),
            estimated_cost_per_post=metrics.rate_per_post or Decimal("1000"),
            estimated_total_cost=metrics.rate_per_post or Decimal("1000"),
            risk_factors=["Standard risk factors"],
            overall_risk_score=Decimal(str(np.random.uniform(0.1, 0.4)))
        )
    
    def _create_mock_optimization_result(
        self, 
        selected_kol_data: List[Tuple], 
        constraints: Optional[OptimizationConstraints]
    ) -> OptimizationResult:
        """Create mock optimization result."""
        candidates = [
            self._convert_kol_pool_to_candidates([kol_tuple])[0] 
            for kol_tuple in selected_kol_data
        ]
        
        total_cost = sum(c.estimated_total_cost for c in candidates)
        
        return OptimizationResult(
            selected_kols=candidates,
            total_cost=total_cost,
            cost_by_tier={},
            cost_by_category={},
            predicted_total_reach=sum(c.predicted_reach for c in candidates),
            predicted_total_engagement=sum(c.predicted_engagement for c in candidates),
            predicted_total_conversions=sum(c.predicted_conversions for c in candidates),
            predicted_roi=Decimal("0.15"),
            portfolio_risk_score=Decimal("0.3"),
            portfolio_diversity_score=Decimal("0.7"),
            optimization_score=Decimal("0.8"),
            budget_utilization=total_cost / constraints.max_budget if constraints else Decimal("0.8"),
            constraints_satisfied=True,
            constraint_violations=[],
            tier_distribution={},
            alternative_allocations=[],
            algorithm_used="test_algorithm",
            optimization_time_seconds=1.5,
            iterations_performed=10,
            convergence_achieved=True
        )
    
    def _create_mock_db_kol(self) -> MagicMock:
        """Create mock database KOL object."""
        kol = MagicMock(spec=KOLProfile)
        kol.id = "db_kol_123"
        kol.username = "@test_db_kol"
        kol.display_name = "Test DB KOL"
        kol.platform = MagicMock()
        kol.platform.value = "instagram"
        kol.tier = MagicMock()
        kol.tier.value = "micro"
        kol.primary_category = MagicMock()
        kol.primary_category.value = "lifestyle"
        return kol
    
    def _create_mock_db_campaign(self) -> MagicMock:
        """Create mock database campaign object."""
        campaign = MagicMock(spec=Campaign)
        campaign.id = "db_campaign_456"
        campaign.campaign_type = "engagement"
        return campaign
    
    def _create_mock_db_kol_from_factory(self, kol, metrics) -> MagicMock:
        """Create mock database KOL from factory data."""
        db_kol = MagicMock(spec=KOLProfile)
        db_kol.id = kol.id
        db_kol.username = kol.username
        db_kol.display_name = kol.display_name
        db_kol.platform = kol.platform
        db_kol.tier = kol.tier
        db_kol.primary_category = kol.primary_category
        db_kol.latest_metrics = metrics
        return db_kol


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "--durations=15"])