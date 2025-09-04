"""Enhanced KOL matching service for POC2 - Sophisticated AI-powered KOL selection."""

import asyncio
import hashlib
import re
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Set
from decimal import Decimal, ROUND_HALF_UP
from dataclasses import asdict

import numpy as np
from sqlalchemy import select, and_, or_, text, func
from sqlalchemy.ext.asyncio import AsyncSession
import structlog

# AIDEV-NOTE: Import enhanced models and utilities
from kol_api.database.models.kol import KOL, KOLMetrics, KOLContent
from kol_api.database.models.campaign import Campaign
from kol_api.database.models.scoring import KOLScore
from kol_api.services.models import (
    ScoreComponents, KOLMetricsData, CampaignRequirements, 
    KOLCandidate, SemanticMatchingRequest, BriefParsingResult,
    ConstraintViolation, KOLTier, ContentCategory
)
from kol_api.config import settings

logger = structlog.get_logger()


class EnhancedKOLMatchingService:
    """Enhanced service for sophisticated AI-powered KOL matching and ranking."""
    
    def __init__(self, db_session: AsyncSession):
        self.db_session = db_session
        
        # AIDEV-NOTE: POC2 sophisticated scoring weights 
        self.scoring_weights = {
            "roi_score": 0.25,           # ROI Score (25%)
            "audience_quality": 0.25,    # Audience Quality Score (25%)
            "brand_safety": 0.20,        # Brand Safety Score (20%)
            "content_relevance": 0.15,   # Content Relevance Score (15%)
            "demographic_fit": 0.10,     # Demographic Fit Score (10%)
            "reliability": 0.05          # Reliability Score (5%)
        }
        
        # AIDEV-NOTE: Confidence thresholds for missing data handling
        self.confidence_thresholds = {
            "high_confidence": Decimal("0.9"),
            "medium_confidence": Decimal("0.7"),
            "low_confidence": Decimal("0.5")
        }
        
        # AIDEV-NOTE: Cache for performance optimization
        self._embedding_cache = {}
        self._score_cache = {}
    
    async def find_matching_kols_advanced(
        self,
        campaign_requirements: CampaignRequirements,
        limit: int = 50,
        enable_semantic_matching: bool = True,
        confidence_threshold: Decimal = Decimal("0.7")
    ) -> Tuple[List[KOLCandidate], Dict[str, Any]]:
        """
        Find KOLs using sophisticated multi-factor scoring system.
        
        Args:
            campaign_requirements: Comprehensive campaign requirements
            limit: Maximum number of KOL candidates to return
            enable_semantic_matching: Use vector embeddings for content similarity
            confidence_threshold: Minimum confidence score for results
            
        Returns:
            Tuple of (matched_kol_candidates, detailed_matching_metadata)
        """
        
        start_time = datetime.utcnow()
        
        try:
            # AIDEV-NOTE: Phase 1 - Hard constraint filtering
            logger.info(
                "Starting sophisticated KOL matching",
                campaign_id=campaign_requirements.campaign_id,
                target_tiers=campaign_requirements.target_kol_tiers,
                budget=float(campaign_requirements.total_budget)
            )
            
            candidate_kols = await self._get_filtered_candidates(
                campaign_requirements
            )
            
            if not candidate_kols:
                logger.warning(
                    "No candidates passed hard constraints",
                    campaign_id=campaign_requirements.campaign_id
                )
                return [], self._create_empty_metadata(campaign_requirements)
            
            # AIDEV-NOTE: Phase 2 - Advanced scoring with missing data handling
            scored_candidates = []
            
            for kol, metrics in candidate_kols:
                try:
                    candidate = await self._create_kol_candidate_with_scoring(
                        kol=kol,
                        metrics=metrics, 
                        campaign_requirements=campaign_requirements,
                        enable_semantic=enable_semantic_matching
                    )
                    
                    # AIDEV-NOTE: Apply confidence threshold filtering
                    if candidate.score_components.overall_confidence >= confidence_threshold:
                        scored_candidates.append(candidate)
                    
                except Exception as e:
                        logger.warning(
                            "Failed to score KOL candidate",
                            kol_id=kol.id,
                            error=str(e)
                        )
                        continue
            
            # AIDEV-NOTE: Phase 3 - Final ranking and selection
            final_candidates = self._rank_and_select_candidates(
                scored_candidates, 
                limit,
                campaign_requirements.campaign_objective
            )
            
            # AIDEV-NOTE: Generate comprehensive metadata
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            metadata = {
                "campaign_id": campaign_requirements.campaign_id,
                "matching_criteria": self._extract_detailed_criteria(campaign_requirements),
                "total_candidates_evaluated": len(candidate_kols),
                "candidates_passed_scoring": len(scored_candidates), 
                "final_selected": len(final_candidates),
                "scoring_method": "enhanced_multi_factor_v2",
                "weights_used": self.scoring_weights,
                "confidence_threshold": float(confidence_threshold),
                "semantic_matching_enabled": enable_semantic_matching,
                "processing_time_seconds": processing_time,
                "algorithm_version": "2.1",
                "data_quality_summary": self._generate_data_quality_summary(scored_candidates)
            }
            
            logger.info(
                "Enhanced KOL matching completed successfully",
                campaign_id=campaign_requirements.campaign_id,
                final_count=len(final_candidates),
                processing_time=processing_time
            )
            
            return final_candidates, metadata
            
        except Exception as e:
            logger.error(
                "KOL matching failed",
                campaign_id=campaign_requirements.campaign_id,
                error=str(e)
            )
            raise
    
    async def find_similar_kols_semantic(
        self,
        reference_kol_id: str,
        campaign_requirements: Optional[CampaignRequirements] = None,
        limit: int = 20,
        similarity_threshold: Decimal = Decimal("0.75")
    ) -> List[KOLCandidate]:
        """
        Find KOLs semantically similar to reference KOL using enhanced vector search.
        
        Args:
            reference_kol_id: Reference KOL ID for similarity matching
            campaign_requirements: Optional campaign context for relevance
            limit: Maximum number of similar KOL candidates to return
            similarity_threshold: Minimum cosine similarity score (0-1)
            
        Returns:
            List of similar KOL candidates with similarity scores and full scoring
        """
        
        # AIDEV-NOTE: Get reference KOL with comprehensive data
        reference_query = select(KOL).where(KOL.id == reference_kol_id)
        reference_result = await self.db_session.execute(reference_query)
        reference_kol = reference_result.scalar_one_or_none()
        
        if not reference_kol:
            logger.warning("Reference KOL not found", kol_id=reference_kol_id)
            return []
        
        # AIDEV-NOTE: Get or generate content embedding for reference KOL
        reference_embedding = await self._get_or_generate_embedding(reference_kol)
        
        if not reference_embedding:
            logger.warning(
                "Could not generate embedding for reference KOL",
                kol_id=reference_kol_id
            )
            return []
        
        # AIDEV-NOTE: Enhanced semantic similarity query with campaign filtering
        similarity_conditions = [
            "k.id != :reference_id",
            "k.is_active = true", 
            "k.is_brand_safe = true",
            "(k.content_embedding <=> :reference_embedding) <= :threshold"
        ]
        
        # AIDEV-NOTE: Add campaign-specific filters if provided
        query_params = {
            "reference_embedding": reference_embedding,
            "reference_id": reference_kol_id,
            "threshold": float(1.0 - similarity_threshold),
            "limit": limit * 2  # Get more for better filtering
        }
        
        if campaign_requirements:
            if campaign_requirements.target_kol_tiers:
                tier_values = [tier.value for tier in campaign_requirements.target_kol_tiers]
                similarity_conditions.append("k.tier = ANY(:target_tiers)")
                query_params["target_tiers"] = tier_values
                
            if campaign_requirements.target_categories:
                category_values = [cat.value for cat in campaign_requirements.target_categories]
                similarity_conditions.append("k.primary_category = ANY(:target_categories)")
                query_params["target_categories"] = category_values
        
        similarity_query = text(f"""
            SELECT k.*, m.*,
                   (k.content_embedding <=> :reference_embedding) as similarity_distance,
                   (1 - (k.content_embedding <=> :reference_embedding)) as similarity_score
            FROM kols k
            JOIN kol_metrics m ON k.id = m.kol_id
            WHERE {' AND '.join(similarity_conditions)}
            ORDER BY similarity_distance ASC
            LIMIT :limit
        """)
        
        result = await self.db_session.execute(similarity_query, query_params)
        similar_rows = result.fetchall()
        
        # AIDEV-NOTE: Convert to full KOLCandidate objects with scoring
        similar_candidates = []
        
        for row in similar_rows:
            try:
                # AIDEV-NOTE: Reconstruct KOL and metrics from row
                kol_data = {k: v for k, v in row._mapping.items() if hasattr(KOL, k)}
                metrics_data = {k: v for k, v in row._mapping.items() if hasattr(KOLMetrics, k) and k != 'id'}
                
                # AIDEV-NOTE: Create candidate with full scoring if campaign provided
                if campaign_requirements:
                    candidate = await self._create_kol_candidate_with_scoring(
                        kol=self._row_to_kol(kol_data),
                        metrics=self._row_to_metrics(metrics_data),
                        campaign_requirements=campaign_requirements,
                        enable_semantic=True
                    )
                    # Set semantic similarity score from query
                    candidate.semantic_similarity_score = Decimal(str(row.similarity_score))
                else:
                    # AIDEV-NOTE: Create minimal candidate for similarity-only search
                    candidate = await self._create_basic_candidate(
                        self._row_to_kol(kol_data),
                        self._row_to_metrics(metrics_data)
                    )
                    candidate.semantic_similarity_score = Decimal(str(row.similarity_score))
                
                similar_candidates.append(candidate)
                
            except Exception as e:
                logger.warning(
                    "Failed to process similar KOL", 
                    error=str(e),
                    row_keys=list(row._mapping.keys())
                )
                continue
        
        # AIDEV-NOTE: Sort by combined similarity and performance if campaign provided
        if campaign_requirements:
            similar_candidates.sort(
                key=lambda x: (
                    float(x.semantic_similarity_score) * 0.6 + 
                    float(x.overall_score) * 0.4
                ),
                reverse=True
            )
        else:
            similar_candidates.sort(
                key=lambda x: float(x.semantic_similarity_score),
                reverse=True
            )
        
        logger.info(
            "Semantic similarity search completed",
            reference_kol_id=reference_kol_id,
            found_count=len(similar_candidates),
            similarity_threshold=float(similarity_threshold)
        )
        
        return similar_candidates[:limit]
    
    async def calculate_sophisticated_kol_score(
        self,
        kol: KOL,
        metrics: KOLMetricsData,
        campaign_requirements: CampaignRequirements,
        enable_semantic: bool = True
    ) -> ScoreComponents:
        """
        Calculate sophisticated multi-factor KOL score with confidence handling.
        
        Args:
            kol: KOL to score
            metrics: KOL performance metrics 
            campaign_requirements: Detailed campaign requirements
            enable_semantic: Include semantic content matching
            
        Returns:
            Comprehensive score components with confidence metrics
        """
        
        # AIDEV-NOTE: Calculate sophisticated ROI score (25% weight)
        roi_score, roi_confidence = await self._calculate_roi_score(
            kol, metrics, campaign_requirements
        )
        
        # AIDEV-NOTE: Calculate audience quality score (25% weight) 
        audience_quality_score, audience_confidence = self._calculate_audience_quality_score(
            metrics
        )
        
        # AIDEV-NOTE: Calculate brand safety score (20% weight)
        brand_safety_score, brand_safety_confidence = await self._calculate_brand_safety_score(
            kol
        )
        
        # AIDEV-NOTE: Calculate content relevance score (15% weight)
        content_relevance_score, content_confidence = await self._calculate_content_relevance_score(
            kol, campaign_requirements, enable_semantic
        )
        
        # AIDEV-NOTE: Calculate demographic fit score (10% weight)
        demographic_fit_score, demographic_confidence = self._calculate_demographic_fit_score(
            kol, campaign_requirements
        )
        
        # AIDEV-NOTE: Calculate reliability score (5% weight)
        reliability_score, reliability_confidence = self._calculate_reliability_score(
            kol, metrics
        )
        
        # AIDEV-NOTE: Calculate data freshness and sample size for confidence
        data_freshness_days = self._calculate_data_freshness(metrics)
        sample_size = self._estimate_sample_size(metrics)
        
        return ScoreComponents(
            roi_score=roi_score,
            audience_quality_score=audience_quality_score,
            brand_safety_score=brand_safety_score,
            content_relevance_score=content_relevance_score,
            demographic_fit_score=demographic_fit_score,
            reliability_score=reliability_score,
            roi_confidence=roi_confidence,
            audience_confidence=audience_confidence,
            brand_safety_confidence=brand_safety_confidence,
            content_relevance_confidence=content_confidence,
            demographic_confidence=demographic_confidence,
            reliability_confidence=reliability_confidence,
            data_freshness_days=data_freshness_days,
            sample_size=sample_size
        )
    
    async def _get_filtered_candidates(
        self,
        campaign_requirements: CampaignRequirements
    ) -> List[Tuple[KOL, KOLMetrics]]:
        """Get KOL candidates that pass hard constraints."""
        
        # AIDEV-NOTE: Build comprehensive filtering query
        query = (
            select(KOL, KOLMetrics)
            .join(KOLMetrics, KOL.id == KOLMetrics.kol_id)
            .where(
                and_(
                    KOL.is_active == True,
                    KOL.is_brand_safe == campaign_requirements.require_brand_safe,
                    KOL.tier.in_([tier.value for tier in campaign_requirements.target_kol_tiers]),
                    or_(*[
                        KOL.primary_category == category.value 
                        for category in campaign_requirements.target_categories
                    ])
                )
            )
        )
        
        # AIDEV-NOTE: Apply follower count constraints
        if campaign_requirements.min_follower_count:
            query = query.where(KOLMetrics.follower_count >= campaign_requirements.min_follower_count)
            
        if campaign_requirements.max_follower_count:
            query = query.where(KOLMetrics.follower_count <= campaign_requirements.max_follower_count)
        
        # AIDEV-NOTE: Apply engagement rate constraints 
        if campaign_requirements.min_engagement_rate:
            query = query.where(KOLMetrics.engagement_rate >= campaign_requirements.min_engagement_rate)
        
        # AIDEV-NOTE: Apply demographic constraints
        if campaign_requirements.target_locations:
            location_conditions = []
            for location in campaign_requirements.target_locations:
                location_conditions.append(KOL.location.ilike(f"%{location}%"))
            query = query.where(or_(*location_conditions))
        
        if campaign_requirements.target_languages:
            query = query.where(KOL.languages.overlap(campaign_requirements.target_languages))
        
        # AIDEV-NOTE: Apply verification requirement
        if campaign_requirements.require_verified:
            query = query.where(KOL.is_verified == True)
        
        # AIDEV-NOTE: Order by most recent metrics first
        query = query.order_by(KOLMetrics.metrics_date.desc())
        
        result = await self.db_session.execute(query)
        return list(result.fetchall())
    
    async def _build_base_query(
        self,
        campaign: Campaign,
        additional_filters: Optional[Dict[str, Any]] = None
    ):
        """Build base query with campaign requirements and additional filters."""
        
        query = select(KOL).where(
            and_(
                KOL.is_active == True,
                KOL.is_brand_safe == True,
                KOL.tier.in_(campaign.target_kol_tiers),
                or_(*[
                    KOL.primary_category == category
                    for category in campaign.target_categories
                ])
            )
        )
        
        # AIDEV-NOTE: Apply follower count filters
        if campaign.min_follower_count:
            query = query.join(KOLMetrics).where(
                KOLMetrics.follower_count >= campaign.min_follower_count
            )
        
        if campaign.max_follower_count:
            query = query.join(KOLMetrics).where(
                KOLMetrics.follower_count <= campaign.max_follower_count
            )
        
        # AIDEV-NOTE: Apply engagement rate filter
        if campaign.min_engagement_rate:
            query = query.join(KOLMetrics).where(
                KOLMetrics.engagement_rate >= campaign.min_engagement_rate
            )
        
        # AIDEV-NOTE: Apply additional filters if provided
        if additional_filters:
            if additional_filters.get("location"):
                query = query.where(KOL.location.ilike(f"%{additional_filters['location']}%"))
            
            if additional_filters.get("languages"):
                query = query.where(KOL.languages.overlap(additional_filters["languages"]))
        
        return query
    
    async def _apply_ai_scoring(
        self,
        base_query,
        campaign: Campaign,
        limit: int
    ) -> List[KOL]:
        """Apply AI-based multi-factor scoring."""
        
        # AIDEV-NOTE: Execute base query to get candidates
        result = await self.db_session.execute(base_query)
        candidate_kols = list(result.scalars().all())
        
        if not candidate_kols:
            return []
        
        # AIDEV-NOTE: Score each KOL for the campaign
        scored_kols = []
        
        # AIDEV-NOTE: Batch process for performance
        for kol in candidate_kols:
            try:
                score = await self.calculate_kol_score_for_campaign(kol, campaign)
                scored_kols.append((kol, score))
            except Exception as e:
                logger.warning("Failed to score KOL", kol_id=kol.id, error=str(e))
                continue
        
        # AIDEV-NOTE: Sort by score descending and return top results
        scored_kols.sort(key=lambda x: x[1], reverse=True)
        return [kol for kol, score in scored_kols[:limit]]
    
    async def _apply_basic_scoring(
        self,
        base_query,
        campaign: Campaign,
        limit: int
    ) -> List[KOL]:
        """Apply basic scoring without AI factors."""
        
        # AIDEV-NOTE: Simple sorting by follower count and engagement rate
        query = base_query.join(KOLMetrics).order_by(
            KOLMetrics.engagement_rate.desc(),
            KOLMetrics.follower_count.desc()
        ).limit(limit)
        
        result = await self.db_session.execute(query)
        return list(result.scalars().all())
    
    def _extract_campaign_criteria(self, campaign: Campaign) -> Dict[str, Any]:
        """Extract campaign criteria for metadata."""
        return {
            "target_tiers": campaign.target_kol_tiers,
            "target_categories": campaign.target_categories,
            "min_followers": campaign.min_follower_count,
            "max_followers": campaign.max_follower_count,
            "min_engagement_rate": float(campaign.min_engagement_rate) if campaign.min_engagement_rate else None,
            "budget": float(campaign.total_budget),
        }
    
    async def _calculate_roi_score(
        self,
        kol: KOL,
        metrics: KOLMetricsData,
        campaign_requirements: CampaignRequirements
    ) -> Tuple[Decimal, Decimal]:
        """Calculate sophisticated ROI score with confidence."""
        
        try:
            # AIDEV-NOTE: Expected engagement calculation
            if not metrics.engagement_rate or not metrics.follower_count:
                return Decimal("0.3"), Decimal("0.3")  # Low confidence fallback
                
            expected_engagement = (
                Decimal(str(metrics.follower_count)) * 
                metrics.engagement_rate * 
                Decimal("0.15")  # Reach rate assumption
            )
            
            # AIDEV-NOTE: Conversion rate estimation based on campaign objective
            conversion_rates = {
                "maximize_reach": Decimal("0.001"),
                "maximize_engagement": Decimal("0.005"), 
                "maximize_conversions": Decimal("0.02"),
                "maximize_roi": Decimal("0.025")
            }
            
            conversion_rate = conversion_rates.get(
                campaign_requirements.campaign_objective.value,
                Decimal("0.01")
            )
            
            # AIDEV-NOTE: Cost estimation based on tier and performance
            estimated_cost = await self._estimate_kol_cost(kol, metrics)
            
            # AIDEV-NOTE: ROI calculation = (expected_engagement × conversion_rate) / cost
            if estimated_cost > 0:
                roi = (expected_engagement * conversion_rate) / estimated_cost
                # Normalize to 0-1 scale (assuming good ROI is 0.1-1.0)
                roi_score = min(roi / Decimal("1.0"), Decimal("1.0"))
            else:
                roi_score = Decimal("0.0")
                
            # AIDEV-NOTE: Confidence based on data availability
            confidence = Decimal("1.0")
            if not metrics.rate_per_post:
                confidence *= Decimal("0.8")  # Reduce confidence if no pricing data
            if not campaign_requirements.expected_conversion_rate:
                confidence *= Decimal("0.9")  # Slight reduction for estimated conversion
                
            return roi_score, confidence
            
        except Exception as e:
            logger.warning("ROI calculation failed", error=str(e), kol_id=kol.id)
            return Decimal("0.0"), Decimal("0.1")  # Very low confidence
    
    def _calculate_audience_quality_score(
        self,
        metrics: KOLMetricsData
    ) -> Tuple[Decimal, Decimal]:
        """Calculate sophisticated audience quality score."""
        
        try:
            base_score = Decimal("0.7")  # Default moderate quality
            confidence = Decimal("0.8")
            
            # AIDEV-NOTE: Use direct quality score if available
            if metrics.audience_quality_score:
                base_score = metrics.audience_quality_score
                confidence = Decimal("1.0")
            else:
                # AIDEV-NOTE: Estimate from available indicators
                
                # Fake follower penalty (heavy impact)
                if metrics.fake_follower_percentage:
                    fake_penalty = metrics.fake_follower_percentage * Decimal("3.0")
                    base_score = max(Decimal("0.0"), base_score - fake_penalty)
                
                # AIDEV-NOTE: Engagement consistency indicator
                if metrics.engagement_rate and metrics.follower_count:
                    # Higher engagement rate suggests better audience quality
                    if metrics.engagement_rate > Decimal("0.05"):
                        base_score += Decimal("0.1")
                    elif metrics.engagement_rate < Decimal("0.01"):
                        base_score -= Decimal("0.2")
                
                # AIDEV-NOTE: Growth rate indicator (organic vs bought followers)
                if metrics.follower_growth_rate:
                    if metrics.follower_growth_rate > Decimal("0.5"):  # Suspicious rapid growth
                        base_score -= Decimal("0.15")
                        confidence *= Decimal("0.8")
                
                confidence *= Decimal("0.7")  # Lower confidence for estimated scores
            
            # AIDEV-NOTE: Follower authenticity checks
            follower_engagement_ratio = None
            if metrics.follower_count and metrics.avg_likes:
                follower_engagement_ratio = metrics.avg_likes / Decimal(str(metrics.follower_count))
                
                # Typical authentic engagement ratios
                if follower_engagement_ratio < Decimal("0.005"):  # Very low engagement
                    base_score -= Decimal("0.2")
                elif follower_engagement_ratio > Decimal("0.1"):  # Suspiciously high
                    base_score -= Decimal("0.1")  # Could be fake engagement
            
            return min(max(base_score, Decimal("0.0")), Decimal("1.0")), confidence
            
        except Exception as e:
            logger.warning("Audience quality calculation failed", error=str(e))
            return Decimal("0.5"), Decimal("0.3")  # Neutral with low confidence
    
    async def _calculate_content_relevance_score(
        self,
        kol: KOL,
        campaign_requirements: CampaignRequirements,
        enable_semantic: bool = True
    ) -> Tuple[Decimal, Decimal]:
        """Calculate sophisticated content relevance score."""
        
        try:
            relevance_score = Decimal("0.2")  # Base score
            confidence = Decimal("0.8")
            
            # AIDEV-NOTE: Primary category exact match (strong signal)
            target_categories = [cat.value for cat in campaign_requirements.target_categories]
            if kol.primary_category.value in target_categories:
                relevance_score += Decimal("0.4")
            
            # AIDEV-NOTE: Secondary category matches
            if hasattr(kol, 'secondary_categories') and kol.secondary_categories:
                secondary_matches = len(set(kol.secondary_categories) & set(target_categories))
                if secondary_matches > 0:
                    relevance_score += Decimal(str(min(0.1 * secondary_matches, 0.2)))
            
            # AIDEV-NOTE: Hashtag overlap analysis
            if campaign_requirements.required_hashtags:
                hashtag_overlap = await self._analyze_hashtag_overlap(
                    kol.id, campaign_requirements.required_hashtags
                )
                relevance_score += hashtag_overlap * Decimal("0.2")
            
            # AIDEV-NOTE: Semantic content similarity (if enabled and available)
            if enable_semantic:
                try:
                    semantic_score = await self._calculate_semantic_relevance(
                        kol, campaign_requirements
                    )
                    if semantic_score:
                        relevance_score += semantic_score * Decimal("0.2")
                        confidence = min(confidence, Decimal("0.9"))  # High confidence with semantic
                except Exception as e:
                    logger.debug("Semantic relevance calculation failed", error=str(e))
                    confidence *= Decimal("0.8")  # Reduce confidence if semantic fails
            
            return min(relevance_score, Decimal("1.0")), confidence
            
        except Exception as e:
            logger.warning("Content relevance calculation failed", error=str(e), kol_id=kol.id)
            return Decimal("0.3"), Decimal("0.4")  # Conservative fallback
    
    async def _calculate_brand_safety_score(
        self,
        kol: KOL
    ) -> Tuple[Decimal, Decimal]:
        """Calculate comprehensive brand safety score."""
        
        try:
            # AIDEV-NOTE: Hard exclusion for unsafe accounts
            if not kol.is_brand_safe:
                return Decimal("0.0"), Decimal("1.0")  # High confidence in exclusion
            
            base_score = Decimal("0.6")  # Base safety score
            confidence = Decimal("0.8")
            
            # AIDEV-NOTE: Verification status (strong trust signal)
            if kol.is_verified:
                base_score += Decimal("0.3")
                confidence = Decimal("1.0")
            
            # AIDEV-NOTE: Content sentiment analysis from recent posts
            sentiment_score = await self._analyze_content_sentiment(kol.id)
            if sentiment_score:
                if sentiment_score > Decimal("0.7"):  # Positive sentiment
                    base_score += Decimal("0.2")
                elif sentiment_score < Decimal("0.3"):  # Negative sentiment
                    base_score -= Decimal("0.3")
            else:
                confidence *= Decimal("0.7")  # Lower confidence without sentiment data
            
            # AIDEV-NOTE: Check for controversy indicators
            controversy_penalty = await self._assess_controversy_risk(kol.id)
            base_score -= controversy_penalty
            
            # AIDEV-NOTE: Safety notes penalty
            if hasattr(kol, 'safety_notes') and kol.safety_notes:
                base_score -= Decimal("0.1")  # Small penalty for having safety notes
                confidence = min(confidence, Decimal("0.9"))
            
            return min(max(base_score, Decimal("0.0")), Decimal("1.0")), confidence
            
        except Exception as e:
            logger.warning("Brand safety calculation failed", error=str(e), kol_id=kol.id)
            return Decimal("0.5"), Decimal("0.5")  # Conservative neutral score
    
    def _calculate_posting_consistency_score(self, metrics: KOLMetrics) -> Decimal:
        """Calculate posting consistency score."""
        if metrics.posts_last_30_days == 0:
            return Decimal("0.0")
        
        # AIDEV-NOTE: Ideal posting frequency is 3-10 posts per month
        ideal_posts = 7
        actual_posts = metrics.posts_last_30_days
        
        if actual_posts >= ideal_posts:
            consistency = min(1.0, ideal_posts / actual_posts)
        else:
            consistency = actual_posts / ideal_posts
        
        return Decimal(str(consistency))
    
    def _calculate_audience_match_score(
        self,
        kol: KOL,
        campaign: Campaign,
        metrics: KOLMetrics
    ) -> Decimal:
        """Calculate audience match score for campaign."""
        # AIDEV-NOTE: Placeholder for demographic matching
        # In full implementation, this would analyze audience demographics
        # against campaign target demographics
        
        base_score = 0.6  # Default moderate match
        
        # AIDEV-NOTE: Location matching
        target_demographics = campaign.target_demographics
        if target_demographics.get("locations") and kol.location:
            if kol.location.lower() in [loc.lower() for loc in target_demographics["locations"]]:
                base_score += 0.2
        
        # AIDEV-NOTE: Language matching
        if target_demographics.get("languages"):
            language_match = len(set(kol.languages) & set(target_demographics["languages"]))
            if language_match > 0:
                base_score += 0.2
        
        return Decimal(str(min(base_score, 1.0)))
    
    def _calculate_cost_efficiency_score(self, kol: KOL, metrics: KOLMetrics) -> Decimal:
        """Calculate cost efficiency score."""
        # AIDEV-NOTE: Placeholder for cost efficiency calculation
        # In full implementation, this would compare typical rates
        # against performance metrics to calculate value
        
        # AIDEV-NOTE: Use tier-based estimation
        tier_efficiency = {
            "nano": 0.9,    # Best value typically
            "micro": 0.8,
            "mid": 0.6,
            "macro": 0.4,
            "mega": 0.3,
        }
        
        return Decimal(str(tier_efficiency.get(kol.tier.value, 0.5)))
    
    # AIDEV-NOTE: === ADVANCED UTILITY METHODS ===
    
    async def _get_or_generate_embedding(self, kol: KOL) -> Optional[List[float]]:
        """Get existing embedding or generate new one."""
        
        try:
            # AIDEV-NOTE: Check cache first
            cache_key = f"embedding_{kol.id}"
            if cache_key in self._embedding_cache:
                return self._embedding_cache[cache_key]
            
            # AIDEV-NOTE: Use existing embedding if available
            if kol.content_embedding:
                self._embedding_cache[cache_key] = kol.content_embedding
                return kol.content_embedding
            
            # AIDEV-NOTE: Generate embedding from KOL profile
            profile_text = self._create_kol_profile_text(kol)
            embedding = await self._generate_text_embedding(profile_text)
            
            if embedding:
                self._embedding_cache[cache_key] = embedding
                # AIDEV-NOTE: Consider storing back to database
                # await self._store_embedding(kol.id, embedding)
            
            return embedding
            
        except Exception as e:
            logger.debug("Embedding generation failed", error=str(e), kol_id=kol.id)
            return None
    
    def _create_kol_profile_text(self, kol: KOL) -> str:
        """Create text representation of KOL profile for embedding."""
        
        parts = []
        
        # AIDEV-NOTE: Basic info
        parts.append(f"Username: {kol.username}")
        parts.append(f"Platform: {kol.platform}")
        parts.append(f"Tier: {kol.tier.value}")
        parts.append(f"Category: {kol.primary_category.value}")
        
        # AIDEV-NOTE: Bio and description
        if kol.bio:
            parts.append(f"Bio: {kol.bio}")
        
        # AIDEV-NOTE: Location and languages
        if kol.location:
            parts.append(f"Location: {kol.location}")
        
        if kol.languages:
            parts.append(f"Languages: {', '.join(kol.languages)}")
        
        # AIDEV-NOTE: Secondary categories
        if hasattr(kol, 'secondary_categories') and kol.secondary_categories:
            parts.append(f"Secondary categories: {', '.join(kol.secondary_categories)}")
        
        return " ".join(parts)
    
    def _create_campaign_context_text(self, campaign_requirements: CampaignRequirements) -> str:
        """Create text representation of campaign for embedding."""
        
        parts = []
        
        # AIDEV-NOTE: Target categories
        categories = [cat.value for cat in campaign_requirements.target_categories]
        parts.append(f"Target categories: {', '.join(categories)}")
        
        # AIDEV-NOTE: Target tiers
        tiers = [tier.value for tier in campaign_requirements.target_kol_tiers]
        parts.append(f"Target tiers: {', '.join(tiers)}")
        
        # AIDEV-NOTE: Objective
        parts.append(f"Objective: {campaign_requirements.campaign_objective.value}")
        
        # AIDEV-NOTE: Demographics
        if campaign_requirements.target_locations:
            parts.append(f"Target locations: {', '.join(campaign_requirements.target_locations)}")
        
        if campaign_requirements.target_languages:
            parts.append(f"Target languages: {', '.join(campaign_requirements.target_languages)}")
        
        # AIDEV-NOTE: Content requirements
        if campaign_requirements.required_hashtags:
            parts.append(f"Required hashtags: {', '.join(campaign_requirements.required_hashtags)}")
        
        return " ".join(parts)
    
    async def _generate_text_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding for text (placeholder for actual implementation)."""
        
        try:
            # AIDEV-NOTE: This is a placeholder - in production, would use:
            # - sentence-transformers library
            # - OpenAI embeddings API
            # - Hugging Face transformers
            
            # Simple hash-based pseudo-embedding for demonstration
            text_hash = hashlib.md5(text.encode()).hexdigest()
            
            # Convert to pseudo-embedding vector
            embedding = []
            for i in range(0, len(text_hash), 2):
                val = int(text_hash[i:i+2], 16) / 255.0  # Normalize to 0-1
                embedding.append(val)
            
            # Pad to 384 dimensions
            while len(embedding) < 384:
                embedding.extend(embedding[:min(20, 384 - len(embedding))])
            
            return embedding[:384]
            
        except Exception as e:
            logger.debug("Text embedding generation failed", error=str(e))
            return None
    
    def _calculate_cosine_similarity(
        self,
        vec1: List[float],
        vec2: List[float]
    ) -> float:
        """Calculate cosine similarity between two vectors."""
        
        try:
            # Convert to numpy arrays for calculation
            a = np.array(vec1)
            b = np.array(vec2)
            
            # Calculate cosine similarity
            dot_product = np.dot(a, b)
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            
            if norm_a == 0 or norm_b == 0:
                return 0.0
            
            return float(dot_product / (norm_a * norm_b))
            
        except Exception as e:
            logger.debug("Cosine similarity calculation failed", error=str(e))
            return 0.0
    
    def _simple_sentiment_analysis(self, text: str) -> Optional[float]:
        """Simple rule-based sentiment analysis."""
        
        if not text:
            return None
        
        # AIDEV-NOTE: Simple keyword-based sentiment
        positive_words = {'great', 'awesome', 'amazing', 'love', 'happy', 'excellent', 'good', 'best'}
        negative_words = {'hate', 'bad', 'terrible', 'awful', 'sad', 'angry', 'worst', 'disappointing'}
        
        words = set(text.lower().split())
        
        positive_count = len(words & positive_words)
        negative_count = len(words & negative_words)
        
        if positive_count == 0 and negative_count == 0:
            return 0.5  # Neutral
        
        # Calculate sentiment score (0 = negative, 1 = positive)
        sentiment = (positive_count - negative_count + max(positive_count + negative_count, 1)) / (2 * max(positive_count + negative_count, 1))
        
        return max(0.0, min(1.0, sentiment))
    
    # AIDEV-NOTE: === DATA CONVERSION HELPERS ===
    
    def _row_to_kol(self, row_data: Dict[str, Any]) -> KOL:
        """Convert database row to KOL object (placeholder)."""
        # AIDEV-NOTE: In production, would properly reconstruct KOL object
        # This is a simplified version
        kol = KOL(
            id=row_data.get('id'),
            username=row_data.get('username', ''),
            display_name=row_data.get('display_name', ''),
            platform=row_data.get('platform', 'tiktok'),
            tier=row_data.get('tier', 'micro'),
            primary_category=row_data.get('primary_category', 'lifestyle')
        )
        return kol
    
    def _row_to_metrics(self, row_data: Dict[str, Any]) -> KOLMetricsData:
        """Convert database row to KOLMetricsData object."""
        return KOLMetricsData(
            follower_count=row_data.get('follower_count', 0),
            following_count=row_data.get('following_count', 0),
            engagement_rate=row_data.get('engagement_rate'),
            posts_last_30_days=row_data.get('posts_last_30_days', 0),
            fake_follower_percentage=row_data.get('fake_follower_percentage'),
            audience_quality_score=row_data.get('audience_quality_score'),
            avg_likes=row_data.get('avg_likes'),
            avg_comments=row_data.get('avg_comments'),
            avg_views=row_data.get('avg_views'),
            rate_per_post=row_data.get('rate_per_post')
        )


    async def _create_kol_candidate_with_scoring(
        self,
        kol: KOL,
        metrics: KOLMetrics,
        campaign_requirements: CampaignRequirements,
        enable_semantic: bool = True
    ) -> KOLCandidate:
        """
        Create KOL candidate with full scoring components.
        
        Args:
            kol: KOL database model
            metrics: KOL metrics data
            campaign_requirements: Campaign requirements
            enable_semantic: Enable semantic content matching
            
        Returns:
            Fully scored KOL candidate
        """
        
        # AIDEV-NOTE: Convert metrics to service format
        metrics_data = KOLMetricsData(
            follower_count=metrics.follower_count,
            following_count=metrics.following_count,
            avg_likes=metrics.avg_likes,
            avg_comments=metrics.avg_comments,
            avg_views=metrics.avg_views,
            engagement_rate=metrics.engagement_rate,
            fake_follower_percentage=metrics.fake_follower_percentage,
            audience_quality_score=metrics.audience_quality_score,
            posts_last_30_days=metrics.posts_last_30_days,
            avg_posting_frequency=metrics.avg_posting_frequency,
            follower_growth_rate=metrics.follower_growth_rate,
            engagement_trend=metrics.engagement_trend
        )
        
        # AIDEV-NOTE: Calculate sophisticated scoring
        score_components = await self.calculate_sophisticated_kol_score(
            kol=kol,
            metrics=metrics_data,
            campaign_requirements=campaign_requirements,
            enable_semantic=enable_semantic
        )
        
        # AIDEV-NOTE: Calculate overall score using weighted components
        overall_score = (
            score_components.roi_score * Decimal(str(self.scoring_weights["roi_score"])) +
            score_components.audience_quality_score * Decimal(str(self.scoring_weights["audience_quality"])) +
            score_components.brand_safety_score * Decimal(str(self.scoring_weights["brand_safety"])) +
            score_components.content_relevance_score * Decimal(str(self.scoring_weights["content_relevance"])) +
            score_components.demographic_fit_score * Decimal(str(self.scoring_weights["demographic_fit"])) +
            score_components.reliability_score * Decimal(str(self.scoring_weights["reliability"]))
        )
        
        # AIDEV-NOTE: Calculate performance predictions
        predicted_reach = await self._predict_reach_for_kol(kol, metrics_data, campaign_requirements)
        predicted_engagement = await self._predict_engagement_for_kol(kol, metrics_data, campaign_requirements)
        predicted_conversions = await self._predict_conversions_for_kol(kol, metrics_data, campaign_requirements)
        
        # AIDEV-NOTE: Estimate costs
        estimated_cost = await self._estimate_kol_cost(kol, metrics_data)
        
        # AIDEV-NOTE: Calculate risk factors
        risk_factors = await self._calculate_risk_factors(kol, metrics_data)
        overall_risk_score = await self._calculate_overall_risk(risk_factors)
        
        return KOLCandidate(
            kol_id=kol.id,
            username=kol.username,
            display_name=kol.display_name,
            platform=kol.platform.value,
            tier=KOLTier(kol.tier.value),
            primary_category=ContentCategory(kol.primary_category.value),
            metrics=metrics_data,
            score_components=score_components,
            overall_score=overall_score,
            predicted_reach=predicted_reach,
            predicted_engagement=predicted_engagement,
            predicted_conversions=predicted_conversions,
            predicted_roi=None,  # Could be calculated if needed
            estimated_cost_per_post=estimated_cost,
            estimated_total_cost=estimated_cost,
            risk_factors=[rf["description"] for rf in risk_factors],
            overall_risk_score=overall_risk_score,
            content_embedding=kol.content_embedding,
            semantic_similarity_score=None  # Set by semantic matching if applicable
        )
    
    async def _create_basic_candidate(
        self,
        kol: KOL,
        metrics: KOLMetrics
    ) -> KOLCandidate:
        """
        Create basic KOL candidate without full scoring (for similarity-only searches).
        
        Args:
            kol: KOL database model
            metrics: KOL metrics data
            
        Returns:
            Basic KOL candidate with minimal scoring
        """
        
        # AIDEV-NOTE: Convert metrics to service format
        metrics_data = KOLMetricsData(
            follower_count=metrics.follower_count,
            following_count=metrics.following_count,
            avg_likes=metrics.avg_likes,
            avg_comments=metrics.avg_comments,
            avg_views=metrics.avg_views,
            engagement_rate=metrics.engagement_rate,
            fake_follower_percentage=metrics.fake_follower_percentage,
            audience_quality_score=metrics.audience_quality_score,
            posts_last_30_days=metrics.posts_last_30_days
        )
        
        # AIDEV-NOTE: Create basic score components
        basic_score_components = ScoreComponents(
            roi_score=Decimal("0.5"),
            audience_quality_score=metrics_data.audience_quality_score or Decimal("0.5"),
            brand_safety_score=Decimal("0.8") if kol.is_brand_safe else Decimal("0.2"),
            content_relevance_score=Decimal("0.5"),
            demographic_fit_score=Decimal("0.5"),
            reliability_score=Decimal("0.7"),
            data_freshness_days=0,
            sample_size=metrics_data.posts_last_30_days
        )
        
        # AIDEV-NOTE: Calculate basic overall score
        overall_score = (
            basic_score_components.audience_quality_score * Decimal("0.4") +
            basic_score_components.brand_safety_score * Decimal("0.3") +
            basic_score_components.reliability_score * Decimal("0.3")
        )
        
        return KOLCandidate(
            kol_id=kol.id,
            username=kol.username,
            display_name=kol.display_name,
            platform=kol.platform.value,
            tier=KOLTier(kol.tier.value),
            primary_category=ContentCategory(kol.primary_category.value),
            metrics=metrics_data,
            score_components=basic_score_components,
            overall_score=overall_score,
            predicted_reach=int(metrics_data.follower_count * 0.1),  # Basic estimate
            predicted_engagement=int(metrics_data.follower_count * float(metrics_data.engagement_rate or 0.03)),
            predicted_conversions=0,
            estimated_cost_per_post=Decimal("1000"),  # Default estimate
            estimated_total_cost=Decimal("1000"),
            risk_factors=[],
            overall_risk_score=Decimal("0.3"),  # Low risk default
            content_embedding=kol.content_embedding
        )

    def _create_empty_metadata(self, campaign_requirements: CampaignRequirements) -> Dict[str, Any]:
        """Create empty metadata for failed matching attempts."""
        return {
            "campaign_id": campaign_requirements.campaign_id,
            "matching_criteria": self._extract_detailed_criteria(campaign_requirements),
            "total_candidates_evaluated": 0,
            "candidates_passed_scoring": 0,
            "final_selected": 0,
            "scoring_method": "enhanced_multi_factor_v2",
            "algorithm_version": "2.1",
            "error_reason": "No candidates passed hard constraints"
        }
    
    def _extract_detailed_criteria(self, campaign_requirements: CampaignRequirements) -> Dict[str, Any]:
        """Extract detailed matching criteria from campaign requirements."""
        return {
            "target_tiers": [tier.value for tier in campaign_requirements.target_kol_tiers],
            "target_categories": [cat.value for cat in campaign_requirements.target_categories],
            "min_followers": campaign_requirements.min_follower_count,
            "max_followers": campaign_requirements.max_follower_count,
            "min_engagement_rate": float(campaign_requirements.min_engagement_rate) if campaign_requirements.min_engagement_rate else None,
            "budget": float(campaign_requirements.total_budget),
            "objective": campaign_requirements.campaign_objective.value,
            "target_locations": campaign_requirements.target_locations,
            "target_languages": campaign_requirements.target_languages,
            "require_brand_safe": campaign_requirements.require_brand_safe,
            "require_verified": campaign_requirements.require_verified
        }
    
    def _generate_data_quality_summary(self, candidates: List[KOLCandidate]) -> Dict[str, Any]:
        """Generate data quality summary for candidates."""
        if not candidates:
            return {"average_confidence": 0.0, "data_completeness": 0.0}
        
        total_confidence = sum(float(candidate.score_components.overall_confidence) for candidate in candidates)
        average_confidence = total_confidence / len(candidates)
        
        # Calculate data completeness based on available metrics
        completeness_scores = []
        for candidate in candidates:
            score = 0
            total_fields = 5
            
            if candidate.metrics.engagement_rate is not None:
                score += 1
            if candidate.metrics.avg_likes is not None:
                score += 1
            if candidate.metrics.audience_quality_score is not None:
                score += 1
            if candidate.metrics.posts_last_30_days > 0:
                score += 1
            if candidate.content_embedding is not None:
                score += 1
            
            completeness_scores.append(score / total_fields)
        
        average_completeness = sum(completeness_scores) / len(completeness_scores)
        
        return {
            "average_confidence": average_confidence,
            "data_completeness": average_completeness,
            "candidates_with_full_data": sum(1 for score in completeness_scores if score >= 0.8),
            "candidates_with_minimal_data": sum(1 for score in completeness_scores if score < 0.5)
        }
    
    def _rank_and_select_candidates(
        self,
        candidates: List[KOLCandidate],
        limit: int,
        objective: str
    ) -> List[KOLCandidate]:
        """Rank candidates and select top performers based on objective."""
        if not candidates:
            return []
        
        # Sort based on objective
        if objective == "maximize_reach":
            candidates.sort(key=lambda x: x.predicted_reach, reverse=True)
        elif objective == "maximize_engagement":
            candidates.sort(key=lambda x: x.predicted_engagement, reverse=True)
        elif objective == "maximize_conversions":
            candidates.sort(key=lambda x: x.predicted_conversions, reverse=True)
        elif objective == "maximize_roi":
            candidates.sort(key=lambda x: float(x.efficiency_ratio), reverse=True)
        else:  # balanced or other
            candidates.sort(key=lambda x: float(x.overall_score), reverse=True)
        
        return candidates[:limit]
    
    async def _predict_reach_for_kol(
        self,
        kol: KOL,
        metrics: KOLMetricsData,
        campaign_requirements: CampaignRequirements
    ) -> int:
        """Predict reach for KOL based on campaign context."""
        base_reach_rate = 0.15  # 15% of followers see typical post
        
        if metrics.engagement_rate:
            engagement_boost = min(2.0, 1.0 + (float(metrics.engagement_rate) * 5))
            base_reach_rate *= engagement_boost
        
        # Tier adjustments
        tier_multipliers = {
            "nano": 0.8, "micro": 1.0, "mid": 1.1, "macro": 1.3, "mega": 1.5
        }
        multiplier = tier_multipliers.get(kol.tier.value, 1.0)
        
        predicted_reach = int(metrics.follower_count * base_reach_rate * multiplier)
        return min(predicted_reach, metrics.follower_count)
    
    async def _predict_engagement_for_kol(
        self,
        kol: KOL,
        metrics: KOLMetricsData,
        campaign_requirements: CampaignRequirements
    ) -> int:
        """Predict engagement for KOL based on campaign context."""
        predicted_reach = await self._predict_reach_for_kol(kol, metrics, campaign_requirements)
        
        if metrics.engagement_rate:
            engagement_rate = float(metrics.engagement_rate)
        else:
            default_rates = {
                "nano": 0.05, "micro": 0.04, "mid": 0.03, "macro": 0.02, "mega": 0.015
            }
            engagement_rate = default_rates.get(kol.tier.value, 0.03)
        
        return int(predicted_reach * engagement_rate)
    
    async def _predict_conversions_for_kol(
        self,
        kol: KOL,
        metrics: KOLMetricsData,
        campaign_requirements: CampaignRequirements
    ) -> int:
        """Predict conversions for KOL based on campaign context."""
        predicted_engagement = await self._predict_engagement_for_kol(kol, metrics, campaign_requirements)
        
        conversion_rates = {
            "maximize_reach": 0.001,
            "maximize_engagement": 0.0,
            "maximize_conversions": 0.02,
            "maximize_roi": 0.025,
            "balanced": 0.01
        }
        
        conversion_rate = conversion_rates.get(
            campaign_requirements.campaign_objective.value,
            0.01
        )
        
        return int(predicted_engagement * conversion_rate)
    
    async def _estimate_kol_cost(
        self,
        kol: KOL,
        metrics: KOLMetricsData
    ) -> Decimal:
        """Estimate KOL cost based on tier and performance."""
        
        # Base costs by tier
        tier_costs = {
            "nano": 500,
            "micro": 2000,
            "mid": 10000,
            "macro": 50000,
            "mega": 200000
        }
        
        base_cost = Decimal(str(tier_costs.get(kol.tier.value, 5000)))
        
        # Adjust based on engagement rate
        if metrics.engagement_rate:
            engagement_multiplier = min(
                Decimal("2.0"),
                Decimal("1.0") + (metrics.engagement_rate * Decimal("10"))
            )
            base_cost *= engagement_multiplier
        
        # Verified accounts premium
        if kol.is_verified:
            base_cost *= Decimal("1.3")
        
        return base_cost
    
    async def _calculate_risk_factors(
        self,
        kol: KOL,
        metrics: KOLMetricsData
    ) -> List[Dict[str, Any]]:
        """Calculate risk factors for KOL."""
        risk_factors = []
        
        if not kol.is_verified:
            risk_factors.append({
                "type": "verification",
                "severity": 0.2,
                "description": "Account not verified"
            })
        
        if metrics.engagement_rate and metrics.engagement_rate < Decimal("0.01"):
            risk_factors.append({
                "type": "engagement",
                "severity": 0.3,
                "description": "Low engagement rate"
            })
        
        if metrics.posts_last_30_days < 3:
            risk_factors.append({
                "type": "activity",
                "severity": 0.2,
                "description": "Low posting frequency"
            })
        
        if (metrics.fake_follower_percentage and 
            metrics.fake_follower_percentage > Decimal("0.2")):
            risk_factors.append({
                "type": "authenticity",
                "severity": 0.4,
                "description": "High fake follower percentage"
            })
        
        return risk_factors
    
    async def _calculate_overall_risk(
        self,
        risk_factors: List[Dict[str, Any]]
    ) -> Decimal:
        """Calculate overall risk score from individual risk factors."""
        if not risk_factors:
            return Decimal("0.1")  # Minimum risk
        
        total_risk = sum(factor["severity"] for factor in risk_factors)
        return Decimal(str(min(total_risk, 1.0)))
    
    def _calculate_demographic_fit_score(
        self,
        kol: KOL,
        campaign_requirements: CampaignRequirements
    ) -> Tuple[Decimal, Decimal]:
        """Calculate demographic fit score with confidence."""
        try:
            from kol_api.utils.ml_models import DemographicMatcher
            
            demographic_matcher = DemographicMatcher()
            
            target_demographics = {
                'locations': campaign_requirements.target_locations,
                'languages': campaign_requirements.target_languages,
                'age_ranges': campaign_requirements.target_age_ranges
            }
            
            fit_score = demographic_matcher.calculate_demographic_fit(
                kol_location=kol.location,
                kol_languages=kol.languages,
                target_demographics=target_demographics
            )
            
            # Confidence based on data availability
            confidence = Decimal("0.8")
            if not kol.location and campaign_requirements.target_locations:
                confidence *= Decimal("0.7")
            if not kol.languages and campaign_requirements.target_languages:
                confidence *= Decimal("0.7")
            
            return fit_score, confidence
            
        except Exception as e:
            logger.warning("Demographic fit calculation failed", error=str(e), kol_id=kol.id)
            return Decimal("0.5"), Decimal("0.3")
    
    def _calculate_reliability_score(
        self,
        kol: KOL,
        metrics: KOLMetricsData
    ) -> Tuple[Decimal, Decimal]:
        """Calculate reliability score based on consistency and history."""
        try:
            base_score = Decimal("0.7")  # Default moderate reliability
            confidence = Decimal("0.8")
            
            # Posting consistency
            if metrics.posts_last_30_days >= 10:
                base_score += Decimal("0.2")  # Regular poster
            elif metrics.posts_last_30_days < 3:
                base_score -= Decimal("0.3")  # Irregular poster
            
            # Account age and verification (proxies for reliability)
            if kol.is_verified:
                base_score += Decimal("0.1")
            
            # Response rate (if available)
            if hasattr(metrics, 'response_rate') and metrics.response_rate:
                if metrics.response_rate > Decimal("0.8"):
                    base_score += Decimal("0.1")
                elif metrics.response_rate < Decimal("0.5"):
                    base_score -= Decimal("0.2")
            else:
                confidence *= Decimal("0.8")  # Lower confidence without response data
            
            return min(max(base_score, Decimal("0.0")), Decimal("1.0")), confidence
            
        except Exception as e:
            logger.warning("Reliability score calculation failed", error=str(e), kol_id=kol.id)
            return Decimal("0.5"), Decimal("0.3")
    
    def _calculate_data_freshness(self, metrics: KOLMetricsData) -> int:
        """Calculate data freshness in days."""
        # AIDEV-NOTE: In production, would calculate from metrics.metrics_date
        # For now, assume recent data
        return 1
    
    def _estimate_sample_size(self, metrics: KOLMetricsData) -> int:
        """Estimate sample size for confidence calculations."""
        return max(metrics.posts_last_30_days, 1)
    
    async def _analyze_hashtag_overlap(
        self,
        kol_id: str,
        required_hashtags: List[str]
    ) -> Decimal:
        """Analyze hashtag overlap between KOL content and required hashtags."""
        # AIDEV-NOTE: In production, would query KOL content and analyze hashtags
        # For now, return moderate overlap
        return Decimal("0.3")
    
    async def _calculate_semantic_relevance(
        self,
        kol: KOL,
        campaign_requirements: CampaignRequirements
    ) -> Optional[Decimal]:
        """Calculate semantic content relevance using embeddings."""
        try:
            if not kol.content_embedding:
                return None
            
            # AIDEV-NOTE: Create campaign context embedding
            campaign_text = self._create_campaign_context_text(campaign_requirements)
            campaign_embedding = await self._generate_text_embedding(campaign_text)
            
            if not campaign_embedding:
                return None
            
            # Calculate similarity
            similarity = self._calculate_cosine_similarity(
                kol.content_embedding,
                campaign_embedding
            )
            
            return Decimal(str(max(0.0, similarity)))
            
        except Exception as e:
            logger.debug("Semantic relevance calculation failed", error=str(e), kol_id=kol.id)
            return None
    
    async def _analyze_content_sentiment(self, kol_id: str) -> Optional[Decimal]:
        """Analyze content sentiment for brand safety assessment."""
        try:
            # AIDEV-NOTE: In production, would query recent KOL content
            # and perform sentiment analysis using ML models
            from kol_api.database.models.kol import KOLContent
            
            # Query recent content
            content_query = (
                select(KOLContent)
                .where(KOLContent.kol_id == kol_id)
                .order_by(desc(KOLContent.posted_at))
                .limit(10)
            )
            
            result = await self.db_session.execute(content_query)
            recent_content = result.scalars().all()
            
            if not recent_content:
                return Decimal("0.5")  # Neutral default
            
            # Simple sentiment analysis on captions
            sentiment_scores = []
            for content in recent_content:
                if content.caption:
                    sentiment = self._simple_sentiment_analysis(content.caption)
                    if sentiment is not None:
                        sentiment_scores.append(sentiment)
            
            if not sentiment_scores:
                return Decimal("0.5")
            
            average_sentiment = sum(sentiment_scores) / len(sentiment_scores)
            return Decimal(str(average_sentiment))
            
        except Exception as e:
            logger.debug("Content sentiment analysis failed", error=str(e), kol_id=kol_id)
            return None
    
    async def _assess_controversy_risk(self, kol_id: str) -> Decimal:
        """Assess controversy risk based on content analysis."""
        try:
            # AIDEV-NOTE: In production, would analyze recent content for controversial topics
            # For now, return low risk
            return Decimal("0.1")
            
        except Exception as e:
            logger.debug("Controversy risk assessment failed", error=str(e), kol_id=kol_id)
            return Decimal("0.2")  # Conservative moderate risk


# AIDEV-NOTE: Legacy compatibility - maintain old class name
KOLMatchingService = EnhancedKOLMatchingService