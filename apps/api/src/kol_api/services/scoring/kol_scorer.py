"""
Multi-Factor KOL Scoring System

AIDEV-NOTE: 250102121500 - Advanced KOL scoring beyond ROI with confidence handling
"""
import asyncio
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession

from ...database.models.kol import KOLProfile, KOLMetrics, KOLContent
from ...database.models.campaign import Campaign, CampaignRequirements
from ...utils.ml_models import SentimentAnalyzer, DemographicMatcher
from ...utils.text_processing import extract_keywords, calculate_similarity

logger = logging.getLogger(__name__)


class ScoreComponent(Enum):
    """Score component types with weights"""
    ROI = ("roi", 0.25)
    AUDIENCE_QUALITY = ("audience_quality", 0.25) 
    BRAND_SAFETY = ("brand_safety", 0.20)
    CONTENT_RELEVANCE = ("content_relevance", 0.15)
    DEMOGRAPHIC_FIT = ("demographic_fit", 0.10)
    RELIABILITY = ("reliability", 0.05)
    
    def __init__(self, component_name: str, weight: float):
        self.component_name = component_name
        self.weight = weight


@dataclass
class ScoreBreakdown:
    """Detailed score breakdown with confidence levels"""
    roi_score: float
    audience_quality_score: float
    brand_safety_score: float
    content_relevance_score: float
    demographic_fit_score: float
    reliability_score: float
    
    # Confidence levels (0-1) for each component
    roi_confidence: float
    audience_quality_confidence: float
    brand_safety_confidence: float
    content_relevance_confidence: float
    demographic_fit_confidence: float
    reliability_confidence: float
    
    # Overall metrics
    composite_score: float
    overall_confidence: float
    missing_data_penalty: float
    
    @property
    def weighted_score(self) -> float:
        """Calculate weighted composite score"""
        components = [
            (self.roi_score, ScoreComponent.ROI.weight),
            (self.audience_quality_score, ScoreComponent.AUDIENCE_QUALITY.weight),
            (self.brand_safety_score, ScoreComponent.BRAND_SAFETY.weight),
            (self.content_relevance_score, ScoreComponent.CONTENT_RELEVANCE.weight),
            (self.demographic_fit_score, ScoreComponent.DEMOGRAPHIC_FIT.weight),
            (self.reliability_score, ScoreComponent.RELIABILITY.weight),
        ]
        
        weighted_sum = sum(score * weight for score, weight in components)
        return weighted_sum * (1 - self.missing_data_penalty)


class KOLScorer:
    """Advanced multi-factor KOL scoring system"""
    
    def __init__(self):
        self.sentiment_analyzer = SentimentAnalyzer()
        self.demographic_matcher = DemographicMatcher()
    
    async def score_kol(
        self,
        kol: KOLProfile,
        campaign: Campaign,
        db: AsyncSession
    ) -> ScoreBreakdown:
        """
        Calculate comprehensive KOL score for campaign fit
        
        Args:
            kol: KOL profile to score
            campaign: Campaign requirements
            db: Database session
            
        Returns:
            Detailed score breakdown with confidence levels
        """
        logger.info(f"Calculating score for KOL {kol.handle} for campaign {campaign.id}")
        
        # Run all scoring components in parallel for performance
        score_tasks = await asyncio.gather(
            self._calculate_roi_score(kol, campaign, db),
            self._calculate_audience_quality_score(kol, db),
            self._calculate_brand_safety_score(kol, db),
            self._calculate_content_relevance_score(kol, campaign, db),
            self._calculate_demographic_fit_score(kol, campaign, db),
            self._calculate_reliability_score(kol, db),
            return_exceptions=True
        )
        
        # Handle any exceptions in scoring components
        for i, task_result in enumerate(score_tasks):
            if isinstance(task_result, Exception):
                logger.error(f"Scoring component {i} failed: {task_result}")
                score_tasks[i] = (0.0, 0.0)  # Default to 0 score, 0 confidence
        
        # Unpack results
        (roi_score, roi_conf) = score_tasks[0]
        (aud_score, aud_conf) = score_tasks[1]
        (safety_score, safety_conf) = score_tasks[2]
        (relevance_score, relevance_conf) = score_tasks[3]
        (demo_score, demo_conf) = score_tasks[4]
        (rel_score, rel_conf) = score_tasks[5]
        
        # Calculate missing data penalty
        missing_penalty = self._calculate_missing_data_penalty(kol)
        
        # Calculate overall confidence
        confidences = [roi_conf, aud_conf, safety_conf, relevance_conf, demo_conf, rel_conf]
        overall_confidence = np.mean(confidences) * (1 - missing_penalty)
        
        breakdown = ScoreBreakdown(
            roi_score=roi_score,
            audience_quality_score=aud_score,
            brand_safety_score=safety_score,
            content_relevance_score=relevance_score,
            demographic_fit_score=demo_score,
            reliability_score=rel_score,
            roi_confidence=roi_conf,
            audience_quality_confidence=aud_conf,
            brand_safety_confidence=safety_conf,
            content_relevance_confidence=relevance_conf,
            demographic_fit_confidence=demo_conf,
            reliability_confidence=rel_conf,
            composite_score=0.0,  # Will be calculated by property
            overall_confidence=overall_confidence,
            missing_data_penalty=missing_penalty
        )
        
        # Set composite score using the property
        breakdown.composite_score = breakdown.weighted_score
        
        logger.info(f"Final score for {kol.handle}: {breakdown.composite_score:.3f} "
                   f"(confidence: {overall_confidence:.3f})")
        
        return breakdown
    
    async def _calculate_roi_score(
        self, 
        kol: KOLProfile, 
        campaign: Campaign, 
        db: AsyncSession
    ) -> Tuple[float, float]:
        """Calculate ROI score: (expected_engagement × conversion_rate) / cost_per_post"""
        try:
            # Get latest metrics
            metrics = kol.latest_metrics
            if not metrics:
                return 0.0, 0.0
            
            # Calculate expected engagement
            expected_engagement = metrics.average_engagement_rate * metrics.followers
            
            # Estimate conversion rate based on KOL tier and campaign type
            conversion_rate = self._estimate_conversion_rate(kol, campaign)
            
            # Get estimated cost per post
            cost_per_post = self._estimate_cost_per_post(kol)
            
            if cost_per_post <= 0:
                return 0.0, 0.5
            
            # Calculate ROI
            expected_value = expected_engagement * conversion_rate
            roi = expected_value / cost_per_post
            
            # Normalize to 0-1 scale (using log transform for better distribution)
            normalized_roi = min(1.0, np.log10(max(1, roi)) / 4.0)  # Cap at 10^4 ROI
            
            # Confidence based on data availability
            confidence = 0.9 if metrics.followers > 1000 else 0.6
            confidence *= 0.8 if not kol.engagement_history else 1.0
            
            return normalized_roi, confidence
            
        except Exception as e:
            logger.error(f"ROI calculation failed for {kol.handle}: {e}")
            return 0.0, 0.0
    
    async def _calculate_audience_quality_score(
        self, 
        kol: KOLProfile, 
        db: AsyncSession
    ) -> Tuple[float, float]:
        """Calculate audience quality: follower authenticity + engagement consistency"""
        try:
            metrics = kol.latest_metrics
            if not metrics:
                return 0.0, 0.0
            
            # Follower authenticity indicators
            authenticity_score = 0.0
            confidence = 0.5
            
            # Check engagement rate vs follower count (detect potential fake followers)
            if metrics.followers > 0:
                engagement_rate = metrics.average_engagement_rate
                
                # Expected engagement rate ranges by tier
                expected_ranges = {
                    "nano": (0.03, 0.08),     # 3-8%
                    "micro": (0.015, 0.05),   # 1.5-5%
                    "mid": (0.01, 0.03),      # 1-3%
                    "macro": (0.005, 0.02),   # 0.5-2%
                }
                
                tier = self._determine_kol_tier(metrics.followers)
                expected_min, expected_max = expected_ranges.get(tier, (0.01, 0.03))
                
                if expected_min <= engagement_rate <= expected_max:
                    authenticity_score += 0.6
                elif engagement_rate > expected_max:
                    authenticity_score += 0.8  # High engagement is good
                else:
                    authenticity_score += 0.2  # Low engagement is concerning
                
                confidence = 0.8
            
            # Engagement consistency (if we have historical data)
            consistency_score = 0.4  # Default moderate score
            if kol.engagement_history:
                # Calculate coefficient of variation for engagement rates
                rates = [h.engagement_rate for h in kol.engagement_history[-10:]]  # Last 10 posts
                if len(rates) > 3:
                    cv = np.std(rates) / max(np.mean(rates), 0.001)
                    # Lower coefficient of variation = more consistent = higher score
                    consistency_score = max(0.0, 1.0 - cv)
                    confidence = 0.9
            
            total_score = (authenticity_score + consistency_score) / 2.0
            return min(1.0, total_score), confidence
            
        except Exception as e:
            logger.error(f"Audience quality calculation failed for {kol.handle}: {e}")
            return 0.0, 0.0
    
    async def _calculate_brand_safety_score(
        self, 
        kol: KOLProfile, 
        db: AsyncSession
    ) -> Tuple[float, float]:
        """Calculate brand safety: content sentiment + controversy risk assessment"""
        try:
            # Get recent content for analysis
            recent_content = kol.recent_content[:10]  # Analyze last 10 posts
            if not recent_content:
                return 0.7, 0.3  # Default moderate score with low confidence
            
            safety_scores = []
            sentiment_scores = []
            
            for content in recent_content:
                # Analyze content sentiment
                sentiment = await self.sentiment_analyzer.analyze(content.caption or "")
                sentiment_scores.append(sentiment.score)
                
                # Check for controversial topics/keywords
                controversy_score = self._check_controversial_content(content.caption or "")
                safety_scores.append(controversy_score)
            
            # Calculate average scores
            avg_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0.5
            avg_safety = np.mean(safety_scores) if safety_scores else 0.8
            
            # Combine scores (positive sentiment + low controversy = high safety)
            combined_score = (avg_sentiment * 0.3 + avg_safety * 0.7)
            
            # Confidence based on content volume analyzed
            confidence = min(0.9, len(recent_content) / 10.0)
            
            return combined_score, confidence
            
        except Exception as e:
            logger.error(f"Brand safety calculation failed for {kol.handle}: {e}")
            return 0.7, 0.3  # Conservative default
    
    def _check_controversial_content(self, text: str) -> float:
        """Check content for controversial topics (returns safety score 0-1)"""
        controversial_keywords = [
            "war", "racist", "sexist", "political", "violence", 
            "drugs", "alcohol", "controversy", "scandal", "hate"
        ]
        
        text_lower = text.lower()
        controversial_count = sum(1 for keyword in controversial_keywords if keyword in text_lower)
        
        # Return safety score (inverse of controversy)
        return max(0.0, 1.0 - (controversial_count * 0.2))
    
    async def _calculate_content_relevance_score(
        self, 
        kol: KOLProfile, 
        campaign: Campaign, 
        db: AsyncSession
    ) -> Tuple[float, float]:
        """Calculate content relevance: category alignment + hashtag overlap"""
        try:
            # Category alignment
            category_score = 0.0
            if campaign.requirements and campaign.requirements.required_categories:
                kol_categories = set(kol.categories or [])
                required_categories = set(campaign.requirements.required_categories)
                
                if required_categories:
                    overlap = len(kol_categories.intersection(required_categories))
                    category_score = overlap / len(required_categories)
            
            # Hashtag and keyword relevance
            hashtag_score = 0.0
            confidence = 0.5
            
            if campaign.requirements and campaign.requirements.target_keywords:
                # Extract hashtags and keywords from recent content
                kol_keywords = set()
                for content in kol.recent_content[:10]:
                    if content.hashtags:
                        kol_keywords.update(content.hashtags)
                    if content.caption:
                        kol_keywords.update(extract_keywords(content.caption))
                
                target_keywords = set(campaign.requirements.target_keywords)
                
                if kol_keywords and target_keywords:
                    # Calculate semantic similarity
                    hashtag_score = calculate_similarity(kol_keywords, target_keywords)
                    confidence = 0.8
            
            # Combine scores
            total_score = (category_score * 0.6 + hashtag_score * 0.4)
            
            return min(1.0, total_score), confidence
            
        except Exception as e:
            logger.error(f"Content relevance calculation failed for {kol.handle}: {e}")
            return 0.0, 0.3
    
    async def _calculate_demographic_fit_score(
        self, 
        kol: KOLProfile, 
        campaign: Campaign, 
        db: AsyncSession
    ) -> Tuple[float, float]:
        """Calculate demographic fit: age/location/interest alignment"""
        try:
            if not campaign.requirements:
                return 0.5, 0.3
            
            fit_scores = []
            
            # Location alignment
            if campaign.requirements.target_locations:
                if kol.location:
                    target_locations = set(campaign.requirements.target_locations)
                    kol_location = {kol.location}
                    location_overlap = len(kol_location.intersection(target_locations))
                    fit_scores.append(1.0 if location_overlap > 0 else 0.0)
                else:
                    fit_scores.append(0.3)  # Unknown location gets low score
            
            # Age alignment (if available)
            if campaign.requirements.target_age_range and kol.demographics:
                target_min, target_max = campaign.requirements.target_age_range
                kol_age = kol.demographics.get('average_age')
                
                if kol_age:
                    if target_min <= kol_age <= target_max:
                        fit_scores.append(1.0)
                    else:
                        # Partial score based on distance from range
                        distance = min(abs(kol_age - target_min), abs(kol_age - target_max))
                        fit_scores.append(max(0.0, 1.0 - distance / 20.0))  # 20 year tolerance
                else:
                    fit_scores.append(0.4)
            
            # Interest alignment using ML model
            if campaign.requirements.target_interests and kol.demographics:
                interest_score = await self.demographic_matcher.match_interests(
                    kol.demographics.get('interests', []),
                    campaign.requirements.target_interests
                )
                fit_scores.append(interest_score)
            
            if not fit_scores:
                return 0.5, 0.2
            
            avg_score = np.mean(fit_scores)
            confidence = min(0.9, len(fit_scores) / 3.0)  # More factors = higher confidence
            
            return avg_score, confidence
            
        except Exception as e:
            logger.error(f"Demographic fit calculation failed for {kol.handle}: {e}")
            return 0.5, 0.2
    
    async def _calculate_reliability_score(
        self, 
        kol: KOLProfile, 
        db: AsyncSession
    ) -> Tuple[float, float]:
        """Calculate reliability: historical performance + response rate"""
        try:
            reliability_factors = []
            
            # Account age and activity (older, active accounts are more reliable)
            if kol.account_created_at:
                from datetime import datetime, timezone
                account_age_days = (datetime.now(timezone.utc) - kol.account_created_at).days
                age_score = min(1.0, account_age_days / 365.0)  # 1 year = max score
                reliability_factors.append(age_score)
            
            # Posting consistency
            if kol.recent_content:
                # Check posting frequency consistency
                post_dates = [c.created_at for c in kol.recent_content if c.created_at]
                if len(post_dates) >= 3:
                    # Calculate posting interval consistency
                    intervals = []
                    for i in range(1, len(post_dates)):
                        interval = (post_dates[i-1] - post_dates[i]).days
                        intervals.append(abs(interval))
                    
                    # Lower variance in intervals = higher consistency
                    if intervals:
                        consistency = 1.0 - (np.std(intervals) / max(np.mean(intervals), 1))
                        reliability_factors.append(max(0.0, consistency))
            
            # Historical campaign performance (if available)
            # This would typically come from past campaign data
            # For now, use engagement consistency as a proxy
            if kol.engagement_history:
                engagement_trend = self._calculate_engagement_trend(kol.engagement_history)
                reliability_factors.append(max(0.3, engagement_trend))  # Minimum 0.3 for having data
            
            if not reliability_factors:
                return 0.5, 0.2  # Default score with low confidence
            
            avg_score = np.mean(reliability_factors)
            confidence = min(0.8, len(reliability_factors) / 3.0)
            
            return avg_score, confidence
            
        except Exception as e:
            logger.error(f"Reliability calculation failed for {kol.handle}: {e}")
            return 0.5, 0.2
    
    def _calculate_engagement_trend(self, engagement_history: List) -> float:
        """Calculate engagement trend (positive trend = higher reliability)"""
        if len(engagement_history) < 3:
            return 0.5
        
        # Get recent engagement rates
        recent_rates = [h.engagement_rate for h in engagement_history[-10:]]
        
        # Calculate trend using linear regression
        x = np.arange(len(recent_rates))
        y = np.array(recent_rates)
        
        if len(recent_rates) > 1:
            slope = np.corrcoef(x, y)[0, 1] if np.std(y) > 0 else 0
            # Normalize slope to 0-1 scale
            trend_score = 0.5 + (slope * 0.5)  # Center at 0.5
            return max(0.0, min(1.0, trend_score))
        
        return 0.5
    
    def _calculate_missing_data_penalty(self, kol: KOLProfile) -> float:
        """Calculate penalty for missing critical data"""
        total_fields = 10
        missing_fields = 0
        
        # Check critical fields
        if not kol.location:
            missing_fields += 1
        if not kol.categories:
            missing_fields += 1
        if not kol.latest_metrics:
            missing_fields += 2  # Metrics are very important
        if not kol.recent_content:
            missing_fields += 2  # Content is very important
        if not kol.demographics:
            missing_fields += 1
        if not kol.engagement_history:
            missing_fields += 1
        if not kol.account_created_at:
            missing_fields += 1
        if not getattr(kol.latest_metrics, 'followers', None):
            missing_fields += 1
        
        # Return penalty as percentage (0-0.5 max penalty)
        return min(0.5, missing_fields / total_fields)
    
    def _estimate_conversion_rate(self, kol: KOLProfile, campaign: Campaign) -> float:
        """Estimate conversion rate based on KOL tier and campaign type"""
        if not kol.latest_metrics:
            return 0.001  # Very low default
        
        # Base conversion rates by tier
        tier = self._determine_kol_tier(kol.latest_metrics.followers)
        base_rates = {
            "nano": 0.03,    # 3%
            "micro": 0.02,   # 2%
            "mid": 0.015,    # 1.5%
            "macro": 0.01    # 1%
        }
        
        base_rate = base_rates.get(tier, 0.015)
        
        # Adjust based on campaign type (if specified)
        if campaign.campaign_type == "awareness":
            return base_rate * 0.5
        elif campaign.campaign_type == "conversion":
            return base_rate * 1.5
        
        return base_rate
    
    def _estimate_cost_per_post(self, kol: KOLProfile) -> float:
        """Estimate cost per post based on followers and engagement"""
        if not kol.latest_metrics or not kol.latest_metrics.followers:
            return 1000.0  # Default cost
        
        followers = kol.latest_metrics.followers
        engagement_rate = kol.latest_metrics.average_engagement_rate or 0.02
        
        # Base cost calculation: followers * rate * engagement multiplier
        base_cost = followers * 0.01  # $0.01 per follower base rate
        engagement_multiplier = 1 + (engagement_rate * 10)  # Higher engagement = higher cost
        
        estimated_cost = base_cost * engagement_multiplier
        
        # Apply tier-based minimums and caps
        tier = self._determine_kol_tier(followers)
        tier_ranges = {
            "nano": (50, 500),
            "micro": (500, 2000),
            "mid": (2000, 10000),
            "macro": (10000, 100000)
        }
        
        min_cost, max_cost = tier_ranges.get(tier, (500, 5000))
        return max(min_cost, min(max_cost, estimated_cost))
    
    def _determine_kol_tier(self, followers: int) -> str:
        """Determine KOL tier based on follower count"""
        if followers < 1000:
            return "nano"
        elif followers < 10000:
            return "micro"
        elif followers < 100000:
            return "mid"
        else:
            return "macro"