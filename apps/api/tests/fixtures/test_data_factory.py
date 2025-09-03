"""
Comprehensive Test Data Factory - Mock Data Generators for POC2 and POC4 Testing

AIDEV-NOTE: Production-ready test data generators that create realistic KOL profiles,
campaign scenarios, and optimization test cases for comprehensive algorithm testing.
"""
import random
import string
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import List, Dict, Any, Optional, Tuple
from unittest.mock import MagicMock

import numpy as np

from kol_api.services.models import (
    KOLCandidate,
    KOLMetricsData, 
    ScoreComponents,
    CampaignRequirements,
    OptimizationConstraints,
    OptimizationObjective,
    KOLTier,
    ContentCategory
)
from kol_api.database.models.kol import KOLProfile, KOLMetrics, KOLContent
from kol_api.database.models.campaign import Campaign


class KOLDataFactory:
    """Factory for generating diverse KOL test data."""
    
    # AIDEV-NOTE: Realistic data distributions based on industry benchmarks
    TIER_FOLLOWER_RANGES = {
        "nano": (100, 10000),
        "micro": (10000, 100000),
        "mid": (100000, 1000000),
        "macro": (1000000, 10000000),
        "mega": (10000000, 50000000)
    }
    
    TIER_ENGAGEMENT_RANGES = {
        "nano": (0.04, 0.12),     # 4-12% (highest engagement)
        "micro": (0.02, 0.08),    # 2-8%
        "mid": (0.015, 0.05),     # 1.5-5%
        "macro": (0.01, 0.03),    # 1-3%
        "mega": (0.005, 0.02)     # 0.5-2% (lowest engagement)
    }
    
    PLATFORMS = ["tiktok", "instagram", "youtube", "twitter", "facebook"]
    
    CATEGORIES = [
        ContentCategory.LIFESTYLE, ContentCategory.FASHION, ContentCategory.BEAUTY,
        ContentCategory.FITNESS, ContentCategory.FOOD, ContentCategory.TRAVEL,
        ContentCategory.TECH, ContentCategory.GAMING, ContentCategory.EDUCATION,
        ContentCategory.ENTERTAINMENT, ContentCategory.BUSINESS, ContentCategory.HEALTH
    ]
    
    LOCATIONS = [
        "Bangkok, Thailand", "Singapore", "Manila, Philippines", "Kuala Lumpur, Malaysia",
        "Jakarta, Indonesia", "Ho Chi Minh City, Vietnam", "Phnom Penh, Cambodia",
        "Yangon, Myanmar", "Vientiane, Laos", "Bandar Seri Begawan, Brunei"
    ]
    
    LANGUAGES = [
        ["th", "en"], ["en"], ["en", "zh"], ["en", "ms"], ["id", "en"],
        ["vi", "en"], ["km", "en"], ["my", "en"], ["lo", "en"], ["ms", "en"]
    ]
    
    @classmethod
    def create_kol_profile(
        cls,
        tier: Optional[str] = None,
        category: Optional[ContentCategory] = None,
        quality_level: str = "medium",  # low, medium, high
        data_completeness: str = "complete",  # minimal, partial, complete
        **kwargs
    ) -> Tuple[MagicMock, MagicMock]:
        """
        Create a realistic KOL profile with controllable quality and completeness.
        
        Args:
            tier: KOL tier (nano, micro, mid, macro, mega)
            category: Primary content category
            quality_level: Overall quality (low, medium, high)
            data_completeness: Data completeness level
            **kwargs: Override specific fields
            
        Returns:
            Tuple of (KOL profile mock, KOL metrics mock)
        """
        
        # Determine tier
        if not tier:
            tier = random.choice(list(cls.TIER_FOLLOWER_RANGES.keys()))
        
        # Determine category
        if not category:
            category = random.choice(cls.CATEGORIES)
        
        # Generate KOL profile
        kol_id = kwargs.get("kol_id", f"{tier}_{category.value}_{cls._generate_id()}")
        username = kwargs.get("username", f"@{tier}_{cls._generate_username()}")
        display_name = kwargs.get("display_name", cls._generate_display_name(category))
        
        kol = MagicMock(spec=KOLProfile)
        kol.id = kol_id
        kol.username = username
        kol.display_name = display_name
        kol.platform = MagicMock()
        kol.platform.value = kwargs.get("platform", random.choice(cls.PLATFORMS))
        
        # Tier setup
        kol.tier = MagicMock()
        kol.tier.value = tier
        
        # Category setup
        kol.primary_category = MagicMock()
        kol.primary_category.value = category.value
        
        # Account status based on quality level
        if quality_level == "high":
            kol.is_verified = True
            kol.is_brand_safe = True
            kol.is_active = True
        elif quality_level == "medium":
            kol.is_verified = random.choice([True, False])
            kol.is_brand_safe = True
            kol.is_active = True
        else:  # low quality
            kol.is_verified = False
            kol.is_brand_safe = random.choice([True, False])
            kol.is_active = random.choice([True, False])
        
        # Data completeness handling
        if data_completeness == "complete":
            kol.location = kwargs.get("location", random.choice(cls.LOCATIONS))
            kol.languages = kwargs.get("languages", random.choice(cls.LANGUAGES))
            kol.bio = kwargs.get("bio", cls._generate_bio(category, quality_level))
            kol.categories = kwargs.get("categories", [category.value, cls._get_related_category(category)])
            kol.demographics = kwargs.get("demographics", cls._generate_demographics())
            kol.account_created_at = kwargs.get("account_created_at", 
                                                cls._generate_account_creation_date(quality_level))
            kol.content_embedding = kwargs.get("content_embedding", cls._generate_content_embedding())
        elif data_completeness == "partial":
            kol.location = kwargs.get("location", random.choice([None, random.choice(cls.LOCATIONS)]))
            kol.languages = kwargs.get("languages", random.choice([None, random.choice(cls.LANGUAGES)]))
            kol.bio = kwargs.get("bio", random.choice([None, cls._generate_bio(category, quality_level)]))
            kol.categories = kwargs.get("categories", random.choice([[], [category.value]]))
            kol.demographics = kwargs.get("demographics", None)
            kol.account_created_at = kwargs.get("account_created_at", 
                                                random.choice([None, cls._generate_account_creation_date(quality_level)]))
            kol.content_embedding = kwargs.get("content_embedding", None)
        else:  # minimal
            kol.location = None
            kol.languages = None
            kol.bio = None
            kol.categories = []
            kol.demographics = None
            kol.account_created_at = None
            kol.content_embedding = None
        
        # Generate metrics
        metrics = cls._generate_kol_metrics(
            kol_id, tier, quality_level, data_completeness, **kwargs
        )
        
        # Generate content
        if data_completeness != "minimal":
            kol.recent_content = cls._generate_recent_content(
                category, quality_level, int(metrics.posts_last_30_days) if metrics.posts_last_30_days else 0
            )
            if data_completeness == "complete":
                kol.engagement_history = cls._generate_engagement_history(
                    float(metrics.engagement_rate) if metrics.engagement_rate else 0.03, quality_level
                )
            else:
                kol.engagement_history = random.choice([None, []])
        else:
            kol.recent_content = []
            kol.engagement_history = None
        
        return kol, metrics
    
    @classmethod
    def _generate_kol_metrics(
        cls, 
        kol_id: str, 
        tier: str, 
        quality_level: str, 
        data_completeness: str,
        **kwargs
    ) -> MagicMock:
        """Generate realistic KOL metrics based on tier and quality."""
        
        metrics = MagicMock(spec=KOLMetrics)
        metrics.kol_id = kol_id
        
        # Follower count based on tier
        follower_range = cls.TIER_FOLLOWER_RANGES[tier]
        if quality_level == "high":
            # Upper portion of tier range
            metrics.follower_count = kwargs.get("follower_count", 
                                                random.randint(int(follower_range[1] * 0.7), follower_range[1]))
        elif quality_level == "medium":
            # Middle portion of tier range
            metrics.follower_count = kwargs.get("follower_count",
                                                random.randint(int(follower_range[1] * 0.3), int(follower_range[1] * 0.7)))
        else:  # low quality
            # Lower portion of tier range
            metrics.follower_count = kwargs.get("follower_count",
                                                random.randint(follower_range[0], int(follower_range[1] * 0.3)))
        
        # Following count (realistic ratio)
        following_ratio = random.uniform(0.01, 0.1) if quality_level == "high" else random.uniform(0.1, 0.5)
        metrics.following_count = kwargs.get("following_count", 
                                            int(metrics.follower_count * following_ratio))
        
        # Engagement rate based on tier and quality
        if data_completeness != "minimal" and random.random() > 0.2:  # 80% have engagement rate
            engagement_range = cls.TIER_ENGAGEMENT_RANGES[tier]
            if quality_level == "high":
                base_engagement = random.uniform(engagement_range[1] * 0.8, engagement_range[1])
            elif quality_level == "medium":
                base_engagement = random.uniform(
                    (engagement_range[0] + engagement_range[1]) / 2, 
                    engagement_range[1] * 0.8
                )
            else:  # low quality
                base_engagement = random.uniform(engagement_range[0], engagement_range[0] * 1.5)
            
            metrics.engagement_rate = kwargs.get("engagement_rate", Decimal(str(round(base_engagement, 4))))
        else:
            metrics.engagement_rate = None
        
        # Derive other metrics from engagement rate and followers
        if metrics.engagement_rate:
            total_engagement = int(metrics.follower_count * float(metrics.engagement_rate))
            
            # Distribute engagement across likes, comments, shares
            likes_ratio = random.uniform(0.7, 0.9)
            comments_ratio = random.uniform(0.05, 0.15)
            shares_ratio = 1.0 - likes_ratio - comments_ratio
            
            metrics.avg_likes = Decimal(str(int(total_engagement * likes_ratio)))
            metrics.avg_comments = Decimal(str(int(total_engagement * comments_ratio)))
            metrics.avg_shares = Decimal(str(int(total_engagement * shares_ratio)))
            metrics.avg_views = Decimal(str(int(metrics.follower_count * random.uniform(0.3, 0.8))))
        else:
            metrics.avg_likes = None
            metrics.avg_comments = None
            metrics.avg_shares = None
            metrics.avg_views = None
        
        # Activity level
        if quality_level == "high":
            posts_range = (15, 30)
        elif quality_level == "medium":
            posts_range = (8, 20)
        else:  # low quality
            posts_range = (0, 10)
        
        metrics.posts_last_30_days = kwargs.get("posts_last_30_days", 
                                                random.randint(posts_range[0], posts_range[1]))
        
        # Quality indicators
        if data_completeness != "minimal" and random.random() > 0.3:  # 70% have quality data
            if quality_level == "high":
                metrics.fake_follower_percentage = Decimal(str(random.uniform(0.01, 0.08)))
                metrics.audience_quality_score = Decimal(str(random.uniform(0.8, 0.95)))
                metrics.campaign_success_rate = Decimal(str(random.uniform(0.85, 0.98)))
                metrics.response_rate = Decimal(str(random.uniform(0.9, 1.0)))
            elif quality_level == "medium":
                metrics.fake_follower_percentage = Decimal(str(random.uniform(0.05, 0.15)))
                metrics.audience_quality_score = Decimal(str(random.uniform(0.65, 0.85)))
                metrics.campaign_success_rate = Decimal(str(random.uniform(0.7, 0.9)))
                metrics.response_rate = Decimal(str(random.uniform(0.75, 0.95)))
            else:  # low quality
                metrics.fake_follower_percentage = Decimal(str(random.uniform(0.15, 0.5)))
                metrics.audience_quality_score = Decimal(str(random.uniform(0.3, 0.7)))
                metrics.campaign_success_rate = Decimal(str(random.uniform(0.4, 0.75)))
                metrics.response_rate = Decimal(str(random.uniform(0.5, 0.8)))
        else:
            metrics.fake_follower_percentage = None
            metrics.audience_quality_score = None
            metrics.campaign_success_rate = None
            metrics.response_rate = None
        
        # Posting frequency
        if metrics.posts_last_30_days > 0:
            metrics.avg_posting_frequency = Decimal(str(round(metrics.posts_last_30_days / 30, 2)))
        else:
            metrics.avg_posting_frequency = None
        
        # Growth metrics
        if quality_level == "high":
            metrics.follower_growth_rate = Decimal(str(random.uniform(0.05, 0.2)))  # 5-20% monthly
            metrics.engagement_trend = random.choice(["increasing", "stable"])
        elif quality_level == "medium":
            metrics.follower_growth_rate = Decimal(str(random.uniform(0.0, 0.1)))  # 0-10% monthly
            metrics.engagement_trend = random.choice(["stable", "slightly_decreasing"])
        else:  # low quality
            metrics.follower_growth_rate = Decimal(str(random.uniform(-0.05, 0.05)))  # -5% to +5%
            metrics.engagement_trend = random.choice(["decreasing", "volatile"])
        
        # Cost data
        base_cost = cls._estimate_base_cost_by_tier(tier)
        
        # Adjust cost based on quality and engagement
        cost_multiplier = 1.0
        if quality_level == "high":
            cost_multiplier *= random.uniform(1.3, 2.0)
        elif quality_level == "low":
            cost_multiplier *= random.uniform(0.5, 0.8)
        
        if metrics.engagement_rate and float(metrics.engagement_rate) > cls.TIER_ENGAGEMENT_RANGES[tier][1] * 0.8:
            cost_multiplier *= random.uniform(1.2, 1.5)  # Premium for high engagement
        
        metrics.rate_per_post = Decimal(str(int(base_cost * cost_multiplier)))
        metrics.rate_per_video = Decimal(str(int(base_cost * cost_multiplier * 1.5)))
        metrics.min_budget = Decimal(str(int(base_cost * cost_multiplier * 0.8)))
        
        # Timestamps
        metrics.metrics_date = kwargs.get("metrics_date", 
                                         datetime.now(timezone.utc) - timedelta(days=random.randint(0, 30)))
        metrics.created_at = datetime.now(timezone.utc)
        
        return metrics
    
    @staticmethod
    def _estimate_base_cost_by_tier(tier: str) -> int:
        """Estimate base cost per post by tier."""
        base_costs = {
            "nano": 500,
            "micro": 2500,
            "mid": 12500,
            "macro": 62500,
            "mega": 250000
        }
        return base_costs.get(tier, 2500)
    
    @staticmethod
    def _generate_id() -> str:
        """Generate random ID."""
        return ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
    
    @staticmethod
    def _generate_username() -> str:
        """Generate realistic username."""
        adjectives = ["creative", "lifestyle", "trendy", "daily", "authentic", "modern", "stylish"]
        nouns = ["creator", "life", "style", "vibes", "world", "journey", "stories"]
        
        return f"{random.choice(adjectives)}_{random.choice(nouns)}_{random.randint(1, 999)}"
    
    @staticmethod
    def _generate_display_name(category: ContentCategory) -> str:
        """Generate display name based on category."""
        category_prefixes = {
            ContentCategory.LIFESTYLE: ["Lifestyle", "Daily", "Modern", "Urban"],
            ContentCategory.FASHION: ["Fashion", "Style", "Trendy", "Chic"],
            ContentCategory.BEAUTY: ["Beauty", "Glam", "Makeup", "Skincare"],
            ContentCategory.FITNESS: ["Fit", "Health", "Active", "Strong"],
            ContentCategory.FOOD: ["Foodie", "Culinary", "Tasty", "Chef"],
            ContentCategory.TRAVEL: ["Travel", "Adventure", "Explorer", "Wanderer"],
            ContentCategory.TECH: ["Tech", "Digital", "Innovation", "Gadget"],
            ContentCategory.GAMING: ["Gaming", "Player", "Gamer", "Stream"]
        }
        
        prefixes = category_prefixes.get(category, ["Content", "Creative"])
        suffixes = ["Creator", "Influencer", "Expert", "Enthusiast", "Guide"]
        
        return f"{random.choice(prefixes)} {random.choice(suffixes)}"
    
    @staticmethod
    def _generate_bio(category: ContentCategory, quality_level: str) -> str:
        """Generate realistic bio based on category and quality."""
        category_phrases = {
            ContentCategory.LIFESTYLE: [
                "Sharing daily inspiration and lifestyle tips",
                "Living authentically and inspiring others",
                "Lifestyle content creator and influencer"
            ],
            ContentCategory.FASHION: [
                "Fashion enthusiast sharing style inspiration",
                "Outfit ideas and fashion trends",
                "Style blogger and fashion influencer"
            ],
            ContentCategory.BEAUTY: [
                "Beauty tips, tutorials, and product reviews",
                "Makeup artist and beauty enthusiast",
                "Skincare and beauty content creator"
            ]
        }
        
        phrases = category_phrases.get(category, ["Content creator and influencer"])
        
        if quality_level == "high":
            return f"{random.choice(phrases)} | Brand partnerships | {random.choice(['📧 contact@email.com', '💌 dm for collabs'])}"
        elif quality_level == "medium":
            return f"{random.choice(phrases)} | {random.choice(['✨ inspiring daily', '🌟 authentic content'])}"
        else:
            return random.choice(phrases)
    
    @staticmethod
    def _get_related_category(category: ContentCategory) -> str:
        """Get related category for diversity."""
        related_map = {
            ContentCategory.LIFESTYLE: "travel",
            ContentCategory.FASHION: "beauty",
            ContentCategory.BEAUTY: "fashion",
            ContentCategory.FITNESS: "health",
            ContentCategory.FOOD: "lifestyle",
            ContentCategory.TRAVEL: "lifestyle",
            ContentCategory.TECH: "gaming",
            ContentCategory.GAMING: "tech"
        }
        return related_map.get(category, "lifestyle")
    
    @staticmethod
    def _generate_demographics() -> Dict[str, Any]:
        """Generate realistic demographic data."""
        return {
            "average_age": random.randint(22, 35),
            "gender_split": {
                "female": round(random.uniform(0.3, 0.8), 2),
                "male": round(random.uniform(0.2, 0.7), 2)
            },
            "top_countries": random.sample(["TH", "SG", "MY", "ID", "PH", "VN"], 3),
            "interests": random.sample([
                "fashion", "beauty", "lifestyle", "travel", "food", "fitness", 
                "technology", "entertainment", "music", "art"
            ], random.randint(3, 6))
        }
    
    @staticmethod
    def _generate_account_creation_date(quality_level: str) -> datetime:
        """Generate account creation date based on quality."""
        if quality_level == "high":
            # Established accounts (1-5 years old)
            days_ago = random.randint(365, 1825)
        elif quality_level == "medium":
            # Moderate age (6 months - 3 years)
            days_ago = random.randint(180, 1095)
        else:  # low quality
            # Newer or very old accounts
            days_ago = random.choice([
                random.randint(1, 90),     # Very new
                random.randint(2190, 2920) # Very old but inactive
            ])
        
        return datetime.now(timezone.utc) - timedelta(days=days_ago)
    
    @staticmethod
    def _generate_content_embedding() -> List[float]:
        """Generate realistic content embedding vector."""
        # Create embedding with some structure (not completely random)
        embedding = []
        
        # Add some clustered values to simulate semantic meaning
        for _ in range(384):  # Standard embedding size
            if random.random() < 0.3:
                # Some values clustered around positive
                embedding.append(random.uniform(0.1, 0.8))
            elif random.random() < 0.3:
                # Some values clustered around negative  
                embedding.append(random.uniform(-0.8, -0.1))
            else:
                # Rest distributed normally
                embedding.append(random.uniform(-0.5, 0.5))
        
        # Normalize to unit vector
        norm = np.linalg.norm(embedding)
        return [x / norm for x in embedding]
    
    @classmethod
    def _generate_recent_content(
        cls,
        category: ContentCategory,
        quality_level: str,
        num_posts: int
    ) -> List[MagicMock]:
        """Generate recent content posts."""
        if num_posts == 0:
            return []
        
        content_posts = []
        
        category_templates = {
            ContentCategory.LIFESTYLE: [
                "Daily {activity} routine! {hashtags}",
                "Morning inspiration to start your day right ✨ {hashtags}",
                "Sharing my {item} favorites this week {hashtags}"
            ],
            ContentCategory.FASHION: [
                "Today's outfit inspiration! {hashtags}",
                "New {season} collection haul {hashtags}",
                "Styling tips for {occasion} {hashtags}"
            ],
            ContentCategory.BEAUTY: [
                "Step by step {makeup} tutorial {hashtags}",
                "My skincare routine for {skin_type} {hashtags}",
                "Product review: {product} {hashtags}"
            ]
        }
        
        hashtag_sets = {
            ContentCategory.LIFESTYLE: ["#lifestyle", "#daily", "#inspiration", "#authentic"],
            ContentCategory.FASHION: ["#fashion", "#ootd", "#style", "#outfit"],
            ContentCategory.BEAUTY: ["#beauty", "#makeup", "#skincare", "#tutorial"]
        }
        
        templates = category_templates.get(category, ["Daily content! {hashtags}"])
        hashtags = hashtag_sets.get(category, ["#content", "#creator"])
        
        for i in range(min(num_posts, 10)):  # Limit for test performance
            content = MagicMock(spec=KOLContent)
            
            # Generate caption
            template = random.choice(templates)
            selected_hashtags = random.sample(hashtags, random.randint(2, 4))
            
            content.caption = template.format(
                activity=random.choice(["morning", "workout", "skincare"]),
                hashtags=" ".join(selected_hashtags),
                item=random.choice(["fashion", "beauty", "lifestyle"]),
                season=random.choice(["spring", "summer", "fall", "winter"]),
                occasion=random.choice(["work", "weekend", "date night"]),
                makeup=random.choice(["natural", "glam", "everyday"]),
                skin_type=random.choice(["oily", "dry", "combination"]),
                product=random.choice(["foundation", "moisturizer", "lipstick"])
            )
            
            content.hashtags = [tag.replace("#", "") for tag in selected_hashtags]
            content.created_at = datetime.now(timezone.utc) - timedelta(days=i * 2 + random.randint(0, 2))
            
            # Add quality-based sentiment
            if quality_level == "low":
                if random.random() < 0.3:  # 30% chance of negative content
                    content.caption = "Having a terrible day... " + content.caption
            
            content_posts.append(content)
        
        return content_posts
    
    @staticmethod
    def _generate_engagement_history(
        base_engagement_rate: float,
        quality_level: str,
        history_length: int = 30
    ) -> List[MagicMock]:
        """Generate engagement history with realistic variance."""
        history = []
        
        # Quality affects variance
        if quality_level == "high":
            variance = 0.1  # Low variance (consistent)
        elif quality_level == "medium":
            variance = 0.2  # Medium variance
        else:  # low quality
            variance = 0.4  # High variance (inconsistent)
        
        for i in range(history_length):
            entry = MagicMock()
            
            # Add some trend and seasonality
            trend_factor = 1.0 + (i * 0.001)  # Slight upward trend
            seasonal_factor = 1.0 + 0.1 * np.sin(i * 0.2)  # Weekly seasonality
            noise = random.uniform(-variance, variance)
            
            entry.engagement_rate = max(0.001, 
                                       base_engagement_rate * trend_factor * seasonal_factor * (1 + noise))
            entry.date = datetime.now(timezone.utc) - timedelta(days=history_length - i)
            
            history.append(entry)
        
        return history


class CampaignDataFactory:
    """Factory for generating diverse campaign test data."""
    
    @classmethod
    def create_campaign_requirements(
        cls,
        campaign_type: str = "engagement",  # engagement, awareness, conversion
        budget_size: str = "medium",        # small, medium, large
        complexity: str = "medium",         # simple, medium, complex
        **kwargs
    ) -> CampaignRequirements:
        """Create campaign requirements with controllable complexity."""
        
        # Budget ranges
        budget_ranges = {
            "small": (5000, 25000),
            "medium": (25000, 100000),
            "large": (100000, 500000)
        }
        
        # Objective mapping
        objective_map = {
            "engagement": OptimizationObjective.MAXIMIZE_ENGAGEMENT,
            "awareness": OptimizationObjective.MAXIMIZE_REACH,
            "conversion": OptimizationObjective.MAXIMIZE_CONVERSIONS,
            "roi": OptimizationObjective.MAXIMIZE_ROI,
            "cost": OptimizationObjective.MINIMIZE_COST
        }
        
        # Generate base requirements
        campaign_id = kwargs.get("campaign_id", f"test_campaign_{cls._generate_id()}")
        total_budget = kwargs.get("total_budget", 
                                 Decimal(str(random.randint(*budget_ranges[budget_size]))))
        
        # Tier selection based on budget and complexity
        if complexity == "simple":
            target_tiers = [random.choice(list(KOLTier))]
        elif complexity == "medium":
            target_tiers = random.sample(list(KOLTier), 2)
        else:  # complex
            target_tiers = random.sample(list(KOLTier), random.randint(3, 4))
        
        # Category selection
        if complexity == "simple":
            target_categories = [random.choice(KOLDataFactory.CATEGORIES)]
        elif complexity == "medium":
            target_categories = random.sample(KOLDataFactory.CATEGORIES, 2)
        else:  # complex
            target_categories = random.sample(KOLDataFactory.CATEGORIES, random.randint(3, 5))
        
        # Follower constraints based on budget
        if budget_size == "small":
            min_followers, max_followers = 1000, 50000
        elif budget_size == "medium":
            min_followers, max_followers = 10000, 500000
        else:  # large
            min_followers, max_followers = 50000, 5000000
        
        # Create requirements
        requirements = CampaignRequirements(
            campaign_id=campaign_id,
            target_kol_tiers=kwargs.get("target_kol_tiers", target_tiers),
            target_categories=kwargs.get("target_categories", target_categories),
            total_budget=total_budget,
            min_follower_count=kwargs.get("min_follower_count", min_followers),
            max_follower_count=kwargs.get("max_follower_count", max_followers),
            min_engagement_rate=kwargs.get("min_engagement_rate", Decimal("0.015")),
            target_locations=kwargs.get("target_locations", 
                                       random.sample(KOLDataFactory.LOCATIONS, random.randint(1, 3))),
            target_languages=kwargs.get("target_languages", 
                                       random.choice(KOLDataFactory.LANGUAGES)),
            campaign_objective=kwargs.get("campaign_objective", 
                                         objective_map.get(campaign_type, OptimizationObjective.MAXIMIZE_ENGAGEMENT))
        )
        
        # Add complexity-based constraints
        if complexity == "complex":
            requirements.require_verified = kwargs.get("require_verified", True)
            requirements.required_hashtags = kwargs.get("required_hashtags", 
                                                       ["campaign", "brand", "sponsored"])
            requirements.excluded_hashtags = kwargs.get("excluded_hashtags", 
                                                       ["competitor", "negative"])
            requirements.expected_conversion_rate = kwargs.get("expected_conversion_rate", 
                                                              Decimal("0.02"))
        
        return requirements
    
    @classmethod
    def create_optimization_constraints(
        cls,
        strictness: str = "medium",  # loose, medium, strict
        risk_tolerance: str = "medium",  # low, medium, high
        **kwargs
    ) -> OptimizationConstraints:
        """Create optimization constraints with controllable strictness."""
        
        # Base constraints
        max_budget = kwargs.get("max_budget", Decimal("50000"))
        
        if strictness == "loose":
            min_kols = kwargs.get("min_kols", 1)
            max_kols = kwargs.get("max_kols", 20)
            min_budget_utilization = kwargs.get("min_budget_utilization", Decimal("0.6"))
        elif strictness == "medium":
            min_kols = kwargs.get("min_kols", 3)
            max_kols = kwargs.get("max_kols", 12)
            min_budget_utilization = kwargs.get("min_budget_utilization", Decimal("0.8"))
        else:  # strict
            min_kols = kwargs.get("min_kols", 5)
            max_kols = kwargs.get("max_kols", 8)
            min_budget_utilization = kwargs.get("min_budget_utilization", Decimal("0.9"))
        
        # Risk constraints
        if risk_tolerance == "low":
            max_risk_per_kol = kwargs.get("max_risk_per_kol", Decimal("0.3"))
            max_portfolio_risk = kwargs.get("max_portfolio_risk", Decimal("0.2"))
        elif risk_tolerance == "medium":
            max_risk_per_kol = kwargs.get("max_risk_per_kol", Decimal("0.6"))
            max_portfolio_risk = kwargs.get("max_portfolio_risk", Decimal("0.4"))
        else:  # high risk tolerance
            max_risk_per_kol = kwargs.get("max_risk_per_kol", Decimal("0.8"))
            max_portfolio_risk = kwargs.get("max_portfolio_risk", Decimal("0.6"))
        
        return OptimizationConstraints(
            max_budget=max_budget,
            min_budget_utilization=min_budget_utilization,
            min_kols=min_kols,
            max_kols=max_kols,
            max_risk_per_kol=max_risk_per_kol,
            max_portfolio_risk=max_portfolio_risk,
            **kwargs
        )
    
    @staticmethod
    def _generate_id() -> str:
        """Generate random ID."""
        return ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))


class TestScenarioFactory:
    """Factory for generating comprehensive test scenarios."""
    
    @classmethod
    def create_mixed_quality_kol_pool(cls, size: int = 20) -> List[Tuple]:
        """Create diverse pool of KOLs with mixed quality levels."""
        kol_pool = []
        
        # Distribution: 20% high, 60% medium, 20% low quality
        high_count = max(1, int(size * 0.2))
        low_count = max(1, int(size * 0.2))
        medium_count = size - high_count - low_count
        
        # High quality KOLs
        for _ in range(high_count):
            kol, metrics = KOLDataFactory.create_kol_profile(
                quality_level="high",
                data_completeness="complete"
            )
            kol_pool.append((kol, metrics))
        
        # Medium quality KOLs
        for _ in range(medium_count):
            kol, metrics = KOLDataFactory.create_kol_profile(
                quality_level="medium",
                data_completeness=random.choice(["partial", "complete"])
            )
            kol_pool.append((kol, metrics))
        
        # Low quality KOLs
        for _ in range(low_count):
            kol, metrics = KOLDataFactory.create_kol_profile(
                quality_level="low",
                data_completeness=random.choice(["minimal", "partial"])
            )
            kol_pool.append((kol, metrics))
        
        return kol_pool
    
    @classmethod
    def create_tier_specific_kol_pool(cls, tier: str, size: int = 10) -> List[Tuple]:
        """Create KOL pool with specific tier focus."""
        kol_pool = []
        
        for _ in range(size):
            quality = random.choice(["low", "medium", "high"])
            completeness = random.choice(["partial", "complete"])
            
            kol, metrics = KOLDataFactory.create_kol_profile(
                tier=tier,
                quality_level=quality,
                data_completeness=completeness
            )
            kol_pool.append((kol, metrics))
        
        return kol_pool
    
    @classmethod
    def create_missing_data_scenarios(cls) -> Dict[str, List[Tuple]]:
        """Create specific missing data test scenarios."""
        scenarios = {}
        
        # Scenario 1: Missing engagement rates
        scenarios["missing_engagement"] = []
        for i in range(5):
            kol, metrics = KOLDataFactory.create_kol_profile(
                data_completeness="partial",
                engagement_rate=None
            )
            scenarios["missing_engagement"].append((kol, metrics))
        
        # Scenario 2: No recent content
        scenarios["no_content"] = []
        for i in range(5):
            kol, metrics = KOLDataFactory.create_kol_profile(
                data_completeness="complete",
                posts_last_30_days=0
            )
            kol.recent_content = []
            scenarios["no_content"].append((kol, metrics))
        
        # Scenario 3: Inconsistent data
        scenarios["inconsistent"] = []
        for i in range(5):
            kol, metrics = KOLDataFactory.create_kol_profile(
                quality_level="low"
            )
            # Make data inconsistent
            metrics.follower_count = 1000000  # High followers
            metrics.engagement_rate = Decimal("0.001")  # Very low engagement
            metrics.fake_follower_percentage = Decimal("0.8")  # High fake percentage
            scenarios["inconsistent"].append((kol, metrics))
        
        return scenarios
    
    @classmethod
    def create_budget_optimization_scenarios(cls) -> Dict[str, Dict[str, Any]]:
        """Create various budget optimization test scenarios."""
        scenarios = {}
        
        # Scenario 1: Small budget, high requirements
        scenarios["constrained_budget"] = {
            "campaign_requirements": CampaignDataFactory.create_campaign_requirements(
                budget_size="small",
                complexity="complex"
            ),
            "constraints": CampaignDataFactory.create_optimization_constraints(
                strictness="strict",
                max_budget=Decimal("15000")
            ),
            "kol_pool": cls.create_mixed_quality_kol_pool(15)
        }
        
        # Scenario 2: Large budget, flexible requirements
        scenarios["abundant_budget"] = {
            "campaign_requirements": CampaignDataFactory.create_campaign_requirements(
                budget_size="large",
                complexity="simple"
            ),
            "constraints": CampaignDataFactory.create_optimization_constraints(
                strictness="loose",
                max_budget=Decimal("200000")
            ),
            "kol_pool": cls.create_mixed_quality_kol_pool(25)
        }
        
        # Scenario 3: Balanced scenario
        scenarios["balanced"] = {
            "campaign_requirements": CampaignDataFactory.create_campaign_requirements(
                budget_size="medium",
                complexity="medium"
            ),
            "constraints": CampaignDataFactory.create_optimization_constraints(
                strictness="medium",
                max_budget=Decimal("75000")
            ),
            "kol_pool": cls.create_mixed_quality_kol_pool(20)
        }
        
        return scenarios


# AIDEV-NOTE: Utility functions for test data validation

def validate_kol_data_realism(kol_data: Tuple) -> Dict[str, bool]:
    """Validate that generated KOL data is realistic."""
    kol, metrics = kol_data
    
    validation_results = {}
    
    # Check follower/following ratio
    if metrics.following_count and metrics.follower_count:
        ratio = metrics.following_count / metrics.follower_count
        validation_results["realistic_follow_ratio"] = 0.001 <= ratio <= 1.0
    
    # Check engagement rate reasonableness
    if metrics.engagement_rate:
        validation_results["realistic_engagement"] = 0.001 <= float(metrics.engagement_rate) <= 0.2
    
    # Check posting frequency
    validation_results["realistic_posting"] = 0 <= metrics.posts_last_30_days <= 60
    
    # Check cost reasonableness
    if hasattr(metrics, 'rate_per_post') and metrics.rate_per_post:
        validation_results["realistic_cost"] = 50 <= float(metrics.rate_per_post) <= 1000000
    
    return validation_results


def generate_performance_test_data(kol_count: int = 1000) -> List[Tuple]:
    """Generate large dataset for performance testing."""
    kol_pool = []
    
    # Generate in batches for memory efficiency
    batch_size = 100
    for batch_start in range(0, kol_count, batch_size):
        batch_end = min(batch_start + batch_size, kol_count)
        batch_size_actual = batch_end - batch_start
        
        batch_pool = TestScenarioFactory.create_mixed_quality_kol_pool(batch_size_actual)
        kol_pool.extend(batch_pool)
    
    return kol_pool


if __name__ == "__main__":
    # Example usage and testing
    print("Testing KOL Data Factory...")
    
    # Generate sample KOL
    kol, metrics = KOLDataFactory.create_kol_profile(
        tier="micro",
        category=ContentCategory.LIFESTYLE,
        quality_level="high",
        data_completeness="complete"
    )
    
    print(f"Generated KOL: {kol.display_name}")
    print(f"Tier: {kol.tier.value}")
    print(f"Followers: {metrics.follower_count}")
    print(f"Engagement Rate: {metrics.engagement_rate}")
    
    # Validate realism
    validation = validate_kol_data_realism((kol, metrics))
    print(f"Validation Results: {validation}")
    
    # Generate campaign requirements
    campaign = CampaignDataFactory.create_campaign_requirements(
        campaign_type="engagement",
        budget_size="medium",
        complexity="complex"
    )
    
    print(f"Generated Campaign: {campaign.campaign_id}")
    print(f"Budget: {campaign.total_budget}")
    print(f"Target Tiers: {[t.value for t in campaign.target_kol_tiers]}")
    
    print("Test data factory validation completed successfully!")