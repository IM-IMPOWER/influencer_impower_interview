"""ML models and stubs for KOL services."""

import hashlib
from typing import Optional, List, Dict, Any
from decimal import Decimal
import structlog

logger = structlog.get_logger()


class SentimentAnalyzer:
    """Simple sentiment analysis for content and KOL profiles."""
    
    def __init__(self):
        """Initialize sentiment analyzer."""
        # AIDEV-NOTE: In production, would load actual ML model
        self._positive_words = {
            'amazing', 'awesome', 'beautiful', 'best', 'brilliant', 'excellent',
            'fantastic', 'good', 'great', 'happy', 'incredible', 'love', 'perfect',
            'wonderful', 'outstanding', 'superb', 'magnificent', 'delightful'
        }
        self._negative_words = {
            'awful', 'bad', 'boring', 'disappointing', 'hate', 'horrible',
            'sad', 'terrible', 'ugly', 'worst', 'annoying', 'frustrating',
            'disgusting', 'pathetic', 'useless', 'worthless', 'stupid'
        }
    
    def analyze_sentiment(self, text: str) -> Optional[Decimal]:
        """
        Analyze sentiment of text content.
        
        Args:
            text: Text content to analyze
            
        Returns:
            Sentiment score from 0.0 (negative) to 1.0 (positive), or None if no text
        """
        if not text:
            return None
            
        # AIDEV-NOTE: Simple keyword-based analysis (production would use transformer model)
        words = set(text.lower().split())
        positive_count = len(words & self._positive_words)
        negative_count = len(words & self._negative_words)
        
        if positive_count == 0 and negative_count == 0:
            return Decimal("0.5")  # Neutral
        
        total_sentiment_words = positive_count + negative_count
        if total_sentiment_words == 0:
            return Decimal("0.5")
        
        # Calculate sentiment score (0 = negative, 1 = positive)
        sentiment_score = positive_count / total_sentiment_words
        return Decimal(str(sentiment_score))
    
    def batch_analyze(self, texts: List[str]) -> List[Optional[Decimal]]:
        """Batch analyze multiple texts for efficiency."""
        return [self.analyze_sentiment(text) for text in texts]


class DemographicMatcher:
    """Demographic matching for KOL audience analysis."""
    
    def __init__(self):
        """Initialize demographic matcher."""
        # AIDEV-NOTE: In production, would load demographic models and data
        self._location_groups = {
            'urban': ['bangkok', 'phuket', 'chiang mai', 'pattaya'],
            'rural': ['isaan', 'northern', 'southern'],
            'international': ['singapore', 'malaysia', 'vietnam', 'philippines']
        }
        
    def calculate_demographic_fit(
        self,
        kol_location: Optional[str],
        kol_languages: List[str],
        target_demographics: Dict[str, Any]
    ) -> Decimal:
        """
        Calculate demographic fit score for KOL against target demographics.
        
        Args:
            kol_location: KOL's location
            kol_languages: KOL's languages
            target_demographics: Target demographic requirements
            
        Returns:
            Fit score from 0.0 to 1.0
        """
        fit_score = Decimal("0.5")  # Base neutral score
        
        # AIDEV-NOTE: Location matching
        target_locations = target_demographics.get('locations', [])
        if target_locations and kol_location:
            location_lower = kol_location.lower()
            
            # Exact location match
            if any(loc.lower() in location_lower for loc in target_locations):
                fit_score += Decimal("0.3")
            
            # Regional group match
            for group_name, locations in self._location_groups.items():
                if (group_name in target_locations and 
                    any(loc in location_lower for loc in locations)):
                    fit_score += Decimal("0.2")
                    break
        
        # AIDEV-NOTE: Language matching
        target_languages = target_demographics.get('languages', [])
        if target_languages and kol_languages:
            language_overlap = set(lang.lower() for lang in kol_languages) & \
                             set(lang.lower() for lang in target_languages)
            if language_overlap:
                # Bonus for each matching language
                language_bonus = min(len(language_overlap) * Decimal("0.1"), Decimal("0.2"))
                fit_score += language_bonus
        
        # AIDEV-NOTE: Age range matching (simplified)
        target_age_ranges = target_demographics.get('age_ranges', [])
        if target_age_ranges:
            # In production, would use actual audience age data
            fit_score += Decimal("0.1")  # Assume reasonable age overlap
        
        return min(fit_score, Decimal("1.0"))
    
    def estimate_audience_overlap(
        self,
        kol_demographics: Dict[str, Any],
        target_demographics: Dict[str, Any]
    ) -> Decimal:
        """
        Estimate audience demographic overlap percentage.
        
        Args:
            kol_demographics: KOL's audience demographics 
            target_demographics: Target audience demographics
            
        Returns:
            Estimated overlap percentage (0.0 to 1.0)
        """
        # AIDEV-NOTE: Placeholder for audience overlap calculation
        # In production, would use actual audience demographic data
        base_overlap = Decimal("0.4")  # Default moderate overlap
        
        # Adjust based on location match
        kol_locations = kol_demographics.get('top_locations', [])
        target_locations = target_demographics.get('locations', [])
        
        if kol_locations and target_locations:
            location_matches = sum(1 for loc in kol_locations 
                                 if any(target.lower() in loc.lower() 
                                       for target in target_locations))
            location_factor = min(location_matches * Decimal("0.1"), Decimal("0.3"))
            base_overlap += location_factor
        
        return min(base_overlap, Decimal("1.0"))


class ContentRelevanceAnalyzer:
    """Analyze content relevance for KOL matching."""
    
    def __init__(self):
        """Initialize content relevance analyzer."""
        # AIDEV-NOTE: Category keyword mappings for relevance scoring
        self._category_keywords = {
            'lifestyle': ['life', 'daily', 'routine', 'living', 'home', 'personal'],
            'fashion': ['fashion', 'style', 'outfit', 'clothing', 'trend', 'wear'],
            'beauty': ['beauty', 'makeup', 'skincare', 'cosmetics', 'hair', 'nail'],
            'fitness': ['fitness', 'workout', 'exercise', 'gym', 'health', 'training'],
            'food': ['food', 'recipe', 'cooking', 'restaurant', 'meal', 'cuisine'],
            'travel': ['travel', 'trip', 'vacation', 'destination', 'journey', 'explore'],
            'tech': ['technology', 'tech', 'gadget', 'digital', 'software', 'app'],
            'gaming': ['gaming', 'game', 'esports', 'player', 'console', 'stream']
        }
    
    def calculate_content_relevance(
        self,
        kol_content: List[str],
        target_categories: List[str],
        kol_hashtags: List[str] = None
    ) -> Decimal:
        """
        Calculate content relevance score for KOL against target categories.
        
        Args:
            kol_content: List of KOL content texts
            target_categories: Target content categories
            kol_hashtags: KOL's commonly used hashtags
            
        Returns:
            Relevance score from 0.0 to 1.0
        """
        if not kol_content and not kol_hashtags:
            return Decimal("0.3")  # Neutral score with no data
        
        relevance_score = Decimal("0.0")
        
        # AIDEV-NOTE: Analyze content text
        all_text = " ".join(kol_content).lower() if kol_content else ""
        
        for category in target_categories:
            category_keywords = self._category_keywords.get(category.lower(), [])
            if not category_keywords:
                continue
                
            # Count keyword matches in content
            keyword_matches = sum(1 for keyword in category_keywords 
                                if keyword in all_text)
            
            if keyword_matches > 0:
                category_score = min(keyword_matches * Decimal("0.1"), Decimal("0.4"))
                relevance_score += category_score
        
        # AIDEV-NOTE: Analyze hashtags if available
        if kol_hashtags:
            hashtag_text = " ".join(kol_hashtags).lower()
            for category in target_categories:
                category_keywords = self._category_keywords.get(category.lower(), [])
                hashtag_matches = sum(1 for keyword in category_keywords 
                                    if keyword in hashtag_text)
                if hashtag_matches > 0:
                    hashtag_score = min(hashtag_matches * Decimal("0.05"), Decimal("0.2"))
                    relevance_score += hashtag_score
        
        return min(relevance_score, Decimal("1.0"))


def generate_content_embedding(text: str, model_name: str = "default") -> Optional[List[float]]:
    """
    Generate content embedding vector for semantic similarity.
    
    Args:
        text: Text content to embed
        model_name: Embedding model to use
        
    Returns:
        384-dimensional embedding vector or None if error
    """
    if not text:
        return None
        
    try:
        # AIDEV-NOTE: Placeholder implementation using hash-based pseudo-embedding
        # In production, would use sentence-transformers or OpenAI embeddings
        text_hash = hashlib.md5(text.encode()).hexdigest()
        
        # Convert hash to pseudo-embedding vector
        embedding = []
        for i in range(0, len(text_hash), 2):
            val = int(text_hash[i:i+2], 16) / 255.0  # Normalize to 0-1
            embedding.append(val)
        
        # Pad/truncate to 384 dimensions (sentence transformer standard)
        target_dim = 384
        while len(embedding) < target_dim:
            embedding.extend(embedding[:min(20, target_dim - len(embedding))])
        
        return embedding[:target_dim]
        
    except Exception as e:
        logger.warning("Content embedding generation failed", error=str(e))
        return None


def calculate_cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Cosine similarity score (-1 to 1)
    """
    try:
        import numpy as np
        
        a = np.array(vec1)
        b = np.array(vec2)
        
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
            
        return float(dot_product / (norm_a * norm_b))
        
    except Exception as e:
        logger.warning("Cosine similarity calculation failed", error=str(e))
        return 0.0