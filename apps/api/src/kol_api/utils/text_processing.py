"""Text processing utilities for KOL platform."""

import re
import hashlib
from typing import List, Dict, Set, Tuple, Optional, Any
from decimal import Decimal
from collections import Counter
import structlog

logger = structlog.get_logger()


def extract_keywords(text: str, min_length: int = 3, max_keywords: int = 20) -> List[str]:
    """
    Extract keywords from text content.
    
    Args:
        text: Input text to process
        min_length: Minimum keyword length
        max_keywords: Maximum number of keywords to return
        
    Returns:
        List of extracted keywords
    """
    if not text:
        return []
    
    # AIDEV-NOTE: Clean and normalize text
    text = text.lower().strip()
    
    # Remove special characters and extra whitespace
    text = re.sub(r'[^\w\s#@]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    
    # Extract words (excluding hashtags and mentions for now)
    words = [word for word in text.split() 
             if len(word) >= min_length and not word.startswith(('#', '@'))]
    
    # AIDEV-NOTE: Remove common stop words
    stop_words = {
        'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
        'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before',
        'after', 'above', 'below', 'between', 'among', 'is', 'are', 'was',
        'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does',
        'did', 'will', 'would', 'should', 'could', 'can', 'may', 'might',
        'must', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she',
        'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your',
        'his', 'her', 'its', 'our', 'their', 'a', 'an'
    }
    
    filtered_words = [word for word in words if word not in stop_words]
    
    # Get most common words
    word_counts = Counter(filtered_words)
    keywords = [word for word, count in word_counts.most_common(max_keywords)]
    
    return keywords


def extract_hashtags(text: str) -> List[str]:
    """
    Extract hashtags from text content.
    
    Args:
        text: Input text to process
        
    Returns:
        List of hashtags (without # symbol)
    """
    if not text:
        return []
    
    # Find all hashtags
    hashtag_pattern = r'#(\w+)'
    hashtags = re.findall(hashtag_pattern, text, re.IGNORECASE)
    
    # Clean and normalize
    hashtags = [tag.lower().strip() for tag in hashtags if len(tag) > 1]
    
    # Remove duplicates while preserving order
    seen = set()
    unique_hashtags = []
    for tag in hashtags:
        if tag not in seen:
            seen.add(tag)
            unique_hashtags.append(tag)
    
    return unique_hashtags


def extract_mentions(text: str) -> List[str]:
    """
    Extract user mentions from text content.
    
    Args:
        text: Input text to process
        
    Returns:
        List of mentioned usernames (without @ symbol)
    """
    if not text:
        return []
    
    # Find all mentions
    mention_pattern = r'@(\w+)'
    mentions = re.findall(mention_pattern, text, re.IGNORECASE)
    
    # Clean and normalize
    mentions = [mention.lower().strip() for mention in mentions if len(mention) > 1]
    
    # Remove duplicates while preserving order
    seen = set()
    unique_mentions = []
    for mention in mentions:
        if mention not in seen:
            seen.add(mention)
            unique_mentions.append(mention)
    
    return unique_mentions


def calculate_text_similarity(text1: str, text2: str) -> Decimal:
    """
    Calculate similarity between two texts using simple word overlap.
    
    Args:
        text1: First text to compare
        text2: Second text to compare
        
    Returns:
        Similarity score from 0.0 to 1.0
    """
    if not text1 or not text2:
        return Decimal("0.0")
    
    # Extract keywords from both texts
    keywords1 = set(extract_keywords(text1, min_length=2, max_keywords=50))
    keywords2 = set(extract_keywords(text2, min_length=2, max_keywords=50))
    
    if not keywords1 and not keywords2:
        return Decimal("0.0")
    
    if not keywords1 or not keywords2:
        return Decimal("0.0")
    
    # Calculate Jaccard similarity
    intersection = keywords1 & keywords2
    union = keywords1 | keywords2
    
    if len(union) == 0:
        return Decimal("0.0")
    
    jaccard_sim = len(intersection) / len(union)
    return Decimal(str(jaccard_sim))


def calculate_hashtag_overlap(hashtags1: List[str], hashtags2: List[str]) -> Decimal:
    """
    Calculate overlap between two sets of hashtags.
    
    Args:
        hashtags1: First set of hashtags
        hashtags2: Second set of hashtags
        
    Returns:
        Overlap ratio from 0.0 to 1.0
    """
    if not hashtags1 or not hashtags2:
        return Decimal("0.0")
    
    set1 = set(tag.lower() for tag in hashtags1)
    set2 = set(tag.lower() for tag in hashtags2)
    
    intersection = set1 & set2
    union = set1 | set2
    
    if len(union) == 0:
        return Decimal("0.0")
    
    return Decimal(str(len(intersection) / len(union)))


def normalize_text(text: str) -> str:
    """
    Normalize text for consistent processing.
    
    Args:
        text: Input text to normalize
        
    Returns:
        Normalized text
    """
    if not text:
        return ""
    
    # Convert to lowercase
    text = text.lower().strip()
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Clean up remaining text
    text = re.sub(r'[^\w\s#@]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def extract_content_themes(content_list: List[str], max_themes: int = 10) -> List[Tuple[str, int]]:
    """
    Extract common themes from a list of content.
    
    Args:
        content_list: List of content texts
        max_themes: Maximum number of themes to return
        
    Returns:
        List of (theme, frequency) tuples
    """
    if not content_list:
        return []
    
    # Extract all keywords from all content
    all_keywords = []
    for content in content_list:
        keywords = extract_keywords(content, min_length=3, max_keywords=30)
        all_keywords.extend(keywords)
    
    if not all_keywords:
        return []
    
    # Find most common themes
    keyword_counts = Counter(all_keywords)
    themes = keyword_counts.most_common(max_themes)
    
    return themes


def calculate_content_diversity(content_list: List[str]) -> Decimal:
    """
    Calculate diversity score for a list of content.
    
    Args:
        content_list: List of content texts
        
    Returns:
        Diversity score from 0.0 to 1.0
    """
    if len(content_list) <= 1:
        return Decimal("0.0")
    
    # Extract keywords from all content
    all_keyword_sets = []
    for content in content_list:
        keywords = set(extract_keywords(content, min_length=3, max_keywords=20))
        if keywords:
            all_keyword_sets.append(keywords)
    
    if len(all_keyword_sets) <= 1:
        return Decimal("0.0")
    
    # Calculate average pairwise similarity
    total_similarity = Decimal("0.0")
    comparison_count = 0
    
    for i in range(len(all_keyword_sets)):
        for j in range(i + 1, len(all_keyword_sets)):
            set1, set2 = all_keyword_sets[i], all_keyword_sets[j]
            
            intersection = set1 & set2
            union = set1 | set2
            
            if len(union) > 0:
                similarity = Decimal(str(len(intersection) / len(union)))
                total_similarity += similarity
                comparison_count += 1
    
    if comparison_count == 0:
        return Decimal("0.0")
    
    average_similarity = total_similarity / comparison_count
    
    # Diversity is inverse of similarity
    diversity = Decimal("1.0") - average_similarity
    return max(Decimal("0.0"), diversity)


def detect_language(text: str) -> str:
    """
    Simple language detection (placeholder).
    
    Args:
        text: Input text to analyze
        
    Returns:
        Detected language code (default: 'th' for Thai)
    """
    if not text:
        return "unknown"
    
    # AIDEV-NOTE: Simplified language detection based on character patterns
    text = text.lower()
    
    # Thai language indicators
    thai_chars = re.findall(r'[\u0e00-\u0e7f]', text)
    
    # English language indicators
    english_words = len(re.findall(r'\b[a-z]+\b', text))
    
    if len(thai_chars) > english_words:
        return "th"
    elif english_words > 0:
        return "en"
    else:
        return "unknown"


def clean_bio_text(bio: str) -> str:
    """
    Clean and normalize KOL bio text.
    
    Args:
        bio: Raw bio text
        
    Returns:
        Cleaned bio text
    """
    if not bio:
        return ""
    
    # Remove excessive emojis (keep some for personality)
    bio = re.sub(r'[\U0001F600-\U0001F64F]{3,}', '😊', bio)  # Emoticons
    bio = re.sub(r'[\U0001F300-\U0001F5FF]{3,}', '🌟', bio)  # Misc symbols
    bio = re.sub(r'[\U0001F680-\U0001F6FF]{3,}', '🚀', bio)  # Transport
    bio = re.sub(r'[\U0001F1E0-\U0001F1FF]{3,}', '🌍', bio)  # Flags
    
    # Clean up whitespace
    bio = re.sub(r'\s+', ' ', bio).strip()
    
    # Remove excessive punctuation
    bio = re.sub(r'[!]{3,}', '!!', bio)
    bio = re.sub(r'[?]{3,}', '??', bio)
    bio = re.sub(r'[.]{3,}', '...', bio)
    
    return bio[:500]  # Limit length


def extract_contact_info(bio: str) -> Dict[str, str]:
    """
    Extract contact information from KOL bio.
    
    Args:
        bio: KOL bio text
        
    Returns:
        Dictionary with extracted contact info
    """
    contact_info = {}
    
    if not bio:
        return contact_info
    
    # Extract email addresses
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    emails = re.findall(email_pattern, bio)
    if emails:
        contact_info['email'] = emails[0]  # Take first email
    
    # Extract phone numbers (Thai format)
    phone_patterns = [
        r'(\+66|66)[\s-]?[0-9][\s-]?[0-9]{4}[\s-]?[0-9]{4}',  # +66 format
        r'0[0-9][\s-]?[0-9]{4}[\s-]?[0-9]{4}'  # 0X format
    ]
    
    for pattern in phone_patterns:
        phones = re.findall(pattern, bio)
        if phones:
            # Clean up the phone number
            phone = re.sub(r'[\s-]', '', str(phones[0]))
            contact_info['phone'] = phone
            break
    
    # Extract LINE ID
    line_patterns = [
        r'line[\s:@]*([a-zA-Z0-9._-]+)',
        r'@([a-zA-Z0-9._-]+)',
    ]
    
    for pattern in line_patterns:
        matches = re.findall(pattern, bio, re.IGNORECASE)
        if matches:
            contact_info['line_id'] = matches[0]
            break
    
    return contact_info