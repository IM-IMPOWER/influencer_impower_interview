"""
Brief Parser Service for Markdown Campaign Briefs

AIDEV-NOTE: 250102122100 - Enhanced markdown parsing with NLP extraction for POC2
"""
import re
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from decimal import Decimal

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from ..database.models.campaign import Campaign, CampaignRequirements
from ..database.models.kol import KOLTier, ContentCategory
from ..services.models import BriefParsingResult, CampaignObjective

logger = structlog.get_logger()


@dataclass
class ParsedBriefData:
    """Structured data extracted from campaign brief."""
    campaign_title: Optional[str] = None
    campaign_description: Optional[str] = None
    target_audience: Optional[str] = None
    campaign_objectives: List[str] = None
    target_categories: List[str] = None
    target_tiers: List[str] = None
    budget_information: Optional[Dict[str, Any]] = None
    geographic_targets: List[str] = None
    demographic_requirements: Optional[Dict[str, Any]] = None
    content_requirements: Optional[Dict[str, Any]] = None
    timeline_information: Optional[Dict[str, Any]] = None
    success_metrics: List[str] = None
    brand_safety_requirements: Optional[Dict[str, Any]] = None
    additional_notes: List[str] = None
    confidence_score: float = 0.0
    
    def __post_init__(self):
        if self.campaign_objectives is None:
            self.campaign_objectives = []
        if self.target_categories is None:
            self.target_categories = []
        if self.target_tiers is None:
            self.target_tiers = []
        if self.geographic_targets is None:
            self.geographic_targets = []
        if self.success_metrics is None:
            self.success_metrics = []
        if self.additional_notes is None:
            self.additional_notes = []


class BriefParserService:
    """Service for parsing and extracting structured data from campaign briefs."""
    
    def __init__(self, db_session: AsyncSession):
        self.db_session = db_session
        
        # AIDEV-NOTE: Pattern definitions for information extraction
        self.patterns = {
            'title': [
                r'^#\s+(.+)$',
                r'(?i)^title:\s*(.+)$',
                r'(?i)^campaign:\s*(.+)$',
                r'(?i)^project:\s*(.+)$'
            ],
            'budget': [
                r'(?i)budget:?\s*\$?([\d,]+(?:\.\d{2})?)',
                r'(?i)\$\s?([\d,]+(?:\.\d{2})?)',
                r'(?i)([\d,]+(?:\.\d{2})?)\s*(?:dollars?|usd|thb|baht)',
                r'(?i)total\s+budget:?\s*\$?([\d,]+(?:\.\d{2})?)'
            ],
            'tiers': [
                r'(?i)(nano|micro|mid-?tier|macro|mega)\s*(?:influencers?|kols?)?',
                r'(?i)\b(\d+k?-\d+k?)\s*followers?',
                r'(?i)(small|medium|large)\s*(?:influencers?|accounts?)'
            ],
            'categories': [
                r'(?i)(fashion|beauty|lifestyle|tech|food|travel|fitness|gaming|music|art|sports|education|parenting|business|finance|health|comedy|dance|diy|pets|automotive)',
                r'(?i)niche:?\s*([a-zA-Z\s,]+)',
                r'(?i)category:?\s*([a-zA-Z\s,]+)'
            ],
            'demographics': [
                r'(?i)age:?\s*(\d{1,2}[-–]\d{1,2})',
                r'(?i)(\d{1,2}[-–]\d{1,2})\s*(?:years?\s*old|yo)',
                r'(?i)(male|female|non-binary|all\s+genders?)',
                r'(?i)gender:?\s*(male|female|non-binary|mixed|all)'
            ],
            'locations': [
                r'(?i)(bangkok|thailand|singapore|malaysia|indonesia|philippines|vietnam|cambodia|laos|myanmar|brunei)',
                r'(?i)(?:location|region|country|city):?\s*([a-zA-Z\s,]+)',
                r'(?i)(asia|southeast\s+asia|asean)'
            ],
            'objectives': [
                r'(?i)(brand\s+awareness|engagement|conversions?|sales?|leads?|installs?|traffic)',
                r'(?i)goal:?\s*([a-zA-Z\s,]+)',
                r'(?i)objective:?\s*([a-zA-Z\s,]+)',
                r'(?i)kpi:?\s*([a-zA-Z\s,]+)'
            ]
        }
        
        # AIDEV-NOTE: Tier mapping for standardization
        self.tier_mapping = {
            'nano': ['nano', 'small', '1k-10k', 'micro-nano'],
            'micro': ['micro', 'medium', '10k-100k'],
            'mid': ['mid-tier', 'mid', 'medium-large', '100k-1m'],
            'macro': ['macro', 'large', '1m-10m'],
            'mega': ['mega', 'celebrity', '10m+', 'very-large']
        }
        
        # AIDEV-NOTE: Category mapping for standardization
        self.category_mapping = {
            ContentCategory.LIFESTYLE: ['lifestyle', 'life', 'daily', 'personal'],
            ContentCategory.FASHION: ['fashion', 'style', 'clothing', 'outfit'],
            ContentCategory.BEAUTY: ['beauty', 'makeup', 'skincare', 'cosmetics'],
            ContentCategory.FOOD: ['food', 'cooking', 'recipe', 'culinary', 'restaurant'],
            ContentCategory.TRAVEL: ['travel', 'tourism', 'vacation', 'adventure'],
            ContentCategory.FITNESS: ['fitness', 'workout', 'gym', 'health', 'exercise'],
            ContentCategory.TECH: ['tech', 'technology', 'gadgets', 'apps', 'digital'],
            ContentCategory.GAMING: ['gaming', 'games', 'esports', 'streamer'],
            ContentCategory.MUSIC: ['music', 'songs', 'artist', 'musician', 'band'],
            ContentCategory.COMEDY: ['comedy', 'funny', 'humor', 'memes', 'jokes'],
            ContentCategory.EDUCATION: ['education', 'learning', 'tutorial', 'teaching'],
            ContentCategory.BUSINESS: ['business', 'entrepreneur', 'startup', 'finance']
        }
    
    async def parse_markdown_brief(
        self,
        file_content: str,
        filename: str,
        user_id: str,
        campaign_id: Optional[str] = None
    ) -> BriefParsingResult:
        """
        Parse markdown campaign brief and extract structured information.
        
        Args:
            file_content: Raw markdown content
            filename: Original filename
            user_id: User uploading the brief
            campaign_id: Optional existing campaign to update
            
        Returns:
            Comprehensive parsing result with extracted data and confidence metrics
        """
        
        try:
            logger.info(
                "Starting brief parsing",
                filename=filename,
                content_length=len(file_content),
                user_id=user_id,
                campaign_id=campaign_id
            )
            
            # AIDEV-NOTE: Clean and normalize content
            normalized_content = self._normalize_content(file_content)
            
            # AIDEV-NOTE: Extract structured data
            parsed_data = self._extract_structured_data(normalized_content)
            
            # AIDEV-NOTE: Validate and standardize extracted data
            standardized_data = self._standardize_extracted_data(parsed_data)
            
            # AIDEV-NOTE: Calculate overall confidence score
            confidence_score = self._calculate_parsing_confidence(
                parsed_data, standardized_data, len(file_content)
            )
            
            # AIDEV-NOTE: Create campaign requirements object
            campaign_requirements = await self._create_campaign_requirements(
                standardized_data, confidence_score
            )
            
            # AIDEV-NOTE: Generate parsing metadata
            parsing_metadata = {
                "filename": filename,
                "content_length": len(file_content),
                "parsing_timestamp": datetime.now(timezone.utc).isoformat(),
                "parser_version": "2.1",
                "user_id": user_id,
                "campaign_id": campaign_id,
                "sections_found": self._identify_sections(normalized_content),
                "extraction_quality": self._assess_extraction_quality(parsed_data),
                "standardization_applied": True,
                "confidence_breakdown": self._get_confidence_breakdown(parsed_data)
            }
            
            result = BriefParsingResult(
                success=True,
                campaign_requirements=campaign_requirements,
                parsed_data=asdict(standardized_data),
                confidence_score=confidence_score,
                parsing_metadata=parsing_metadata,
                error_message=None,
                validation_warnings=self._get_validation_warnings(standardized_data),
                extracted_text_sections=self._extract_text_sections(normalized_content)
            )
            
            logger.info(
                "Brief parsing completed successfully",
                filename=filename,
                confidence_score=confidence_score,
                campaign_id=campaign_id,
                extracted_categories=len(standardized_data.target_categories),
                extracted_tiers=len(standardized_data.target_tiers)
            )
            
            return result
            
        except Exception as e:
            logger.error(
                "Brief parsing failed",
                filename=filename,
                error=str(e),
                user_id=user_id
            )
            
            return BriefParsingResult(
                success=False,
                campaign_requirements=None,
                parsed_data={},
                confidence_score=0.0,
                parsing_metadata={"error": str(e), "filename": filename},
                error_message=f"Failed to parse brief: {str(e)}",
                validation_warnings=[],
                extracted_text_sections={}
            )
    
    def _normalize_content(self, content: str) -> str:
        """Clean and normalize markdown content."""
        
        # AIDEV-NOTE: Remove excessive whitespace and normalize line endings
        content = re.sub(r'\r\n', '\n', content)
        content = re.sub(r'\n{3,}', '\n\n', content)
        content = re.sub(r'[ \t]+', ' ', content)
        
        # AIDEV-NOTE: Normalize markdown headers
        content = re.sub(r'^#{1,6}\s*', '# ', content, flags=re.MULTILINE)
        
        # AIDEV-NOTE: Remove markdown formatting that interferes with parsing
        content = re.sub(r'\*\*(.*?)\*\*', r'\1', content)  # Bold
        content = re.sub(r'\*(.*?)\*', r'\1', content)      # Italic
        content = re.sub(r'`(.*?)`', r'\1', content)        # Inline code
        
        return content.strip()
    
    def _extract_structured_data(self, content: str) -> ParsedBriefData:
        """Extract structured information from normalized content."""
        
        parsed_data = ParsedBriefData()
        lines = content.split('\n')
        
        # AIDEV-NOTE: Extract title (usually first header)
        for line in lines:
            for pattern in self.patterns['title']:
                match = re.match(pattern, line.strip(), re.MULTILINE)
                if match:
                    parsed_data.campaign_title = match.group(1).strip()
                    break
            if parsed_data.campaign_title:
                break
        
        # AIDEV-NOTE: Extract budget information
        budget_matches = []
        for pattern in self.patterns['budget']:
            matches = re.findall(pattern, content)
            budget_matches.extend(matches)
        
        if budget_matches:
            # Take the largest budget mentioned (likely the total)
            budgets = [float(b.replace(',', '')) for b in budget_matches]
            parsed_data.budget_information = {
                "total_budget": max(budgets),
                "currency": self._detect_currency(content),
                "budget_mentions": budgets
            }
        
        # AIDEV-NOTE: Extract tier information
        tier_matches = []
        for pattern in self.patterns['tiers']:
            matches = re.findall(pattern, content, re.IGNORECASE)
            tier_matches.extend([m.lower() for m in matches])
        parsed_data.target_tiers = list(set(tier_matches))
        
        # AIDEV-NOTE: Extract category information
        category_matches = []
        for pattern in self.patterns['categories']:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    category_matches.extend([m.strip().lower() for m in match if m.strip()])
                else:
                    category_matches.extend([m.strip().lower() for m in match.split(',') if m.strip()])
        parsed_data.target_categories = list(set(category_matches))
        
        # AIDEV-NOTE: Extract demographic information
        demo_data = {}
        for pattern in self.patterns['demographics']:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                if '-' in match and match.replace('-', '').isdigit():
                    demo_data['age_range'] = match
                elif match.lower() in ['male', 'female', 'non-binary', 'all', 'mixed']:
                    demo_data['gender'] = match.lower()
        
        if demo_data:
            parsed_data.demographic_requirements = demo_data
        
        # AIDEV-NOTE: Extract location information
        location_matches = []
        for pattern in self.patterns['locations']:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                locations = [loc.strip() for loc in match.split(',') if loc.strip()]
                location_matches.extend(locations)
        parsed_data.geographic_targets = list(set([loc.title() for loc in location_matches]))
        
        # AIDEV-NOTE: Extract objectives
        objective_matches = []
        for pattern in self.patterns['objectives']:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                objectives = [obj.strip().lower() for obj in match.split(',') if obj.strip()]
                objective_matches.extend(objectives)
        parsed_data.campaign_objectives = list(set(objective_matches))
        
        # AIDEV-NOTE: Extract description (first paragraph after title)
        description_match = re.search(r'^#.*?\n\n(.*?)(?:\n\n|\n#|$)', content, re.DOTALL | re.MULTILINE)
        if description_match:
            parsed_data.campaign_description = description_match.group(1).strip()
        
        return parsed_data
    
    def _standardize_extracted_data(self, parsed_data: ParsedBriefData) -> ParsedBriefData:
        """Standardize extracted data to match database enums."""
        
        # AIDEV-NOTE: Standardize tiers
        standardized_tiers = []
        for tier in parsed_data.target_tiers:
            for standard_tier, variations in self.tier_mapping.items():
                if any(var in tier.lower() for var in variations):
                    if standard_tier not in standardized_tiers:
                        standardized_tiers.append(standard_tier)
                    break
        parsed_data.target_tiers = standardized_tiers
        
        # AIDEV-NOTE: Standardize categories
        standardized_categories = []
        for category in parsed_data.target_categories:
            for standard_category, variations in self.category_mapping.items():
                if any(var in category.lower() for var in variations):
                    if standard_category.value not in standardized_categories:
                        standardized_categories.append(standard_category.value)
                    break
        parsed_data.target_categories = standardized_categories
        
        # AIDEV-NOTE: Standardize objectives
        objective_mapping = {
            'brand awareness': 'maximize_reach',
            'awareness': 'maximize_reach',
            'engagement': 'maximize_engagement',
            'conversions': 'maximize_conversions',
            'sales': 'maximize_conversions',
            'leads': 'maximize_conversions',
            'traffic': 'maximize_reach',
            'installs': 'maximize_conversions'
        }
        
        standardized_objectives = []
        for objective in parsed_data.campaign_objectives:
            for key, value in objective_mapping.items():
                if key in objective.lower():
                    if value not in standardized_objectives:
                        standardized_objectives.append(value)
                    break
        
        parsed_data.campaign_objectives = standardized_objectives
        
        return parsed_data
    
    def _detect_currency(self, content: str) -> str:
        """Detect currency from content."""
        
        if re.search(r'(?i)\bthb\b|baht|thai', content):
            return 'THB'
        elif re.search(r'(?i)\busd\b|\$|dollar', content):
            return 'USD'
        elif re.search(r'(?i)\bsgd\b|singapore', content):
            return 'SGD'
        else:
            return 'USD'  # Default
    
    def _calculate_parsing_confidence(
        self,
        parsed_data: ParsedBriefData,
        standardized_data: ParsedBriefData,
        content_length: int
    ) -> float:
        """Calculate overall confidence score for parsing results."""
        
        confidence_factors = []
        
        # AIDEV-NOTE: Content completeness factors
        if standardized_data.campaign_title:
            confidence_factors.append(0.15)
        if standardized_data.campaign_description and len(standardized_data.campaign_description) > 50:
            confidence_factors.append(0.10)
        if standardized_data.budget_information:
            confidence_factors.append(0.20)
        if standardized_data.target_categories:
            confidence_factors.append(0.15)
        if standardized_data.target_tiers:
            confidence_factors.append(0.15)
        if standardized_data.campaign_objectives:
            confidence_factors.append(0.10)
        if standardized_data.geographic_targets:
            confidence_factors.append(0.05)
        if standardized_data.demographic_requirements:
            confidence_factors.append(0.10)
        
        base_confidence = sum(confidence_factors)
        
        # AIDEV-NOTE: Content quality adjustments
        if content_length < 200:
            base_confidence *= 0.7  # Very short content
        elif content_length > 1000:
            base_confidence *= 1.1  # Detailed content
        
        # AIDEV-NOTE: Standardization success bonus
        standardization_ratio = len(standardized_data.target_categories) / max(len(parsed_data.target_categories), 1)
        if standardization_ratio > 0.8:
            base_confidence *= 1.05
        
        return min(1.0, base_confidence)
    
    async def _create_campaign_requirements(
        self,
        standardized_data: ParsedBriefData,
        confidence_score: float
    ) -> Optional[CampaignRequirements]:
        """Create CampaignRequirements object from parsed data."""
        
        if confidence_score < 0.3:  # Too low confidence
            return None
        
        try:
            # AIDEV-NOTE: Map standardized data to campaign requirements
            return CampaignRequirements(
                campaign_id=None,  # Will be set when creating campaign
                target_kol_tiers=[KOLTier(tier) for tier in standardized_data.target_tiers],
                target_categories=[ContentCategory(cat) for cat in standardized_data.target_categories],
                total_budget=Decimal(str(standardized_data.budget_information.get('total_budget', 100000))) if standardized_data.budget_information else Decimal('100000'),
                campaign_objective=CampaignObjective(standardized_data.campaign_objectives[0]) if standardized_data.campaign_objectives else CampaignObjective.MAXIMIZE_ENGAGEMENT,
                target_locations=standardized_data.geographic_targets,
                target_languages=['en', 'th'] if any('thailand' in loc.lower() for loc in standardized_data.geographic_targets) else ['en'],
                min_follower_count=1000,  # Default
                max_follower_count=None,
                min_engagement_rate=Decimal('0.01'),  # 1% minimum
                require_verified=False,
                require_brand_safe=True,
                target_demographics=standardized_data.demographic_requirements or {},
                required_hashtags=[],
                excluded_keywords=[],
                expected_conversion_rate=Decimal('0.02'),  # 2% default
                max_cost_per_engagement=Decimal('1.00'),
                preferred_content_types=['video', 'image'],
                campaign_duration_days=30,
                content_requirements=standardized_data.content_requirements or {}
            )
            
        except Exception as e:
            logger.warning("Failed to create campaign requirements", error=str(e))
            return None
    
    def _identify_sections(self, content: str) -> List[str]:
        """Identify different sections in the content."""
        sections = []
        
        if re.search(r'(?i)(budget|cost|price)', content):
            sections.append('budget')
        if re.search(r'(?i)(tier|follower|audience size)', content):
            sections.append('tiers')
        if re.search(r'(?i)(category|niche|vertical)', content):
            sections.append('categories')
        if re.search(r'(?i)(demographic|age|gender|location)', content):
            sections.append('demographics')
        if re.search(r'(?i)(objective|goal|kpi|metric)', content):
            sections.append('objectives')
        if re.search(r'(?i)(timeline|duration|schedule)', content):
            sections.append('timeline')
        if re.search(r'(?i)(brand safety|content guideline)', content):
            sections.append('brand_safety')
        
        return sections
    
    def _assess_extraction_quality(self, parsed_data: ParsedBriefData) -> Dict[str, Any]:
        """Assess the quality of data extraction."""
        return {
            "title_extracted": bool(parsed_data.campaign_title),
            "description_extracted": bool(parsed_data.campaign_description),
            "budget_extracted": bool(parsed_data.budget_information),
            "categories_count": len(parsed_data.target_categories),
            "tiers_count": len(parsed_data.target_tiers),
            "objectives_count": len(parsed_data.campaign_objectives),
            "locations_count": len(parsed_data.geographic_targets),
            "demographics_extracted": bool(parsed_data.demographic_requirements)
        }
    
    def _get_confidence_breakdown(self, parsed_data: ParsedBriefData) -> Dict[str, float]:
        """Get confidence breakdown by component."""
        return {
            "title_confidence": 0.9 if parsed_data.campaign_title else 0.0,
            "budget_confidence": 0.8 if parsed_data.budget_information else 0.0,
            "categories_confidence": min(0.9, len(parsed_data.target_categories) * 0.3),
            "tiers_confidence": min(0.9, len(parsed_data.target_tiers) * 0.3),
            "objectives_confidence": min(0.9, len(parsed_data.campaign_objectives) * 0.4),
            "demographics_confidence": 0.7 if parsed_data.demographic_requirements else 0.0
        }
    
    def _get_validation_warnings(self, standardized_data: ParsedBriefData) -> List[str]:
        """Generate validation warnings for incomplete or ambiguous data."""
        warnings = []
        
        if not standardized_data.campaign_title:
            warnings.append("Campaign title not found - please specify a clear title")
        
        if not standardized_data.budget_information:
            warnings.append("Budget information not found - using default budget")
        
        if not standardized_data.target_categories:
            warnings.append("No content categories specified - may result in broad matching")
        
        if not standardized_data.target_tiers:
            warnings.append("No influencer tiers specified - will include all tiers")
        
        if not standardized_data.campaign_objectives:
            warnings.append("Campaign objectives unclear - defaulting to engagement maximization")
        
        if standardized_data.budget_information and standardized_data.budget_information.get('total_budget', 0) < 1000:
            warnings.append("Budget seems low - please verify the amount")
        
        return warnings
    
    def _extract_text_sections(self, content: str) -> Dict[str, str]:
        """Extract relevant text sections for reference."""
        sections = {}
        
        # AIDEV-NOTE: Extract first paragraph as summary
        first_para_match = re.search(r'^.*?\n\n(.*?)(?:\n\n|$)', content, re.DOTALL)
        if first_para_match:
            sections['summary'] = first_para_match.group(1).strip()
        
        # AIDEV-NOTE: Extract any bullet points as requirements
        bullets = re.findall(r'^[-*+]\s+(.+)$', content, re.MULTILINE)
        if bullets:
            sections['requirements'] = bullets
        
        # AIDEV-NOTE: Extract numbered lists as steps or priorities
        numbered = re.findall(r'^\d+\.\s+(.+)$', content, re.MULTILINE)
        if numbered:
            sections['priorities'] = numbered
        
        return sections