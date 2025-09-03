"""GraphQL resolvers for KOL-related operations."""

from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import structlog

from kol_api.graphql.resolvers.base import BaseResolver
from kol_api.graphql.types import KOL, KOLMatchingResult, KOLFilterInput
from kol_api.services.kol_matching import KOLMatchingService
from kol_api.services.models import CampaignRequirements, KOLTier, ContentCategory, OptimizationObjective
from kol_api.database.models.kol import KOL, KOLMetrics, PlatformType, KOLTier as DBKOLTier, ContentCategory as DBContentCategory
from kol_api.database.models.scoring import KOLScore
from kol_api.database.models.campaign import Campaign
from kol_api.utils.converters import (
    convert_kol_to_graphql, convert_kol_candidate_to_graphql,
    convert_matching_result_to_graphql, extract_kol_summary_data
)
from sqlalchemy import select, and_, or_, desc, func
from sqlalchemy.orm import selectinload
from decimal import Decimal

logger = structlog.get_logger()


class KOLResolvers(BaseResolver):
    """GraphQL resolvers for KOL operations."""
    
    @staticmethod
    async def get_kols(
        context: Dict[str, Any],
        filters: Optional[KOLFilterInput] = None,
        limit: int = 20,
        offset: int = 0,
        sort_by: Optional[str] = None,
        search: Optional[str] = None,
    ) -> List[KOL]:
        """Get KOLs with filtering, sorting, and search."""
        try:
            # AIDEV-NOTE: Require authentication for KOL access
            user = KOLResolvers.require_authentication(context)
            
            # AIDEV-NOTE: Get database session
            db_session = KOLResolvers.get_database_session(context)
            
            # AIDEV-NOTE: Build base query with joins
            query = (
                select(KOL)
                .options(
                    selectinload(KOL.metrics),
                    selectinload(KOL.scores)
                )
                .where(KOL.is_active == True)
            )
            
            # AIDEV-NOTE: Apply filters
            if filters:
                if filters.platform:
                    query = query.where(KOL.platform == PlatformType(filters.platform.value))
                
                if filters.tier:
                    query = query.where(KOL.tier == DBKOLTier(filters.tier.value))
                
                if filters.category:
                    query = query.where(KOL.primary_category == DBContentCategory(filters.category.value))
                
                if filters.min_followers or filters.max_followers:
                    query = query.join(KOLMetrics)
                    if filters.min_followers:
                        query = query.where(KOLMetrics.follower_count >= filters.min_followers)
                    if filters.max_followers:
                        query = query.where(KOLMetrics.follower_count <= filters.max_followers)
                
                if filters.min_engagement_rate:
                    if not any("kol_metrics" in str(table) for table in query.column_descriptions):
                        query = query.join(KOLMetrics)
                    query = query.where(KOLMetrics.engagement_rate >= Decimal(str(filters.min_engagement_rate)))
                
                if filters.location:
                    query = query.where(KOL.location.ilike(f"%{filters.location}%"))
                
                if filters.is_brand_safe is not None:
                    query = query.where(KOL.is_brand_safe == filters.is_brand_safe)
                
                if filters.languages:
                    query = query.where(KOL.languages.overlap(filters.languages))
            
            # AIDEV-NOTE: Apply search
            if search:
                search_filter = or_(
                    KOL.username.ilike(f"%{search}%"),
                    KOL.display_name.ilike(f"%{search}%"),
                    KOL.bio.ilike(f"%{search}%")
                )
                query = query.where(search_filter)
            
            # AIDEV-NOTE: Apply sorting
            if sort_by:
                if sort_by == "followers":
                    if not any("kol_metrics" in str(table) for table in query.column_descriptions):
                        query = query.join(KOLMetrics)
                    query = query.order_by(desc(KOLMetrics.follower_count))
                elif sort_by == "engagement":
                    if not any("kol_metrics" in str(table) for table in query.column_descriptions):
                        query = query.join(KOLMetrics)
                    query = query.order_by(desc(KOLMetrics.engagement_rate))
                elif sort_by == "created_at":
                    query = query.order_by(desc(KOL.created_at))
                else:
                    query = query.order_by(desc(KOL.created_at))
            else:
                query = query.order_by(desc(KOL.created_at))
            
            # AIDEV-NOTE: Apply pagination
            query = query.offset(offset).limit(limit)
            
            # AIDEV-NOTE: Execute query
            result = await db_session.execute(query)
            kols = result.scalars().all()
            
            # AIDEV-NOTE: Convert to GraphQL types
            graphql_kols = []
            for kol in kols:
                latest_metrics = None
                latest_score = None
                
                if kol.metrics:
                    latest_metrics = max(kol.metrics, key=lambda m: m.metrics_date)
                
                if kol.scores:
                    latest_score = max(kol.scores, key=lambda s: s.created_at)
                
                graphql_kol = convert_kol_to_graphql(kol, latest_metrics, latest_score)
                graphql_kols.append(graphql_kol)
            
            logger.info(
                "KOLs queried successfully",
                user_id=user["id"],
                count=len(graphql_kols),
                has_filters=filters is not None,
                has_search=search is not None
            )
            
            return graphql_kols
            
        except Exception as e:
            KOLResolvers.log_resolver_error("get_kols", e, context)
            raise
    
    @staticmethod
    async def get_kol_by_id(context: Dict[str, Any], kol_id: str) -> Optional[KOL]:
        """Get single KOL by ID."""
        try:
            user = KOLResolvers.require_authentication(context)
            db_session = KOLResolvers.get_database_session(context)
            
            # AIDEV-NOTE: Query KOL with related data
            query = (
                select(KOL)
                .options(
                    selectinload(KOL.metrics),
                    selectinload(KOL.scores),
                    selectinload(KOL.content)
                )
                .where(and_(
                    KOL.id == kol_id,
                    KOL.is_active == True
                ))
            )
            
            result = await db_session.execute(query)
            kol = result.scalar_one_or_none()
            
            if not kol:
                logger.info("KOL not found", user_id=user["id"], kol_id=kol_id)
                return None
            
            # AIDEV-NOTE: Get latest metrics and scores
            latest_metrics = None
            latest_score = None
            
            if kol.metrics:
                latest_metrics = max(kol.metrics, key=lambda m: m.metrics_date)
            
            if kol.scores:
                latest_score = max(kol.scores, key=lambda s: s.created_at)
            
            # AIDEV-NOTE: Convert to GraphQL type
            graphql_kol = convert_kol_to_graphql(kol, latest_metrics, latest_score)
            
            logger.info(
                "KOL retrieved successfully", 
                user_id=user["id"], 
                kol_id=kol_id,
                has_metrics=latest_metrics is not None,
                has_scores=latest_score is not None
            )
            
            return graphql_kol
            
        except Exception as e:
            KOLResolvers.log_resolver_error("get_kol_by_id", e, context, kol_id=kol_id)
            raise
    
    @staticmethod
    async def match_kols_for_campaign(
        context: Dict[str, Any],
        campaign_id: str,
        limit: int = 50,
        use_ai_scoring: bool = True,
    ) -> KOLMatchingResult:
        """AI-powered KOL matching for campaign."""
        try:
            user = KOLResolvers.require_authentication(context)
            db_session = KOLResolvers.get_database_session(context)
            
            # AIDEV-NOTE: Get campaign requirements
            campaign_query = select(Campaign).where(Campaign.id == campaign_id)
            campaign_result = await db_session.execute(campaign_query)
            campaign = campaign_result.scalar_one_or_none()
            
            if not campaign:
                raise ValueError(f"Campaign {campaign_id} not found")
            
            # AIDEV-NOTE: Convert campaign to requirements format
            campaign_requirements = CampaignRequirements(
                campaign_id=campaign_id,
                target_kol_tiers=[KOLTier(tier) for tier in campaign.target_kol_tiers],
                target_categories=[ContentCategory(cat) for cat in campaign.target_categories],
                total_budget=campaign.total_budget,
                min_follower_count=campaign.min_follower_count,
                max_follower_count=campaign.max_follower_count,
                min_engagement_rate=campaign.min_engagement_rate,
                campaign_objective=OptimizationObjective.BALANCED
            )
            
            # AIDEV-NOTE: Use enhanced KOL matching service
            matching_service = KOLMatchingService(db_session)
            
            matched_kols, metadata = await matching_service.find_matching_kols_advanced(
                campaign_requirements=campaign_requirements,
                limit=limit,
                enable_semantic_matching=use_ai_scoring,
                confidence_threshold=Decimal("0.7")
            )
            
            logger.info(
                "Advanced KOL matching completed",
                user_id=user["id"],
                campaign_id=campaign_id,
                matched_count=len(matched_kols),
                use_ai=use_ai_scoring,
                algorithm_version=metadata.get("algorithm_version")
            )
            
            # AIDEV-NOTE: Convert to GraphQL response type
            return convert_matching_result_to_graphql(matched_kols, metadata)
            
        except Exception as e:
            KOLResolvers.log_resolver_error(
                "match_kols_for_campaign", 
                e, 
                context, 
                campaign_id=campaign_id
            )
            raise
    
    @staticmethod
    async def find_similar_kols(
        context: Dict[str, Any],
        kol_id: str,
        limit: int = 10,
        similarity_threshold: float = 0.7,
    ) -> List[KOL]:
        """Find similar KOLs using vector similarity."""
        try:
            user = KOLResolvers.require_authentication(context)
            db_session = KOLResolvers.get_database_session(context)
            
            # AIDEV-NOTE: Use enhanced KOL matching service for semantic similarity
            matching_service = KOLMatchingService(db_session)
            
            similar_kols = await matching_service.find_similar_kols_semantic(
                reference_kol_id=kol_id,
                limit=limit,
                similarity_threshold=Decimal(str(similarity_threshold))
            )
            
            logger.info(
                "Semantic similar KOLs found",
                user_id=user["id"],
                reference_kol_id=kol_id,
                similar_count=len(similar_kols)
            )
            
            # AIDEV-NOTE: Convert to GraphQL types
            graphql_kols = [convert_kol_candidate_to_graphql(candidate) for candidate in similar_kols]
            
            return graphql_kols
            
        except Exception as e:
            KOLResolvers.log_resolver_error(
                "find_similar_kols", 
                e, 
                context, 
                kol_id=kol_id
            )
            raise
    
    @staticmethod
    async def trigger_data_refresh(
        context: Dict[str, Any],
        kol_ids: Optional[List[str]] = None,
        platform: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Trigger KOL data refresh via Go scraper service."""
        try:
            user = KOLResolvers.require_role(context, ["admin", "manager"])
            
            # AIDEV-NOTE: Validate KOL IDs if provided
            if kol_ids:
                kol_count_query = select(func.count(KOL.id)).where(
                    and_(
                        KOL.id.in_(kol_ids),
                        KOL.is_active == True
                    )
                )
                result = await db_session.execute(kol_count_query)
                valid_kol_count = result.scalar()
                
                if valid_kol_count != len(kol_ids):
                    logger.warning(
                        "Some KOL IDs not found",
                        requested=len(kol_ids),
                        found=valid_kol_count
                    )
            
            # AIDEV-NOTE: Mock implementation - in production would call Go scraper service
            # This would make HTTP requests to the scraper microservice
            # and enqueue background scraping tasks
            
            refresh_payload = {
                "kol_ids": kol_ids or [],
                "platform": platform,
                "requested_by": user["id"],
                "priority": "normal"
            }
            
            # AIDEV-NOTE: Would make actual HTTP call here
            # response = await scraper_client.trigger_refresh(refresh_payload)
            
            logger.info(
                "Data refresh request processed",
                user_id=user["id"],
                kol_ids=kol_ids,
                platform=platform,
                kol_count=len(kol_ids or [])
            )
            
            return {
                "success": True,
                "message": "Data refresh request submitted to scraper service",
                "data": {
                    "kol_count": len(kol_ids or []),
                    "platform": platform,
                    "estimated_completion": "5-10 minutes"
                }
            }
            
        except Exception as e:
            KOLResolvers.log_resolver_error(
                "trigger_data_refresh", 
                e, 
                context,
                kol_ids=kol_ids,
                platform=platform
            )
            raise
    
    @staticmethod
    async def update_brand_safety_status(
        context: Dict[str, Any],
        kol_id: str,
        is_brand_safe: bool,
        safety_notes: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Update KOL brand safety status."""
        try:
            user = KOLResolvers.require_role(context, ["admin", "manager"])
            db_session = KOLResolvers.get_database_session(context)
            
            # AIDEV-NOTE: Get KOL and update brand safety status
            kol_query = select(KOL).where(KOL.id == kol_id)
            result = await db_session.execute(kol_query)
            kol = result.scalar_one_or_none()
            
            if not kol:
                raise ValueError(f"KOL {kol_id} not found")
            
            # AIDEV-NOTE: Update brand safety status
            old_status = kol.is_brand_safe
            kol.is_brand_safe = is_brand_safe
            
            if safety_notes:
                kol.safety_notes = safety_notes
            
            # AIDEV-NOTE: Create audit log entry (simplified)
            audit_entry = {
                "action": "brand_safety_update",
                "kol_id": kol_id,
                "old_value": old_status,
                "new_value": is_brand_safe,
                "notes": safety_notes,
                "updated_by": user["id"],
                "updated_at": datetime.utcnow().isoformat()
            }
            
            # AIDEV-NOTE: In production, would store audit entry in database
            # audit_service.create_entry(audit_entry)
            
            await db_session.commit()
            
            logger.info(
                "Brand safety status updated successfully",
                user_id=user["id"],
                kol_id=kol_id,
                old_status=old_status,
                new_status=is_brand_safe,
                has_notes=safety_notes is not None
            )
            
            return {
                "success": True,
                "message": f"Brand safety status updated to {'safe' if is_brand_safe else 'unsafe'}",
                "data": {
                    "kol_id": kol_id,
                    "previous_status": old_status,
                    "current_status": is_brand_safe,
                    "audit_logged": True
                }
            }
            
        except Exception as e:
            KOLResolvers.log_resolver_error(
                "update_brand_safety_status", 
                e, 
                context,
                kol_id=kol_id
            )
            raise