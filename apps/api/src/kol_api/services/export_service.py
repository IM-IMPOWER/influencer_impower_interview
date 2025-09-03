"""
CSV Export Service for Budget Plans and KOL Data

AIDEV-NOTE: 250102122000 - Production-ready CSV export with enhanced data formatting
"""
import csv
import io
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from decimal import Decimal

import structlog
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from ..database.models.budget import BudgetPlan, BudgetAllocation
from ..database.models.kol import KOL, KOLMetrics
from ..database.models.campaign import Campaign
from ..database.models.scoring import KOLScore

logger = structlog.get_logger()


class CsvExportService:
    """Service for generating CSV exports of campaign data."""
    
    def __init__(self, db_session: AsyncSession):
        self.db_session = db_session
    
    async def export_budget_plan_csv(
        self,
        budget_plan_id: str,
        include_kol_details: bool = True,
        include_performance_predictions: bool = True
    ) -> Dict[str, Any]:
        """
        Export budget plan with KOL allocations to CSV format.
        
        Args:
            budget_plan_id: Budget plan to export
            include_kol_details: Include detailed KOL profile data
            include_performance_predictions: Include performance forecasts
            
        Returns:
            Dict with CSV content and metadata
        """
        
        try:
            # AIDEV-NOTE: Get budget plan with all related data
            plan_query = (
                select(BudgetPlan, Campaign)
                .join(Campaign, BudgetPlan.campaign_id == Campaign.id)
                .where(BudgetPlan.id == budget_plan_id)
            )
            
            plan_result = await self.db_session.execute(plan_query)
            plan_row = plan_result.first()
            
            if not plan_row:
                raise ValueError(f"Budget plan {budget_plan_id} not found")
            
            budget_plan, campaign = plan_row
            
            # AIDEV-NOTE: Get all allocations with KOL and metrics data
            allocations_query = (
                select(BudgetAllocation, KOL, KOLMetrics, KOLScore)
                .join(KOL, BudgetAllocation.kol_id == KOL.id)
                .outerjoin(KOLMetrics, KOL.id == KOLMetrics.kol_id)
                .outerjoin(KOLScore, KOL.id == KOLScore.kol_id)
                .where(BudgetAllocation.budget_plan_id == budget_plan_id)
                .order_by(BudgetAllocation.allocated_amount.desc())
            )
            
            allocations_result = await self.db_session.execute(allocations_query)
            allocation_rows = allocations_result.fetchall()
            
            # AIDEV-NOTE: Generate CSV content
            csv_content = self._generate_budget_plan_csv(
                budget_plan=budget_plan,
                campaign=campaign,
                allocation_rows=allocation_rows,
                include_kol_details=include_kol_details,
                include_performance_predictions=include_performance_predictions
            )
            
            # AIDEV-NOTE: Generate filename
            export_timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            filename = f"budget_plan_{budget_plan.name.replace(' ', '_')}_{export_timestamp}.csv"
            
            logger.info(
                "Budget plan CSV exported",
                budget_plan_id=budget_plan_id,
                filename=filename,
                allocations_count=len(allocation_rows)
            )
            
            return {
                "content": csv_content,
                "filename": filename,
                "content_type": "text/csv",
                "size": len(csv_content.encode('utf-8')),
                "metadata": {
                    "budget_plan_id": budget_plan_id,
                    "budget_plan_name": budget_plan.name,
                    "campaign_name": campaign.name,
                    "allocations_count": len(allocation_rows),
                    "total_budget": float(budget_plan.total_budget),
                    "optimization_score": float(budget_plan.optimization_score or 0),
                    "export_timestamp": export_timestamp,
                    "includes_kol_details": include_kol_details,
                    "includes_performance_predictions": include_performance_predictions
                }
            }
            
        except Exception as e:
            logger.error(
                "Budget plan CSV export failed",
                budget_plan_id=budget_plan_id,
                error=str(e)
            )
            raise
    
    async def export_kol_list_csv(
        self,
        kol_ids: List[str],
        include_detailed_metrics: bool = True,
        include_scores: bool = True,
        campaign_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Export list of KOLs to CSV format.
        
        Args:
            kol_ids: List of KOL IDs to export
            include_detailed_metrics: Include comprehensive metrics
            include_scores: Include scoring components
            campaign_id: Optional campaign context for scoring
            
        Returns:
            Dict with CSV content and metadata
        """
        
        try:
            # AIDEV-NOTE: Get KOLs with all related data
            kols_query = (
                select(KOL, KOLMetrics, KOLScore)
                .outerjoin(KOLMetrics, KOL.id == KOLMetrics.kol_id)
                .outerjoin(KOLScore, KOL.id == KOLScore.kol_id)
                .where(KOL.id.in_(kol_ids))
                .order_by(KOLMetrics.follower_count.desc().nullslast())
            )
            
            kols_result = await self.db_session.execute(kols_query)
            kol_rows = kols_result.fetchall()
            
            if not kol_rows:
                raise ValueError("No KOLs found for export")
            
            # AIDEV-NOTE: Generate CSV content
            csv_content = self._generate_kol_list_csv(
                kol_rows=kol_rows,
                include_detailed_metrics=include_detailed_metrics,
                include_scores=include_scores,
                campaign_id=campaign_id
            )
            
            # AIDEV-NOTE: Generate filename
            export_timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            filename = f"kol_list_{export_timestamp}.csv"
            
            logger.info(
                "KOL list CSV exported",
                kol_count=len(kol_rows),
                filename=filename,
                campaign_id=campaign_id
            )
            
            return {
                "content": csv_content,
                "filename": filename,
                "content_type": "text/csv",
                "size": len(csv_content.encode('utf-8')),
                "metadata": {
                    "kol_count": len(kol_rows),
                    "campaign_id": campaign_id,
                    "export_timestamp": export_timestamp,
                    "includes_detailed_metrics": include_detailed_metrics,
                    "includes_scores": include_scores
                }
            }
            
        except Exception as e:
            logger.error(
                "KOL list CSV export failed",
                kol_ids=kol_ids,
                error=str(e)
            )
            raise
    
    def _generate_budget_plan_csv(
        self,
        budget_plan: BudgetPlan,
        campaign: Campaign,
        allocation_rows: List,
        include_kol_details: bool,
        include_performance_predictions: bool
    ) -> str:
        """Generate CSV content for budget plan."""
        
        output = io.StringIO()
        
        # AIDEV-NOTE: Define CSV headers based on options
        headers = [
            "KOL_ID",
            "KOL_Username",
            "KOL_Display_Name", 
            "Platform",
            "Tier",
            "Primary_Category",
            "Allocated_Amount",
            "Priority_Score",
            "Expected_Reach",
            "Expected_Engagement",
            "Expected_Conversions",
            "Efficiency_Score"
        ]
        
        if include_kol_details:
            headers.extend([
                "Followers",
                "Following",
                "Engagement_Rate",
                "Posts_Last_30_Days",
                "Location",
                "Languages",
                "Is_Verified",
                "Is_Brand_Safe"
            ])
        
        if include_performance_predictions:
            headers.extend([
                "Risk_Score",
                "Cost_Per_Engagement",
                "Cost_Per_Reach",
                "Predicted_ROI",
                "Confidence_Score"
            ])
        
        # AIDEV-NOTE: Add campaign and plan metadata headers
        headers.extend([
            "Campaign_Name",
            "Budget_Plan_Name",
            "Optimization_Objective",
            "Allocation_Strategy"
        ])
        
        writer = csv.writer(output)
        writer.writerow(headers)
        
        # AIDEV-NOTE: Write allocation rows
        for allocation, kol, metrics, score in allocation_rows:
            row = [
                kol.id,
                kol.username,
                kol.display_name or kol.username,
                kol.platform.value,
                kol.tier.value,
                kol.primary_category.value,
                float(allocation.allocated_amount),
                float(allocation.priority_score or 0),
                allocation.expected_reach or 0,
                allocation.expected_engagement or 0,
                allocation.expected_conversions or 0,
                float(allocation.efficiency_score or 0)
            ]
            
            if include_kol_details and metrics:
                row.extend([
                    metrics.follower_count or 0,
                    metrics.following_count or 0,
                    float(metrics.engagement_rate or 0),
                    metrics.posts_last_30_days or 0,
                    kol.location or "",
                    ",".join(kol.languages or []),
                    kol.is_verified,
                    kol.is_brand_safe
                ])
            elif include_kol_details:
                row.extend(["", "", "", "", "", "", "", ""])
            
            if include_performance_predictions:
                cost_per_engagement = (
                    float(allocation.allocated_amount) / max(allocation.expected_engagement or 1, 1)
                )
                cost_per_reach = (
                    float(allocation.allocated_amount) / max(allocation.expected_reach or 1, 1)
                )
                predicted_roi = (
                    (allocation.expected_conversions or 0) * 10 / float(allocation.allocated_amount)
                )  # Assuming $10 value per conversion
                
                row.extend([
                    float(allocation.risk_score or 0),
                    cost_per_engagement,
                    cost_per_reach,
                    predicted_roi,
                    float(score.overall_confidence if score else 0.5)
                ])
            
            # AIDEV-NOTE: Add campaign and plan metadata
            row.extend([
                campaign.name,
                budget_plan.name,
                budget_plan.optimization_objective.value,
                budget_plan.allocation_strategy.value
            ])
            
            writer.writerow(row)
        
        return output.getvalue()
    
    def _generate_kol_list_csv(
        self,
        kol_rows: List,
        include_detailed_metrics: bool,
        include_scores: bool,
        campaign_id: Optional[str]
    ) -> str:
        """Generate CSV content for KOL list."""
        
        output = io.StringIO()
        
        # AIDEV-NOTE: Define headers based on options
        headers = [
            "KOL_ID",
            "Username",
            "Display_Name",
            "Platform",
            "Tier",
            "Primary_Category",
            "Followers",
            "Engagement_Rate",
            "Is_Verified",
            "Is_Brand_Safe",
            "Location",
            "Bio",
            "Profile_URL"
        ]
        
        if include_detailed_metrics:
            headers.extend([
                "Following_Count",
                "Posts_Last_30_Days",
                "Avg_Likes",
                "Avg_Comments",
                "Avg_Views",
                "Fake_Follower_Percentage",
                "Languages",
                "Account_Created_Date",
                "Last_Updated"
            ])
        
        if include_scores:
            headers.extend([
                "Overall_Score",
                "ROI_Score",
                "Audience_Quality_Score",
                "Brand_Safety_Score", 
                "Content_Relevance_Score",
                "Demographic_Fit_Score",
                "Reliability_Score",
                "Overall_Confidence"
            ])
        
        writer = csv.writer(output)
        writer.writerow(headers)
        
        # AIDEV-NOTE: Write KOL rows
        for kol, metrics, score in kol_rows:
            row = [
                kol.id,
                kol.username,
                kol.display_name or kol.username,
                kol.platform.value,
                kol.tier.value,
                kol.primary_category.value,
                metrics.follower_count if metrics else 0,
                float(metrics.engagement_rate) if metrics and metrics.engagement_rate else 0.0,
                kol.is_verified,
                kol.is_brand_safe,
                kol.location or "",
                kol.bio or "",
                f"https://{kol.platform.value}.com/@{kol.username}"
            ]
            
            if include_detailed_metrics and metrics:
                row.extend([
                    metrics.following_count or 0,
                    metrics.posts_last_30_days or 0,
                    metrics.avg_likes or 0,
                    metrics.avg_comments or 0,
                    metrics.avg_views or 0,
                    float(metrics.fake_follower_percentage or 0),
                    ",".join(kol.languages or []),
                    kol.account_created_at.isoformat() if kol.account_created_at else "",
                    metrics.metrics_date.isoformat() if metrics.metrics_date else ""
                ])
            elif include_detailed_metrics:
                row.extend(["", "", "", "", "", "", "", "", ""])
            
            if include_scores and score:
                row.extend([
                    float(score.overall_score or 0),
                    float(score.roi_score or 0),
                    float(score.audience_quality_score or 0),
                    float(score.brand_safety_score or 0),
                    float(score.content_relevance_score or 0),
                    float(score.demographic_fit_score or 0),
                    float(score.reliability_score or 0),
                    float(score.overall_confidence or 0)
                ])
            elif include_scores:
                row.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            
            writer.writerow(row)
        
        return output.getvalue()