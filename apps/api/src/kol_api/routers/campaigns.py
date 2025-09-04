"""Campaign management REST endpoints."""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from sqlalchemy.ext.asyncio import AsyncSession

from kol_api.database.connection import get_session

router = APIRouter(prefix="/campaigns", tags=["Campaigns"])


class CampaignSummary(BaseModel):
    """Summary campaign information."""
    id: str
    name: str
    brand_name: str
    status: str
    start_date: str
    end_date: str
    total_budget: float
    currency: str


@router.get("/", response_model=List[CampaignSummary])
async def list_campaigns(
    status: Optional[str] = None,
    limit: int = 20,
    offset: int = 0,
    db_session: AsyncSession = Depends(get_session)
):
    """List campaigns with filtering."""
    # AIDEV-NOTE: Implement with SQLAlchemy queries
    return []


@router.get("/{campaign_id}", response_model=dict)
async def get_campaign(
    campaign_id: str,
    db_session: AsyncSession = Depends(get_session)
):
    """Get detailed campaign information."""
    # AIDEV-NOTE: Implement with full campaign details
    return {}


@router.get("/{campaign_id}/performance")
async def get_campaign_performance(
    campaign_id: str,
    db_session: AsyncSession = Depends(get_session)
):
    """Get campaign performance analytics."""
    # AIDEV-NOTE: Implement performance analytics aggregation
    return {}


@router.post("/{campaign_id}/kols/{kol_id}/invite")
async def invite_kol(
    campaign_id: str,
    kol_id: str,
    proposed_rate: Optional[float] = None,
    db_session: AsyncSession = Depends(get_session)
):
    """Invite KOL to campaign."""
    # AIDEV-NOTE: Implement KOL invitation with notification
    return {"message": "KOL invitation sent"}