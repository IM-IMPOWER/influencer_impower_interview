"""KOL management REST endpoints."""

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional
from sqlalchemy.ext.asyncio import AsyncSession

from kol_api.database.connection import get_session

router = APIRouter(prefix="/kols", tags=["KOLs"])


class KOLSummary(BaseModel):
    """Summary KOL information for REST responses."""
    id: str
    username: str
    display_name: str
    platform: str
    tier: str
    follower_count: Optional[int] = None
    engagement_rate: Optional[float] = None
    is_brand_safe: bool


@router.get("/", response_model=List[KOLSummary])
async def list_kols(
    platform: Optional[str] = Query(None),
    tier: Optional[str] = Query(None),
    limit: int = Query(20, le=100),
    offset: int = Query(0, ge=0),
    db_session: AsyncSession = Depends(get_session)
):
    """List KOLs with filtering and pagination."""
    # AIDEV-NOTE: Implement with SQLAlchemy queries
    return []


@router.get("/{kol_id}", response_model=dict)
async def get_kol(
    kol_id: str,
    db_session: AsyncSession = Depends(get_session)
):
    """Get detailed KOL information."""
    # AIDEV-NOTE: Implement with full KOL details including metrics and scores
    return {}


@router.post("/refresh")
async def trigger_data_refresh(
    kol_ids: Optional[List[str]] = None,
    platform: Optional[str] = None,
    db_session: AsyncSession = Depends(get_session)
):
    """Trigger KOL data refresh via Go scraper service."""
    # AIDEV-NOTE: Implement integration with Go scraper service
    return {"message": "Data refresh triggered", "kol_count": len(kol_ids or [])}


@router.put("/{kol_id}/brand-safety")
async def update_brand_safety(
    kol_id: str,
    is_brand_safe: bool,
    safety_notes: Optional[str] = None,
    db_session: AsyncSession = Depends(get_session)
):
    """Update KOL brand safety status."""
    # AIDEV-NOTE: Implement brand safety status update with audit trail
    return {"message": "Brand safety status updated"}