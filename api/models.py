from pydantic import BaseModel, Field
from typing import List, Optional, Any

class KOLIn(BaseModel):
    platform: str
    username: str
    display_name: Optional[str] = None
    category: Optional[List[str]] = None
    followers: Optional[int] = None
    country: Optional[str] = None
    contact: Optional[str] = None
    sample_links: Optional[List[str]] = None
    extra: Optional[Any] = None

class KOLOut(KOLIn):
    id: int

class KOLQuery(BaseModel):
    q: Optional[str] = None
    platform: Optional[str] = None
    category: Optional[str] = None
    country: Optional[str] = None
    tier: Optional[str] = None  # nano/micro/mid/macro/mega
    limit: int = Field(20, ge=1, le=100)
    offset: int = Field(0, ge=0)

class ImportSummary(BaseModel):
    inserted: int
    updated: int
    skipped: int
