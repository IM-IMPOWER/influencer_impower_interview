"""FastAPI routers package."""

from kol_api.routers.health import router as health_router
from kol_api.routers.auth import router as auth_router
from kol_api.routers.kols import router as kols_router
from kol_api.routers.campaigns import router as campaigns_router
from kol_api.routers.budget_optimizer import router as budget_optimizer_router
from kol_api.routers.scoring import router as scoring_router
from kol_api.routers.upload import router as upload_router

__all__ = [
    "health_router",
    "auth_router", 
    "kols_router",
    "campaigns_router",
    "budget_optimizer_router",
    "scoring_router",
    "upload_router",
]