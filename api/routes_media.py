from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from pathlib import Path

router = APIRouter(prefix="/api", tags=["media"])
BASE = Path("data")  # adjust if different; mount into container if needed

@router.get("/media")
def media(path: str = Query(...)):
    p = Path(path)
    # allow only under data/
    full = (p if p.is_absolute() else (BASE / p)).resolve()
    if BASE.resolve() not in full.parents and BASE.resolve() != full:
        raise HTTPException(status_code=400, detail="Invalid path")
    if not full.exists():
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(str(full))
