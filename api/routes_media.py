import os
import psycopg
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from psycopg.rows import dict_row

router = APIRouter(prefix="/api", tags=["media"])


def get_conn():
    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        raise RuntimeError("DATABASE_URL not set")
    return psycopg.connect(dsn, row_factory=dict_row)


@router.get("/kols/{kol_id}/media")
def list_kol_media(kol_id: int):
    """
    Return all media (profile + thumbs) for a given KOL.
    Each row includes media_url if set, otherwise UI can call /api/media/{id}.
    """
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, kol_id, kind, media_url, path
            FROM kol_media
            WHERE kol_id = %s AND kind IN ('profile','thumb')
            ORDER BY CASE kind WHEN 'profile' THEN 0 ELSE 1 END, id
            """,
            (kol_id,),
        )
        rows = cur.fetchall()
    return rows


@router.get("/media/{media_id}")
def serve_media(media_id: int):
    """
    Stream an image file from disk (using kol_media.path).
    Useful when you don't have public URLs for KOL images.
    """
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT path FROM kol_media WHERE id = %s", (media_id,)
        )
        row = cur.fetchone()

    if not row:
        raise HTTPException(404, "Media not found")
    
    path = row["path"]
    print("Path is", path)
    if not path or not os.path.isfile(path):
        raise HTTPException(404, f"File not found on server: {path}")

    return FileResponse(path, media_type="image/jpeg")  # assume jpeg for now
