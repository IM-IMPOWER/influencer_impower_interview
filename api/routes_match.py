from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
import os, psycopg
from psycopg.rows import dict_row
from pgvector.psycopg import register_vector

from services.clip_embed import embed_text

router = APIRouter(prefix="/api", tags=["match"])

def get_conn():
    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        raise RuntimeError("DATABASE_URL not set")
    conn = psycopg.connect(dsn)
    register_vector(conn)
    return conn

@router.get("/match")
def match_kols(
    brief: str = Query(..., min_length=1),
    limit: int = Query(12, ge=1, le=50),
):
    # 1) embed the brief
    bvec = embed_text(brief)

    # 2) fetch nearest by centroid if present
    with get_conn() as conn, conn.cursor(row_factory=dict_row) as cur:
        # Try centroid first
        cur.execute("""
            WITH ranked AS (
                SELECT
                    k.id,
                    k.username,
                    k.display_name,
                    k.followers,
                    k.platform,
                    k.country,
                    k.category,
                    ks.image_centroid AS emb,
                    1 - (ks.image_centroid <#> %(b)s) AS sim  -- cosine-like (if normalized)
                FROM kol_signals ks
                JOIN kols k ON k.id = ks.kol_id
                WHERE ks.image_centroid IS NOT NULL
                ORDER BY sim DESC
                LIMIT %(lim)s
            )
            SELECT * FROM ranked;
        """, {"b": bvec, "lim": limit*3})  # grab extra for re-scoring
        rows = cur.fetchall()

        # If no centroids yet, fall back to image-level (avg per KOL)
        if not rows:
            cur.execute("""
                WITH img AS (
                    SELECT
                        m.kol_id,
                        AVG(1 - (m.emb <#> %(b)s)) AS sim
                    FROM kol_media m
                    WHERE m.emb IS NOT NULL
                    GROUP BY m.kol_id
                    ORDER BY sim DESC
                    LIMIT %(lim)s
                )
                SELECT
                    k.id,
                    k.username,
                    k.display_name,
                    k.followers,
                    k.platform,
                    k.country,
                    k.category,
                    NULL AS emb,
                    img.sim
                FROM img
                JOIN kols k ON k.id = img.kol_id
                ORDER BY img.sim DESC
                LIMIT %(lim2)s;
            """, {"b": bvec, "lim": limit*5, "lim2": limit*3})
            rows = cur.fetchall()

        if not rows:
            return []

        # 3) light rule boosts (keywords in username/display_name/category)
        brief_lc = brief.lower()
        def rule_boost(r):
            boost = 0.0
            uname = (r["username"] or "").lower()
            dname = (r["display_name"] or "").lower()
            cats = r["category"] or []
            if any(tok in uname or tok in dname for tok in brief_lc.split()):
                boost += 0.05
            if any(c.lower() in brief_lc or brief_lc in c.lower() for c in cats):
                boost += 0.10
            # small follower prior so bigger accounts float slightly
            if r["followers"]:
                if r["followers"] >= 1_000_000: boost += 0.05
                elif r["followers"] >= 100_000: boost += 0.02
            return boost

        scored = []
        for r in rows:
            base = float(r["sim"]) if r["sim"] is not None else 0.0
            s = base + rule_boost(r)
            scored.append((s, r))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = [r for (_, r) in scored[:limit]]

        # 4) fetch 1–2 best sample images per KOL for "reasons"
        kol_ids = tuple([r["id"] for r in top])
        samples = {}
        if kol_ids:
            cur.execute(f"""
                SELECT
                    m.kol_id,
                    m.path,
                    (1 - (m.emb <#> %(b)s)) AS sim
                FROM kol_media m
                WHERE m.kol_id = ANY(%(ids)s)
                ORDER BY m.kol_id, sim DESC
            """, {"b": bvec, "ids": list(kol_ids)})
            for row in cur.fetchall():
                lst = samples.setdefault(row["kol_id"], [])
                if len(lst) < 2:  # keep top-2
                    lst.append(row["path"])

        # 5) format reasons
        out = []
        for r in top:
            imgs = samples.get(r["id"], [])[:2]
            reason = "High visual similarity to the brief"
            if r["category"]:
                # highlight any overlapping category words
                overlaps = [c for c in r["category"] if c and c.lower() in brief_lc]
                if overlaps:
                    reason += f"; category overlap: {', '.join(overlaps)}"
            out.append({
                "id": r["id"],
                "username": r["username"],
                "display_name": r["display_name"] or r["username"],
                "platform": r["platform"],
                "followers": r["followers"],
                "country": r["country"],
                "category": r["category"],
                "score": round(float(r["sim"]), 3) if r["sim"] is not None else None,
                "sample_images": imgs,
                "reason": reason,
                "profile_url": f"https://www.{r['platform']}.com/@{r['username']}",
            })
        return out
