from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import os, psycopg
from psycopg.rows import dict_row
import torch
import clip
from PIL import Image

router = APIRouter(prefix="/api", tags=["plan"])
_device = "cuda" if torch.cuda.is_available() else "cpu"
_clip_model, _clip_preprocess = clip.load("ViT-B/32", device=_device)

def _encode_text(text: str):
    with torch.no_grad():
        tokens = clip.tokenize([text]).to(_device)
        feats = _clip_model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats[0].cpu().numpy()  # 512-d np.array

def get_conn():
    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        raise RuntimeError("DATABASE_URL not set")
    return psycopg.connect(dsn, row_factory=dict_row)

# ----------- Inputs / Outputs -----------

class PlanRequest(BaseModel):
    total_budget: int = Field(gt=0, description="Total budget in THB")
    min_reach: Optional[int] = Field(default=None, ge=0)
    categories: Optional[List[str]] = None          # match if overlaps any
    min_th_audience: Optional[int] = Field(default=None, ge=0, le=100)  # audience_th_pct >= this

class PlanItem(BaseModel):
    kol_id: int
    username: str
    display_name: Optional[str] = None
    item: str                      # 'bundle' or 'tiktok_video'
    price: int
    est_reach: int
    match_score: float             # from your PoC #2 if you later include it (placeholder here)
    roi: float                     # est_reach / price
    rationale: str

class PlanResponse(BaseModel):
    brief: Optional[str] = None
    total_budget: int
    total_spend: int
    est_total_reach: int
    items: List[PlanItem]

class PlanRequest(BaseModel):
    total_budget: int
    brief: Optional[str] = None          # NEW: free-text brief
    top_k: int = 100                     # NEW: shortlist size before budgeting
    min_reach: Optional[int] = None
    categories: Optional[List[str]] = None  # keep if you still want it
    min_th_audience: Optional[int] = None

# ----------- Endpoint -----------

@router.post("/plan", response_model=PlanResponse)
def build_plan(req: PlanRequest):
    # ---- optional: encode brief to vector (for shortlist) ----
    brief_vec = None
    if req.brief and req.brief.strip():
        brief_vec = _encode_text(req.brief.strip()).tolist()  # 512-d list

    # ---- filters shared across both paths ----
    args = {}
    where_extra = ["c.status = 'confirmed'"]  # only confirmed KOLs
    if req.categories:
        where_extra.append("k.category && %(cats)s::text[]")
        args["cats"] = req.categories
    if req.min_th_audience is not None:
        where_extra.append("COALESCE(k.audience_th_pct, 0) >= %(min_th)s")
        args["min_th"] = req.min_th_audience

    where_sql = " AND ".join(where_extra)

    with get_conn() as conn, conn.cursor() as cur:
        if brief_vec is not None:
            # ---- Path A: shortlist by CLIP similarity to the brief, then prefer bundle>video ----
            sql = f"""
            WITH sims AS (
              SELECT km.kol_id, 
              1 - (km.emb <#> %(vec)s::vector) AS score
              FROM kol_media km
              WHERE km.kind IN ('profile','thumb')
            ),
            agg AS (
              SELECT kol_id, MAX(score) AS match_score
              FROM sims
              GROUP BY kol_id
            ),
            confirmed AS (
              SELECT k.id AS kol_id, k.username, k.display_name, k.followers, k.category,
                     COALESCE(k.audience_th_pct, 100) AS th_pct
              FROM kols k
              JOIN conversations c ON c.kol_id = k.id
              WHERE {where_sql}
            ),
            preferred_rate AS (
              SELECT
                cf.kol_id, cf.username, cf.display_name, cf.followers, cf.category, cf.th_pct,
                a.match_score,
                (SELECT rc.item FROM rate_cards rc
                  WHERE rc.kol_id = cf.kol_id AND rc.item IN ('bundle','tiktok_video')
                  ORDER BY CASE rc.item WHEN 'bundle' THEN 0 ELSE 1 END, rc.id
                  LIMIT 1) AS item,
                (SELECT rc.price_integer FROM rate_cards rc
                  WHERE rc.kol_id = cf.kol_id AND rc.item IN ('bundle','tiktok_video')
                  ORDER BY CASE rc.item WHEN 'bundle' THEN 0 ELSE 1 END, rc.id
                  LIMIT 1) AS price,
                (SELECT COALESCE(rc.est_reach, GREATEST(1, (cf.followers/5))) FROM rate_cards rc
                  WHERE rc.kol_id = cf.kol_id AND rc.item IN ('bundle','tiktok_video')
                  ORDER BY CASE rc.item WHEN 'bundle' THEN 0 ELSE 1 END, rc.id
                  LIMIT 1) AS est_reach
              FROM confirmed cf
              JOIN agg a ON a.kol_id = cf.kol_id
            )
            SELECT *
            FROM preferred_rate
            WHERE price IS NOT NULL AND price > 0 AND est_reach IS NOT NULL AND est_reach > 0
            ORDER BY match_score DESC, (est_reach::numeric / price::numeric) DESC, followers DESC
            LIMIT %(top_k)s;
            """
            cur.execute(sql, {**args, "vec": brief_vec, "top_k": getattr(req, "top_k", 100)})
        else:
            # ---- Path B: no brief → rank by ROI (your original logic) ----
            sql = f"""
            WITH preferred_rate AS (
              SELECT
                k.id        AS kol_id,
                k.username,
                k.display_name,
                k.followers,
                k.category,
                COALESCE(k.audience_th_pct, 100) AS th_pct,
                1.0 AS match_score,
                (SELECT rc.item
                   FROM rate_cards rc
                  WHERE rc.kol_id = k.id
                    AND rc.item IN ('bundle','tiktok_video')
                  ORDER BY CASE rc.item WHEN 'bundle' THEN 0 ELSE 1 END, rc.id
                  LIMIT 1) AS item,
                (SELECT rc.price_integer
                   FROM rate_cards rc
                  WHERE rc.kol_id = k.id
                    AND rc.item IN ('bundle','tiktok_video')
                  ORDER BY CASE rc.item WHEN 'bundle' THEN 0 ELSE 1 END, rc.id
                  LIMIT 1) AS price,
                (SELECT COALESCE(rc.est_reach, GREATEST(1, (k.followers/5)))
                   FROM rate_cards rc
                  WHERE rc.kol_id = k.id
                    AND rc.item IN ('bundle','tiktok_video')
                  ORDER BY CASE rc.item WHEN 'bundle' THEN 0 ELSE 1 END, rc.id
                  LIMIT 1) AS est_reach
              FROM kols k
              JOIN conversations c ON c.kol_id = k.id
              WHERE {where_sql}
            )
            SELECT *
            FROM preferred_rate
            WHERE price IS NOT NULL AND price > 0 AND est_reach IS NOT NULL AND est_reach > 0
            ORDER BY (est_reach::numeric / price::numeric) DESC, followers DESC, kol_id ASC;
            """
            cur.execute(sql, args)

        rows = cur.fetchall()

    if not rows:
        raise HTTPException(400, "No eligible KOLs found. Confirm some in /crm, add rate cards, or relax filters.")

    # ---- Greedy allocation by ROI within the shortlist ----
    budget_left = req.total_budget
    items: List[PlanItem] = []
    total_spend = 0
    total_reach = 0

    for r in rows:
        if budget_left <= 0:
            break
        price = int(r["price"])
        est_reach = int(r["est_reach"])
        if price <= 0 or est_reach <= 0:
            continue
        if price > budget_left:
            # skip partial buys for PoC simplicity
            continue

        roi = float(est_reach) / float(price)
        match_score = float(r.get("match_score", 1.0))
        rationale = rationale_for(r, roi, req)  # consider showing match score inside this

        items.append(PlanItem(
            kol_id = r["kol_id"],
            username = r["username"],
            display_name = r["display_name"],
            item = r["item"],
            price = price,
            est_reach = est_reach,
            match_score = round(match_score, 4),
            roi = round(roi, 4),
            rationale = rationale
        ))
        budget_left -= price
        total_spend += price
        total_reach += est_reach

        if req.min_reach is not None and total_reach >= req.min_reach:
            break

    if not items:
        raise HTTPException(400, "Budget too small for available rate cards under current filters. Try increasing budget or relaxing filters.")

    return PlanResponse(
        brief=(req.brief or None),
        total_budget=req.total_budget,
        total_spend=total_spend,
        est_total_reach=total_reach,
        items=items
    )


def rationale_for(r, roi: float, req: PlanRequest) -> str:
    bits = []
    if r.get("category"):
        bits.append(f"Categories: {', '.join(r['category'])}")
    th_pct = r.get("th_pct")
    if th_pct is not None:
        bits.append(f"TH audience ≈ {th_pct}%")
    if r.get("item"):
        bits.append(f"Picked {r['item']}")
    if "match_score" in r and r["match_score"] is not None:
        try:
            bits.append(f"Match {float(r['match_score']):.2f}")
        except Exception:
            pass
    bits.append(f"ROI {roi:.2f} reach/THB")
    if req.min_reach:
        bits.append(f"toward {req.min_reach:,} reach target")
    return " · ".join(bits)

