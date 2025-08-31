from fastapi import APIRouter, UploadFile, File, Query, Depends
from models import KOLIn, KOLOut, KOLQuery, ImportSummary
import csv, io, json
import psycopg
from typing import List, Optional
import os

router = APIRouter(prefix="/api/kols", tags=["kols"])

def get_conn():
    # read from env: PGHOST, PGUSER, PGPASSWORD, PGDATABASE
    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        raise RuntimeError("DATABASE_URL is not set")
    return psycopg.connect(dsn)

@router.post("/import", response_model=ImportSummary)
def import_kols(file: UploadFile = File(...)):
    content = file.file.read().decode("utf-8")
    reader = csv.DictReader(io.StringIO(content))
    inserted = updated = skipped = 0
    with get_conn() as conn, conn.cursor() as cur:
        for row in reader:
            # parse array-ish fields
            def parse_arr(v):
                if not v: return None
                v = v.strip()
                if v.startswith("{") and v.endswith("}"):
                    # postgres array literal already
                    return v
                try:
                    # try json array
                    arr = json.loads(v)
                    return "{" + ",".join([str(x) for x in arr]) + "}"
                except:
                    return "{" + ",".join([s.strip() for s in v.split(",") if s.strip()]) + "}"
            params = {
                "platform": row.get("platform","").strip().lower(),
                "username": row.get("username","").strip().lstrip("@").lower(),
                "display_name": row.get("display_name") or None,
                "category": parse_arr(row.get("category")),
                "followers": int(row["followers"]) if row.get("followers") else None,
                "country": row.get("country") or None,
                "contact": row.get("contact") or None,
                "sample_links": parse_arr(row.get("sample_links")),
                "extra": row.get("extra") or None,
            }
            if not params["platform"] or not params["username"]:
                skipped += 1
                continue
            cur.execute("""
                INSERT INTO kols (platform, username, display_name, category, followers, country, contact, sample_links, extra)
                VALUES (%(platform)s, %(username)s, %(display_name)s, %(category)s::text[], %(followers)s, %(country)s, %(contact)s, %(sample_links)s::text[], %(extra)s::jsonb)
                ON CONFLICT (platform, username) DO UPDATE
                SET display_name = EXCLUDED.display_name,
                    category     = EXCLUDED.category,
                    followers    = COALESCE(EXCLUDED.followers, kols.followers),
                    country      = COALESCE(EXCLUDED.country,  kols.country),
                    contact      = COALESCE(EXCLUDED.contact,  kols.contact),
                    sample_links = EXCLUDED.sample_links,
                    extra        = COALESCE(EXCLUDED.extra,     kols.extra)
                RETURNING (xmax = 0) AS inserted_flag;
            """, params)
            inserted_flag = cur.fetchone()[0]
            if inserted_flag:
                inserted += 1
            else:
                updated += 1
    return ImportSummary(inserted=inserted, updated=updated, skipped=skipped)

def tier_from_followers(f: Optional[int]) -> Optional[str]:
    if f is None: return None
    if f < 10_000: return "nano"
    if f < 100_000: return "micro"
    if f < 500_000: return "mid"
    if f < 1_000_000: return "macro"
    return "mega"

@router.get("", response_model=List[KOLOut])
def list_kols(
    q: Optional[str] = None,
    platform: Optional[str] = None,
    category: Optional[str] = None,
    country: Optional[str] = None,
    tier: Optional[str] = None,
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    wh, args = [], {}
    if q:
        wh.append("unaccent(display_name || ' ' || username) ILIKE unaccent(%(q)s)")
        args["q"] = f"%{q}%"
    if platform:
        wh.append("platform = %(platform)s")
        args["platform"] = platform
    if category:
        wh.append("%(category)s = ANY(category)")
        args["category"] = category
    if country:
        wh.append("country = %(country)s")
        args["country"] = country
    if tier:
        # emulate tier via followers
        bounds = {
            "nano": (0, 10_000),
            "micro": (10_000, 100_000),
            "mid": (100_000, 500_000),
            "macro": (500_000, 1_000_000),
            "mega": (1_000_000, 10_000_000_000),
        }[tier]
        wh.append("followers BETWEEN %(fmin)s AND %(fmax)s")
        args["fmin"], args["fmax"] = bounds
    where = ("WHERE " + " AND ".join(wh)) if wh else ""
    sql = f"""
        SELECT id, platform, username, display_name, category, followers, country, contact, sample_links, extra
        FROM kols
        {where}
        ORDER BY followers DESC NULLS LAST, id DESC
        LIMIT %(limit)s OFFSET %(offset)s;
    """
    args["limit"], args["offset"] = limit, offset
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, args)
        rows = cur.fetchall()
    out = []
    for r in rows:
        out.append({
            "id": r[0], "platform": r[1], "username": r[2], "display_name": r[3],
            "category": r[4], "followers": r[5], "country": r[6],
            "contact": r[7], "sample_links": r[8], "extra": r[9],
        })
    return out
