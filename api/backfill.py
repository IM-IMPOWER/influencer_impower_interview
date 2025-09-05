#!/usr/bin/env python3
import os
import sys
import argparse
from typing import Optional

import psycopg
from psycopg.rows import dict_row


DDL_RATE_CARDS = """
CREATE TABLE IF NOT EXISTS rate_cards (
  id BIGSERIAL PRIMARY KEY,
  kol_id BIGINT NOT NULL REFERENCES kols(id) ON DELETE CASCADE,
  item TEXT NOT NULL,                    -- e.g. 'tiktok_video','bundle','live'
  price_integer INTEGER NOT NULL,        -- THB
  est_reach INTEGER,                     -- optional heuristic
  usage_rights TEXT,
  exclusivity TEXT,
  lead_time_days INTEGER,
  payment_terms TEXT,
  notes TEXT,
  UNIQUE (kol_id, item)
);

CREATE INDEX IF NOT EXISTS idx_rate_cards_kol ON rate_cards(kol_id);
"""

VIEW_CONFIRMED = """
CREATE OR REPLACE VIEW confirmed_kols AS
SELECT
  k.id AS kol_id,
  k.platform,
  k.username,
  k.display_name,
  k.followers,
  k.country,
  k.category,
  c.id AS conversation_id,
  c.status,
  c.agreed_price_integer,
  c.agreed_deliverables,
  c.agreed_timeline
FROM kols k
JOIN conversations c ON c.kol_id = k.id
WHERE c.status = 'confirmed';
"""


def connect(dsn: Optional[str]):
    dsn = dsn or os.getenv("DATABASE_URL")
    if not dsn:
        print("ERROR: Provide --dsn or set DATABASE_URL", file=sys.stderr)
        sys.exit(1)
    return psycopg.connect(dsn, row_factory=dict_row)


def heur_price(followers: Optional[int]) -> int:
    f = int(followers or 0)
    return max(5000, int(round(f * 0.20)))  # floor 5k, else 0.20 THB per follower


def heur_reach_video(followers: Optional[int]) -> int:
    f = int(followers or 0)
    return int(round(f * 0.20))  # ~20% of followers


def heur_reach_bundle(followers: Optional[int]) -> int:
    f = int(followers or 0)
    return int(round(f * 0.25))  # slightly higher for bundle


def upsert_rate(cur, kol_id: int, item: str, price: int, est_reach: Optional[int], notes: str, update_existing: bool):
    if update_existing:
        cur.execute(
            """
            INSERT INTO rate_cards (kol_id, item, price_integer, est_reach, notes)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (kol_id, item)
            DO UPDATE SET
              price_integer = EXCLUDED.price_integer,
              est_reach = EXCLUDED.est_reach,
              notes = EXCLUDED.notes
            """,
            (kol_id, item, price, est_reach, notes),
        )
    else:
        cur.execute(
            """
            INSERT INTO rate_cards (kol_id, item, price_integer, est_reach, notes)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (kol_id, item) DO NOTHING
            """,
            (kol_id, item, price, est_reach, notes),
        )


def main():
    ap = argparse.ArgumentParser(description="Backfill default rate cards for confirmed KOLs")
    ap.add_argument("--dsn", help="Postgres DSN, e.g. postgresql://postgres:postgres@localhost:5432/app")
    ap.add_argument("--dry-run", action="store_true", help="Print actions without writing")
    ap.add_argument("--update-existing", action="store_true", help="Update existing rate card rows (otherwise DO NOTHING)")
    args = ap.parse_args()

    with connect(args.dsn) as conn:
        with conn.cursor() as cur:
            # Ensure schema pieces
            cur.execute(DDL_RATE_CARDS)
            cur.execute(VIEW_CONFIRMED)

        with conn.cursor() as cur:
            # Pull confirmed KOLs
            cur.execute("""
                SELECT kol_id, username, display_name, followers, agreed_price_integer
                FROM confirmed_kols
                ORDER BY kol_id
            """)
            rows = cur.fetchall()

        if not rows:
            print("No confirmed KOLs found. Confirm some in /crm first.")
            return

        created = updated = skipped = bundle_created = 0

        with conn.cursor() as cur:
            for r in rows:
                kol_id = r["kol_id"]
                username = r["username"]
                dname = r["display_name"] or username
                followers = r["followers"] or 0
                agreed = r["agreed_price_integer"]

                # Default tiktok_video rate
                price = heur_price(followers)
                reach = heur_reach_video(followers)
                note = "Auto-estimated from followers"

                if args.dry_run:
                    print(f"DRY-RUN video: kol={kol_id} @{username} price={price} reach={reach}")
                else:
                    before = cur.rowcount
                    upsert_rate(cur, kol_id, "tiktok_video", price, reach, note, args.update_existing)
                    # We can't trust rowcount across upsert reliably; just count actions
                    created += 1

                # Optional 'bundle' from agreed price if present
                if agreed is not None:
                    b_reach = heur_reach_bundle(followers)
                    b_note = "From confirmed agreement"
                    if args.dry_run:
                        print(f"DRY-RUN bundle: kol={kol_id} @{username} price={agreed} reach={b_reach}")
                    else:
                        upsert_rate(cur, kol_id, "bundle", int(agreed), b_reach, b_note, args.update_existing)
                        bundle_created += 1

        if not args.dry_run:
            conn.commit()

    if args.dry_run:
        print("Done (dry-run).")
    else:
        print(f"Done. Upserted video rates for {created} KOLs; bundle rows (from agreed) for {bundle_created} KOLs.")
        

if __name__ == "__main__":
    main()
