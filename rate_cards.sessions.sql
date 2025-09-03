INSERT INTO rate_cards (kol_id, item, price_integer, est_reach, notes)
SELECT
  ck.kol_id,
  'tiktok_video' AS item,
  GREATEST(5000, ROUND(ck.followers * 0.20))::int AS price_integer,
  (ck.followers * 0.20)::int                      AS est_reach,
  'Auto-estimated from followers'                 AS notes
FROM confirmed_kols ck
ON CONFLICT (kol_id, item) DO NOTHING;

WITH base AS (
  SELECT
    k.id,
    k.username,
    k.display_name,
    k.followers,
    k.category,
    COALESCE(rc.price_integer, 0)          AS price,
    COALESCE(rc.est_reach, k.followers/5)  AS est_reach  -- fallback heuristic
    -- COALESCE(k.audience_th_pct, 100)       AS th_pct
  FROM confirmed_kols ck
  JOIN kols k ON k.id = ck.kol_id
  LEFT JOIN LATERAL (
    SELECT price_integer, est_reach
    FROM rate_cards
    WHERE rate_cards.kol_id = k.id
      AND item IN ('bundle','tiktok_video')          -- preference order handled in ORDER below
    ORDER BY CASE item WHEN 'bundle' THEN 0 ELSE 1 END, id
    LIMIT 1
  ) rc ON TRUE
)
SELECT * FROM base;

SELECT k.display_name, rc.item, rc.price_integer, rc.est_reach
FROM kols k
JOIN rate_cards rc ON rc.kol_id = k.id
ORDER BY k.id, rc.item;
