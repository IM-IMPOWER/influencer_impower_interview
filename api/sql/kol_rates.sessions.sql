-- One KOL can have multiple items (video, story, live, etc.)
CREATE TABLE IF NOT EXISTS rate_cards (
  id BIGSERIAL PRIMARY KEY,
  kol_id BIGINT NOT NULL REFERENCES kols(id) ON DELETE CASCADE,
  item TEXT NOT NULL,                   -- 'tiktok_video','tiktok_story','live','bundle', etc.
  price_integer INTEGER NOT NULL,       -- THB (whole number)
  est_reach INTEGER,                    -- optional per-item reach heuristic
  usage_rights TEXT,                    -- optional: usage/whitelisting terms
  exclusivity TEXT,                     -- optional: exclusivity notes
  lead_time_days INTEGER,               -- optional
  payment_terms TEXT,                   -- optional
  notes TEXT,                           -- optional free text
  UNIQUE (kol_id, item)
);

CREATE INDEX IF NOT EXISTS idx_rate_cards_kol ON rate_cards(kol_id);

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
