CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS unaccent;

CREATE TABLE IF NOT EXISTS kols (
  id BIGSERIAL PRIMARY KEY,
  platform TEXT NOT NULL,                -- 'tiktok', 'instagram', etc.
  username TEXT NOT NULL,                  -- username
  display_name TEXT,
  category TEXT[],                       -- e.g. {'beauty','cooking'}
  followers INTEGER,
  country TEXT,                          -- ISO-2 if known, e.g. 'TH'
  contact TEXT,                          -- email/line/url
  sample_links TEXT[],                   -- e.g. {'https://...','https://...'}
  extra JSONB,
  created_at TIMESTAMPTZ DEFAULT now()
);

-- Uniqueness for upsert
CREATE UNIQUE INDEX IF NOT EXISTS ux_kols_platform_username
  ON kols (platform, username);

-- Optional helpful indexes
CREATE INDEX IF NOT EXISTS idx_kols_text
  ON kols USING GIN ((unaccent(coalesce(display_name,'') || ' ' || coalesce(extra->>'bio',''))) gin_trgm_ops);

CREATE INDEX IF NOT EXISTS idx_kols_category ON kols USING GIN (category);