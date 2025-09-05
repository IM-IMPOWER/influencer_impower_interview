CREATE TABLE kol_media (
  id BIGSERIAL PRIMARY KEY,
  kol_id BIGINT REFERENCES kols(id),
  kind TEXT CHECK (kind IN ('profile','thumb')),
  path TEXT NOT NULL,
  emb VECTOR(512)    -- CLIP ViT-B/32 gives 512-dim
);
-- Fast ANN index (safe default: L2)
CREATE INDEX IF NOT EXISTS idx_kol_media_emb
  ON kol_media USING ivfflat (emb vector_l2_ops) WITH (lists = 100);

-- Prevent duplicate rows per image
CREATE UNIQUE INDEX IF NOT EXISTS ux_kol_media_kol_path
  ON kol_media (kol_id, path);

CREATE TABLE IF NOT EXISTS kol_signals (
  kol_id BIGINT PRIMARY KEY REFERENCES kols(id) ON DELETE CASCADE,
  image_centroid VECTOR(512),  -- mean embedding across all this KOL’s images
  image_count INTEGER DEFAULT 0
);

