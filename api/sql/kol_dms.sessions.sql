-- Conversation log with each KOL
CREATE TABLE IF NOT EXISTS conversations (
  id BIGSERIAL PRIMARY KEY,
  kol_id BIGINT NOT NULL REFERENCES kols(id) ON DELETE CASCADE,
  status TEXT NOT NULL CHECK (status IN ('contacted','negotiating','confirmed','closed'))
    DEFAULT 'contacted',
  channel TEXT NOT NULL DEFAULT 'dm',
  last_message_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ DEFAULT now()
);

-- Messages inside each conversation
CREATE TABLE IF NOT EXISTS messages (
  id BIGSERIAL PRIMARY KEY,
  conversation_id BIGINT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
  direction TEXT NOT NULL CHECK (direction IN ('out','in')),
  body TEXT NOT NULL,
  created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_conv_kol ON conversations(kol_id);
CREATE INDEX IF NOT EXISTS idx_msg_conv ON messages(conversation_id);

ALTER TABLE conversations
  ADD COLUMN IF NOT EXISTS proposed_deliverables TEXT,
  ADD COLUMN IF NOT EXISTS proposed_timeline TEXT,
  ADD COLUMN IF NOT EXISTS proposed_price_integer INTEGER;  -- THB

ALTER TABLE conversations
  ADD COLUMN IF NOT EXISTS agreed_deliverables TEXT,
  ADD COLUMN IF NOT EXISTS agreed_timeline TEXT,
  ADD COLUMN IF NOT EXISTS agreed_price_integer INTEGER;