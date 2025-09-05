# 📊 Database Schema

This document describes the core tables and relationships for the **KOL Automation Platform**.  
The schema is implemented in **PostgreSQL 15** with the **pgvector** extension enabled.

---

## Extensions

```sql
CREATE EXTENSION IF NOT EXISTS vector;  -- required for CLIP embeddings

Tables
1. kols

Stores the base profile information for each KOL.

Column	Type	Notes
id	BIGSERIAL PK	Unique identifier (auto-increment)
platform	TEXT	Social platform (tiktok, etc.)
username	TEXT	Normalized username (lowercased, without @)
display_name	TEXT	Display name (Porji, KOL_1, etc.)
category	TEXT[]	Categories/tags ({beauty, skincare})
followers	INTEGER	Current follower count
country	TEXT	ISO country code (e.g. TH)
contact	TEXT	Contact link (usually profile URL)
sample_links	TEXT[]	Example post/profile links
extra	JSONB	Arbitrary extra data
audience_th_pct	INTEGER	% of audience in Thailand (for budget optimizer)
created_at	TIMESTAMPTZ	Default now()

Indexes:

CREATE UNIQUE INDEX ON kols (platform, username);

2. kol_media

Stores profile images and thumbnails for each KOL.
Images are either stored in Cloud Storage (media_url) or referenced by a local file_path.
Embeddings are generated with CLIP and saved to pgvector for semantic search.

Column	Type	Notes
id	BIGSERIAL PK	Unique identifier
kol_id	BIGINT FK	References kols(id)
kind	TEXT	profile or thumb
path	TEXT	Local dev path (e.g., /data/split_images/...)
media_url	TEXT	Public GCS/Supabase URL
emb	VECTOR(512)	CLIP embedding vector

Indexes:

CREATE INDEX ON kol_media (kol_id);
CREATE INDEX ON kol_media USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

3. conversations

Tracks outreach conversations with KOLs.
Each KOL can have at most one active conversation in the PoC design.

Column	Type	Notes
id	BIGSERIAL PK	Unique identifier
kol_id	BIGINT FK	References kols(id)
channel	TEXT	Outreach channel (dm, email, etc.)
status	TEXT	Enum: contacted, negotiating, confirmed, closed
created_at	TIMESTAMPTZ	Default now()
last_meddage_at	TIMESTAMPTZ	Auto-updated with NOW()
proposed_deliverables TEXT proposed deliverables for creators
proposed_timeline TEXT proposed time period for content delivery
proposed _price_integer INT proposed price for creators
agreed_deliverables TEXT agreed deliverables for creators
agreed_timeline TEXT agreed time period for content delivery
agreed _price_integer INT agreed price for creators

Indexes:

CREATE INDEX ON conversations (kol_id);

4. messages

Individual messages inside a conversation.

Column	Type	Notes
id	BIGSERIAL PK	Unique identifier
conversation_id	BIGINT FK	References conversations(id)
direction	TEXT	out (our message) or in (their reply)
body	TEXT	Message text
created_at	TIMESTAMPTZ	Default now()

Indexes:

CREATE INDEX ON messages (conversation_id);

5. rate_cards

Stores KOL pricing and estimated reach for different deliverables.

Column	Type	Notes
id	BIGSERIAL PK	Unique identifier
kol_id	BIGINT FK	References kols(id)
item	TEXT	Deliverable: bundle, tiktok_video, etc.
price_integer	INTEGER	Price in local currency (THB)
est_reach	INTEGER	Estimated reach (usually derived from followers × heuristic %)

Indexes:

CREATE INDEX ON rate_cards (kol_id);

Relationships
erDiagram
    KOLS ||--o{ KOL_MEDIA : has
    KOLS ||--o{ CONVERSATIONS : has
    CONVERSATIONS ||--o{ MESSAGES : contains
    KOLS ||--o{ RATE_CARDS : offers

Notes

All IDs are BIGSERIAL primary keys.

All timestamps use TIMESTAMPTZ DEFAULT now().

pgvector used for semantic similarity search between briefs and media.

Images in production are served via GCS public URLs; file_path is only used in dev.

For PoC simplicity, optimizer uses greedy allocation on rate_cards.


