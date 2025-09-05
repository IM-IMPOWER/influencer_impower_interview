# KOL Automation Platform

An experimental outreach and budget planning tool for managing Key Opinion Leaders (KOLs).  
Built with **FastAPI**, **Next.js**, **PostgreSQL + pgvector**, and containerized with **Docker**.  
Deployed on **Google Cloud (Cloud SQL + Cloud Run + Firebase Hosting)**.

---

## ✨ Features

- **PoC #1 – Basic Directory**
  - Seed & store KOL profiles in PostgreSQL
  - Expose API endpoints (`/api/kols`) to list/search KOLs
  - Simple Next.js UI page to view KOLs

- **PoC #2 – Matching Engine**
  - Pre-scraped images (profile + thumbnails) embedded with [OpenAI CLIP](https://github.com/openai/CLIP)
  - Store image embeddings in `kol_media` table (pgvector)
  - Text-to-image semantic search (match KOLs to a brief)

- **PoC #3 – Outreach CRM**
  - Manage outreach conversations with KOLs
  - CRUD for conversations and messages
  - UI: inbox-like CRM view with stages (`contacted`, `negotiating`, `confirmed`, `closed`)
  - Message templates for faster outreach

- **PoC #4 – Budget Optimizer**
  - Collect KOL rate cards (`bundle` / `tiktok_video`)
  - Input a total budget, category filters, Thai audience %, and brief text
  - Greedy allocator suggests plan: which KOLs to book, at what spend, with estimated reach
  - UI: interactive planner with inputs + CSV export

---

## 🏗️ Architecture

+-------------------+ +------------------+ +-------------------+
| Next.js (UI) | <--> | FastAPI (API) | <--> | PostgreSQL+pgvector |
| Firebase Hosting | | Cloud Run | | Cloud SQL (GCP) |
+-------------------+ +------------------+ +-------------------+

## Media assets:
Dev: local /data/split_images

---

## ⚙️ Local Development

### 1. Prereqs
- Docker & Docker Compose
- Python 3.10+ (for scripts)
- Node.js 18+ (Next.js UI)

### 2. Run services
```bash
docker compose up --build
```
### This will start
Postgres with pgvector
FastAPI at http://localhost:8000
Next.js UI at http://localhost:8000

### Load seed data 
python data/seed_kols.py --dsn postgresql://postgres:postgres@localhost:5432/app
python scripts/ingest_images.py --dsn postgresql://postgres:postgres@localhost:5432/app

## Key Directories
api/ FastAPI service
ui/ Next.js frontend
data/split_images - image folders (profile + thubnails)
