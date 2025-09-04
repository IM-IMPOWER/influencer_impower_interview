from fastapi import FastAPI
from routes_kols import router as kols_router
from routes_match import router as match_router
from routes_media import router as media_router
from routes_crm import router as crm_router
from routes_plan import router as plan_router




app = FastAPI(title="Influencer PoC API")

app.include_router(kols_router)
app.include_router(match_router)
app.include_router(media_router)
app.include_router(crm_router)
app.include_router(plan_router)

@app.get("/health")
def health():
    return {"status": "ok"}

# @app.get("/kols")
# def list_kols():
#     return [
#         {"handle": "@anahpfai", "platform": "tik_tok", "followers": 120000, "categories": ["beauty"]},
#         {"handle": "@ice_supathanes3", "platform": "tik_tok", "followers": 87000, "categories": ["food", "condo_cooking"]}
#     ]

@app.post("/match")
def match_brief(brief: dict):
    b = brief.get("brief", "").lower()
    if "beauty" in b:
        return {"results": ["@anahpfai"]}
    if "cooking" in b:
        return {"results": ["@ice_supathanes3"]}
    return {"results": []}