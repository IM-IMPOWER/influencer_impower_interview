from fastapi import FastAPI

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/kols")
def list_kols():
    return [
        {"handle": "@anahpfai", "platform": "tik_tok", "followers": 120000, "categories": ["beauty"]},
        {"handle": "@ice_supathanes3", "platform": "tik_tok", "followers": 87000, "categories": ["food", "condo_cooking"]}
    ]

@app.post("/match")
def match_brief(brief: dict):
    b = brief.get("brief", "").lower()
    if "beauty" in b:
        return {"results": ["@anahpfai"]}
    if "cooking" in b:
        return {"results": ["@ice_supathanes3"]}
    return {"results": []}