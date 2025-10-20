import os
import json
import requests
from google import genai
from pydantic import BaseModel
from typing import List, Literal

# ====== Konfiguration laden ======
with open("config.json", "r", encoding="utf-8") as f:
    CONFIG = json.load(f)

PORTFOLIO = {t.upper(): 0 for t in CONFIG["portfolio"]}
KEYWORDS = [k.lower() for k in CONFIG["keywords"]]
PROMPT = CONFIG["gemini_prompt"]

# ====== Modelldefinition for Gemini ======
class ActionItem(BaseModel):
    ticker: str
    action: Literal["BUY", "SELL", "HOLD"]
    confidence: float
    rationale: str

class ModelOutput(BaseModel):
    actions: List[ActionItem]

# ====== Funktionen zum Einlesen der Quellen ======
def fetch_truthsocial_posts(profile: str, api_key: str, limit: int = 10):
    # Beispiel-URL einer Scraper-API; du musst sie auf deine API anpassen
    url = f"https://api.example-scraper.com/truthsocial/profile/{profile}"
    params = {"api_key": api_key, "limit": limit}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    # Angenommen: data["posts"] ist Liste von Beiträgen mit Feldern ["id","content","date"]
    posts = []
    for p in data.get("posts", []):
        posts.append({"id": p.get("id"), "title": p.get("content"), "url": p.get("url", ""), "tickers": []})
    return posts

def get_all_items():
    all_items = []
    for src in CONFIG["sources"]:
        if src["type"] == "truthsocial_profile":
            api_key = os.getenv(src["api_key_env"])
            items = fetch_truthsocial_posts(src["profile"], api_key, src.get("limit", 10))
            all_items += items
        # Hier weitere Quellen ergänzen, wenn nötig
    return all_items

# ====== Relevanzprüfung (Pre-Filter) ======
def is_potentially_relevant(item: dict) -> bool:
    title = item.get("title", "").lower()
    # prüfe Keywords
    if any(k in title for k in KEYWORDS):
        return True
    # prüfe Portfolio-Ticker im Text (ganz einfach)
    for t in PORTFOLIO:
        if t.lower() in title:
            return True
    return False

# ====== Analyse mit Gemini ======
def analyze_with_gemini(items: List[dict]) -> ModelOutput:
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    bullet_lines = []
    for i in items:
        bullet_lines.append(f"- {i.get('title')} (url: {i.get('url')})")
    content = PROMPT + "\n\n" + "\n".join(bullet_lines)
    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=content,
        config={
            "response_mime_type": "application/json",
            "response_schema": ModelOutput,
        },
    )
    return ModelOutput.model_validate_json(resp.text)

# ====== Push via Pushover ======
def send_pushover(message: str):
    token = os.getenv(CONFIG["pushover_token_env"])
    user = os.getenv(CONFIG["pushover_user_env"])
    url = "https://api.pushover.net/1/messages.json"
    payload = {"token": token, "user": user, "message": message, "title": "Finanzbot Alert"}
    r = requests.post(url, data=payload, timeout=20)
    r.raise_for_status()

# ====== Hauptfunktion ======
def main():
    items = get_all_items()
    relevant = [it for it in items if is_potentially_relevant(it)]
    if not relevant:
        return
    result = analyze_with_gemini(relevant)
    if not result.actions:
        return
    msg_lines = []
    for a in result.actions:
        msg_lines.append(f"{a.ticker}: {a.action} ({a.confidence:.0%}) – {a.rationale}")
    send_pushover("\n".join(msg_lines))

if __name__ == "__main__":
    main()
