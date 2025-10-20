import os, json, requests
from google import genai
from pydantic import BaseModel
from typing import List, Literal

# ======= KONFIGURATION LADEN =======
with open("config.json", "r", encoding="utf-8") as f:
    CONFIG = json.load(f)

PORTFOLIO = {t.upper(): 0 for t in CONFIG["portfolio"]}
KEYWORDS = [k.lower() for k in CONFIG["keywords"]]
PROMPT = CONFIG["gemini_prompt"]

# ======= MODELLE =======
class ActionItem(BaseModel):
    ticker: str
    action: Literal["BUY", "SELL", "HOLD"]
    confidence: float
    rationale: str

class ModelOutput(BaseModel):
    actions: List[ActionItem]

# ======= ALPHA VANTAGE NEWS =======
def fetch_news_alphavantage(api_key: str, limit: int = 30):
    url = (
        f"https://www.alphavantage.co/query?"
        f"function=NEWS_SENTIMENT&sort=LATEST&limit={limit}&apikey={api_key}"
    )
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json().get("feed", [])

# ======= RELEVANZ-PRÜFUNG =======
def is_relevant(item: dict) -> bool:
    title = item.get("title", "").lower()
    tickers = [x.get("ticker", "").upper() for x in item.get("ticker_sentiment", [])]
    # Portfolio-Ticker?
    if any(t in PORTFOLIO for t in tickers):
        return True
    # Keyword?
    if any(k in title for k in KEYWORDS):
        return True
    return False

# ======= GEMINI ANALYSE =======
def analyze_with_gemini(news_items: List[dict]) -> ModelOutput:
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    bullets = []
    for n in news_items:
        tickers = ", ".join([x.get("ticker") for x in n.get("ticker_sentiment", []) if x.get("ticker")]) or "—"
        bullets.append(f"- {n.get('title')} (tickers: {tickers})")
    content = PROMPT + "\n\n" + "\n".join(bullets)
    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=content,
        config={
            "response_mime_type": "application/json",
            "response_schema": ModelOutput,
        },
    )
    return ModelOutput.model_validate_json(resp.text)

# ======= PUSHOVER =======
def send_pushover(message: str):
    token = os.getenv(CONFIG["pushover_token_env"])
    user = os.getenv(CONFIG["pushover_user_env"])
    payload = {"token": token, "user": user, "message": message, "title": "Finanzbot"}
    r = requests.post("https://api.pushover.net/1/messages.json", data=payload, timeout=20)
    r.raise_for_status()

# ======= MAIN =======
def main():
    # Quelle: Alpha Vantage
    api_key = os.getenv(CONFIG["sources"][0]["api_key_env"])
    news = fetch_news_alphavantage(api_key, CONFIG["sources"][0]["limit"])
    relevant = [n for n in news if is_relevant(n)]
    if not relevant:
        return

    result = analyze_with_gemini(relevant)
    if not result.actions:
        return

    msg = "\n".join(
        f"{a.ticker}: {a.action} ({a.confidence:.0%}) – {a.rationale}"
        for a in result.actions
    )
    send_pushover(msg)

if __name__ == "__main__":
    main()
