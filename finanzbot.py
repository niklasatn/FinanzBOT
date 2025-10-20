import os, json, requests
import time
from google import genai
from pydantic import BaseModel
from typing import List, Literal

# ===== KONFIGURATION LADEN =====
with open("config.json", "r", encoding="utf-8") as f:
    CONFIG = json.load(f)

PORTFOLIO = {t.upper(): 0 for t in CONFIG["portfolio"]}
KEYWORDS = [k.lower() for k in CONFIG["keywords"]]
PROMPT = CONFIG["gemini_prompt"]

# ===== MODELDEFINITION =====
class ActionItem(BaseModel):
    position: str
    entscheidung: Literal["KAUFEN", "HALTEN", "VERKAUFEN"]
    vertrauen: float
    begruendung: str

class ModelOutput(BaseModel):
    analysen: List[ActionItem]

# ===== NEWS AUS ALPHAVANTAGE =====
def fetch_news_alphavantage(api_key: str, limit: int = 30):
    url = (
        f"https://www.alphavantage.co/query?"
        f"function=NEWS_SENTIMENT&sort=LATEST&limit={limit}&apikey={api_key}"
    )
    for attempt in range(3):  # bis zu 3 Versuche
        try:
            r = requests.get(url, timeout=20)
            r.raise_for_status()
            data = r.json()
            if "feed" in data:
                return data["feed"]
            print("⚠️ AlphaVantage liefert keine News:", data)
        except requests.exceptions.ReadTimeout:
            print(f"⏳ Timeout bei AlphaVantage (Versuch {attempt+1}/3)")
            time.sleep(10)
        except Exception as e:
            print("⚠️ API-Fehler:", e)
            time.sleep(5)
    print("❌ AlphaVantage nicht erreichbar.")
    return []

# ===== FILTERLOGIK =====
def is_relevant(item: dict) -> bool:
    title = item.get("title", "").lower()
    summary = item.get("summary", "").lower()
    combined = title + " " + summary
    tickers = [x.get("ticker", "").upper() for x in item.get("ticker_sentiment", [])]
    if any(t in PORTFOLIO for t in tickers):
        return True
    if any(k in combined for k in KEYWORDS):
        return True
    if any(term in combined for term in [
        "markets", "bond", "etf", "crypto", "real estate", "bank", "insurance", "government debt", "funds"
    ]):
        return True
    if any(term in combined for term in ["europe", "germany", "eu", "eurozone", "ecb", "eur"]):
        return True
    return False

# ===== GEMINI ANALYSE =====
def analyze_with_gemini(news_items: List[dict]) -> ModelOutput:
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    bullets = []
    for n in news_items:
        title = n.get("title", "")
        url = n.get("url", "")
        tickers = ", ".join(
            [x.get("ticker") for x in n.get("ticker_sentiment", []) if x.get("ticker")]
        )
        bullets.append(f"- {title} ({tickers or '—'})\n  Quelle: {url}")
    full_prompt = (
        PROMPT
        + "\n\nBitte gib die Antwort als JSON-Liste 'analysen' zurück, "
          "mit Feldern: position (string), entscheidung (KAUFEN/HALTEN/VERKAUFEN), "
          "vertrauen (0–100, Zahl ohne Prozentzeichen), begruendung (string, kurz und prägnant). "
          "Antworte ausschließlich in Deutsch.\n\n"
        + "\n".join(bullets)
    )
    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=full_prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": ModelOutput,
        },
    )
    return ModelOutput.model_validate_json(resp.text)

# ===== PUSHOVER =====
def send_pushover(message: str):
    token = os.getenv(CONFIG["pushover_token_env"])
    user = os.getenv(CONFIG["pushover_user_env"])
    payload = {"token": token, "user": user, "message": message}
    r = requests.post("https://api.pushover.net/1/messages.json", data=payload, timeout=20)
    r.raise_for_status()

# ===== MAIN =====
def main():
    api_key = os.getenv(CONFIG["sources"][0]["api_key_env"])
    news = fetch_news_alphavantage(api_key, CONFIG["sources"][0]["limit"])
    relevant = [n for n in news if is_relevant(n)]

    print(f"🔎 Relevante News: {len(relevant)}")
    if not relevant:
        print("Keine relevanten News, kein Push.")
        return

    result = analyze_with_gemini(relevant)
    if not result.analysen:
        print("Keine relevanten Analysen.")
        return

    analysierte_pos = {a.position.upper() for a in result.analysen}
    fehlende_pos = [p for p in PORTFOLIO if p not in analysierte_pos]
    for pos in fehlende_pos:
        result.analysen.append(
            ActionItem(
                position=pos,
                entscheidung="HALTEN",
                vertrauen=0,
                begruendung="Keine neuen Nachrichten."
            )
        )

    parts = []
    for a in result.analysen:
        v = a.vertrauen
        if v > 100: v /= 100
        if v <= 1: v *= 100
        v = f"{v:.0f}%"
        if a.entscheidung == "KAUFEN":
            s = "🟢"
        elif a.entscheidung == "VERKAUFEN":
            s = "🔴"
        else:
            s = "🟡"
        parts.append(f"{s} {a.position}: {a.entscheidung} ({v}) - {a.begruendung.strip()}")

    msg = "\n".join(parts)
    send_pushover(msg)
    print("✅ Push gesendet.")

if __name__ == "__main__":
    main()
