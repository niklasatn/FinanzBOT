import os, json, requests
from google import genai
from pydantic import BaseModel
from typing import List

# ===== KONFIGURATION LADEN =====
with open("config.json", "r", encoding="utf-8") as f:
    CONFIG = json.load(f)

KEYWORDS = [k.lower() for k in CONFIG["keywords"]]
PROMPT = CONFIG["gemini_prompt"]

# ===== MODELDEFINITION =====
class IdeaItem(BaseModel):
    name: str
    typ: str
    begruendung: str
    vertrauen: float

class IdeaOutput(BaseModel):
    ideen: List[IdeaItem]

# ===== NEWS AUS ALPHAVANTAGE =====
def fetch_news_alphavantage(api_key: str, limit: int = 30):
    url = (
        f"https://www.alphavantage.co/query?"
        f"function=NEWS_SENTIMENT&sort=LATEST&limit={limit}&apikey={api_key}"
    )
    try:
        r = requests.get(url, timeout=25)
        r.raise_for_status()
        return r.json().get("feed", [])
    except Exception as e:
        print("⚠️ Fehler beim Abrufen der News:", e)
        return []

# ===== FILTERLOGIK =====
def is_relevant(item: dict) -> bool:
    text = (item.get("title", "") + " " + item.get("summary", "")).lower()
    return any(k in text for k in KEYWORDS)

# ===== GEMINI ANALYSE =====
def analyze_with_gemini(news_items: List[dict]) -> IdeaOutput:
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
        + "\n\nLies die News und schlage bis zu 5 neue, interessante Finanzprodukte vor "
          "(Aktien, ETFs, Kryptos, Fonds usw.), die aktuell Potenzial haben könnten. "
          "Gib die Antwort als JSON-Liste 'ideen' mit Feldern: "
          "name (string), typ (z. B. Aktie/ETF/Krypto/Fonds), begruendung (kurz, deutsch), "
          "vertrauen (0–100, Zahl ohne %). Antworte nur im JSON-Format.\n\n"
        + "\n".join(bullets)
    )

    resp = client.generate_content(
        model="gemini-2.5-flash",
        contents=full_prompt,
        response_mime_type="application/json"
    )

    try:
        return IdeaOutput.model_validate_json(resp.candidates[0].content.parts[0].text)
    except Exception as e:
        print("⚠️ Fehler beim Parsen der Gemini-Antwort:", e)
        print("Antwort:", resp.candidates[0].content.parts[0].text)
        return IdeaOutput(ideen=[])

# ===== PUSHOVER =====
def send_pushover(message: str):
    token = os.getenv(CONFIG["pushover_token_env"])
    user = os.getenv(CONFIG["pushover_user_env"])
    payload = {"token": token, "user": user, "message": message}
    try:
        r = requests.post("https://api.pushover.net/1/messages.json", data=payload, timeout=15)
        r.raise_for_status()
    except Exception as e:
        print("⚠️ Fehler beim Senden an Pushover:", e)

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
    if not result.ideen:
        print("Keine neuen Anlageideen erkannt.")
        return

    # Nachricht komprimiert aufbauen
    parts = []
    for i in result.ideen:
        v = i.vertrauen
        if v > 100: v /= 100
        if v <= 1: v *= 100
        parts.append(f"🟢 {i.name} ({i.typ}) {v:.0f}% - {i.begruendung.strip()}")

    msg = "\n".join(parts)
    send_pushover(msg)
    print("✅ Neue Anlageideen gesendet.")

if __name__ == "__main__":
    main()
