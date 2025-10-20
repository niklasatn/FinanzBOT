import os, json, requests
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
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json().get("feed", [])

# ===== FILTERLOGIK =====
def is_relevant(item: dict) -> bool:
    title = item.get("title", "").lower()
    summary = item.get("summary", "").lower()
    combined = title + " " + summary

    tickers = [x.get("ticker", "").upper() for x in item.get("ticker_sentiment", [])]

    # Portfolio-Bezug
    if any(t in PORTFOLIO for t in tickers):
        return True
    # Keywords
    if any(k in combined for k in KEYWORDS):
        return True
    # Branchen / Makro
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

    # Hinweis an Gemini für sauberes JSON + Prozentwerte 0–100
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

# ===== PUSHOVER BENACHRICHTIGUNG =====
def send_pushover(message: str):
    token = os.getenv(CONFIG["pushover_token_env"])
    user = os.getenv(CONFIG["pushover_user_env"])
    payload = {
        "token": token,
        "user": user,
        "message": message,
        "title": "Finanzbot – Neue Analyse",
        "html": 1  # Aktiviert HTML/Formatierung in Pushover
    }
    r = requests.post("https://api.pushover.net/1/messages.json", data=payload, timeout=20)
    r.raise_for_status()

# ===== HAUPTLAUF =====
def main():
    # === 1. News abrufen ===
    api_key = os.getenv(CONFIG["sources"][0]["api_key_env"])
    news = fetch_news_alphavantage(api_key, CONFIG["sources"][0]["limit"])
    relevant = [n for n in news if is_relevant(n)]

    print(f"🔎 Relevante News gefunden: {len(relevant)}")
    if not relevant:
        print("Keine relevanten News, kein Push gesendet.")
        return

    # === 2. Gemini-Analyse ===
    result = analyze_with_gemini(relevant)
    if not result.analysen:
        print("Keine relevanten Analysen erhalten.")
        return

    # === 2b. Fehlende Positionen markieren ===
    analysierte_pos = {a.position.upper() for a in result.analysen}
    fehlende_pos = [p for p in PORTFOLIO if p not in analysierte_pos]

    for pos in fehlende_pos:
        result.analysen.append(
            ActionItem(
                position=pos,
                entscheidung="HALTEN",
                vertrauen=0,
                begruendung="Keine aktuellen Nachrichten oder Signale verfügbar."
            )
        )

    # === 3. Nachricht zusammenbauen ===
    message_parts = ["📊 <b>Finanzbot – Portfolio-Update</b>\n"]

    for a in result.analysen:
        vertr = a.vertrauen
        if vertr > 100:
            vertr = vertr / 100
        if vertr <= 1:
            vertr = vertr * 100
        vertr_str = f"{vertr:.0f}%"

        if a.entscheidung == "KAUFEN":
            symbol = "🟢"
        elif a.entscheidung == "VERKAUFEN":
            symbol = "🔴"
        else:
            symbol = "🟡"  # Halten = Gelb

        message_parts.append(
            f"{symbol} <b>{a.position}</b>: {a.entscheidung} ({vertr_str})\n"
            f"{a.begruendung.strip()}\n"
        )

    final_message = "\n\n".join(message_parts)

    # === 4. Push senden ===
    send_pushover(final_message)
    print("✅ Nachricht erfolgreich gesendet (inkl. Hinweis auf fehlende News).")

if __name__ == "__main__":
    main()
