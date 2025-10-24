import os, json, requests, importlib, time

from pydantic import BaseModel
from typing import List


# ===== KONFIGURATION LADEN =====
with open("config.json", "r", encoding="utf-8") as f:
    CONFIG = json.load(f)

KEYWORDS = [k.lower() for k in CONFIG["keywords"]]
PROMPTS = CONFIG["prompts"]
PUSHOVER_TOKEN = os.getenv(CONFIG["pushover_token_env"])
PUSHOVER_USER = os.getenv(CONFIG["pushover_user_env"])
STATE_FILE = "last_sent.json"
CHAR_LIMIT = 950  # Sicherheitslimit für Pushover


# ===== MODELDEFINITION =====
class IdeaItem(BaseModel):
    name: str
    typ: str
    begruendung: str
    vertrauen: float


class IdeaOutput(BaseModel):
    ideen: List[IdeaItem]


# ===== LETZTE NACHRICHTEN SPEICHERN =====
def load_last_ids():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return set(json.load(f))
    return set()


def save_last_ids(ids):
    with open(STATE_FILE, "w") as f:
        json.dump(list(ids), f)


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


# ===== RELEVANZFILTER =====
def is_relevant(item: dict) -> bool:
    title = (item.get("title", "") + " " + item.get("summary", "")).lower()
    sentiment = float(item.get("overall_sentiment_score", 0))
    relevance = float(item.get("relevance_score", 0))
    # nur News mit deutlicher Auswirkung
    return (
        abs(sentiment) > 0.35
        or relevance > 0.5
        or any(k in title for k in KEYWORDS)
    )


# ===== GEMINI ANALYSE =====
def analyze_with_gemini(news_items: List[dict]) -> IdeaOutput:
    # dynamische Unterstützung beider Gemini-SDKs
    raw = ""
    try:
        if importlib.util.find_spec("google.genai"):
            from google import genai
            client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
            bullets = []
            for n in news_items:
                title = n.get("title", "")
                url = n.get("url", "")
                tickers = ", ".join(
                    [x.get("ticker") for x in n.get("ticker_sentiment", []) if x.get("ticker")]
                )
                bullets.append(f"- {title} ({tickers or '—'})\n  Quelle: {url}")

            prompt = (
                PROMPTS["main"]
                + "\n\n"
                + PROMPTS["format"]
                + "\n\n"
                + "\n".join(bullets)
            )

            resp = client.models.generate_content(
                model="gemini-2.0-flash-exp",
                contents=prompt,
                config={"response_mime_type": "application/json"},
            )
            raw = resp.text
        else:
            import google.generativeai as genai
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            model = genai.GenerativeModel("gemini-1.5-flash")

            bullets = []
            for n in news_items:
                title = n.get("title", "")
                url = n.get("url", "")
                tickers = ", ".join(
                    [x.get("ticker") for x in n.get("ticker_sentiment", []) if x.get("ticker")]
                )
                bullets.append(f"- {title} ({tickers or '—'})\n  Quelle: {url}")

            prompt = (
                PROMPTS["main"]
                + "\n\n"
                + PROMPTS["format"]
                + "\n\n"
                + "\n".join(bullets)
            )
            resp = model.generate_content(prompt)
            raw = resp.text
    except Exception as e:
        print("⚠️ Gemini-Fehler:", e)
        return IdeaOutput(ideen=[])

    try:
        return IdeaOutput.model_validate_json(raw)
    except Exception as e:
        print("⚠️ Fehler beim Parsen der Gemini-Antwort:", e)
        print("Antwort:", raw)
        return IdeaOutput(ideen=[])


# ===== PUSHOVER =====
def send_pushover(message: str):
    if not PUSHOVER_TOKEN or not PUSHOVER_USER:
        print("⚠️ Pushover-Umgebungsvariablen fehlen.")
        return
    # falls Nachricht zu lang ist → splitten
    chunks = [message[i:i + CHAR_LIMIT] for i in range(0, len(message), CHAR_LIMIT)]
    for chunk in chunks:
        payload = {
            "token": PUSHOVER_TOKEN,
            "user": PUSHOVER_USER,
            "message": chunk,
            "html": 1  # HTML aktiv für fette Prozentzahlen
        }
        try:
            r = requests.post("https://api.pushover.net/1/messages.json", data=payload, timeout=15)
            r.raise_for_status()
            time.sleep(1)
        except Exception as e:
            print("⚠️ Fehler beim Senden an Pushover:", e)


# ===== MAIN =====
def main():
    api_key = os.getenv(CONFIG["sources"][0]["api_key_env"])
    news = fetch_news_alphavantage(api_key, CONFIG["sources"][0]["limit"])
    relevant = [n for n in news if is_relevant(n)]
    print(f"🔎 Relevante News gefunden: {len(relevant)}")

    if not relevant:
        print("Keine relevanten News, kein Push.")
        return

    # nur neue Nachrichten (gegen gespeicherte URLs prüfen)
    current_ids = {n["url"] for n in relevant}
    last_ids = load_last_ids()
    new_ids = current_ids - last_ids
    if not new_ids:
        print("Keine neuen relevanten News – kein Push.")
        return

    new_news = [n for n in relevant if n["url"] in new_ids]
    result = analyze_with_gemini(new_news)
    if not result.ideen:
        print("Keine neuen Anlageideen erkannt.")
        save_last_ids(current_ids)
        return

    # Nachricht zusammenbauen
    parts = []
    for i in result.ideen:
        v = i.vertrauen
        if v > 100: v /= 100
        if v <= 1: v *= 100
        parts.append(f"🟢 {i.name} ({i.typ}) <b>{v:.0f}%</b> – {i.begruendung.strip()}")

    msg = "\n".join(parts)
    send_pushover(msg)
    save_last_ids(current_ids)
    print("✅ Neue Anlageideen gesendet.")


if __name__ == "__main__":
    main()
