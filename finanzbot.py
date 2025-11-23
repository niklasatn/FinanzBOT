import os, json, requests, importlib, time, re, hashlib
import feedparser
from datetime import datetime, timezone, timedelta
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

# ===== KONFIGURATION LADEN =====
with open("config.json", "r", encoding="utf-8") as f:
    CONFIG = json.load(f)

KEYWORDS = [k.lower() for k in CONFIG["keywords"]]
PROMPTS = CONFIG["prompts"]
PUSHOVER_TOKEN = os.getenv(CONFIG["pushover_token_env"])
PUSHOVER_USER = os.getenv(CONFIG["pushover_user_env"])
STATE_FILE = "last_sent.json"
CHAR_LIMIT = 1024  # Pushover-Hardlimit
MAX_NEWS_AGE_HOURS = 4  # Wie alt dürfen News maximal sein

# ===== MODELDEFINITION =====
class IdeaItem(BaseModel):
    name: str
    typ: str
    begruendung: str
    vertrauen: float

class IdeaOutput(BaseModel):
    ideen: List[IdeaItem]

# ===== STATE MANAGEMENT =====
def load_last_ids():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                return set(json.load(f))
        except:
            return set()
    return set()

def save_last_ids(ids):
    with open(STATE_FILE, "w") as f:
        # Konvertiere Set in Liste für JSON
        json.dump(list(ids), f)

# ===== UTILS =====
def clean_html(raw_html: str) -> str:
    """Entfernt HTML-Tags aus RSS-Beschreibungen."""
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext.strip()

def normalize_title(t: str) -> str:
    t = re.sub(r"[^A-Za-z0-9äöüÄÖÜß ]+", " ", t.lower())
    t = re.sub(r"\s+", " ", t).strip()
    return t

# ===== RSS FEED HOLEN =====
def fetch_news_rss(url: str, limit: int = 30) -> List[Dict[str, Any]]:
    """
    Holt News von einem RSS Feed.
    """
    print(f"📡 Lade RSS Feed: {url} ...")
    feed = feedparser.parse(url)
    
    collected = []
    
    # Einträge durchgehen
    for entry in feed.entries[:limit]:
        title = entry.get("title", "")
        link = entry.get("link", "")
        # RSS hat oft 'summary' oder 'description'
        summary = entry.get("summary") or entry.get("description") or ""
        summary = clean_html(summary)
        
        # Zeitstempel parsen
        published_dt = None
        if hasattr(entry, "published_parsed") and entry.published_parsed:
            # struct_time in datetime UTC umwandeln
            published_dt = datetime.fromtimestamp(time.mktime(entry.published_parsed), timezone.utc)
        elif hasattr(entry, "updated_parsed") and entry.updated_parsed:
            published_dt = datetime.fromtimestamp(time.mktime(entry.updated_parsed), timezone.utc)
        
        # Fallback: Jetzt, falls kein Datum im Feed
        if not published_dt:
            published_dt = datetime.now(timezone.utc)

        collected.append({
            "title": title,
            "url": link,
            "summary": summary,
            "source": "finanzen.net", # oder feed.feed.title
            "time_published": published_dt
        })

    return collected

# ===== FILTER & RELEVANZ =====
def is_recent(item: dict) -> bool:
    published = item.get("time_published")
    if not published:
        return True
    now = datetime.now(timezone.utc)
    # Puffer, falls Serverzeiten abweichen
    return (now - published) <= timedelta(hours=MAX_NEWS_AGE_HOURS)

def relevance_score(item: dict) -> int:
    text = (item.get("title", "") + " " + item.get("summary", "")).lower()
    score = 0
    
    # Keyword Match
    for k in KEYWORDS:
        if k in text:
            score += 1
            
    # News ohne echten Inhalt filtern (oft nur Tabellen-Updates im RSS)
    if "dgap-news" in text or "original-research" in text:
        score -= 2 # Herabstufen von reinen Pressemitteilungen, außer sie enthalten Keywords
        
    return score

# ===== GEMINI ANALYSE =====
def analyze_with_gemini(news_items: List[dict]) -> IdeaOutput:
    print(f"🧠 Sende {len(news_items)} News an Gemini...")
    
    # Zusammenstellen der Daten für den Prompt
    bullets = []
    for n in news_items:
        t = n.get("title", "")
        s = n.get("summary", "")[:200] # Summary kürzen
        u = n.get("url", "")
        bullets.append(f"- TITEL: {t}\n  SUMMARY: {s}\n  LINK: {u}")
    
    bullet_text = "\n".join(bullets)
    
    # Prompt zusammenbauen
    full_prompt = (
        PROMPTS["main"] + "\n\n" + 
        PROMPTS["format"] + "\n\n" + 
        "HIER SIND DIE NEWS:\n" + bullet_text
    )

    raw_response = ""
    try:
        # Versuche google.genai (neues SDK)
        if importlib.util.find_spec("google.genai"):
            from google import genai
            client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
            
            resp = client.models.generate_content(
                model="gemini-2.0-flash-exp", # oder gemini-1.5-flash
                contents=full_prompt,
                config={"response_mime_type": "application/json"}
            )
            raw_response = resp.text
            
        # Fallback auf google.generativeai (altes SDK)
        else:
            import google.generativeai as genai
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            model = genai.GenerativeModel("gemini-1.5-flash")
            resp = model.generate_content(full_prompt)
            # Markdown Code Blöcke entfernen, falls vorhanden
            raw_response = resp.text.replace("```json", "").replace("```", "")
            
    except Exception as e:
        print(f"⚠️ Gemini API Fehler: {e}")
        return IdeaOutput(ideen=[])

    try:
        # Parsing
        return IdeaOutput.model_validate_json(raw_response)
    except Exception as e:
        print(f"⚠️ JSON Parsing Fehler: {e}")
        print("Raw Response:", raw_response)
        return IdeaOutput(ideen=[])

# ===== MAIN =====
def main():
    # 1. Konfiguration laden (erstes Element aus sources)
    src_config = CONFIG["sources"][0]
    
    if src_config["type"] != "rss":
        print("❌ Fehler: Dieses Skript erwartet 'type': 'rss' in der config.json")
        return

    # 2. News holen
    news = fetch_news_rss(src_config["url"], src_config.get("limit", 30))
    print(f"📥 {len(news)} Einträge geladen.")

    # 3. Filtern (Zeit & Relevanz)
    recent_news = [n for n in news if is_recent(n)]
    relevant_news = [n for n in recent_news if relevance_score(n) >= 1]
    
    print(f"🔎 Nach Filterung (Zeit/Relevanz): {len(relevant_news)} News übrig.")

    if not relevant_news:
        print("😴 Keine relevanten News gefunden.")
        return

    # 4. Dubletten-Check (nur neue News senden)
    last_ids = load_last_ids()
    current_ids = {n["url"] for n in relevant_news} # URL als ID
    
    new_ids = current_ids - last_ids
    
    # Filterliste auf nur die NEUEN Items reduzieren
    final_news_list = [n for n in relevant_news if n["url"] in new_ids]
    
    if not final_news_list:
        print("🔄 Alle relevanten News wurden bereits gesendet.")
        # IDs trotzdem updaten, damit wir 'state' behalten
        save_last_ids(last_ids.union(current_ids))
        return

    # 5. KI-Analyse
    # Wir senden maximal die Top 10 neusten an Gemini, um Token zu sparen
    ai_result = analyze_with_gemini(final_news_list[:10])
    
    if not ai_result.ideen:
        print("🤷 Gemini hat keine handelbaren Ideen gefunden.")
        save_last_ids(last_ids.union(new_ids))
        return

    # 6. Nachricht bauen
    msg_parts = []
    for idee in ai_result.ideen:
        # Vertrauens-Score formatieren
        score = idee.vertrauen
        # Normalisierung falls die KI 0.8 statt 80 schickt
        if score <= 1: score *= 100
            
        icon = "🟢" if score > 75 else "🟡"
        msg_parts.append(
            f"{icon} <b>{idee.name}</b> ({idee.typ}) - {score:.0f}%\n"
            f"{idee.begruendung}"
        )

    full_msg = "\n\n".join(msg_parts)
    
    # Pushover Limit Check
    if len(full_msg) > CHAR_LIMIT:
        full_msg = full_msg[:CHAR_LIMIT-10] + "..."

    # 7. Senden
    payload = {
        "token": PUSHOVER_TOKEN,
        "user": PUSHOVER_USER,
        "message": full_msg,
        "html": 1,
        "title": "FinanzBot News"
    }

    try:
        r = requests.post("https://api.pushover.net/1/messages.json", data=payload, timeout=10)
        r.raise_for_status()
        print("✅ Pushover Nachricht gesendet!")
        
        # Erst speichern, wenn erfolgreich gesendet
        save_last_ids(last_ids.union(new_ids))
        
    except Exception as e:
        print(f"❌ Fehler beim Senden an Pushover: {e}")

if __name__ == "__main__":
    main()
