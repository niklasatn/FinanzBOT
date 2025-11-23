import os, json, requests, importlib, time, re, smtplib
import feedparser
from datetime import datetime, timezone, timedelta
from email.message import EmailMessage
from pydantic import BaseModel
from typing import List, Dict, Any

# ===== KONFIGURATION LADEN =====
with open("config.json", "r", encoding="utf-8") as f:
    CONFIG = json.load(f)

KEYWORDS = [k.lower() for k in CONFIG["keywords"]]
PROMPTS = CONFIG["prompts"]

# E-Mail Konfiguration
EMAIL_USER = os.getenv(CONFIG["email_user_env"])
EMAIL_PASSWORD = os.getenv(CONFIG["email_password_env"])
EMAIL_RECIPIENT_RAW = os.getenv(CONFIG["email_recipient_env"])

STATE_FILE = "last_sent.json"
MAX_NEWS_AGE_HOURS = 4

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
        json.dump(list(ids), f)

# ===== UTILS =====
def clean_html(raw_html: str) -> str:
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext.strip()

# ===== RSS FEED HOLEN =====
def fetch_news_rss(url: str, limit: int = 30) -> List[Dict[str, Any]]:
    print(f"📡 Lade RSS Feed: {url} ...")
    feed = feedparser.parse(url)
    collected = []
    
    for entry in feed.entries[:limit]:
        title = entry.get("title", "")
        link = entry.get("link", "")
        summary = entry.get("summary") or entry.get("description") or ""
        summary = clean_html(summary)
        
        published_dt = None
        # Verschiedene Zeitformate im RSS prüfen
        if hasattr(entry, "published_parsed") and entry.published_parsed:
            published_dt = datetime.fromtimestamp(time.mktime(entry.published_parsed), timezone.utc)
        elif hasattr(entry, "updated_parsed") and entry.updated_parsed:
            published_dt = datetime.fromtimestamp(time.mktime(entry.updated_parsed), timezone.utc)
        
        if not published_dt:
            published_dt = datetime.now(timezone.utc)

        collected.append({
            "title": title,
            "url": link,
            "summary": summary,
            "source": "finanzen.net",
            "time_published": published_dt
        })
    return collected

# ===== FILTER & RELEVANZ =====
def is_recent(item: dict) -> bool:
    published = item.get("time_published")
    if not published: return True
    now = datetime.now(timezone.utc)
    return (now - published) <= timedelta(hours=MAX_NEWS_AGE_HOURS)

def relevance_score(item: dict) -> int:
    text = (item.get("title", "") + " " + item.get("summary", "")).lower()
    score = 0
    for k in KEYWORDS:
        if k in text: score += 1
    
    # Unwichtiges abwerten
    if "dgap-news" in text or "original-research" in text:
        score -= 2 
    return score

# ===== GEMINI ANALYSE =====
def analyze_with_gemini(news_items: List[dict]) -> IdeaOutput:
    # Hier werden alle News in EINEN String gepackt -> Nur 1 Anfrage!
    print(f"🧠 Sende {len(news_items)} News gebündelt an Gemini (Modell: gemini-1.5-flash)...")
    
    bullets = []
    for n in news_items:
        t = n.get("title", "")
        s = n.get("summary", "")[:250] # Limit pro News, um Tokens zu sparen
        u = n.get("url", "")
        bullets.append(f"- TITEL: {t}\n  SUMMARY: {s}\n  LINK: {u}")
    
    bullet_text = "\n".join(bullets)
    full_prompt = (PROMPTS["main"] + "\n\n" + PROMPTS["format"] + "\n\n" + "HIER SIND DIE NEWS:\n" + bullet_text)

    # Retry-Logik (3 Versuche)
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Nutzung des neuen Google Gen AI SDKs
            if importlib.util.find_spec("google.genai"):
                from google import genai
                client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
                
                # WICHTIG: Hier nutzen wir jetzt das stabile Modell
                resp = client.models.generate_content(
                    model="gemini-1.5-flash", 
                    contents=full_prompt,
                    config={"response_mime_type": "application/json"}
                )
                raw_response = resp.text
                
            # Fallback für älteres SDK
            else:
                import google.generativeai as genai
                genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
                model = genai.GenerativeModel("gemini-1.5-flash")
                resp = model.generate_content(full_prompt)
                raw_response = resp.text.replace("```json", "").replace("```", "")

            # Erfolgreich -> Parsen
            return IdeaOutput.model_validate_json(raw_response)

        except Exception as e:
            print(f"⚠️ Versuch {attempt+1}/{max_retries} fehlgeschlagen: {e}")
            if "429" in str(e):
                print("⏳ Warte 10 Sekunden wegen Rate Limit...")
                time.sleep(10)
            else:
                break # Bei anderen Fehlern abbrechen

    return IdeaOutput(ideen=[])

# ===== E-MAIL SENDEN =====
def send_email(subject: str, html_content: str):
    if not EMAIL_USER or not EMAIL_PASSWORD or not EMAIL_RECIPIENT_RAW:
        print("❌ E-Mail-Zugangsdaten oder Empfänger fehlen!")
        return

    recipients_list = [email.strip() for email in EMAIL_RECIPIENT_RAW.split(",") if email.strip()]

    msg = EmailMessage()
    msg['Subject'] = subject
    msg['From'] = EMAIL_USER
    msg['To'] = ", ".join(recipients_list)
    
    msg.set_content("Dein E-Mail Client unterstützt kein HTML.") 
    msg.add_alternative(html_content, subtype='html')

    try:
        # Hier bei Bedarf 'mail.gmx.net' eintragen, falls kein Gmail
        smtp_server = 'smtp.gmail.com' 
        with smtplib.SMTP(smtp_server, 587) as server:
            server.starttls()
            server.login(EMAIL_USER, EMAIL_PASSWORD)
            server.send_message(msg)
        print(f"📧 E-Mail erfolgreich an {len(recipients_list)} Empfänger gesendet!")
    except Exception as e:
        print(f"❌ Fehler beim E-Mail-Versand: {e}")

# ===== MAIN =====
def main():
    if not CONFIG.get("sources"):
        print("❌ Keine Quellen in der Config.")
        return

    src_config = CONFIG["sources"][0]
    # Limit auf 15 News setzen, um Request-Größe überschaubar zu halten
    news_limit = src_config.get("limit", 15)
    
    news = fetch_news_rss(src_config["url"], news_limit)
    
    recent_news = [n for n in news if is_recent(n)]
    # Relevanz-Filter etwas strenger (>= 1)
    relevant_news = [n for n in recent_news if relevance_score(n) >= 1]
    
    print(f"🔎 Relevant: {len(relevant_news)}")

    if not relevant_news:
        print("😴 Keine relevanten News.")
        return

    last_ids = load_last_ids()
    current_ids = {n["url"] for n in relevant_news}
    new_ids = current_ids - last_ids
    
    final_news_list = [n for n in relevant_news if n["url"] in new_ids]
    
    if not final_news_list:
        print("🔄 Alle News schon bekannt.")
        save_last_ids(last_ids.union(current_ids))
        return

    # Wir nehmen max 10 News mit in die Analyse, um Tokens zu sparen
    ai_result = analyze_with_gemini(final_news_list[:10])
    
    if not ai_result.ideen:
        print("🤷 Keine Handelsideen.")
        save_last_ids(last_ids.union(new_ids))
        return

    # E-Mail HTML zusammenbauen
    html_body = "<h2>🚀 Neue Finanz-Ideen</h2><hr>"
    for idee in ai_result.ideen:
        score = idee.vertrauen
        if score <= 1: score *= 100
        color = "green" if score > 75 else "orange"
        
        html_body += f"""
        <div style="margin-bottom: 20px; padding: 10px; border-left: 5px solid {color}; background-color: #f9f9f9; font-family: Arial, sans-serif;">
            <h3 style="margin: 0;">{idee.name} <span style="font-size: 0.8em; color: #666;">({idee.typ})</span></h3>
            <p style="font-weight: bold; color: {color};">Vertrauen: {score:.0f}%</p>
            <p>{idee.begruendung}</p>
        </div>
        """
    
    html_body += f"<hr><p style='font-size:small; color:gray;'>Generiert vom FinanzBot am {datetime.now().strftime('%d.%m.%Y %H:%M')}</p>"

    subject = f"FinanzBot: {len(ai_result.ideen)} neue Chancen 📈"
    send_email(subject, html_body)
    
    save_last_ids(last_ids.union(new_ids))

if __name__ == "__main__":
    main()
