import os, json, requests, time, re, smtplib
import feedparser
import google.generativeai as genai
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
    
    if "dgap-news" in text or "original-research" in text:
        score -= 2 
    return score

# ===== GEMINI ANALYSE (ROBUST) =====
def analyze_with_gemini(news_items: List[dict]) -> IdeaOutput:
    print(f"🧠 Sende {len(news_items)} News an Gemini...")
    
    bullets = []
    for n in news_items:
        t = n.get("title", "")
        s = n.get("summary", "")[:250]
        u = n.get("url", "")
        bullets.append(f"- TITEL: {t}\n  SUMMARY: {s}\n  LINK: {u}")
    
    bullet_text = "\n".join(bullets)
    full_prompt = (PROMPTS["main"] + "\n\n" + PROMPTS["format"] + "\n\n" + "HIER SIND DIE NEWS:\n" + bullet_text)

    # API Konfigurieren
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    
    # Liste der Modelle zum Durchprobieren (Fallback-Strategie)
    # 1. Flash-001 (Spezifisch, schnell)
    # 2. Flash (Alias)
    # 3. Pro (Langsam, aber sehr stabil)
    models_to_try = ["gemini-1.5-flash-001", "gemini-1.5-flash", "gemini-pro"]

    for model_name in models_to_try:
        try:
            print(f"🤖 Versuche Modell: {model_name} ...")
            model = genai.GenerativeModel(model_name)
            resp = model.generate_content(full_prompt)
            
            # Markdown bereinigen
            raw_response = resp.text.replace("```json", "").replace("```", "").strip()
            
            # Validieren und zurückgeben
            return IdeaOutput.model_validate_json(raw_response)

        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg:
                print(f"⏳ Rate Limit bei {model_name}. Warte kurz...")
                time.sleep(5) # Kurzer Backoff
            elif "404" in error_msg:
                print(f"❌ Modell {model_name} nicht gefunden (404). Versuche nächstes...")
            else:
                print(f"⚠️ Fehler bei {model_name}: {error_msg}")
    
    print("❌ Alle KI-Modelle fehlgeschlagen.")
    return IdeaOutput(ideen=[])

# ===== E-MAIL SENDEN =====
def send_email(subject: str, html_content: str):
    if not EMAIL_USER or not EMAIL_PASSWORD or not EMAIL_RECIPIENT_RAW:
        print("❌ E-Mail-Zugangsdaten fehlen!")
        return

    recipients_list = [email.strip() for email in EMAIL_RECIPIENT_RAW.split(",") if email.strip()]

    msg = EmailMessage()
    msg['Subject'] = subject
    msg['From'] = EMAIL_USER
    msg['To'] = ", ".join(recipients_list)
    
    msg.set_content("HTML nicht unterstützt.") 
    msg.add_alternative(html_content, subtype='html')

    try:
        smtp_server = 'smtp.gmail.com' # GMX: 'mail.gmx.net'
        with smtplib.SMTP(smtp_server, 587) as server:
            server.starttls()
            server.login(EMAIL_USER, EMAIL_PASSWORD)
            server.send_message(msg)
        print(f"📧 E-Mail an {len(recipients_list)} Empfänger gesendet!")
    except Exception as e:
        print(f"❌ Fehler beim E-Mail-Versand: {e}")

# ===== MAIN =====
def main():
    if not CONFIG.get("sources"):
        print("❌ Config Fehler.")
        return

    src_config = CONFIG["sources"][0]
    news = fetch_news_rss(src_config["url"], src_config.get("limit", 15))
    
    recent_news = [n for n in news if is_recent(n)]
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
        print("🔄 Nichts Neues.")
        save_last_ids(last_ids.union(current_ids))
        return

    # KI Analyse
    ai_result = analyze_with_gemini(final_news_list[:10])
    
    if not ai_result.ideen:
        print("🤷 Keine Ergebnisse.")
        save_last_ids(last_ids.union(new_ids))
        return

    # HTML Email bauen
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
    html_body += f"<hr><p style='font-size:small; color:gray;'>Bot Run: {datetime.now().strftime('%H:%M')}</p>"

    send_email(f"FinanzBot: {len(ai_result.ideen)} Ideen 📈", html_body)
    save_last_ids(last_ids.union(new_ids))

if __name__ == "__main__":
    main()
