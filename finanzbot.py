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

EMAIL_USER = os.getenv(CONFIG["email_user_env"])
EMAIL_PASSWORD = os.getenv(CONFIG["email_password_env"])
EMAIL_RECIPIENT_RAW = os.getenv(CONFIG["email_recipient_env"])
USER_PORTFOLIO = os.getenv(CONFIG.get("portfolio_env"), "Kein Portfolio")

STATE_FILE = "last_sent.json"
MAX_NEWS_AGE_HOURS = 4

# --- FILTER EINSTELLUNGEN ---
MIN_CONF_PORTFOLIO = 70   # Portfolio-News ab 70%
MIN_CONF_NEW_GEM = 90     # Neue Chancen ab 90%

# ===== MODELDEFINITION =====
class IdeaItem(BaseModel):
    name: str
    typ: str
    signal: str
    begruendung: str
    vertrauen: float
    betrifft_portfolio: bool

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

# ===== RSS FEED =====
def fetch_news_rss(url: str, limit: int = 30) -> List[Dict[str, Any]]:
    print(f"📡 Lade RSS Feed: {url} ...")
    feed = feedparser.parse(url)
    collected = []
    for entry in feed.entries[:limit]:
        title = entry.get("title", "")
        link = entry.get("link", "")
        summary = clean_html(entry.get("summary") or entry.get("description") or "")
        
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

# ===== FILTER =====
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

# ===== GEMINI ANALYSE =====
def analyze_with_gemini(news_items: List[dict]) -> IdeaOutput:
    print(f"🧠 Analysiere {len(news_items)} News...")
    
    bullets = []
    for n in news_items:
        t = n.get("title", "")
        s = n.get("summary", "")[:250]
        u = n.get("url", "")
        bullets.append(f"- {t}\n  {s}\n  Quelle: {u}")
    
    bullet_text = "\n".join(bullets)
    prompt_intro = PROMPTS["main"].replace("{portfolio}", USER_PORTFOLIO)
    full_prompt = (prompt_intro + "\n\n" + PROMPTS["format"] + "\n\n" + "NEWS:\n" + bullet_text)

    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    models_to_try = ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-1.5-flash"]

    for model_name in models_to_try:
        try:
            model = genai.GenerativeModel(model_name)
            resp = model.generate_content(full_prompt)
            raw_response = resp.text.replace("```json", "").replace("```", "").strip()
            return IdeaOutput.model_validate_json(raw_response)
        except Exception as e:
            if "429" in str(e): time.sleep(5)
    
    return IdeaOutput(ideen=[])

# ===== E-MAIL SENDEN =====
def send_email(subject: str, html_content: str):
    if not EMAIL_USER or not EMAIL_RECIPIENT_RAW: return
    recipients = [e.strip() for e in EMAIL_RECIPIENT_RAW.split(",") if e.strip()]

    msg = EmailMessage()
    msg['Subject'] = subject
    msg['From'] = EMAIL_USER
    msg['To'] = ", ".join(recipients)
    msg.set_content("Kein HTML.") 
    msg.add_alternative(html_content, subtype='html')

    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(EMAIL_USER, EMAIL_PASSWORD)
            server.send_message(msg)
        print(f"📧 Gesendet an {len(recipients)} Empfänger.")
    except Exception as e:
        print(f"❌ Mail-Fehler: {e}")

# ===== MAIN =====
def main():
    if not CONFIG.get("sources"): return
    src_config = CONFIG["sources"][0]
    
    news = fetch_news_rss(src_config["url"], src_config.get("limit", 30))
    news = [n for n in news if is_recent(n) and relevance_score(n) >= 1]
    
    last_ids = load_last_ids()
    current_ids = {n["url"] for n in news}
    new_ids = current_ids - last_ids
    
    final_news = [n for n in news if n["url"] in new_ids]
    
    if not final_news:
        print("🔄 Keine neuen relevanten News.")
        save_last_ids(last_ids.union(current_ids))
        return

    # KI Analyse
    ai_result = analyze_with_gemini(final_news[:12])
    
    if not ai_result.ideen:
        print("🤷 KI hat keine Signale gefunden.")
        save_last_ids(last_ids.union(new_ids))
        return

    # ===== FILTER LOGIK =====
    relevant_for_mail = []

    for idee in ai_result.ideen:
        score = idee.vertrauen
        sig_upper = idee.signal.upper()
        
        is_action_signal = ("KAUF" in sig_upper) or ("VERKAUF" in sig_upper)

        # 1. Fall: Portfolio > 70% + Action
        if idee.betrifft_portfolio and score >= MIN_CONF_PORTFOLIO:
            if is_action_signal:
                relevant_for_mail.append(idee)
            
        # 2. Fall: Neu > 90% + Kauf
        elif (not idee.betrifft_portfolio) and score >= MIN_CONF_NEW_GEM:
            if "KAUF" in sig_upper or "NACHKAUFEN" in sig_upper:
                relevant_for_mail.append(idee)

    if not relevant_for_mail:
        print(f"📉 Keine Signale mit Handlungsbedarf.")
        save_last_ids(last_ids.union(current_ids))
        return

    # --- BETREFF GENERIEREN (NEU) ---
    subject_actions = []
    
    for item in relevant_for_mail:
        # Namen bereinigen: "NVIDIA Corp. (NVDA)" -> "NVIDIA Corp."
        short_name = item.name.split("(")[0].strip()
        # Falls immer noch sehr lang, kürzen
        if len(short_name) > 15:
            short_name = short_name[:12] + ".."
            
        if "VERKAUF" in item.signal.upper():
            subject_actions.append(f"⚠️ Verkauf {short_name}")
        else:
            subject_actions.append(f"💰 Kauf {short_name}")
    
    # Verbinden mit Pipe |
    subject = " | ".join(subject_actions)
    
    # Fallback falls leer (sollte nicht passieren) oder zu lang
    if not subject:
        subject = "🚨 Handlungsbedarf erkannt"

    # --- HTML BODY ---
    html_body = """
    <div style="font-family: Helvetica, Arial, sans-serif; color: #333;">
        <h2 style="border-bottom: 2px solid #333; padding-bottom: 10px;">⚡ Handlungsbedarf</h2>
    """
    
    for idee in relevant_for_mail:
        score = idee.vertrauen * 100 if idee.vertrauen <= 1 else idee.vertrauen
        sig = idee.signal.upper()
        
        if "VERKAUF" in sig:
            color = "#d32f2f" # Rot
            bg = "#ffebee"
            icon = "📉 VERKAUF"
        else: 
            color = "#2e7d32" # Grün
            bg = "#e8f5e9"
            icon = "📈 KAUF"

        badge = ""
        if idee.betrifft_portfolio:
            badge = f'<span style="background-color:#333; color:white; padding:2px 6px; border-radius:4px; font-size:0.7em; margin-left:10px;">PORTFOLIO</span>'
        else:
            badge = f'<span style="background-color:#0288d1; color:white; padding:2px 6px; border-radius:4px; font-size:0.7em; margin-left:10px;">NEU</span>'

        html_body += f"""
        <div style="margin-bottom: 15px; border-left: 6px solid {color}; background-color: {bg}; padding: 15px; border-radius: 4px;">
            <div style="color: {color}; font-weight: bold; font-size: 0.9em; margin-bottom: 5px;">
                {icon} {badge}
            </div>
            <h3 style="margin: 0; font-size: 1.2em;">{idee.name}</h3>
            <div style="font-size: 0.85em; color: #666; margin-bottom: 8px;">{idee.typ} • Signal: {sig} • Konfidenz: {score:.0f}%</div>
            <p style="margin: 0; line-height: 1.4;">{idee.begruendung}</p>
        </div>
        """

    html_body += f"""
        <hr style="border: 0; border-top: 1px solid #eee; margin-top: 20px;">
        <p style="font-size: 0.8em; color: #999;">Generiert am {datetime.now().strftime('%d.%m.%Y %H:%M')}</p>
    </div>
    """

    send_email(subject, html_body)
    save_last_ids(last_ids.union(new_ids))

if __name__ == "__main__":
    main()
