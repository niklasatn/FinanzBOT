import os, json, requests, time, re
import feedparser
import yfinance as yf
import matplotlib
# Backend auf 'Agg' setzen, damit kein Fenster geöffnet wird (wichtig für GitHub Actions)
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import io
import base64
import google.generativeai as genai
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from pydantic import BaseModel
from typing import List, Dict, Any

# ===== KONFIGURATION =====
with open("config.json", "r", encoding="utf-8") as f:
    CONFIG = json.load(f)

KEYWORDS = [k.lower() for k in CONFIG["keywords"]]
PROMPTS = CONFIG["prompts"]
USER_PORTFOLIO = os.getenv(CONFIG.get("portfolio_env"), "Kein Portfolio")

STATE_FILE = "last_sent.json"
MAX_NEWS_AGE_HOURS = 4

# Filter
MIN_CONF_PORTFOLIO = 70
MIN_CONF_NEW_GEM = 90

# ===== PORTFOLIO MAPPING (TICKER SYMBOLE) =====
# Hier ordnen wir deinen Namen die Yahoo-Finance Ticker zu.
# .DE = Xetra/Deutschland, -EUR = Krypto in Euro
PORTFOLIO_MAPPING = {
    "iShares MSCI World": "EUNL.DE",
    "Vanguard FTSE All-World": "VWCE.DE",
    "Allianz SE": "ALV.DE",
    "Münchener Rück": "MUV2.DE",
    "Snowflake": "SNOW",
    "Bitcoin": "BTC-EUR"
}

# ===== MODELLE =====
class IdeaItem(BaseModel):
    name: str
    typ: str
    signal: str
    begruendung: str
    vertrauen: float
    betrifft_portfolio: bool

class IdeaOutput(BaseModel):
    ideen: List[IdeaItem]

class MarketData(BaseModel):
    name: str
    price: str
    change_percent: float
    graph_base64: str

# ===== STATE MANAGEMENT =====
def load_last_ids():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f: return set(json.load(f))
        except: return set()
    return set()

def save_last_ids(ids):
    with open(STATE_FILE, "w") as f: json.dump(list(ids), f)

# ===== UTILS =====
def clean_html(raw_html: str) -> str:
    cleanr = re.compile('<.*?>')
    return re.sub(cleanr, '', raw_html).strip()

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
        if not published_dt: published_dt = datetime.now(timezone.utc)

        collected.append({"title": title, "url": link, "summary": summary, "time_published": published_dt})
    return collected

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
    if "dgap-news" in text or "original-research" in text: score -= 2 
    return score

# ===== FINANCE DATA & PLOTTING =====
def get_market_data() -> List[MarketData]:
    print("📈 Lade Marktdaten & erstelle Graphen...")
    data_list = []

    for name, ticker_symbol in PORTFOLIO_MAPPING.items():
        try:
            # Daten holen (1 Tag, 15 Min Interval)
            ticker = yf.Ticker(ticker_symbol)
            hist = ticker.history(period="1d", interval="15m")
            
            if hist.empty:
                continue

            # Aktueller Preis & Change
            current = hist['Close'].iloc[-1]
            open_price = hist['Open'].iloc[0]
            change_pct = ((current - open_price) / open_price) * 100
            
            # Währungssymbol
            currency = "€" if "EUR" in ticker_symbol or ".DE" in ticker_symbol else "$"
            price_fmt = f"{current:.2f} {currency}"

            # Graph erstellen (Matplotlib)
            fig, ax = plt.subplots(figsize=(4, 1.5))
            # Hintergrund dunkel
            fig.patch.set_facecolor('#1e1e1e')
            ax.set_facecolor('#1e1e1e')
            
            # Farbe je nach Trend
            line_color = '#4caf50' if change_pct >= 0 else '#e57373'
            
            ax.plot(hist.index, hist['Close'], color=line_color, linewidth=2)
            
            # Alles ausblenden (Achsen, Rahmen)
            ax.axis('off')
            plt.tight_layout(pad=0)
            
            # Als Base64 speichern
            buf = io.BytesIO()
            plt.savefig(buf, format='png', facecolor=fig.get_facecolor())
            plt.close(fig)
            buf.seek(0)
            img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            
            data_list.append(MarketData(
                name=name,
                price=price_fmt,
                change_percent=change_pct,
                graph_base64=img_base64
            ))

        except Exception as e:
            print(f"⚠️ Fehler bei {name} ({ticker_symbol}): {e}")
            continue
            
    return data_list

# ===== GEMINI =====
def analyze_with_gemini(news_items: List[dict]) -> IdeaOutput:
    print(f"🧠 Analysiere {len(news_items)} News...")
    bullets = [f"- {n['title']}\n  {n['summary']}" for n in news_items]
    
    prompt_intro = PROMPTS["main"].replace("{portfolio}", USER_PORTFOLIO)
    full_prompt = (prompt_intro + "\n\n" + PROMPTS["format"] + "\n\nNEWS:\n" + "\n".join(bullets))

    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    models = ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-1.5-flash"]

    for m in models:
        try:
            model = genai.GenerativeModel(m)
            resp = model.generate_content(full_prompt)
            raw = resp.text.replace("```json", "").replace("```", "").strip()
            return IdeaOutput.model_validate_json(raw)
        except Exception as e:
            if "429" in str(e): time.sleep(5)
    return IdeaOutput(ideen=[])

# ===== HTML GENERATOR =====
def generate_dashboard(items: List[IdeaItem] = None, market_data: List[MarketData] = None):
    now_str = datetime.now(ZoneInfo("Europe/Berlin")).strftime('%d.%m.%Y %H:%M')
    
    css = """
    <style>
        body { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; background-color: #121212; color: #e0e0e0; margin: 0; padding: 20px; }
        .container { max-width: 800px; margin: 0 auto; }
        header { border-bottom: 1px solid #333; padding-bottom: 20px; margin-bottom: 30px; display: flex; justify-content: space-between; align-items: center; }
        h1 { margin: 0; font-size: 1.5rem; letter-spacing: -0.5px; }
        h2 { border-bottom: 1px solid #333; padding-bottom: 10px; margin-top: 40px; color: #888; font-size: 1.2rem;}
        .timestamp { font-size: 0.9rem; color: #888; }
        
        .status-card { background: #1e1e1e; border-radius: 12px; padding: 20px; text-align: center; border: 1px solid #333; margin-bottom: 30px; }
        .status-ok { color: #4caf50; font-size: 1.2rem; font-weight: bold; }
        .status-alert { color: #e57373; font-size: 1.2rem; font-weight: bold; }
        
        /* News Cards */
        .card { background: #1e1e1e; border-radius: 12px; padding: 20px; margin-bottom: 20px; border: 1px solid #333; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
        .card-header { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 10px; }
        .asset-name { font-size: 1.2rem; font-weight: 700; color: #fff; }
        .asset-type { font-size: 0.8rem; color: #888; background: #333; padding: 2px 6px; border-radius: 4px; }
        .badge { padding: 4px 8px; border-radius: 6px; font-size: 0.8rem; font-weight: bold; text-transform: uppercase; }
        .bg-red { background: rgba(211, 47, 47, 0.2); color: #ef9a9a; border: 1px solid #d32f2f; }
        .bg-green { background: rgba(56, 142, 60, 0.2); color: #a5d6a7; border: 1px solid #388e3c; }
        .bg-blue { background: rgba(25, 118, 210, 0.2); color: #90caf9; border: 1px solid #1976d2; }
        .portfolio-tag { background: linear-gradient(45deg, #6a1b9a, #4a148c); color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.7rem; font-weight: bold; letter-spacing: 0.5px; }
        
        /* Market Data Grid */
        .market-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap: 15px; }
        .market-card { background: #1e1e1e; border: 1px solid #333; border-radius: 12px; padding: 15px; display: flex; flex-direction: column; }
        .mc-top { display: flex; justify-content: space-between; margin-bottom: 5px; }
        .mc-name { font-weight: bold; font-size: 0.95rem; }
        .mc-price { font-family: monospace; font-size: 1rem; }
        .mc-change { font-size: 0.8rem; font-weight: bold; }
        .col-green { color: #4caf50; }
        .col-red { color: #e57373; }
        .graph-img { width: 100%; height: auto; margin-top: 5px; border-radius: 4px; opacity: 0.9; }

        footer { text-align: center; margin-top: 50px; font-size: 0.8rem; color: #555; border-top: 1px solid #333; padding-top: 20px;}
    </style>
    """

    html = f"""<!DOCTYPE html>
<html lang="de">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FinanzBot Dashboard</title>
    {css}
</head>
<body>
    <div class="container">
        <header>
            <h1>FinanzBot <span style="color:#666">Dashboard</span></h1>
            <div class="timestamp">Stand: {now_str} (Berlin)</div>
        </header>
    """

    # 1. STATUS
    if not items:
        html += """
        <div class="status-card">
            <div class="status-ok">✅ Alles ruhig</div>
            <p style="color:#888; margin-top:10px;">Keine kritischen Signale.</p>
        </div>
        """
    else:
        html += f"""
        <div class="status-card" style="border-color: #d32f2f;">
            <div class="status-alert">🚨 {len(items)} Signale erkannt</div>
        </div>
        """

    # 2. SIGNALE (KI)
    if items:
        for i in items:
            score = i.vertrauen * 100 if i.vertrauen <= 1 else i.vertrauen
            sig_upper = i.signal.upper()
            
            if "VERKAUF" in sig_upper:
                badge_class = "bg-red"
                icon = "📉"
            elif "KAUF" in sig_upper or "NACHKAUFEN" in sig_upper:
                badge_class = "bg-green"
                icon = "💰"
            else:
                badge_class = "bg-blue"
                icon = "ℹ️"
            
            port_html = '<span class="portfolio-tag">MEIN PORTFOLIO</span>' if i.betrifft_portfolio else ""

            html += f"""
            <div class="card">
                <div class="card-header">
                    <div><span class="asset-name">{i.name}</span> {port_html}</div>
                    <span class="asset-type">{i.typ}</span>
                </div>
                <div style="margin-bottom: 10px;">
                    <span class="badge {badge_class}">{icon} {i.signal}</span>
                    <span style="font-size:0.9rem; margin-left:10px; color:{'#a5d6a7' if score>80 else '#ef9a9a'}">{score:.0f}% Konfidenz</span>
                </div>
                <p style="line-height:1.5; color:#ccc;">{i.begruendung}</p>
            </div>
            """
    
    # 3. MARKTÜBERSICHT (LIVE)
    if market_data:
        html += "<h2>📊 Live Portfolio-Check (Tagesverlauf)</h2>"
        html += '<div class="market-grid">'
        for m in market_data:
            color_class = "col-green" if m.change_percent >= 0 else "col-red"
            prefix = "+" if m.change_percent >= 0 else ""
            
            html += f"""
            <div class="market-card">
                <div class="mc-top">
                    <span class="mc-name">{m.name}</span>
                    <span class="mc-change {color_class}">{prefix}{m.change_percent:.2f}%</span>
                </div>
                <div class="mc-price">{m.price}</div>
                <img src="data:image/png;base64,{m.graph_base64}" class="graph-img" alt="Graph">
            </div>
            """
        html += '</div>'

    html += """
        <footer>&copy; 2025 All rights to NiklasATN</footer>
    </div>
</body>
</html>
    """

    with open("index.html", "w", encoding="utf-8") as f:
        f.write(html)
    print("✅ Dashboard (index.html) mit Marktdaten aktualisiert.")

# ===== MAIN =====
def main():
    if not CONFIG.get("sources"): return
    src = CONFIG["sources"][0]
    
    # A. News verarbeiten
    news = fetch_news_rss(src["url"], src.get("limit", 30))
    news = [n for n in news if is_recent(n) and relevance_score(n) >= 1]
    last_ids = load_last_ids()
    current_ids = {n["url"] for n in news}
    new_ids = current_ids - last_ids
    final_news = [n for n in news if n["url"] in new_ids]

    relevant_items = []
    
    # Nur KI fragen, wenn News da sind
    if final_news:
        ai_result = analyze_with_gemini(final_news[:12])
        if ai_result.ideen:
            for idee in ai_result.ideen:
                score = idee.vertrauen
                sig = idee.signal.upper()
                is_action = ("KAUF" in sig) or ("VERKAUF" in sig)

                if idee.betrifft_portfolio and score >= MIN_CONF_PORTFOLIO and is_action:
                    relevant_items.append(idee)
                elif (not idee.betrifft_portfolio) and score >= MIN_CONF_NEW_GEM and ("KAUF" in sig or "NACHKAUFEN" in sig):
                    relevant_items.append(idee)
            save_last_ids(last_ids.union(current_ids))
    else:
        print("🔄 Keine neuen News für KI.")

    # B. Marktdaten IMMER holen (damit die Graphen aktuell sind)
    market_data = get_market_data()

    # C. Dashboard bauen
    generate_dashboard(items=relevant_items if relevant_items else None, market_data=market_data)

if __name__ == "__main__":
    main()
