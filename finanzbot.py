import os, json, requests, time, re
import feedparser
import yfinance as yf
import matplotlib
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

STATE_FILE = "last_sent.json"
MAX_NEWS_AGE_HOURS = 12

# Filter
MIN_CONF_PORTFOLIO = 60
MIN_CONF_NEW_GEM = 85

# ===== PORTFOLIO MAPPING =====
PORTFOLIO_MAPPING = {
    "iShares MSCI World": "EUNL.DE",
    "Vanguard FTSE All-World": "VWCE.DE",
    "MSCI ACWI EUR": "IUSQ.DE",
    "Nasdaq 100": "EQQQ.DE",
    "Allianz SE": "ALV.DE",
    "Münchener Rück": "MUV2.DE",
    "BMW": "BMW.DE",
    "Berkshire Hathaway": "BRK-B",
    "Realty Income": "O",
    "Carnival": "CCL",
    "Snowflake": "SNOW",
    "Highland Copper": "HIC.F",
    "Bitcoin": "BTC-EUR"
}

# ===== LOGO MAPPING (DOMAINS) =====
# Präzise Domains für die Google Favicon API
LOGO_DOMAINS = {
    "iShares MSCI World": "blackrock.com",
    "Vanguard FTSE All-World": "vanguardinvestor.co.uk",
    "MSCI ACWI EUR": "blackrock.com",
    "Nasdaq 100": "invesco.com",
    "Allianz SE": "allianz.com",
    "Münchener Rück": "munichre.com",
    "BMW": "bmw.de",
    "Berkshire Hathaway": "berkshirehathaway.com",
    "Realty Income": "realtyincome.com",
    "Carnival": "carnivalcorp.com",
    "Snowflake": "snowflake.com",
    "Highland Copper": "highlandcopper.com",
    "Bitcoin": "bitcoin.org"
}

USER_PORTFOLIO = ", ".join(PORTFOLIO_MAPPING.keys())

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
    price_fmt: str
    change_pct: float
    change_abs: float
    currency_symbol: str
    graph_base64: str
    logo_url: str

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

def fetch_news_rss(url: str, limit: int = 50) -> List[Dict[str, Any]]:
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

# ===== FINANCE DATA =====
def get_market_data() -> List[MarketData]:
    print("📈 Lade Marktdaten...")
    data_list = []

    for name, ticker_symbol in PORTFOLIO_MAPPING.items():
        try:
            ticker = yf.Ticker(ticker_symbol)
            hist = ticker.history(period="1d", interval="15m")
            
            if hist.empty: continue

            current = hist['Close'].iloc[-1]
            open_price = hist['Open'].iloc[0]
            
            change_pct = ((current - open_price) / open_price) * 100
            change_abs = current - open_price
            
            currency = "€" if "EUR" in ticker_symbol or ".DE" in ticker_symbol or ".F" in ticker_symbol else "$"
            price_fmt = f"{current:.2f} {currency}"

            # Graph erstellen
            fig, ax = plt.subplots(figsize=(3.5, 1.2)) # Etwas breiter/höher
            fig.patch.set_facecolor('#1e1e1e')
            ax.set_facecolor('#1e1e1e')
            
            line_color = '#4caf50' if change_pct >= 0 else '#e57373'
            
            ax.plot(hist.index, hist['Close'], color=line_color, linewidth=2.5) # Linie etwas dicker
            
            # WICHTIG: Ränder hinzufügen, damit Spitzen nicht abgeschnitten werden
            ax.margins(x=0, y=0.2) 
            ax.axis('off')
            
            # Tight layout mit kleinem Padding, damit nichts den Rand berührt
            plt.tight_layout(pad=0.1)
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png', facecolor=fig.get_facecolor(), dpi=100)
            plt.close(fig)
            buf.seek(0)
            img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            
            # Google Favicon API (Zuverlässiger als Clearbit)
            domain = LOGO_DOMAINS.get(name, "google.com")
            logo_url = f"https://t2.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://{domain}&size=64"
            
            data_list.append(MarketData(
                name=name,
                price_fmt=price_fmt,
                change_pct=change_pct,
                change_abs=change_abs,
                currency_symbol=currency,
                graph_base64=img_base64,
                logo_url=logo_url
            ))

        except Exception as e:
            print(f"⚠️ Fehler bei {name}: {e}")
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
    # Zeit ohne "Berlin" Text
    now_str = datetime.now(ZoneInfo("Europe/Berlin")).strftime('%d.%m.%Y %H:%M')
    
    css = """
    <style>
        body { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; background-color: #121212; color: #e0e0e0; margin: 0; padding: 20px; }
        .container { max-width: 900px; margin: 0 auto; }
        
        header { border-bottom: 1px solid #333; padding-bottom: 20px; margin-bottom: 30px; display: flex; justify-content: space-between; align-items: center; }
        h1 { margin: 0; font-size: 1.5rem; letter-spacing: -0.5px; }
        h2 { border-bottom: 1px solid #333; padding-bottom: 10px; margin-top: 40px; color: #bbb; font-size: 1.1rem; display: flex; justify-content: space-between; align-items: center;}
        .timestamp { font-size: 0.9rem; color: #888; }
        
        .toggle-btn { background: #333; border: 1px solid #555; color: #eee; padding: 4px 10px; border-radius: 6px; cursor: pointer; font-size: 0.75rem; transition: background 0.2s; }
        .toggle-btn:hover { background: #444; }

        /* Status */
        .status-card { background: #1e1e1e; border-radius: 12px; padding: 15px; text-align: center; border: 1px solid #333; margin-bottom: 20px; }
        .status-ok { color: #4caf50; font-size: 1.1rem; font-weight: bold; }
        .status-alert { color: #e57373; font-size: 1.1rem; font-weight: bold; }
        
        /* Grid Layout */
        .grid-container { display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 15px; }
        
        /* Unified Card Style */
        .card { background: #1e1e1e; border: 1px solid #333; border-radius: 12px; padding: 15px; display: flex; flex-direction: column; height: 100%; box-sizing: border-box; transition: transform 0.2s; position: relative; overflow: hidden; }
        .card:hover { transform: translateY(-2px); border-color: #444; }
        
        /* Signal Specifics */
        .sig-header { display: flex; justify-content: space-between; margin-bottom: 8px; font-size: 0.9rem; font-weight: bold;}
        .sig-body { font-size: 0.85rem; color: #ccc; line-height: 1.4; flex-grow: 1; }
        .badge { padding: 3px 6px; border-radius: 4px; font-size: 0.7rem; font-weight: bold; text-transform: uppercase; }
        .bg-red { background: rgba(211, 47, 47, 0.2); color: #ef9a9a; border: 1px solid #d32f2f; }
        .bg-green { background: rgba(56, 142, 60, 0.2); color: #a5d6a7; border: 1px solid #388e3c; }

        /* Market Specifics */
        .mc-top { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 8px; z-index: 2; }
        .mc-info { display: flex; flex-direction: column; }
        .mc-name { font-weight: bold; font-size: 0.9rem; margin-bottom: 2px; }
        .mc-price { font-family: monospace; font-size: 1rem; color: #fff; }
        .mc-change { font-size: 0.75rem; font-weight: bold; }
        .col-green { color: #4caf50; }
        .col-red { color: #e57373; }
        
        .logo-img { width: 28px; height: 28px; border-radius: 6px; object-fit: contain; }
        
        /* Graph Styling */
        .graph-container { margin-top: auto; position: relative; height: 50px; width: 100%; overflow: hidden; }
        .graph-img { width: 110%; height: 100%; object-fit: cover; margin-left: -5%; } /* Leichter Zoom damit es randlos wirkt */

        footer { text-align: center; margin-top: 50px; font-size: 0.8rem; color: #555; border-top: 1px solid #333; padding-top: 20px;}
    </style>
    """
    
    js = """
    <script>
        let showPercent = true;
        function toggleCurrency() {
            showPercent = !showPercent;
            document.getElementById('toggleBtn').innerText = showPercent ? "Anzeige: %" : "Anzeige: €/$";
            document.querySelectorAll('.dynamic-change').forEach(el => {
                el.innerText = showPercent ? el.dataset.pct : el.dataset.abs;
            });
        }
    </script>
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
            <div class="timestamp">Stand: {now_str}</div>
        </header>
    """

    # 1. STATUS
    if not items:
        html += """
        <div class="status-card">
            <div class="status-ok">✅ Alles ruhig</div>
            <p style="color:#888; font-size: 0.9rem; margin-top:5px;">Kein akuter Handlungsbedarf.</p>
        </div>
        """
    else:
        html += f"""
        <div class="status-card" style="border-color: #d32f2f;">
            <div class="status-alert">🚨 {len(items)} Signale erkannt</div>
        </div>
        """

    # 2. SIGNALE
    if items:
        html += "<h2>⚡ Handlungsbedarf (KI)</h2>"
        html += '<div class="grid-container">'
        for i in items:
            score = i.vertrauen * 100 if i.vertrauen <= 1 else i.vertrauen
            sig_upper = i.signal.upper()
            
            if "VERKAUF" in sig_upper:
                badge_class = "bg-red"; icon = "📉"
            else:
                badge_class = "bg-green"; icon = "💰"
            
            short_desc = (i.begruendung[:100] + '..') if len(i.begruendung) > 100 else i.begruendung

            html += f"""
            <div class="card">
                <div class="sig-header">
                    <span style="color:#fff">{i.name}</span>
                    <span class="badge {badge_class}">{icon} {i.signal}</span>
                </div>
                <div style="font-size:0.75rem; color:#888; margin-bottom:5px;">Konfidenz: {score:.0f}%</div>
                <div class="sig-body">{short_desc}</div>
            </div>
            """
        html += '</div>'
    
    # 3. MARKET DATA
    if market_data:
        html += """
        <h2>
            <span>📊 Portfolio Live-Check</span>
            <button id="toggleBtn" class="toggle-btn" onclick="toggleCurrency()">Anzeige: %</button>
        </h2>
        <div class="grid-container">
        """
        for m in market_data:
            color_class = "col-green" if m.change_pct >= 0 else "col-red"
            prefix = "+" if m.change_pct >= 0 else ""
            
            pct_str = f"{prefix}{m.change_pct:.2f}%"
            abs_str = f"{prefix}{m.change_abs:.2f} {m.currency_symbol}"
            
            html += f"""
            <div class="card">
                <div class="mc-top">
                    <div class="mc-info">
                        <span class="mc-name">{m.name}</span>
                        <span class="mc-price">{m.price_fmt}</span>
                        <span class="mc-change {color_class} dynamic-change" 
                              data-pct="{pct_str}" 
                              data-abs="{abs_str}">
                            {pct_str}
                        </span>
                    </div>
                    <img src="{m.logo_url}" class="logo-img" onerror="this.style.display='none'">
                </div>
                <div class="graph-container">
                    <img src="data:image/png;base64,{m.graph_base64}" class="graph-img" alt="Chart">
                </div>
            </div>
            """
        html += '</div>'

    html += """
        <footer>
            all rights to niklasatn
        </footer>
    </div>
    """
    html += js
    html += """
</body>
</html>
    """

    with open("index.html", "w", encoding="utf-8") as f:
        f.write(html)
    print("✅ Dashboard aktualisiert.")

# ===== MAIN =====
def main():
    if not CONFIG.get("sources"): return
    src = CONFIG["sources"][0]
    
    news = fetch_news_rss(src["url"], src.get("limit", 40))
    news = [n for n in news if is_recent(n) and relevance_score(n) >= 1]
    last_ids = load_last_ids()
    current_ids = {n["url"] for n in news}
    new_ids = current_ids - last_ids
    final_news = [n for n in news if n["url"] in new_ids]

    relevant_items = []
    
    if final_news:
        ai_result = analyze_with_gemini(final_news[:15])
        if ai_result.ideen:
            for idee in ai_result.ideen:
                score = idee.vertrauen
                sig = idee.signal.upper()
                is_action = ("KAUF" in sig) or ("VERKAUF" in sig)

                if is_action:
                    if idee.betrifft_portfolio and score >= MIN_CONF_PORTFOLIO:
                        relevant_items.append(idee)
                    elif (not idee.betrifft_portfolio) and score >= MIN_CONF_NEW_GEM:
                        relevant_items.append(idee)
            
            save_last_ids(last_ids.union(current_ids))

    market_data = get_market_data()
    generate_dashboard(items=relevant_items if relevant_items else None, market_data=market_data)

if __name__ == "__main__":
    main()
