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

            # Graph erstellen (Extrem kompakt)
            fig, ax = plt.subplots(figsize=(3, 1)) 
            fig.patch.set_facecolor('#1e1e1e')
            ax.set_facecolor('#1e1e1e')
            
            line_color = '#4caf50' if change_pct >= 0 else '#e57373'
            
            # Dynamische Limits, damit die Linie den Platz nutzt
            y_vals = hist['Close']
            y_min, y_max = y_vals.min(), y_vals.max()
            margin = (y_max - y_min) * 0.05 # Nur 5% Rand
            if margin == 0: margin = 0.1
            
            ax.set_ylim(y_min - margin, y_max + margin)
            ax.set_xlim(hist.index[0], hist.index[-1])
            
            ax.plot(hist.index, y_vals, color=line_color, linewidth=2)
            ax.axis('off')
            
            buf = io.BytesIO()
            # pad_inches=0 entfernt ALLE weißen Ränder
            plt.savefig(buf, format='png', facecolor=fig.get_facecolor(), bbox_inches='tight', pad_inches=0)
            plt.close(fig)
            buf.seek(0)
            img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            
            data_list.append(MarketData(
                name=name,
                price_fmt=price_fmt,
                change_pct=change_pct,
                change_abs=change_abs,
                currency_symbol=currency,
                graph_base64=img_base64
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
        .grid-container { display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 15px; }
        
        /* Unified Card Style */
        .card-base { background: #1e1e1e; border: 1px solid #333; border-radius: 12px; padding: 15px; display: flex; flex-direction: column; transition: all 0.2s ease; position: relative; overflow: hidden; }
        
        /* Signal Card */
        .sig-card { cursor: pointer; min-height: 100px; max-height: 140px; }
        .sig-card:hover { border-color: #555; transform: translateY(-2px); box-shadow: 0 4px 10px rgba(0,0,0,0.4); }
        
        /* Neue Chancen: Blauer Rahmen */
        .sig-card.card-new { border: 1px solid #2196f3; box-shadow: 0 0 8px rgba(33, 150, 243, 0.2); }
        
        .sig-card.expanded { grid-row: span 2; max-height: none; background: #252525; border-color: #777; z-index: 10; }
        
        .sig-header { display: flex; justify-content: space-between; margin-bottom: 8px; font-size: 0.9rem; font-weight: bold; align-items: center; }
        
        .sig-body { font-size: 0.85rem; color: #999; line-height: 1.4; 
                    display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden; }
        .sig-card.expanded .sig-body { -webkit-line-clamp: unset; overflow: visible; color: #fff; }
        
        .expand-hint { font-size: 0.7rem; color: #555; text-align: center; margin-top: auto; padding-top: 5px; }
        .sig-card.expanded .expand-hint { display: none; }

        .badge { padding: 3px 6px; border-radius: 4px; font-size: 0.7rem; font-weight: bold; text-transform: uppercase; }
        .bg-red { background: rgba(211, 47, 47, 0.2); color: #ef9a9a; border: 1px solid #d32f2f; }
        .bg-green { background: rgba(56, 142, 60, 0.2); color: #a5d6a7; border: 1px solid #388e3c; }

        .tag-portfolio { background: linear-gradient(45deg, #6a1b9a, #4a148c); color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.65rem; font-weight: bold; letter-spacing: 0.5px; }
        .tag-new { background: linear-gradient(45deg, #0288d1, #01579b); color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.65rem; font-weight: bold; letter-spacing: 0.5px; }

        /* Market Card */
        .market-card { height: 160px; justify-content: space-between; padding-bottom: 0; }
        .mc-top { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 5px; z-index: 2; padding-top:5px; }
        .mc-info { display: flex; flex-direction: column; width: 100%; }
        .mc-name { font-weight: bold; font-size: 0.9rem; margin-bottom: 2px; }
        .mc-price { font-family: monospace; font-size: 1rem; color: #fff; }
        .mc-change { font-size: 0.75rem; font-weight: bold; }
        .col-green { color: #4caf50; }
        .col-red { color: #e57373; }
        
        /* Graph Container - Randlos */
        .graph-container { 
            position: absolute; 
            bottom: 0; left: 0; right: 0; 
            height: 70px; /* Höhe des Graphen */
            overflow: hidden; 
            border-bottom-left-radius: 12px;
            border-bottom-right-radius: 12px;
            opacity: 0.8;
        }
        .graph-img { width: 100%; height: 100%; object-fit: cover; } 

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
        function toggleCard(el) {
            el.classList.toggle('expanded');
        }
    </script>
    """

    html = f"""<!DOCTYPE html>
<html lang="de">
<head>
    <meta charset="UTF-8">
    <meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate">
    <meta http-equiv="Pragma" content="no-cache">
    <meta http-equiv="Expires" content="0">
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
            <p style="color:#888; font-size: 0.9rem; margin-top:5px;">Kein akuter Handlungsbedarf (Kauf/Verkauf).</p>
        </div>
        """
    else:
        count = len(items)
        label = "Signal" if count == 1 else "Signale"
        html += f"""
        <div class="status-card" style="border-color: #d32f2f;">
            <div class="status-alert">🚨 {count} {label} erkannt</div>
        </div>
        """

    # 2. SIGNALE
    if items:
        html += "<h2>⚡ Handlungsbedarf (KI)</h2>"
        html += '<div class="grid-container">'
        for i in items:
            score = i.vertrauen * 100 if i.vertrauen <= 1 else i.vertrauen
            sig_upper = i.signal.upper()
            
            badge_class = "bg-red" if "VERKAUF" in sig_upper else "bg-green"
            icon = "📉" if "VERKAUF" in sig_upper else "💰"
            
            # Unterscheidung: Portfolio vs Neu
            if i.betrifft_portfolio:
                tag_html = '<span class="tag-portfolio">MEIN PORTFOLIO</span>'
                extra_class = ""
            else:
                tag_html = '<span class="tag-new">💎 NEU ENTDECKT</span>'
                extra_class = "card-new"

            html += f"""
            <div class="card-base sig-card {extra_class}" onclick="toggleCard(this)">
                <div class="sig-header">
                    <span style="color:#fff">{i.name}</span>
                    <span class="badge {badge_class}">{icon} {i.signal}</span>
                </div>
                
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                    {tag_html}
                    <div style="font-size:0.75rem; color:#666;">Konfidenz: {score:.0f}%</div>
                </div>

                <div class="sig-body">{i.begruendung}</div>
                <div class="expand-hint">▼ klick</div>
            </div>
            """
        html += '</div>'
    
    # 3. MARKET DATA (Clean & Full Width Graph)
    if market_data:
        html += """
        <h2>
            <span>📊 Portfolio Tagesverlauf</span>
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
            <div class="card-base market-card">
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
                </div>
                <div class="graph-container">
                    <img src="data:image/png;base64,{m.graph_base64}" class="graph-img" alt="Chart">
                </div>
            </div>
            """
        html += '</div>'

    html += """
        <footer>
            all rights to niklasatn | Version: Clean & Tight
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
                    # Regel 1: Portfolio >= 60%
                    if idee.betrifft_portfolio and score >= MIN_CONF_PORTFOLIO:
                        relevant_items.append(idee)
                    # Regel 2: Neu >= 95% (Sehr streng)
                    elif (not idee.betrifft_portfolio) and score >= MIN_CONF_NEW_GEM:
                        relevant_items.append(idee)
            
            save_last_ids(last_ids.union(current_ids))

    market_data = get_market_data()
    generate_dashboard(items=relevant_items if relevant_items else None, market_data=market_data)

if __name__ == "__main__":
    main()
