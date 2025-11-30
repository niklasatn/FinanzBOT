import os, json, requests, time, re
import feedparser
import yfinance as yf
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import io
import base64
import pandas as pd
import google.generativeai as genai
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

# ===== KONFIGURATION =====
with open("config.json", "r", encoding="utf-8") as f:
    CONFIG = json.load(f)

KEYWORDS = [k.lower() for k in CONFIG["keywords"]]
PROMPTS = CONFIG["prompts"]

STATE_FILE = "last_sent.json"
OUTPUT_FILE = "index.html"
MAX_NEWS_AGE_HOURS = 12

# Filter
MIN_CONF_PORTFOLIO = 60
MIN_CONF_NEW_GEM = 95

# ===== PORTFOLIO MAPPING =====
PORTFOLIO_MAPPING = {
    "iShares MSCI World": "EUNL.DE",
    "Vanguard FTSE All-World": "VWCE.DE",
    "MSCI ACWI EUR": "IUSQ.DE",
    "Nasdaq 100": "EQQQ.DE",
    "Core Euro Gov Bond": "EUNH.DE",
    "Allianz SE": "ALV.DE",
    "Münchener Rück": "MUV2.DE",
    "BMW": "BMW.DE",
    "Berkshire Hathaway": "BRK-B",
    "Realty Income": "O",
    "Carnival": "CCL",
    "Snowflake": "SNOW",
    "Highland Copper": "HIC.V",
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
    rsi: Optional[float] = None
    sma200_dist_pct: Optional[float] = None

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

def fetch_news_rss(url: str, limit: int = 20) -> List[Dict[str, Any]]:
    print(f"📡 Lade RSS Feed: {url} ...")
    try:
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
    except Exception as e:
        print(f"⚠️ Fehler beim Laden von {url}: {e}")
        return []

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

# ===== CALCULATIONS =====
def calculate_rsi(series, period=14):
    if len(series) < period: return None
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# ===== FINANCE DATA =====
def get_market_data() -> List[MarketData]:
    print("📈 Lade Marktdaten (Graph Fix V3)...")
    data_list = []

    for name, ticker_symbol in PORTFOLIO_MAPPING.items():
        try:
            ticker = yf.Ticker(ticker_symbol)
            
            # 1. Intraday
            hist_intra = ticker.history(period="1d", interval="15m")
            if hist_intra.empty: continue

            current = hist_intra['Close'].iloc[-1]
            open_price = hist_intra['Open'].iloc[0]
            change_pct = ((current - open_price) / open_price) * 100
            change_abs = current - open_price
            currency = "€" if "EUR" in ticker_symbol or ".DE" in ticker_symbol or ".F" in ticker_symbol else "$"
            price_fmt = f"{current:.2f} {currency}"

            # 2. Indikatoren
            rsi_val = None
            sma200_dist = None
            try:
                hist_long = ticker.history(period="1y")
                if not hist_long.empty and len(hist_long) > 50:
                    rsi_series = calculate_rsi(hist_long['Close'])
                    if rsi_series is not None and not pd.isna(rsi_series.iloc[-1]):
                        rsi_val = rsi_series.iloc[-1]
                    if len(hist_long) >= 200:
                        sma200 = hist_long['Close'].rolling(window=200).mean().iloc[-1]
                        if not pd.isna(sma200):
                            sma200_dist = ((current - sma200) / sma200) * 100
            except: pass

            # --- GRAPH FIX ---
            fig, ax = plt.subplots(figsize=(3, 1))
            fig.patch.set_facecolor('#1e1e1e')
            ax.set_facecolor('#1e1e1e')
            
            line_color = '#4caf50' if change_pct >= 0 else '#e57373'
            y_vals = hist_intra['Close']
            
            # 1. Plotten
            ax.plot(hist_intra.index, y_vals, color=line_color, linewidth=2.5) # Linie dicker
            
            # 2. Limits MANUELL setzen mit 20% Puffer oben/unten
            y_min = y_vals.min()
            y_max = y_vals.max()
            rng = y_max - y_min
            if rng == 0: rng = y_max * 0.01 if y_max != 0 else 1.0 # Schutz vor Flatline
            
            # Puffer addieren
            buffer = rng * 0.3 # 30% Puffer!
            ax.set_ylim(y_min - buffer, y_max + buffer)
            ax.set_xlim(hist_intra.index[0], hist_intra.index[-1])
            
            # 3. Alles ausblenden
            ax.axis('off')
            
            buf = io.BytesIO()
            # 4. Speichern mit Pad Inches (der "unsichtbare Rand")
            plt.savefig(buf, format='png', facecolor=fig.get_facecolor(), bbox_inches='tight', pad_inches=0.1)
            plt.close(fig)
            buf.seek(0)
            img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            
            data_list.append(MarketData(
                name=name, price_fmt=price_fmt, change_pct=change_pct, change_abs=change_abs,
                currency_symbol=currency, graph_base64=img_base64, rsi=rsi_val, sma200_dist_pct=sma200_dist
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
    
    # ---------------------------------------------------------
    # HTML TEMPLATE
    # ---------------------------------------------------------
    HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="de">
<head>
    <meta charset="UTF-8">
    <meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate">
    <meta http-equiv="Pragma" content="no-cache">
    <meta http-equiv="Expires" content="0">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FinanzBot Dashboard</title>
    <style>
        body { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; background-color: #121212; color: #e0e0e0; margin: 0; padding: 20px; }
        .container { max-width: 900px; margin: 0 auto; }
        header { border-bottom: 1px solid #333; padding-bottom: 20px; margin-bottom: 30px; display: flex; justify-content: space-between; align-items: center; }
        h1 { margin: 0; font-size: 1.5rem; letter-spacing: -0.5px; }
        .timestamp { font-size: 0.9rem; color: #888; }
        .toggle-btn { background: #333; border: 1px solid #555; color: #eee; padding: 4px 10px; border-radius: 6px; cursor: pointer; font-size: 0.75rem; transition: background 0.2s; }
        .toggle-btn:hover { background: #444; }
        .status-card { background: #1e1e1e; border-radius: 12px; padding: 15px; text-align: center; border: 1px solid #333; margin-bottom: 20px; }
        .status-ok { color: #4caf50; font-size: 1.1rem; font-weight: bold; }
        .status-alert { color: #e57373; font-size: 1.1rem; font-weight: bold; }
        .grid-container { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 15px; }
        .card-base { background: #1e1e1e; border: 1px solid #333; border-radius: 12px; padding: 15px; display: flex; flex-direction: column; transition: all 0.2s ease; position: relative; overflow: hidden; }
        .sig-card { cursor: pointer; min-height: 100px; max-height: 140px; }
        .sig-card:hover { border-color: #555; transform: translateY(-2px); box-shadow: 0 4px 10px rgba(0,0,0,0.4); }
        .sig-card.card-new { border: 1px solid #2196f3; box-shadow: 0 0 8px rgba(33, 150, 243, 0.2); }
        .sig-card.expanded { grid-row: span 2; max-height: none; background: #252525; border-color: #777; z-index: 10; }
        .sig-header { display: flex; justify-content: space-between; margin-bottom: 8px; font-size: 0.9rem; font-weight: bold; align-items: center; }
        .sig-body { font-size: 0.85rem; color: #999; line-height: 1.4; display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden; }
        .sig-card.expanded .sig-body { -webkit-line-clamp: unset; overflow: visible; color: #fff; }
        .expand-hint { font-size: 0.7rem; color: #555; text-align: center; margin-top: auto; padding-top: 5px; }
        .sig-card.expanded .expand-hint { display: none; }
        .badge { padding: 3px 6px; border-radius: 4px; font-size: 0.7rem; font-weight: bold; text-transform: uppercase; }
        .bg-red { background: rgba(211, 47, 47, 0.2); color: #ef9a9a; border: 1px solid #d32f2f; }
        .bg-green { background: rgba(56, 142, 60, 0.2); color: #a5d6a7; border: 1px solid #388e3c; }
        .tag-portfolio { background: linear-gradient(45deg, #6a1b9a, #4a148c); color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.65rem; font-weight: bold; letter-spacing: 0.5px; }
        .tag-new { background: linear-gradient(45deg, #0288d1, #01579b); color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.65rem; font-weight: bold; letter-spacing: 0.5px; }
        .market-card { height: 180px; justify-content: space-between; padding-bottom: 0; }
        .mc-top { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 5px; z-index: 2; padding-top:5px; }
        .mc-info { display: flex; flex-direction: column; width: 100%; }
        .mc-name { font-weight: bold; font-size: 0.9rem; margin-bottom: 2px; }
        .mc-price { font-family: monospace; font-size: 1rem; color: #fff; }
        .mc-change { font-size: 0.75rem; font-weight: bold; }
        .indicator-row { display: flex; gap: 8px; margin-top: 5px; margin-bottom: 5px; font-size: 0.7rem; font-weight: bold; flex-wrap: wrap; }
        .ind-pill { padding: 2px 6px; border-radius: 4px; background: #333; color: #ccc; border: 1px solid #444; }
        .ind-warn { color: #ef9a9a; border-color: #d32f2f; }
        .ind-good { color: #a5d6a7; border-color: #388e3c; }
        .ind-mid  { color: #ffe082; border-color: #ffca28; }
        .col-green { color: #4caf50; }
        .col-red { color: #e57373; }
        .graph-container { position: absolute; bottom: 0; left: 0; right: 0; height: 60px; overflow: hidden; border-bottom-left-radius: 12px; border-bottom-right-radius: 12px; opacity: 0.8; }
        /* Contain sorgt dafür dass das komplette Bild inkl. Padding sichtbar ist */
        .graph-img { width: 100%; height: 100%; object-fit: contain; object-position: center bottom; } 
        .legend-box { margin-top: 50px; border-top: 1px solid #333; padding-top: 20px; color: #888; font-size: 0.85rem; }
        .legend-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 20px; margin-top: 10px; }
        .legend-item h4 { color: #ccc; margin: 0 0 8px 0; font-size: 0.9rem; border-bottom: 1px solid #444; display: inline-block; padding-bottom: 2px; }
        .legend-item ul { list-style: none; padding: 0; margin: 0; }
        .legend-item li { margin-bottom: 6px; display: flex; align-items: center; gap: 8px; line-height: 1.3; }
        h2 { border-bottom: 1px solid #333; padding-bottom: 10px; margin-top: 40px; color: #bbb; font-size: 1.1rem; display: flex; justify-content: space-between; align-items: center;}
        footer { text-align: center; margin-top: 40px; font-size: 0.8rem; color: #555; padding-bottom: 20px;}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>FinanzBot <span style="color:#666">Dashboard</span></h1>
            <div class="timestamp">Stand: {{TIMESTAMP}}</div>
        </header>

        {{STATUS_SECTION}}

        {{SIGNALS_SECTION}}
    
        {{MARKET_SECTION}}

        <div class="legend-box">
            <div style="font-weight:bold; margin-bottom:15px; color:#eee;">Erklärung der Indikatoren</div>
            <div class="legend-grid">
                <div class="legend-item">
                    <h4>RSI (14)</h4>
                    <ul>
                        <li><span class="ind-pill ind-warn">🔥 > 70</span>: Markt "heiß" (Verkaufsrisiko)</li>
                        <li><span class="ind-pill ind-mid">⚖️ 30-70</span>: Neutral</li>
                        <li><span class="ind-pill ind-good">❄️ < 30</span>: "Billig" (Kaufchance)</li>
                    </ul>
                </div>
                <div class="legend-item">
                    <h4>Trend (SMA 200)</h4>
                    <ul>
                        <li><span class="ind-pill ind-good">📈 Aufwärts</span>: Kurs > 200-Tage-Linie.</li>
                        <li><span class="ind-pill ind-warn">📉 Abwärts</span>: Kurs < 200-Tage-Linie.</li>
                    </ul>
                </div>
            </div>
        </div>

        <footer>
            all rights to niklasatn | Version: 1.0
        </footer>
    </div>

    <script>
        let showPercent = true;
        function toggleCurrency() {
            showPercent = !showPercent;
            document.getElementById('toggleBtn').innerText = showPercent ? "Anzeige: %" : "Anzeige: €/$";
            document.querySelectorAll('.dynamic-change').forEach(el => {
                el.innerText = showPercent ? el.dataset.pct : el.dataset.abs;
            });
        }
        function toggleCard(el) { el.classList.toggle('expanded'); }
    </script>
</body>
</html>"""

    # -- SECTIONS --
    if not items:
        status_html = """
        <div class="status-card">
            <div class="status-ok">✅ Alles ruhig</div>
            <p style="color:#888; font-size: 0.9rem; margin-top:5px;">Kein akuter Handlungsbedarf (Kauf/Verkauf).</p>
        </div>
        """
    else:
        count = len(items)
        label = "Signal" if count == 1 else "Signale"
        status_html = f"""
        <div class="status-card" style="border-color: #d32f2f;">
            <div class="status-alert">🚨 {count} {label} erkannt</div>
        </div>
        """

    if items:
        signals_html = '<h2>⚡ Handlungsbedarf (KI)</h2><div class="grid-container">'
        for i in items:
            score = i.vertrauen * 100 if i.vertrauen <= 1 else i.vertrauen
            sig_upper = i.signal.upper()
            badge_class = "bg-red" if "VERKAUF" in sig_upper else "bg-green"
            icon = "📉" if "VERKAUF" in sig_upper else "💰"
            
            if i.betrifft_portfolio:
                tag_html = '<span class="tag-portfolio">MEIN PORTFOLIO</span>'
                extra_class = ""
            else:
                tag_html = '<span class="tag-new">💎 NEU ENTDECKT</span>'
                extra_class = "card-new"

            signals_html += f"""
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
        signals_html += '</div>'
    else:
        signals_html = ""

    market_html = ""
    if market_data:
        market_html = """
        <h2>
            <span>📊 Portfolio Tagesverlauf</span>
            <button id="toggleBtn" class="toggle-btn" onclick="toggleCurrency()">Anzeige: %</button>
        </h2>
        <div class="grid-container">
        """
        for m in market_data:
            color_class = "col-green" if m.change_pct >= 0 else "col-red"
            prefix = "+" if m.change_pct >= 0 else ""
            
            ind_html = '<div class="indicator-row">'
            if m.rsi is not None:
                if m.rsi > 70: rsi_cls="ind-warn"; rsi_ico="🔥"
                elif m.rsi < 30: rsi_cls="ind-good"; rsi_ico="❄️"
                else: rsi_cls="ind-mid"; rsi_ico="⚖️"
                ind_html += f'<span class="ind-pill {rsi_cls}">{rsi_ico} RSI {m.rsi:.0f}</span>'
            
            if m.sma200_dist_pct is not None:
                if m.sma200_dist_pct > 0: sma_cls="ind-good"; sma_txt="📈 Trend"
                else: sma_cls="ind-warn"; sma_txt="📉 Trend"
                ind_html += f'<span class="ind-pill {sma_cls}">{sma_txt}</span>'
            ind_html += '</div>'

            pct_str = f"{prefix}{m.change_pct:.2f}%"
            abs_str = f"{prefix}{m.change_abs:.2f} {m.currency_symbol}"
            
            market_html += f"""
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
                        {ind_html}
                    </div>
                </div>
                <div class="graph-container">
                    <img src="data:image/png;base64,{m.graph_base64}" class="graph-img" alt="Chart">
                </div>
            </div>
            """
        market_html += '</div>'

    final_html = HTML_TEMPLATE.replace("{{TIMESTAMP}}", now_str)
    final_html = final_html.replace("{{STATUS_SECTION}}", status_html)
    final_html = final_html.replace("{{SIGNALS_SECTION}}", signals_html)
    final_html = final_html.replace("{{MARKET_SECTION}}", market_html)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(final_html)
    print(f"✅ Dashboard ({OUTPUT_FILE}) aus internem Template erstellt.")

# ===== MAIN =====
def main():
    if not CONFIG.get("sources"): return
    
    # News sammeln
    all_news = []
    seen_links = set()
    for src in CONFIG["sources"]:
        news = fetch_news_rss(src["url"], src.get("limit", 15))
        for n in news:
            if n["url"] not in seen_links:
                all_news.append(n)
                seen_links.add(n["url"])
    all_news.sort(key=lambda x: x["time_published"], reverse=True)
    
    # Filtern
    filtered_news = [n for n in all_news if is_recent(n) and relevance_score(n) >= 1]
    
    last_ids = load_last_ids()
    current_ids = {n["url"] for n in filtered_news}
    new_ids = current_ids - last_ids
    final_news_for_ai = [n for n in filtered_news if n["url"] in new_ids]

    relevant_items = []
    if final_news_for_ai:
        ai_result = analyze_with_gemini(final_news_for_ai[:20])
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
