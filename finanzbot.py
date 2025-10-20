import os, requests, smtplib, ssl, json
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from google import genai
from pydantic import BaseModel
from typing import List, Literal

# --- Konfiguration ---
PORTFOLIO = {"AAPL": 0.25, "MSFT": 0.2, "SPY": 0.3, "BTC": 0.1}
KEYWORDS = ["dividend", "earnings", "profit", "guidance", "merger", "acquisition"]

ALPHAVANTAGE_KEY = os.getenv("ALPHAVANTAGE_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("EMAIL_PASS")
EMAIL_TO   = os.getenv("EMAIL_TO")

# --- Modelle für Gemini ---
class ActionItem(BaseModel):
    ticker: str
    action: Literal["BUY", "SELL", "HOLD"]
    confidence: float
    rationale: str

class ModelOutput(BaseModel):
    actions: List[ActionItem]

# --- Hilfsfunktionen ---
def is_potentially_relevant(news_item):
    tickers = [x.get("ticker") for x in news_item.get("ticker_sentiment", [])]
    title = news_item.get("title", "").lower()
    return any(t in PORTFOLIO for t in tickers) or any(k in title for k in KEYWORDS)

def fetch_news(limit=40):
    url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&sort=LATEST&limit={limit}&apikey={ALPHAVANTAGE_KEY}"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return [n for n in r.json().get("feed", []) if is_potentially_relevant(n)]

def analyze_with_gemini(news_list):
    client = genai.Client(api_key=GEMINI_API_KEY)
    prompt = f"""
Analysiere folgende Finanznachrichten in Bezug auf mein Portfolio: {', '.join(PORTFOLIO.keys())}.
Empfiehl für betroffene Ticker BUY, SELL oder HOLD mit confidence (0..1) und kurzer Begründung.
Rückgabe als JSON-Liste unter 'actions'.
"""
    summaries = "\n".join([f"- {n['title']} ({', '.join([x.get('ticker') for x in n.get('ticker_sentiment', []) if x.get('ticker')] or ['—'])})" for n in news_list])
    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt + "\n" + summaries,
        config={
            "response_mime_type": "application/json",
            "response_schema": ModelOutput,
        },
    )
    return ModelOutput.model_validate_json(resp.text)

def send_email(subject, html):
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = EMAIL_USER
    msg["To"] = EMAIL_TO
    msg.attach(MIMEText(html, "html"))
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=ssl.create_default_context()) as server:
        server.login(EMAIL_USER, EMAIL_PASS)
        server.sendmail(EMAIL_USER, EMAIL_TO, msg.as_string())

def main():
    news = fetch_news()
    if not news:
        return
    result = analyze_with_gemini(news)
    if not result.actions:
        return
    html = "<h3>Finanzbot – Neue Signale</h3><ul>"
    for a in result.actions:
        html += f"<li><b>{a.ticker}</b>: {a.action} ({a.confidence:.0%}) – {a.rationale}</li>"
    html += "</ul>"
    send_email("Finanzbot: Neue Handlungssignale", html)

if __name__ == "__main__":
    main()
