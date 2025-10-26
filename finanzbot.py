import os, json, requests, importlib, time, re, hashlib
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
MAX_NEWS_AGE_HOURS = 6  # Nur News der letzten x Stunden prüfen

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

# ===== UTILS =====
def parse_iso_dt(value: str) -> Optional[datetime]:
    """Versucht diverse Meta-Formate in UTC zu parsen."""
    if not value:
        return None
    fmts = [
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S.%f%z",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%d %H:%M:%S%z",
        "%Y-%m-%d %H:%M:%S",
    ]
    for fmt in fmts:
        try:
            dt = datetime.strptime(value, fmt)
            if not dt.tzinfo:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except Exception:
            continue
    return None

def normalize_title(t: str) -> str:
    t = re.sub(r"[^A-Za-z0-9äöüÄÖÜß ]+", " ", t.lower())
    t = re.sub(r"\s+", " ", t).strip()
    return t

def hash_id(url_or_title: str) -> str:
    return hashlib.sha256(url_or_title.encode("utf-8")).hexdigest()[:16]

# ===== GOOGLE CSE: NEWS HOLEN =====
def fetch_news_google(api_key: str, cx: str, queries: List[str], limit: int = 40) -> List[Dict[str, Any]]:
    """
    Holt News über Google Programmable Search (CSE) mit dateRestrict=h6.
    Dedup über Link. Liefert vereinheitlichte Felder: title, url, summary, source, time_published
    """
    base = "https://www.googleapis.com/customsearch/v1"
    collected: Dict[str, Dict[str, Any]] = {}
    per_page = 10

    for q in queries:
        start = 1
        while start <= 100 and len(collected) < limit:
            params = {
                "key": api_key,
                "cx": cx,
                "q": q,
                "num": per_page,
                "start": start,
                "dateRestrict": f"h{MAX_NEWS_AGE_HOURS}",
                "safe": "off",
            }
            try:
                r = requests.get(base, params=params, timeout=25)
                r.raise_for_status()
                data = r.json()
            except Exception as e:
                print("⚠️ Fehler bei Google CSE:", e)
                break

            items = data.get("items", []) or []
            if not items:
                break

            for it in items:
                link = it.get("link")
                title = it.get("title") or ""
                snippet = it.get("snippet") or ""
                pagemap = (it.get("pagemap") or {})
                metatags = (pagemap.get("metatags") or [{}])[0] if pagemap.get("metatags") else {}
                source = (it.get("displayLink") or "").lower()

                # Datum aus Metadaten ziehen
                published = None
                candidates = [
                    metatags.get("article:published_time"),
                    metatags.get("og:published_time"),
                    metatags.get("article:modified_time"),
                    metatags.get("og:updated_time"),
                ]
                # NewsArticle-Strukturen
                for na in pagemap.get("newsarticle", []) or []:
                    if not published and na.get("datepublished"):
                        published = parse_iso_dt(na.get("datepublished"))
                if not published:
                    for c in candidates:
                        if c:
                            published = parse_iso_dt(c)
                            if published:
                                break

                if not link or not title:
                    continue

                nid = link  # dedup über URL
                if nid in collected:
                    continue

                collected[nid] = {
                    "title": title,
                    "url": link,
                    "summary": snippet,
                    "source": source,
                    "time_published": published.isoformat() if published else None,
                }

            start += per_page
            if not data.get("queries", {}).get("nextPage"):
                break

    return list(collected.values())[:limit]

# ===== ZEITFILTER =====
def is_recent(item: dict) -> bool:
    ts = item.get("time_published")
    if not ts:
        # Falls kein Zeitstempel – CSE ist bereits auf hN eingeschränkt → zulassen
        return True
    try:
        published = datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(timezone.utc)
        now = datetime.now(timezone.utc)
        return (now - published) <= timedelta(hours=MAX_NEWS_AGE_HOURS)
    except Exception:
        return True

# ===== RELEVANZBEWERTUNG =====
GLOBAL_SIGNAL_TERMS = [
    "fed", "ecb", "boj", "boe", "interest rate", "rate hike", "rate cut", "policy rate",
    "cpi", "ppi", "inflation", "deflation", "recession", "gdp", "jobs report", "unemployment",
    "credit rating", "downgrade", "default", "bankruptcy", "bailout",
    "opec", "oil", "gas", "geopolitical", "attack", "war", "escalation",
    "merger", "acquisition", "profit warning", "guidance cut", "earnings beat", "earnings miss",
    "bond yield", "treasury yield", "real estate", "housing", "china", "europe", "us economy", "eurozone"
]

def relevance_score(item: dict) -> int:
    text = (item.get("title", "") + " " + item.get("summary", "")).lower()

    score = 0
    # Basisscore: KEYWORDS aus config
    for k in KEYWORDS:
        if k in text:
            score += 1

    # Globale Signalbegriffe
    for g in GLOBAL_SIGNAL_TERMS:
        if g in text:
            score += 2

    # Starke Trigger
    strong_triggers = [
        "rate decision", "emergency meeting", "flash crash", "limit down", "halted trading",
        "terror", "invasion", "sanction", "capital controls", "liquidity crisis",
        "profit warning", "bank run"
    ]
    for s in strong_triggers:
        if s in text:
            score += 3

    # Headline-Länge (zu kurz = eher Rauschen)
    if len(item.get("title","")) >= 60:
        score += 1

    return score

def group_topics(items: List[dict]) -> Dict[str, List[dict]]:
    """
    Clustert grob nach normalisiertem Titelkern (bis zu ersten 10 Wörter).
    """
    buckets: Dict[str, List[dict]] = {}
    for it in items:
        key = " ".join(normalize_title(it.get("title","")).split()[:10])
        buckets.setdefault(key, []).append(it)
    return buckets

def should_notify(relevant_items: List[dict]) -> bool:
    """
    Benachrichtigen nur, wenn:
    - mind. 1 Item mit sehr hohem Score (>=6), ODER
    - mind. 3 unabhängige Quellen berichten über dasselbe Thema (Cluster mit Größe >=3) UND Durchschnittsscore im Cluster >=3, ODER
    - mind. 6 relevante Items gesamt (breite Marktrelevanz).
    """
    if not relevant_items:
        return False

    # Einzelner sehr hoher Score
    if any(relevance_score(it) >= 6 for it in relevant_items):
        return True

    # Clusteranalyse
    clusters = group_topics(relevant_items)
    for _, lst in clusters.items():
        if len(lst) >= 3:
            avg_score = sum(relevance_score(x) for x in lst) / len(lst)
            if avg_score >= 3:
                return True

    # Breites Signal
    if len(relevant_items) >= 6:
        return True

    return False

def filter_relevant(items: List[dict]) -> List[dict]:
    return [n for n in items if is_recent(n) and relevance_score(n) >= 2]

# ===== GEMINI ANALYSE =====
def analyze_with_gemini(news_items: List[dict]) -> IdeaOutput:
    raw = ""
    try:
        if importlib.util.find_spec("google.genai"):
            from google import genai
            client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
            bullets = []
            for n in news_items:
                title = n.get("title", "")
                url = n.get("url", "")
                bullets.append(f"- {title}\n  Quelle: {url}")

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
                bullets.append(f"- {title}\n  Quelle: {url}")

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

# ===== MAIN =====
def main():
    src = CONFIG["sources"][0]
    api_key = os.getenv(src["api_key_env"])
    cx = os.getenv(src["cx_env"])
    queries = src["queries"]
    limit = src.get("limit", 40)

    if not api_key or not cx:
        print("⚠️ GOOGLE CSE Keys fehlen (GOOGLE_CSE_KEY / GOOGLE_CSE_CX).")
        return

    news = fetch_news_google(api_key, cx, queries, limit)

    # Nur aktuelle + relevante News
    filtered = filter_relevant(news)
    print(f"🔎 Relevante aktuelle News gefunden: {len(filtered)}")

    if not filtered:
        print("Keine relevanten News – kein Push.")
        return

    # Nur neue Nachrichten (gegen gespeicherte IDs prüfen)
    # Wir nehmen URL-IDs
    current_ids = {n["url"] for n in filtered}
    last_ids = load_last_ids()
    new_ids = current_ids - last_ids
    if not new_ids:
        print("Keine neuen relevanten News – kein Push.")
        return

    new_news = [n for n in filtered if n["url"] in new_ids]

    # Prüfe Signifikanz – nur bei „wirklich wichtig“
    if not should_notify(new_news):
        print("Relevanz-Schwelle nicht erreicht – kein Push.")
        # Trotzdem IDs speichern, um Spam bei kleinen Schwankungen zu vermeiden
        save_last_ids(current_ids)
        return

    # Gemini-Bewertung
    result = analyze_with_gemini(new_news)
    if not result.ideen:
        print("Keine neuen Anlageideen erkannt.")
        save_last_ids(current_ids)
        return

    # Nachricht zusammenbauen (Name + Typ + % fett, darunter Begründung)
    parts = []
    for i in result.ideen:
        v = i.vertrauen
        if v > 100:  # falls Modell 0–1 oder 0–1000 ausspuckt
            v /= 100
        if v <= 1:
            v *= 100
        parts.append(f"🟢 {i.name} ({i.typ}) – <b>{v:.0f}%</b>\n{i.begruendung.strip()}")

    msg = "\n\n".join(parts)

    # Auf 1024 Zeichen begrenzen (Pushover-Limit)
    if len(msg) > CHAR_LIMIT:
        print(f"⚠️ Nachricht zu lang ({len(msg)} Zeichen) – auf {CHAR_LIMIT} gekürzt.")
        msg = msg[:CHAR_LIMIT]

    # Eine einzige Nachricht senden
    payload = {
        "token": PUSHOVER_TOKEN,
        "user": PUSHOVER_USER,
        "message": msg,
        "html": 1
    }

    try:
        r = requests.post("https://api.pushover.net/1/messages.json", data=payload, timeout=15)
        r.raise_for_status()
        print(f"✅ Pushover gesendet ({len(msg)} Zeichen).")
    except Exception as e:
        print("⚠️ Fehler beim Senden an Pushover:", e)

    save_last_ids(current_ids)
    print("✅ Neue Anlageideen gesendet und IDs gespeichert.")

if __name__ == "__main__":
    main()
