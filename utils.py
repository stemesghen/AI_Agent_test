# ingest/utils.py
import re, hashlib, datetime
from pathlib import Path

def iso_now():
    """Return current time in ISO 8601 format with timezone."""
    return datetime.datetime.now().astimezone().isoformat()

def make_doc_id(title, url, text=""):
    """Stable hash ID for a document."""
    key = (title + url + text[:500]).encode("utf-8", errors="ignore")
    return hashlib.sha1(key).hexdigest()

def looks_maritime(text: str) -> bool:
    """Simple heuristic to keep maritime-related articles."""
    maritime_keywords = [
        "vessel","ship","tanker","cargo","port","harbor","marine",
        "maritime","imo","grounding","collision","piracy","coast guard",
        "seafarer","crew","anchorage","container","engine room"
    ]
    return any(k in text.lower() for k in maritime_keywords)

def safe_fname(s: str) -> str:
    """Filename-safe version of a string."""
    return re.sub(r"[^A-Za-z0-9_.-]", "_", s)[:100]
