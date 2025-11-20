# ingest/run_ingest.py
import csv, json, os, time, hashlib
from pathlib import Path
from urllib.parse import urlparse, urljoin

import requests
import feedparser
import trafilatura
from dateutil import parser as dtp
from langdetect import detect as lang_detect
from bs4 import BeautifulSoup


import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# --- your existing utils (kept as-is) ---
from utils import iso_now, make_doc_id, looks_maritime, safe_fname

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
NORM = DATA / "normalized"
CATALOG = DATA / "catalog.jsonl"
for p in (RAW, NORM):
    p.mkdir(parents=True, exist_ok=True)


# -------- helpers --------
def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()


def read_sources():
    with open(Path(__file__).parent / "sources.csv", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            yield {k: (v or "").strip() for k, v in row.items()}


def fetch_rss(url: str):
    """Fetch→parse for robustness against slightly malformed feeds."""
    resp = requests.get(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/rss+xml,application/xml;q=0.9,*/*;q=0.8",
        },
        timeout=20,
    )
    text = resp.text.replace("&nbsp;", " ")
    return feedparser.parse(text)


def list_page_links(
    url: str,
    item_selector: str,
    link_selector: str,
    max_pages: int = 1,
    page_pattern: str = "/page/{n}/",
    source_id: str = "",
):
    """
    Robust HTML list scraper:
    - Supports multiple CSS selectors (pipe-separated with "||")
    - Supports simple pagination patterns (e.g., '?page={n}', '/page/{n}/')
    - Falls back to scanning anchors on the same host if selectors fail
    - Saves a debug snapshot if page 1 yields 0 links
    """
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }

    item_sets = [s.strip() for s in (item_selector or "").split("||") if s.strip()]
    link_sets = [s.strip() for s in (link_selector or "").split("||") if s.strip()]
    base_host = urlparse(url).netloc

    seen, out, snapshots = set(), []

    for p in range(1, max_pages + 1):
        page_url = url if p == 1 else (url.rstrip("/") + page_pattern.format(n=p))
        try:
            r = requests.get(page_url, headers=headers, timeout=20)
            r.raise_for_status()
        except Exception as e:
            print(f"  [WARN] {source_id} page {p}: {e}")
            continue

        soup = BeautifulSoup(r.text, "html.parser")
        found = []

        # Primary: use selectors if provided
        if item_sets and link_sets:
            for it_sel in item_sets:
                for ln_sel in link_sets:
                    for card in soup.select(it_sel):
                        a = card.select_one(ln_sel)
                        if a and a.get("href"):
                            found.append(a["href"])

        # Fallback: all anchors with some minimal filtering
        if not found:
            for a in soup.find_all("a", href=True):
                txt = a.get_text(strip=True)
                href = a["href"].strip()
                if len(txt) < 6 or href.startswith("#"):
                    continue
                found.append(href)

        # Normalize and dedupe
        for href in found:
            full = href if href.startswith("http") else urljoin(page_url, href)
            if urlparse(full).netloc != base_host:
                continue
            if full in seen:
                continue
            seen.add(full)
            out.append({"link": full, "title": ""})

        # Save snapshot if empty on first page
        if p == 1 and not found:
            Path("debug").mkdir(exist_ok=True)
            snap = Path("debug") / f"debug_{(source_id or 'page')}_p{p}.html"
            snap.write_text(r.text, encoding="utf-8")
            snapshots.append(str(snap))

        time.sleep(0.3)

    if not out and snapshots:
        print(f"  ↳ saved snapshot(s): {', '.join(snapshots)}")

    return out


def clean_html_to_text(url: str) -> str:
    """Download article HTML and extract readable text."""
    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
        r.raise_for_status()
        txt = trafilatura.extract(r.text, url=url) or ""
        return (txt or "").strip()
    except Exception:
        return ""


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def append_catalog(line_obj):
    CATALOG.parent.mkdir(parents=True, exist_ok=True)
    with open(CATALOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(line_obj, ensure_ascii=False) + "\n")


def already_seen(doc_id: str) -> bool:
    return (NORM / f"{safe_fname(doc_id)}.json").exists()


def prefer_structured_datetime(entry):
    # Best-effort published date
    try:
        if getattr(entry, "published_parsed", None):
            return dtp.parse(
                time.strftime("%Y-%m-%dT%H:%M:%S%z", entry.published_parsed)
            ).astimezone().isoformat()
        if getattr(entry, "updated_parsed", None):
            return dtp.parse(
                time.strftime("%Y-%m-%dT%H:%M:%S%z", entry.updated_parsed)
            ).astimezone().isoformat()
    except Exception:
        pass
    # Fallback to string fields
    pub = entry.get("published") or entry.get("updated") or ""
    try:
        return dtp.parse(pub).astimezone().isoformat() if pub else iso_now()
    except Exception:
        return iso_now()


def norm_item(item, source_id, reliability, default_lang):
    url = (item.get("link") or item.get("id") or "").strip()
    title = (item.get("title") or "").strip()
    if not url or not title:
        return None

    text = clean_html_to_text(url)
    if not text:
        text = (item.get("summary") or "").strip()

    # Filter to maritime-related content only
    if not looks_maritime(f"{title}\n{text}"):
        return None

    # Language guess (robust)
    try:
        lang = default_lang or lang_detect((title + " " + text)[:5000])
    except Exception:
        lang = default_lang or "en"

    published_at = item.get("_published_at") or iso_now()

    doc_id = make_doc_id(title, url, text)
    return {
        "doc_id": doc_id,
        "source_id": source_id,
        "url": url,
        "title": title,
        "published_at": published_at,
        "fetched_at": iso_now(),
        "language": lang,
        "reliability": float(reliability or 0.7),
        "content_text": text,
        "fingerprint": sha1(url + published_at),
    }


def ingest_once():
    new_count, dupes = 0, 0

    for src in read_sources():
        sid = src["source_id"]
        kind = (src["kind"] or "").lower()
        url = src["url"]
        rel = src.get("reliability") or "0.75"
        dlang = src.get("lang") or ""

        entries = []
        try:
            if kind == "rss":
                feed = fetch_rss(url)
                print(
                    f"→ {sid} fetched {len(feed.entries)} entries (bozo={bool(getattr(feed,'bozo',0))})"
                )
                # Normalize to a dict interface like your HTML path
                for e in feed.entries[:300]:
                    entries.append(
                        {
                            "link": e.get("link") or e.get("id"),
                            "title": e.get("title", ""),
                            "summary": e.get("summary", ""),
                            "_published_at": prefer_structured_datetime(e),
                        }
                    )

            elif kind in ("html", "portal"):
                item_sel = src.get("item_selector") or ""
                link_sel = src.get("link_selector") or ""
                max_pages = int(src.get("max_pages") or "1")
                page_pattern = src.get("page_pattern") or "/page/{n}/"

                pairs = list_page_links(
                    url,
                    item_sel,
                    link_sel,
                    max_pages=max_pages,
                    page_pattern=page_pattern,
                    source_id=sid,
                )
                print(f"→ {sid} scraped {len(pairs)} links from {kind.upper()}")
                entries = pairs

            else:
                # Unknown kind: log but do not silently skip
                print(f"→ {sid} kind={kind} not recognized, skipping for now")
                continue

        except Exception as e:
            print(f"[WARN] {sid} failed to fetch list: {e}")
            continue

        # Normalize + store
        for entry in entries:
            doc = norm_item(entry, sid, rel, dlang)
            if not doc:
                continue

            if already_seen(doc["doc_id"]):
                dupes += 1
                continue

            save_json(doc, NORM / f"{safe_fname(doc['doc_id'])}.json")
            append_catalog(
                {
                    "doc_id": doc["doc_id"],
                    "url": doc["url"],
                    "source_id": doc["source_id"],
                    "title": doc["title"],
                    "published_at": doc["published_at"],
                }
            )
            new_count += 1
            time.sleep(0.2)  # politeness

    print(f" new: {new_count} | dupes skipped: {dupes}")


if __name__ == "__main__":
    ingest_once()



