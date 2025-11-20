# ingest/check_feeds.py
# ingest/check_feeds.py
import csv, time
from pathlib import Path
import requests, feedparser
from bs4 import BeautifulSoup

SOURCES = Path(__file__).parent / "sources.csv"

def fetch_rss(url: str):
    """Fetch then parse to tolerate slightly malformed RSS/Atom."""
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

def count_html_list(url: str, item_selector: str, link_selector: str, max_pages: int = 1):
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    total = 0
    from urllib.parse import urlparse, urljoin
    base_host = urlparse(url).netloc

    for p in range(1, max_pages + 1):
        page_url = url if p == 1 else (url.rstrip("/") + f"/page/{p}/")
        try:
            r = requests.get(page_url, headers=headers, timeout=20)
            r.raise_for_status()
        except Exception:
            continue

        soup = BeautifulSoup(r.text, "html.parser")
        found = []

        # primary CSS path
        if item_selector and link_selector:
            for card in soup.select(item_selector):
                a = card.select_one(link_selector)
                if a and a.get("href"):
                    found.append(a["href"])

        # fallback: scan anchors that look like onsite article links
        if not found:
            for a in soup.find_all("a", href=True):
                href = a["href"].strip()
                if href.startswith("#"):
                    continue
                full = href if href.startswith("http") else urljoin(page_url, href)
                if urlparse(full).netloc == base_host and len(a.get_text(strip=True)) >= 6:
                    found.append(full)

        total += len(set(found))
        time.sleep(0.3)

    return total

def main():
    total_rows = rss_checked = html_checked = portal_skipped = 0

    with open(SOURCES, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            total_rows += 1
            sid  = (row.get("source_id") or "").strip()
            kind = (row.get("kind") or "").strip().lower()
            url  = (row.get("url") or "").strip()

            if kind == "rss":
                try:
                    d = fetch_rss(url)
                    bozo = bool(getattr(d, "bozo", 0))
                    print(f"{sid:<24} items={len(d.entries):>4}  bozo={bozo}  url={url}")
                    if bozo and getattr(d, "bozo_exception", None):
                        print(f"  ↳ parse warning: {d.bozo_exception}")
                except Exception as e:
                    print(f"{sid:<24} ERROR: {e}  url={url}")
                rss_checked += 1

            elif kind == "html":
                item_sel = (row.get("item_selector") or "").strip()
                link_sel = (row.get("link_selector") or "").strip()
                max_pages = int((row.get("max_pages") or "1").strip() or 1)
                if item_sel and link_sel:
                    try:
                        n = count_html_list(url, item_sel, link_sel, max_pages)
                        print(f"{sid:<24} links={n:>4}  kind=html  url={url}")
                        html_checked += 1
                    except Exception as e:
                        print(f"{sid:<24} HTML ERROR: {e}  url={url}")
                        html_checked += 1
                else:
                    print(f"{sid:<24} kind=html (no selectors)  url={url}")

            else:
                portal_skipped += 1
                print(f"{sid:<24} kind={kind} (skipped) url={url}")

    print(
        f"\nSummary: total rows={total_rows}, rss checked={rss_checked}, "
        f"html checked={html_checked}, portal/other skipped={portal_skipped}"
    )

if __name__ == "__main__":
    main()

