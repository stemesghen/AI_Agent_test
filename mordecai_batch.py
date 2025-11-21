from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from ftfy import fix_text

import sys
# allow importing utils and geo.* from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import safe_fname
from geo.mordecai_integration import extract_places_with_mordecai


# ---- Simple paths (relative to repo root) ----
INCIDENTS_DIR = Path("data/is_incident")
NORM_DIR = Path("data/normalized")
OUT_CSV = Path("data/extracted/mordecai_ims_hits.csv")


# ---- Safe JSON loader (same as in extract/run.py) ----
def load_json_any(path: Path) -> Any:
    for enc in ("utf-8-sig", "utf-8", "cp1252"):
        try:
            with path.open("r", encoding=enc) as f:
                return json.load(f)
        except UnicodeDecodeError:
            continue
    with path.open("r", encoding="utf-8", errors="replace") as f:
        return json.loads(f.read())


# ---- Get full article text from data/normalized ----
def get_article_text_for_incident(incident_json: Dict[str, Any]) -> Optional[str]:
    """
    Look up the normalized article by doc_id and return title + body text.
    """
    doc_id = incident_json.get("doc_id")
    if not doc_id:
        return None

    safe_id = safe_fname(doc_id)
    norm_path = NORM_DIR / f"{safe_id}.json"

    if not norm_path.exists():
        print(f"[MORDECAI-BATCH] No normalized file for doc_id={doc_id} at {norm_path}")
        return None

    norm = load_json_any(norm_path)

    title = fix_text(norm.get("title", ""))[:500]
    body = fix_text(
        norm.get("content_text", "")
        or norm.get("body", "")
        or norm.get("text", "")
    )

    text = f"{title}\n\n{body}".strip()
    if not text:
        return None

    return text


def process_incidents_with_mordecai() -> None:
    INCIDENTS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    incident_files = sorted(INCIDENTS_DIR.glob("*.classify.json"))
    print(f"[MORDECAI-BATCH] Found {len(incident_files)} incident JSON files in {INCIDENTS_DIR}")

    rows: List[Dict[str, Any]] = []

    for path in incident_files:
        data = load_json_any(path)

        # Only process if flagged is_incident==true
        if str(data.get("is_incident", "")).lower() not in {"true", "1", "yes"}:
            continue

        doc_id = data.get("doc_id") or path.name

        text = get_article_text_for_incident(data)
        if not text:
            print(f"[MORDECAI-BATCH] Skipping {path.name}: no article text found")
            continue

        print(f"[MORDECAI-BATCH] Running Mordecai on doc_id={doc_id}")

        places = extract_places_with_mordecai(text)

        for p in places:
            rows.append(
                {
                    "doc_id": doc_id,
                    "name": p.get("name"),
                    "lat": p.get("lat"),
                    "lon": p.get("lon"),
                    "country_iso3": p.get("country_iso3"),
                    "country_iso2": p.get("country_iso2"),
                    "admin1_name": p.get("admin1_name"),
                    "admin1_code": p.get("admin1_code"),
                    "feature_class": p.get("feature_class"),
                    "feature_code": p.get("feature_code"),
                    "start_char": p.get("start_char"),
                    "end_char": p.get("end_char"),
                    "geonameid": p.get("geonameid"),
                }
            )

    if not rows:
        print("[MORDECAI-BATCH] Done. No places found, nothing to write.")
        return

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"[MORDECAI-BATCH] Wrote {len(rows)} rows → {OUT_CSV}")


if __name__ == "__main__":
    process_incidents_with_mordecai()

