# tools/make_label_pack.py
from __future__ import annotations
import os, json
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
import sys
import traceback

# repo root on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from place_resolver import PlaceResolver
from nlp.country_codes import load_country_codes, map_names_to_iso2
from extract.run import clean_article_text  # your generic cleaner

# ----- Config -----
IN_DIR        = Path(os.getenv("EXTRACT_IN_DIR", "data/is_incident"))
NORM_DIR      = Path("data/normalized")
OUT_CSV       = Path("data/training/label_pack.csv")
UNLOCODE_CSV  = Path("data/raw/unlocode_labeled.csv")
GEONAMES_CSV  = Path("data/raw/geonames_labeled.csv")
COUNTRY_CSV   = Path("data/country_ISO.csv")
REGIONS_JSON  = Path("data/region_lookup.json")
TOP_K         = int(os.getenv("LABEL_TOPK", "5"))

# Optional focus: set LABEL_DOC_ID to a specific doc_id to generate rows only for that one
FOCUS_DOC_ID  = os.getenv("LABEL_DOC_ID", "").strip()

HEADER = [
    "doc_id","title","span","text_country_hint",
    "cand_location_name","cand_country_iso2","cand_unlocode","cand_source",
    "label"
]

def write_header_once(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        pd.DataFrame(columns=HEADER).to_csv(path, index=False)

def append_rows(path: Path, rows: List[Dict[str, Any]]):
    if not rows:
        return
    df = pd.DataFrame(rows, columns=HEADER)
    # append without header
    df.to_csv(path, mode="a", index=False, header=False)

def main():
    print(f"[INFO] make_label_pack starting…", flush=True)
    print(f"[CFG] IN_DIR={IN_DIR} | NORM_DIR={NORM_DIR} | TOP_K={TOP_K}", flush=True)
    if FOCUS_DOC_ID:
        print(f"[CFG] FOCUS_DOC_ID={FOCUS_DOC_ID}", flush=True)
    print(f"[CFG] OUT_CSV={OUT_CSV}", flush=True)

    # Create output with header immediately so you can tail it live
    write_header_once(OUT_CSV)

    # Construct resolver + country map
    try:
        resolver = PlaceResolver(str(UNLOCODE_CSV), str(GEONAMES_CSV), str(COUNTRY_CSV), str(REGIONS_JSON))
        name2iso, _ = load_country_codes(COUNTRY_CSV)
    except Exception as e:
        print("[ERR] Failed to initialize resolver/country map:")
        traceback.print_exc()
        return

    # Collect classify files
    cls_files = sorted([p for p in IN_DIR.glob("*.classify.json")])
    if FOCUS_DOC_ID:
        cls_files = [p for p in cls_files if p.stem.startswith(FOCUS_DOC_ID)]
    print(f"[INFO] Found {len(cls_files)} classified files to label", flush=True)

    processed = 0
    total_written = 0

    for cf in cls_files:
        processed += 1
        try:
            meta = json.loads(cf.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[WARN] {cf.name}: cannot read classify json: {e}", flush=True)
            continue

        if str(meta.get("is_incident","")).lower() not in {"true","1","yes"}:
            print(f"[SKIP] ({processed}/{len(cls_files)}) {cf.name}: not incident", flush=True)
            continue

        doc_id = meta.get("doc_id") or cf.stem
        norm_path = NORM_DIR / f"{doc_id}.json"
        if not norm_path.exists():
            print(f"[WARN] ({processed}/{len(cls_files)}) {doc_id}: normalized file missing → {norm_path}", flush=True)
            # still write a placeholder row so you see the miss
            append_rows(OUT_CSV, [{
                "doc_id": doc_id,
                "title": "",
                "span": "",
                "text_country_hint": "",
                "cand_location_name": "",
                "cand_country_iso2": "",
                "cand_unlocode": "",
                "cand_source": "missing_normalized",
                "label": "",
            }])
            total_written += 1
            continue

        try:
            norm = json.loads(norm_path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[WARN] ({processed}/{len(cls_files)}) {doc_id}: failed reading normalized JSON: {e}", flush=True)
            continue

        title = norm.get("title","") or ""
        text  = norm.get("content_text","") or ""
        cleaned = clean_article_text(title, text)

        # Country hints from full cleaned text
        hints = map_names_to_iso2([cleaned], name2iso)
        iso_hint = hints[0] if hints else None

        # Resolve candidates
        try:
            hits = resolver.resolve(cleaned, country_hint=iso_hint, top_k=TOP_K)
        except Exception:
            print(f"[WARN] ({processed}/{len(cls_files)}) {doc_id}: resolver.resolve crashed:", flush=True)
            traceback.print_exc()
            hits = []

        # Build rows (append immediately so you can tail live)
        out_rows: List[Dict[str, Any]] = []
        if hits:
            for h in hits:
                out_rows.append({
                    "doc_id": doc_id,
                    "title": title[:150],
                    "span": h.get("locationName","") or h.get("unlocode",""),
                    "text_country_hint": iso_hint or "",
                    "cand_location_name": h.get("locationName",""),
                    "cand_country_iso2": h.get("countryISOCode",""),
                    "cand_unlocode": h.get("unlocode",""),
                    "cand_source": h.get("source",""),
                    "label": "",   # fill 1 for correct, 0 for wrong
                })
        else:
            out_rows.append({
                "doc_id": doc_id,
                "title": title[:150],
                "span": "",
                "text_country_hint": iso_hint or "",
                "cand_location_name": "",
                "cand_country_iso2": "",
                "cand_unlocode": "",
                "cand_source": "no_candidates",
                "label": "",
            })

        append_rows(OUT_CSV, out_rows)
        total_written += len(out_rows)

        print(f"[OK] ({processed}/{len(cls_files)}) {doc_id}: wrote {len(out_rows)} row(s) → {OUT_CSV}", flush=True)

    print(f"[DONE] Processed {processed} files; total rows written: {total_written}", flush=True)

if __name__ == "__main__":
    main()


