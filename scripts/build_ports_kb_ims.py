#!/usr/bin/env python3
"""
Build an enriched IMS-only ports knowledge base with auto-generated aliases.

Input:
    data/ims_facilities_lookup.json   (raw IMS export)

Output:
    data/ports_kb_ims_only.json       (list of facilities with aliases)

Each output entry looks like:
{
  "ims_facility_id": "AXO",
  "source": "ims",
  "country_iso2": "ES",
  "location_name": "A Baiuca",
  "city": "",
  "aliases": ["a baiuca", "a-baiuca", "abaiuca", ...]
}
"""

import json
import re
import unicodedata
from pathlib import Path


# ---------- config ----------

IMS_JSON = Path("data/ims_facilities_lookup.json")
OUT_JSON = Path("data/ports_kb_ims_only.json")


# ---------- helpers ----------

def normalize_ascii(s: str) -> str:
    """Normalize to ASCII-only (strip accents, etc.)."""
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = s.encode("ascii", "ignore").decode("ascii")
    return s


def generate_aliases(location_name: str, city: str | None = None) -> list[str]:
    """
    Given a location name and optional city, generate a small set of aliases.

    Idea:
    - Include original + ascii-normalized forms
    - Lowercase variants
    - Replace weird quotes with "'"
    - Generate variants with space / hyphen / apostrophe / no separator

    Example:
        "Ust'Luga" -> ["ust'luga", "ust luga", "ust-luga", "ustluga", ...]
    """
    aliases: set[str] = set()

    base_names: list[str] = []
    if location_name:
        base_names.append(location_name)
    if city:
        c = city.strip()
        if c and c.lower() not in (location_name or "").lower():
            base_names.append(c)

    for name in base_names:
        name = name.strip()
        if not name:
            continue

        # original
        aliases.add(name)

        # ascii-only
        ascii_name = normalize_ascii(name)
        aliases.add(ascii_name)

        # lowercase
        lower = ascii_name.lower()
        aliases.add(lower)

        # normalize apostrophes
        lower = lower.replace("’", "'").replace("`", "'")

        # split on spaces, dashes, apostrophes
        tokens = re.split(r"[\s\-']+", lower)
        tokens = [t for t in tokens if t]

        if len(tokens) >= 2:
            joined_space = " ".join(tokens)     # "ust luga"
            joined_hyphen = "-".join(tokens)    # "ust-luga"
            joined_apost = "'".join(tokens)     # "ust'luga"
            joined_nosym = "".join(tokens)      # "ustluga"
            aliases.update([joined_space, joined_hyphen, joined_apost, joined_nosym])

    # Clean up: drop empties, dedupe, and sort
    aliases = {a.strip() for a in aliases if a and a.strip()}
    # It’s useful to avoid huge alias sets, but here they’re tiny anyway.
    return sorted(aliases)


# ---------- main ----------

def main() -> None:
    if not IMS_JSON.exists():
        raise SystemExit(f"Input JSON not found: {IMS_JSON}")

    print(f"[BUILD] Loading IMS facilities from {IMS_JSON} ...")
    raw = json.loads(IMS_JSON.read_text(encoding="utf-8"))

    if "data" not in raw or not isinstance(raw["data"], list):
        raise SystemExit("Unexpected IMS JSON structure: missing 'data' list")

    facilities = raw["data"]
    print(f"[BUILD] Found {len(facilities)} IMS facilities")

    out_records = []

    for item in facilities:
        attrs = item.get("attributes", {})
        ims_id = item.get("id")

        if ims_id is None:
            # Skip weird entries if any
            continue

        country_iso2 = attrs.get("countryISOCode", "") or ""
        location_name = attrs.get("locationName", "") or ""
        city = attrs.get("city", "") or ""

        aliases = generate_aliases(location_name, city)

        rec = {
            "ims_facility_id": ims_id,
            "source": "ims",
            "country_iso2": country_iso2,
            "location_name": location_name,
            "city": city,
            "aliases": aliases,
        }
        out_records.append(rec)

    print(f"[BUILD] Built {len(out_records)} enriched IMS records")
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(
        json.dumps(out_records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[BUILD] Wrote enriched KB to {OUT_JSON}")


if __name__ == "__main__":
    main()
