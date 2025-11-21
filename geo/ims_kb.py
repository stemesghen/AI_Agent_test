# geo/ims_kb.py

from __future__ import annotations
from typing import Any, Dict, List, Optional
import json
import unicodedata
import re
from pathlib import Path


# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]

PORT_KB_PATH = BASE_DIR / "data" / "ports_kb_ims_only.json"
IMS_FACILITIES_PATH = BASE_DIR / "data" / "ims_facilities_lookup.json"


# -------------------------------------------------------------------
# Normalization helper
# -------------------------------------------------------------------
def _norm(s: str) -> str:
    """
    Normalize strings for alias matching:
      - remove accents
      - lowercase
      - replace dashes/apostrophes with spaces
      - collapse whitespace
    """
    if not isinstance(s, str):
        return ""

    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = s.lower()
    s = re.sub(r"['’\-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# -------------------------------------------------------------------
# Load Knowledge Base files
# -------------------------------------------------------------------

with open(PORT_KB_PATH, "r", encoding="utf-8") as f:
    PORT_KB = json.load(f)

with open(IMS_FACILITIES_PATH, "r", encoding="utf-8") as f:
    IMS_FACILITIES = json.load(f)


# -------------------------------------------------------------------
# Build alias index
# alias_index: normalized_alias → [KB entry, KB entry, ...]
# -------------------------------------------------------------------
ALIAS_INDEX: Dict[str, List[Dict[str, Any]]] = {}

for entry in PORT_KB:
    aliases = entry.get("aliases", []) or []
    for alias in aliases:
        key = _norm(alias)
        if not key:
            continue
        ALIAS_INDEX.setdefault(key, []).append(entry)


# -------------------------------------------------------------------
# Build IMS facility index
# facility_id → facility metadata dict
# -------------------------------------------------------------------
IMS_INDEX: Dict[str, Dict[str, Any]] = {}

if isinstance(IMS_FACILITIES, list):
    for fac in IMS_FACILITIES:
        fac_id = fac.get("id") or fac.get("ims_facility_id")
        if fac_id:
            IMS_INDEX[fac_id] = fac
elif isinstance(IMS_FACILITIES, dict):
    # If already keyed by ID, use directly
    IMS_INDEX = IMS_FACILITIES


# -------------------------------------------------------------------
# Lookups
# -------------------------------------------------------------------

def lookup_alias(
    raw_name: str,
    country_iso2: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Lookup a port by alias from ports_kb_ims_only.json.

    Steps:
      • Normalize the name (case-insensitive, remove accents)
      • Retrieve any KB entries whose aliases match
      • If country_iso2 is provided, filter entries

    Returns:
      List of KB entries (each containing ims_facility_id, aliases, etc.)
    """
    if not raw_name:
        return []

    key = _norm(raw_name)
    hits = ALIAS_INDEX.get(key, [])

    if country_iso2:
        c2 = country_iso2.upper()
        hits = [
            h for h in hits
            if h.get("country_iso2", "").upper() == c2
        ]

    return hits


def get_facility_by_id(facility_id: str) -> Optional[Dict[str, Any]]:
    """
    Get full IMS facility metadata from ims_facilities_lookup.json.
    """
    return IMS_INDEX.get(facility_id)


def facilities_by_country(country_iso2: str) -> List[Dict[str, Any]]:
    """
    Return all IMS facilities belonging to a given country ISO2 code.
    """
    c2 = country_iso2.upper()
    return [
        fac for fac in IMS_INDEX.values()
        if fac.get("attributes", {}).get("countryISOCode", "").upper() == c2
    ]

