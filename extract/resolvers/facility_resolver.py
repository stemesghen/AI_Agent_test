# src/resolvers/facility_resolver.py
from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
import re
from difflib import SequenceMatcher
import unicodedata

def _norm(s: str) -> str:
    if not s:
        return ""
    # lowercase, deaccent, collapse spaces/punct
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def _sim(a: str, b: str) -> float:
    return SequenceMatcher(None, _norm(a), _norm(b)).ratio()

def index_facilities(facilities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Build a normalized index once per run.
    Expected keys (adapt to your payload):
      id / attributes.locationISOCode
      name / attributes.locationName
      country / attributes.countryName
      countryISO / attributes.countryISOCode
      city / attributes.city
    """
    idx = []
    for f in facilities or []:
        # Handle both flat and JSON:API style
        attrs = f.get("attributes") or {}
        entry = {
            "id": f.get("id") or attrs.get("locationISOCode"),
            "name": f.get("name") or attrs.get("locationName"),
            "country": f.get("country") or attrs.get("countryName"),
            "countryISO": f.get("countryISO") or attrs.get("countryISOCode"),
            "city": f.get("city") or attrs.get("city"),
            "raw": f,
        }
        entry["name_norm"] = _norm(entry["name"] or "")
        entry["city_norm"] = _norm(entry["city"] or "")
        entry["country_norm"] = _norm(entry["country"] or "")
        idx.append(entry)
    return idx

def _country_ok(entry: Dict[str, Any], country_hint: Optional[str]) -> bool:
    if not country_hint:
        return True
    ch = _norm(country_hint)
    return ch in {entry["country_norm"], entry.get("countryISO", "") and entry["countryISO"].lower()}

def resolve_facility(
    candidates: List[str],
    fac_idx: List[Dict[str, Any]],
    country_hint: Optional[str] = None,
    min_conf_main: float = 0.85,
    min_conf_with_country: float = 0.80
) -> Optional[Dict[str, Any]]:
    """
    Try multiple candidate strings (port/area/title hints) against the index.
    Returns best match with confidence.
    """
    if not fac_idx or not candidates:
        return None

    best: Tuple[float, Dict[str, Any], str] | None = None
    for cand in candidates:
        if not cand:
            continue
        for e in fac_idx:
            # Soft country gating: reward matches that agree with country hint
            sim = _sim(cand, e["name"] or "")
            if country_hint and _country_ok(e, country_hint):
                sim = sim + 0.03  # small boost

            if not best or sim > best[0]:
                best = (sim, e, cand)

    if not best:
        return None

    score, entry, used_cand = best
    threshold = min_conf_with_country if country_hint else min_conf_main
    if score < threshold:
        return None

    return {
        "resolver": "company_facilities",
        "confidence": round(score, 3),
        "id": entry["id"],
        "name": entry["name"],
        "country": entry["country"],
        "countryISO": entry.get("countryISO"),
        "city": entry.get("city"),
        "used_cand": used_cand,
        "raw": entry["raw"],
    }
