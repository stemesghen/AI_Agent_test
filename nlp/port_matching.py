# nlp/port_matching.py
"""
Port candidate scoring:
1) String/alias similarity against facility name, city, and aliases.
2) Country-first HARD FILTER when hints exist (prevents wrong-country matches).
3) Importance prior (e.g., population/throughput rank) as a small tie-breaker.
4) Guardrail: skip short uppercase spans (e.g., 'US', 'UK') so they don't
   accidentally match ports with similar substrings (like 'Abu Musa').

Required facility keys (each item in facilities_ref):
  {
    country_iso2, location_name, city, unlocode, lat, lon,
    aliases: List[str], subdivision, iata, source, importance (optional 0..1)
  }
"""

from typing import List, Dict, Any, Optional
from rapidfuzz import fuzz

def _normalize(s: str) -> str:
    return " ".join((s or "").split()).lower()

def _is_short_upper_token(s: str, max_len: int = 3) -> bool:
    """
    True for short (<= max_len) all-uppercase tokens (letters/digits and punctuation-free),
    e.g., 'US', 'UK', 'RU', 'CN', 'UAE', 'EU'. We treat these as country/region acronyms,
    not port names, and ignore them as port spans.
    """
    if not s:
        return False
    raw = s.strip()
    if len(raw) == 0 or len(raw) > max_len:
        return False
    # Allow alnum only; no spaces; and all uppercase
    if raw.isalnum() and raw.upper() == raw:
        return True
    return False

def _best_similarity(span: str, name: str, city: str, aliases: List[str]) -> dict:
    """
    Compute best partial-ratio score for span against name/city/aliases.
    Returns: {score, field, matched_alias}
    """
    span_n = _normalize(span)
    best = 0.0
    field = ""
    alias_hit = ""

    if name:
        s = fuzz.partial_ratio(span_n, _normalize(name)) / 100.0
        if s > best:
            best, field = s, "name"
    if city:
        s = fuzz.partial_ratio(span_n, _normalize(city)) / 100.0
        if s > best:
            best, field = s, "city"
    for a in (aliases or []):
        if not a:
            continue
        s = fuzz.partial_ratio(span_n, _normalize(a)) / 100.0
        if s > best:
            best, field, alias_hit = s, "alias", a

    return {"score": best, "field": field, "alias": alias_hit}

def match_ports_to_facilities(
    port_spans: List[str],
    country_iso2_hints: List[str],
    facilities_ref: List[Dict[str, Any]],
    regions: Optional[List[str]] = None,
    region_centroids: Optional[Dict[str, tuple]] = None,
    threshold: float = 0.75,
    top_k: int = 5,
    importance_max_boost: float = 0.12,   # <= 0.12 additive
    skip_short_upper: bool = True,        # guardrail for 'US', 'UK', etc.
) -> List[Dict[str, Any]]:
    """
    Args:
      port_spans: raw strings from NER/regex (“Port of X”, “Y Harbor”, etc.)
      country_iso2_hints: ISO2 codes from text; if present, we hard-filter.
      facilities_ref: list from nlp/port_index.load_ports_index_from_sources
    """
    # --- sanitize spans: dedupe, normalize for comparisons, and drop short-uppercase acronyms
    spans_raw = [s for s in (port_spans or []) if (s or "").strip()]
    if skip_short_upper:
        spans_raw = [s for s in spans_raw if not _is_short_upper_token(s, max_len=3)]

    spans_norm = [_normalize(s) for s in spans_raw]
    # dedupe while preserving original form (keep first appearance)
    seen = set()
    spans = []
    for raw, normed in zip(spans_raw, spans_norm):
        if not normed:
            continue
        if normed not in seen:
            seen.add(normed)
            spans.append(raw)

    hint_set = {(_normalize(c)).upper() for c in (country_iso2_hints or []) if c}

    results: List[Dict[str, Any]] = []

    for f in facilities_ref:
        f_iso = (f.get("country_iso2") or "").upper()

        # HARD FILTER: if we have country hints, only consider those countries
        if hint_set and f_iso and (f_iso not in hint_set):
            continue

        name = f.get("location_name", "") or ""
        city = f.get("city", "") or ""
        aliases = [a for a in (f.get("aliases") or []) if a]

        best_overall = 0.0
        best_field = ""
        best_span = ""
        best_alias = ""

        # Evaluate all kept spans; keep the best match across name/city/aliases
        for raw in (spans or []):
            r = _best_similarity(raw, name, city, aliases)
            if r["score"] > best_overall:
                best_overall = r["score"]
                best_field = r["field"]
                best_span = raw
                best_alias = r["alias"]

        # If we had no usable spans, don't emit (region-only paths are handled by caller)
        if best_overall <= 0 and not spans:
            continue

        # Base score = best string score
        score = best_overall

        # Country presence (light nudge if hints exist and match)
        if hint_set and f_iso in hint_set:
            score += 0.08

        # Importance prior: 0..1 scaled to <= importance_max_boost
        imp = f.get("importance")
        try:
            imp = float(imp) if imp is not None else 0.0
        except Exception:
            imp = 0.0
        if imp > 0:
            score += min(importance_max_boost, max(0.0, imp) * importance_max_boost)

        # (Optional) region/proximity bonuses could go here using region_centroids

        if score >= (threshold - 0.1):
            results.append({
                "facility": f,
                "score": round(float(score), 4),
                "span": best_span,
                "name_score": round(float(best_overall), 4),
                "city_score": 0.0,  # retained for compatibility
                "matched_field": best_field if best_field else ("alias" if best_alias else "name"),
                "matched_alias": best_alias,
                "source": "string+alias+boosts",
            })

    results.sort(key=lambda x: x["score"], reverse=True)
    # Only return those meeting threshold
    return [r for r in results if r["score"] >= threshold][:top_k]

