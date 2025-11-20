# disambiguation/features.py
from __future__ import annotations
from typing import Dict, List, Any
import math
import re
import unicodedata


def _norm(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def _tok(s: str) -> set:
    s = _norm(s)
    if not s:
        return set()
    return set(t for t in re.split(r"[ ,;:/()\\-]+", s) if t)


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _sim(a: str, b: str) -> float:
    ta = _tok(a)
    tb = _tok(b)
    s = _jaccard(ta, tb)
    na = _norm(a)
    nb = _norm(b)
    if na == nb:
        s += 0.4
    elif na.startswith(nb) or nb.startswith(na):
        s += 0.2
    elif na in nb or nb in na:
        s += 0.1
    return float(min(s, 1.0))


def assemble_feature_vector(
    span_text: str,
    full_text: str,
    country_hints: List[str],
    vessels: List[str],
    regions: List[str],
    cand: Dict[str, Any],
) -> Dict[str, float]:
    """
    Build a feature dict matching FEATURE_ORDER in disambiguation/model.py.
    """
    span = span_text or ""
    loc_name = cand.get("location_name") or cand.get("locationName") or ""
    city = cand.get("city") or cand.get("subdivision") or ""
    aliases = cand.get("aliases") or []
    if isinstance(aliases, str):
        aliases = [aliases]

    # string sims
    sim_name = _sim(span, loc_name)
    sim_city = _sim(span, city) if city else 0.0
    sim_alias = max((_sim(span, a) for a in aliases), default=0.0)

    tok_jacc_name = _jaccard(_tok(span), _tok(loc_name))
    tok_jacc_city = _jaccard(_tok(span), _tok(city)) if city else 0.0

    sim_best = max(sim_name, sim_city, sim_alias)

    # country match
    cand_ctry = (cand.get("country_iso2") or cand.get("countryISOCode") or "").upper()
    hints_norm = [c.upper() for c in (country_hints or [])]
    country_match_any = 1.0 if hints_norm and cand_ctry and cand_ctry in hints_norm else 0.0
    country_has_hint = 1.0 if hints_norm else 0.0

    # distance (if present)
    lat = cand.get("lat")
    lon = cand.get("lon")
    dist_km = float(cand.get("dist_km") or 0.0)
    if not dist_km and cand.get("distance_m"):
        try:
            dist_km = float(cand["distance_m"]) / 1000.0
        except Exception:
            dist_km = 0.0
    dist_inv = 1.0 / (1.0 + dist_km) if dist_km > 0 else 0.0

    # context flags
    tnorm = _norm(full_text)
    ctx_vessel = 1.0 if any(_norm(v) in tnorm for v in vessels) else 0.0
    ctx_region = 1.0 if any(_norm(r) in tnorm for r in regions) else 0.0
    harbor_terms = ["port", "harbour", "harbor", "terminal", "jetty", "berth", "dock", "pier", "quay"]
    ctx_harbor_terms = 1.0 if any(ht in tnorm for ht in harbor_terms) else 0.0

    sim_best_x_country = sim_best * (1.0 + country_match_any)
    sim_best_x_ctx = sim_best * (1.0 + ctx_vessel + ctx_region + ctx_harbor_terms)

    return {
        "sim_name": sim_name,
        "sim_city": sim_city,
        "sim_alias": sim_alias,
        "tok_jacc_name": tok_jacc_name,
        "tok_jacc_city": tok_jacc_city,
        "sim_best": sim_best,
        "country_match_any": country_match_any,
        "country_has_hint": country_has_hint,
        "dist_km": float(dist_km),
        "dist_inv": float(dist_inv),
        "ctx_vessel": ctx_vessel,
        "ctx_region": ctx_region,
        "ctx_harbor_terms": ctx_harbor_terms,
        "sim_best_x_country": sim_best_x_country,
        "sim_best_x_ctx": sim_best_x_ctx,
    }

