# ims/facility_resolver.py
from __future__ import annotations
from typing import Optional, Tuple, List, Dict
from pathlib import Path
import json, os, re, unicodedata

from .ims_client import IMSClient

# You renamed the file already:
#   mv data/ims_facilities_lookup data/ims_facilities_lookup.json
IMS_FACILITIES_JSON = os.getenv("IMS_FACILITIES_FILE", "data/ports_kb_ims_only.json")


# ------------------ small helpers ------------------
def _norm(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def _tokenize(s: str) -> set:
    s = _norm(s)
    if not s:
        return set()
    return set(t for t in re.split(r"[ ,;:/()\-]+", s) if t)


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _name_score(q: str, name: str) -> float:
    qt = _tokenize(q)
    nt = _tokenize(name)
    s = _jaccard(qt, nt)
    nq = _norm(q)
    nn = _norm(name)
    if nq and nn:
        if nq == nn:
            s += 0.4
        elif nq.startswith(nn) or nn.startswith(nq):
            s += 0.2
        elif nq in nn or nn in nq:
            s += 0.1
    return min(s, 1.0)


# ------------------ local facilities cache ------------------
_LOCAL_FACILITIES: List[Dict] = []
_LOCAL_LOADED = False


def _load_local_facilities(path: str = IMS_FACILITIES_JSON) -> List[Dict]:
    """
    Load IMS facilities from a local JSON file.

    Supports:
      - list of dicts
      - dict-of-dicts (e.g. pandas .to_json default orient)
    Normalizes everything to:
      {
        "id": "...",
        "attributes": {
            "countryISOCode": "...",
            "locationISOCode": "...",
            "locationName":   "...",
            "city":           "...",
            "aliases":        [...],
            "latitude":       "...",
            "longitude":      "...",
        }
      }
    """
    global _LOCAL_FACILITIES, _LOCAL_LOADED
    if _LOCAL_LOADED and _LOCAL_FACILITIES:
        return _LOCAL_FACILITIES

    p = Path(path)
    if not p.exists():
        _LOCAL_FACILITIES = []
        _LOCAL_LOADED = True
        print(f"[IMS][local] file not found, 0 facilities loaded from {path}")
        return _LOCAL_FACILITIES

    try:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[IMS][local] error loading {path}: {e}")
        _LOCAL_FACILITIES = []
        _LOCAL_LOADED = True
        return _LOCAL_FACILITIES

    # Normalize to iterable of dicts
    if isinstance(data, dict):
        # e.g. pandas .to_json gives {"0": {...}, "1": {...}, ...}
        items = [v for v in data.values()]
    elif isinstance(data, list):
        items = data
    else:
        items = []

    out: List[Dict] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        attr = it.get("attributes") or it

        # aliases may be stored as a comma-separated string
        aliases = attr.get("aliases") or it.get("aliases") or []
        if isinstance(aliases, str):
            aliases = [a.strip() for a in aliases.split(",") if a.strip()]

        out.append(
            {
                "id": it.get("id")
                or it.get("ims_facility_id")
                or attr.get("ims_facility_id")
                or "",
                "attributes": {
                    "countryISOCode": attr.get("countryISOCode")
                    or it.get("country_iso2")
                    or "",
                    "locationISOCode": attr.get("locationISOCode")
                    or it.get("locationISOCode")
                    or "",
                    "locationName": attr.get("locationName")
                    or it.get("location_name")
                    or "",
                    "city": attr.get("city") or it.get("city") or "",
                    "aliases": aliases,
                    "latitude": attr.get("latitude") or it.get("latitude") or "",
                    "longitude": attr.get("longitude") or it.get("longitude") or "",
                },
            }
        )

    _LOCAL_FACILITIES = out
    _LOCAL_LOADED = True
    print(f"[IMS][local] loaded {len(_LOCAL_FACILITIES)} facilities from {path}")
    return _LOCAL_FACILITIES


# ------------------ local search ------------------
def _search_local_facility(
    country_iso2: str,
    location_name: str,
    unlocode: Optional[str] = None,
    city: Optional[str] = None,
) -> Optional[Dict]:
    """
    Try to resolve an IMS facility using the local ims_facilities_lookup.json file.
    """
    facilities = _load_local_facilities()
    if not facilities:
        return None

    ctry = (country_iso2 or "").upper()
    qname = location_name or ""
    qcity = city or ""

    code3 = ""
    if unlocode:
        u = unlocode.strip().upper()
        if len(u) >= 3:
            code3 = u[-3:]

    best = None
    best_score = 0.0

    for fac in facilities:
        attr = fac.get("attributes", {}) or {}
        f_ctry = (attr.get("countryISOCode") or "").upper()
        f_loc = attr.get("locationISOCode") or ""
        f_name = attr.get("locationName") or ""
        f_city = attr.get("city") or ""

        # if we have a country, require match
        if ctry and f_ctry and ctry != f_ctry:
            continue

        score = 0.0

        # strong bonus if locationISOCode matches last 3 of UNLOCODE
        if code3 and f_loc and f_loc.upper() == code3:
            score += 0.8

        # name similarity (port/terminal name)
        score = max(score, _name_score(qname, f_name))

        # optional city similarity
        if qcity:
            score = max(score, _name_score(qcity, f_city))

        if score > best_score:
            best_score = score
            best = fac

    # require reasonably good match
    if best and best_score >= 0.6:
        return best
    return None


# ------------------ live IMS helpers ------------------
def _remote_search_facility(
    ims_client: IMSClient,
    country_iso2: str,
    location_name: str,
    code3: Optional[str],
    city: Optional[str],
) -> Optional[Dict]:
    """
    Very lightweight remote search; tune this to your actual IMSClient/Filters.

    Intended filter logic (pseudo):
      countryISOCode eq 'ES'
      and (locationISOCode eq 'AXO' or contains(tolower(locationName),'ust luga'))
    """
    if ims_client is None or not getattr(ims_client, "request", None):
        return None

    filters = []
    ctry = (country_iso2 or "").upper()
    if ctry:
        filters.append(f"countryISOCode eq '{ctry}'")
    if code3:
        filters.append(f"locationISOCode eq '{code3}'")
    if location_name:
        # very simple contains filter; adapt if API differs
        name_norm = location_name.replace("'", "''")
        filters.append(f"contains(tolower(locationName),'{name_norm.lower()}')")

    if not filters:
        return None

    filter_expr = " and ".join(filters)

    try:
        data = ims_client.request(
            "GET",
            "/Facilities",
            params={"filter": filter_expr, "page[size]": 10},
        )
        items = data.get("data") or data.get("Data") or data.get("items") or []
        return items[0] if items else None
    except Exception as e:
        print(f"[IMS][remote] error during _remote_search_facility: {e}")
        return None


# ------------------ public API ------------------
def resolve_facility_id(
    ims_client: IMSClient,
    country_iso2: str,
    location_name: str,
    unlocode: Optional[str] = None,
    city: Optional[str] = None,
) -> Tuple[Optional[Dict], str]:
    """
    Returns (facility_obj_or_none, reason_str).

    Search order:
      1) Local ims_facilities_lookup.json (fast, stable).
      2) Live IMS API (if configured).
    """
    # 1) local lookup
    local = _search_local_facility(country_iso2, location_name, unlocode, city)
    if local:
        return local, "local_lookup"

    # 2) remote / live IMS
    if ims_client is None or not ims_client.base_url:
        return None, "no_ims_client"

    code3 = ""
    if unlocode:
        u = unlocode.strip().upper()
        if len(u) >= 3:
            code3 = u[-3:]

    remote = _remote_search_facility(ims_client, country_iso2, location_name, code3, city)
    if remote:
        return remote, "remote_lookup"

    return None, "no_match"


def list_facilities_by_country(
    ims_client: IMSClient,
    country_iso2: str,
    limit: int = 50,
) -> List[Dict]:
    """
    First try local ims_facilities_lookup.json for this country.
    If empty, fall back to live IMS /Facilities.
    """
    facilities = _load_local_facilities()
    ctry = (country_iso2 or "").upper()

    local = [
        f
        for f in facilities
        if (f.get("attributes", {}).get("countryISOCode") or "").upper() == ctry
    ]
    if local:
        return local[:limit]

    if ims_client is None or not ims_client.base_url:
        return []

    try:
        # Here we use the generic request() wrapper, which maps "/Facilities"
        # to the correct /api/:locationType/Facilities path.
        data = ims_client.request(
            "GET",
            "/Facilities",
            params={
                # If your IMS API expects filter instead, you can change this to:
                # "filter": f"countryISOCode=={ctry}",
                "countryISOCode": ctry,
                "page[size]": limit,
            },
        )
        return data.get("data") or data.get("Data") or data.get("items") or []
    except Exception as e:
        print(f"[IMS][remote] error during list_facilities_by_country: {e}")
        return []

