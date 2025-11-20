#!/usr/bin/env python3
"""
ais_enrich.py
Enrich maritime incident evidence with AIS.

Outputs:
- vessels (who was there)
- confidence_level (vessel ID confidence: high/medium/low/none)
- location_support (how well AIS activity supports the claimed port/region)

Changes in this version:
- Uses POST with JSON body when sending a polygon (prevents oversized GET URLs).
- Simplifies large sea polygons before sending to keep payloads small.
- Keeps 'position' as a dict (no json.dumps) so requests can send compact JSON.
"""

import json
import requests
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional, Tuple
from difflib import SequenceMatcher

import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon, Point, mapping

# ============================
# CONFIG
# ============================

# TODO: set to your working base URL from Postman
LLI_BASE_URL = "https://api.lloydslistintelligence.com/v1"

# TODO: paste a VALID, CURRENT token here
AIS_BEARER_TOKEN = "eyJhbGciOiJSUzI1NiIsImtpZCI6ImEzck1VZ01Gdjl0UGNsTGE2eUYzekFrZnF1RSIsIng1dCI6ImEzck1VZ01Gdjl0UGNsTGE2eUYzekFrZnF1RSIsInR5cCI6ImF0K2p3dCJ9.eyJpc3MiOiJodHRwOi8vbGxveWRzbGlzdGludGVsbGlnZW5jZS5jb20iLCJuYmYiOjE3NjA0NzM3MTksImlhdCI6MTc2MDQ3MzcxOSwiZXhwIjoxNzYzMDY1NzE5LCJzY29wZSI6WyJsbGl3ZWJhcGkiXSwiYW1yIjpbImN1c3RvbWVyQXBpX2dyYW50Il0sImNsaWVudF9pZCI6IkN1c3RvbWVyQXBpIiwic3ViIjoiaW1zQGluc3VyaXR5LmNvbSIsImF1dGhfdGltZSI6MTc2MDQ3MzcxOSwiaWRwIjoic2FsZXNmb3JjZSIsImFjY2Vzc1Rva2VuIjoiMDBEOGQwMDAwMDlvaTM4IUFRRUFRSFlTLmRiVl9Za3UubThJMjBESHRyNXlIUG5Icks3QVNMelhSMEJHTHFQSHcwVzF0UG5hcXRuZ1hfUno4d2g2QUM3M01kVy5hempuNk9GS3FmemxRczFwM3dSTCIsInNlcnZpY2VJZCI6IiIsImVudGl0bGVtZW50VHlwZSI6IiIsImFjY291bnROYW1lIjoiIiwidXNlcm5hbWUiOiJpbXNAaW5zdXJpdHkuY29tIiwidXNlcklkIjoiMDA1TnowMDAwMEd0Z0RWSUFaIiwiY29udGFjdEFjY291bnRJZCI6IjAwMThkMDAwMDBrZ0I0cEFBRSIsInVzZXJUeXBlIjoiQ3NwTGl0ZVBvcnRhbCIsImVtYWlsIjoiaW1zQGluc3VyaXR5LmNvbSIsImdpdmVuX25hbWUiOiJJTVMiLCJmYW1pbHlfbmFtZSI6IkFQSSIsInNoaXBUbyI6IiIsImp0aSI6IkQyQjg2NEQ4MTM5QkQ3N0YxNDc3Qzg1RUExREZGNDU1In0.HmIGF4mRBM5M5hXR79LbC0ChzzZLlqj1lEXe-BtlT20PNN8XskQPOUoahC8FdHufB7FD7kcGiczRU7_iyIljMf6OTxZovJ7TTlT0kM2jX91yLZY8dVByOe-pJ_ViUr8CgCgRLDT-Hm5C47mbC6hRa2RyLf7SKPIVJhBQGDiIfNTD1aVTZ_rNTtok4o_xyEzjq-M63vPjLH5sL5WbInYe39_s0vPTl-my5XPCPQ_R5LEqgUAyDT7PV6JBS3i4LpTinrOz21kgNSUAYBn6qtIyLAu7lhSUfNJT5YpkN5sgaOvXrDQkJgnD4Hsp_C8qv8gFk_Z_EO7sm3JAZEE_SjpDNQ"

DEFAULT_LOOKBACK_HOURS = 24 * 7  # 7 days

# Use the backfilled file you just generated
PORTS_LOOKUP_PATH = "data/ports_lookup_enriched_with_region.backfilled.json"

# World seas shapefile (IHO v3)
SEA_SHAPEFILE_PATH = "data/world_seas/World_Seas_IHO_v3.shp"
SEA_NAME_FIELD = "NAME"  # column with names like "Baltic Sea", "Gulf of Aden"

# Extra headers if your gateway requires them
EXTRA_AIS_HEADERS: Dict[str, str] = {}

# ============================
# GLOBAL CACHES
# ============================

_PORTS_CACHE: Optional[List[Dict[str, Any]]] = None
_REGION_POLYS_CACHE: Optional[Dict[str, dict]] = None  # name -> GeoJSON polygon dict

# ============================
# HELPERS: text / time
# ============================

def _utc_now():
    return datetime.now(timezone.utc)

def _time_window_from_evidence(evidence: Dict[str, Any]) -> Tuple[str, str]:
    """
    If evidence['incident_time_utc'] is present, use ±48h around it.
    Else, use last DEFAULT_LOOKBACK_HOURS hours.
    """
    incident_time = evidence.get("incident_time_utc")
    if incident_time:
        try:
            center = datetime.fromisoformat(incident_time.replace("Z", "+00:00"))
            start_time = center - timedelta(hours=48)
            end_time = center + timedelta(hours=48)
        except Exception:
            end_time = _utc_now()
            start_time = end_time - timedelta(hours=DEFAULT_LOOKBACK_HOURS)
    else:
        end_time = _utc_now()
        start_time = end_time - timedelta(hours=DEFAULT_LOOKBACK_HOURS)

    return (
        start_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        end_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    )

def _similarity(a: str, b: str) -> float:
    """Fuzzy ratio 0..1 using SequenceMatcher with light normalization."""
    a_norm = "".join(ch for ch in (a or "").lower() if ch.isalnum() or ch.isspace()).strip()
    b_norm = "".join(ch for ch in (b or "").lower() if ch.isalnum() or ch.isspace()).strip()
    if not a_norm or not b_norm:
        return 0.0
    return SequenceMatcher(None, a_norm, b_norm).ratio()

# ============================
# HELPERS: ports / regions
# ============================

def _load_ports_cache() -> List[Dict[str, Any]]:
    """Load enriched IMS facilities (backfilled) once."""
    global _PORTS_CACHE
    if _PORTS_CACHE is None:
        with open(PORTS_LOOKUP_PATH, "r", encoding="utf-8") as f:
            _PORTS_CACHE = json.load(f)
        print(f"[DEBUG] Loaded {len(_PORTS_CACHE)} facilities from {PORTS_LOOKUP_PATH}")
        if _PORTS_CACHE:
            print("[DEBUG] Example facility 0:")
            print(json.dumps(_PORTS_CACHE[0], indent=2)[:500])
    return _PORTS_CACHE

def _largest_polygon(geom) -> Optional[Polygon]:
    if geom is None:
        return None
    if isinstance(geom, Polygon):
        return geom
    if isinstance(geom, MultiPolygon):
        polys = list(geom.geoms)
        if not polys:
            return None
        polys.sort(key=lambda g: g.area, reverse=True)
        return polys[0]
    return None

def _simplify_poly(poly: Polygon, tol: float = 0.05) -> Polygon:
    """
    Simplify polygon (degrees). tol≈0.05 ~ 5–6 km at mid-latitudes.
    Keeps topology; falls back to original if simplification fails.
    """
    try:
        simp = poly.simplify(tol, preserve_topology=True)
        if isinstance(simp, Polygon) and len(list(simp.exterior.coords)) >= 4:
            return simp
    except Exception:
        pass
    return poly

def _geojson_from_poly(poly: Polygon) -> dict:
    """Convert a shapely polygon to compact GeoJSON (lists, not tuples)."""
    gj = mapping(poly)
    def tupl2list(x):
        if isinstance(x, tuple):
            return [tupl2list(v) for v in x]
        if isinstance(x, list):
            return [tupl2list(v) for v in x]
        return x
    gj["coordinates"] = tupl2list(gj["coordinates"])
    return gj

def _load_region_polys_main() -> Dict[str, dict]:
    """
    Load sea polygons, keep the largest per name, simplify, and cache.
    Returns: { "Baltic Sea": {type: "Polygon", coordinates: ...}, ... }
    """
    global _REGION_POLYS_CACHE
    if _REGION_POLYS_CACHE is not None:
        return _REGION_POLYS_CACHE

    gdf = gpd.read_file(SEA_SHAPEFILE_PATH)
    print(f"[DEBUG] Loaded {len(gdf)} sea polygons from {SEA_SHAPEFILE_PATH}")

    region_map = {}
    for _, row in gdf.iterrows():
        name_raw = row.get(SEA_NAME_FIELD)
        if not name_raw:
            continue
        main_poly = _largest_polygon(row.geometry)
        if main_poly is None:
            continue
        simp = _simplify_poly(main_poly, tol=0.05)
        keep = region_map.get(name_raw)
        if (keep is None) or (simp.area > keep["poly"].area):
            region_map[name_raw] = {"poly": simp}

    _REGION_POLYS_CACHE = {name: _geojson_from_poly(data["poly"]) for name, data in region_map.items()}
    print(f"[DEBUG] Region keys sample: {list(_REGION_POLYS_CACHE.keys())[:5]}")
    return _REGION_POLYS_CACHE

def lookup_region_polygon(region_name: str) -> Optional[dict]:
    """Exact match first, otherwise fuzzy ≥ 0.7."""
    if not region_name:
        return None
    region_polys = _load_region_polys_main()

    if region_name in region_polys:
        print(f"[DEBUG] Exact region polygon match '{region_name}'")
        return region_polys[region_name]

    best_key, best_score = None, 0.0
    for name in region_polys.keys():
        sc = _similarity(region_name, name)
        if sc > best_score:
            best_score, best_key = sc, name
    print(f"[DEBUG] Best region match '{region_name}' -> '{best_key}' score={best_score:.3f}")
    if best_key and best_score >= 0.7:
        return region_polys[best_key]
    return None

def _build_bounding_box(lat: float, lon: float, delta_deg: float = 0.5) -> dict:
    """Simple CCW bbox polygon around a point."""
    return {
        "type": "Polygon",
        "coordinates": [[
            [lon - delta_deg, lat - delta_deg],  # BL
            [lon - delta_deg, lat + delta_deg],  # TL
            [lon + delta_deg, lat + delta_deg],  # TR
            [lon + delta_deg, lat - delta_deg],  # BR
            [lon - delta_deg, lat - delta_deg],  # close
        ]]
    }

def lookup_facility_latlon(facility_name: str, country_iso2: Optional[str]) -> Tuple[Optional[float], Optional[float]]:
    """
    Fuzzy facility→(lat,lon) using ports cache; +0.05 bonus if country matches.
    Require score≥0.7 and non-null lat/lon.
    """
    ports = _load_ports_cache()
    best_row, best_score = None, 0.0
    for row in ports:
        cand_name = row.get("location_name") or ""
        sc = _similarity(facility_name, cand_name)
        sc += 0.05 if (country_iso2 and row.get("country_iso2") == country_iso2) else 0.0
        if sc > best_score:
            best_score, best_row = sc, row

    print(f"[DEBUG] Best facility match '{facility_name}' score={best_score:.3f}")
    if not best_row or best_score < 0.7:
        return (None, None)
    lat, lon = best_row.get("lat"), best_row.get("lon")
    if lat is None or lon is None:
        return (None, None)
    print(f"[DEBUG] Facility resolved to lat={lat}, lon={lon}")
    return (lat, lon)

def _polygon_from_facility(evidence: Dict[str, Any]) -> Optional[dict]:
    name = evidence.get("location_facility")
    cc = evidence.get("country")
    if not name:
        return None
    lat, lon = lookup_facility_latlon(name, cc)
    if lat is None or lon is None:
        return None
    return _build_bounding_box(lat, lon, delta_deg=0.5)

# ============================
# AIS CALL + NORMALIZE
# ============================

def _call_ais(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calls /aislatestinformation.
    - If 'position' present (dict polygon), POST with JSON body.
    - Else GET with query params.
    """
    headers = {
        "Accept": "application/json",
    "Content-Type": "application/json",
    "Authorization": AIS_BEARER_TOKEN.strip(),
    }
    headers.update(EXTRA_AIS_HEADERS)

    url = f"{LLI_BASE_URL}/aislatestinformation"

    # Use POST when polygon is included
    position = params.pop("position", None)
    use_post = position is not None

    if use_post:
        body = {**params, "position": position}
        print(f"[DEBUG] AIS POST {url}")
        print(f"[DEBUG] body = {json.dumps(body)[:800]}")
        resp = requests.post(url, headers=headers, json=body, timeout=60)
    else:
        print(f"[DEBUG] AIS GET {url}")
        print(f"[DEBUG] params = {json.dumps(params)[:800]}")
        resp = requests.get(url, headers=headers, params=params, timeout=60)

    print("AIS STATUS:", resp.status_code)
    preview = resp.text[:500] if isinstance(resp.text, str) else str(resp.content)[:500]
    print("AIS RAW (first 500 chars):", preview)

    if resp.status_code in (401, 403):
        raise RuntimeError("AIS auth failed (401/403). Token missing/expired or wrong tenant/base URL.")
    if resp.status_code != 200:
        raise RuntimeError(f"AIS error {resp.status_code}: {resp.text}")

    try:
        return resp.json()
    except Exception:
        raise RuntimeError("AIS returned non-JSON or malformed JSON.")

def _normalize_ais_hits(raw_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Convert Lloyd's JSON to a normalized list.
    Tries "aisMessages" first; falls back to Data.aisMessages.
    """
    if "aisMessages" in raw_json:
        source_list = raw_json.get("aisMessages", [])
    elif "Data" in raw_json and isinstance(raw_json["Data"], dict):
        source_list = raw_json["Data"].get("aisMessages", [])
    else:
        source_list = []

    hits = []
    for msg in source_list:
        hits.append({
            "vesselName": msg.get("vesselName"),
            "imo": msg.get("vesselImo") or msg.get("imo"),
            "mmsi": msg.get("mmsi"),
            "shipType": msg.get("shipType"),
            "lat": msg.get("lat"),
            "lon": msg.get("lon"),
            "timestamp": msg.get("timestamp") or msg.get("receivedDateTimeUtc"),
        })
    print(f"[DEBUG] Normalized {len(hits)} AIS hits")
    return hits

# ============================
# HINTING / SCORING
# ============================

def _filter_by_shiptype(hits: List[Dict[str, Any]], evidence: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    If incident_type hints tanker/container/bulk, prefer those types.
    Otherwise pass through.
    """
    hint = (evidence.get("incident_type") or "").lower()
    if not hint:
        return hits

    tanker_words = ["tanker", "oil", "crude", "lpg", "lng", "gas"]
    container_words = ["container", "boxship"]
    bulk_words = ["bulk", "bulker", "dry cargo", "grain", "ore"]

    def relevant(shiptype: str) -> bool:
        if not shiptype:
            return True
        st = shiptype.lower()
        if any(w in hint for w in tanker_words) and ("tank" in st or "lpg" in st or "lng" in st or "gas" in st):
            return True
        if any(w in hint for w in container_words) and ("container" in st):
            return True
        if any(w in hint for w in bulk_words) and ("bulk" in st):
            return True
        return True  # fallback keep

    return [h for h in hits if relevant(h.get("shipType", ""))]

def _pick_best_name_match(hits: List[Dict[str, Any]], target_name: str, min_ratio: float = 0.8) -> Optional[Dict[str, Any]]:
    best, best_score = None, 0.0
    for h in hits:
        vn = h.get("vesselName") or ""
        sc = _similarity(vn, target_name)
        if sc > best_score:
            best, best_score = h, sc
    if best and best_score >= min_ratio:
        out = dict(best)
        out["match_score"] = best_score
        return out
    return None

def _score_location_support(hits: List[Dict[str, Any]], evidence: Dict[str, Any], polygon_used: Optional[dict]) -> Dict[str, Any]:
    hit_count = len(hits)
    if polygon_used is None:
        confidence = "none"
    else:
        confidence = "high" if hit_count > 5 else ("medium" if hit_count > 0 else "low")
    return {
        "location_match_confidence": confidence,
        "claimed_facility": evidence.get("location_facility"),
        "claimed_region": evidence.get("region_name"),
        "polygon_used": polygon_used,
        "hit_count": hit_count,
    }

# ============================
# MAIN ENRICH
# ============================

def enrich_with_AIS(evidence: Dict[str, Any]) -> Dict[str, Any]:
    """
    evidence can include:
      imo, vessel_name, location_facility, region_name,
      country, incident_time_utc, incident_type
    """
    received_after, received_before = _time_window_from_evidence(evidence)

    # A) Exact IMO (strongest identity)
    if evidence.get("imo"):
        imo = evidence["imo"]
        params = {
            "messageFormat": "decoded",
            "receivedAfter": received_after,
            "receivedBefore": received_before,
            "vesselImo": imo,
            "landFilter": "false",
            "cleansed": "true",
        }
        raw_json = _call_ais(params)
        hits = _normalize_ais_hits(raw_json)
        hits = _filter_by_shiptype(hits, evidence)
        location_support = _score_location_support([], evidence, None)
        return {
            "confidence_level": "high",
            "needs_manual_review": False,
            "vessels": [{**h, "evidence": f"Matched by IMO {imo} directly"} for h in hits],
            "location_support": location_support,
        }

    # B) Vessel name + polygon (facility preferred, else region)
    if evidence.get("vessel_name"):
        polygon_used = _polygon_from_facility(evidence)
        if polygon_used is None:
            polygon_used = lookup_region_polygon(evidence.get("region_name", ""))

        if polygon_used is not None:
            params = {
                "messageFormat": "decoded",
                "receivedAfter": received_after,
                "receivedBefore": received_before,
                "position": polygon_used,  # dict → POST JSON
                "landFilter": "false",
                "cleansed": "true",
            }
            raw_json = _call_ais(params)
            hits = _normalize_ais_hits(raw_json)
            hits = _filter_by_shiptype(hits, evidence)
            best = _pick_best_name_match(hits, evidence["vessel_name"])
            location_support = _score_location_support(hits, evidence, polygon_used)

            if best:
                return {
                    "confidence_level": "medium",
                    "needs_manual_review": False,
                    "vessels": [{**best, "evidence": f"Fuzzy-matched AIS vesselName to '{evidence['vessel_name']}' within polygon"}],
                    "location_support": location_support,
                }

            return {
                "confidence_level": "low",
                "needs_manual_review": True,
                "vessels": [{**h, "evidence": "Candidate vessel seen in reported area/time (no strong name match)"} for h in hits],
                "location_support": location_support,
            }

    # C) Facility polygon only
    polygon_used = _polygon_from_facility(evidence)
    if polygon_used is not None:
        params = {
            "messageFormat": "decoded",
            "receivedAfter": received_after,
            "receivedBefore": received_before,
            "position": polygon_used,
            "landFilter": "false",
            "cleansed": "true",
        }
        raw_json = _call_ais(params)
        hits = _normalize_ais_hits(raw_json)
        hits = _filter_by_shiptype(hits, evidence)
        location_support = _score_location_support(hits, evidence, polygon_used)
        return {
            "confidence_level": "low",
            "needs_manual_review": True,
            "vessels": [{**h, "evidence": "Vessel(s) present near reported facility during incident window"} for h in hits],
            "location_support": location_support,
        }

    # D) Region polygon only
    polygon_used = lookup_region_polygon(evidence.get("region_name", ""))
    if polygon_used is not None:
        params = {
            "messageFormat": "decoded",
            "receivedAfter": received_after,
            "receivedBefore": received_before,
            "position": polygon_used,
            "landFilter": "false",
            "cleansed": "true",
        }
        raw_json = _call_ais(params)
        hits = _normalize_ais_hits(raw_json)
        hits = _filter_by_shiptype(hits, evidence)
        location_support = _score_location_support(hits, evidence, polygon_used)
        return {
            "confidence_level": "low",
            "needs_manual_review": True,
            "vessels": [{**h, "evidence": "Vessel(s) present in reported region during incident window"} for h in hits],
            "location_support": location_support,
        }

    # E) Nothing usable
    location_support = _score_location_support([], evidence, None)
    return {
        "confidence_level": "none",
        "needs_manual_review": True,
        "vessels": [],
        "location_support": location_support,
    }

# ============================
# STANDALONE TEST
# ============================

if __name__ == "__main__":
    # Example from your run (adjust as needed)
    test_evidence = {
        "vessel_name": None,
        "imo": None,
        "location_facility": "Ust-Luga",
        "region_name": "Baltic Sea",
        "incident_time_utc": "2025-10-28T12:00:00Z",
        "country": "RU",
        "incident_type": "LPG tanker spoofing / sanctions evasion",
    }

    print("[INFO] Running AIS enrichment test on:")
    print(json.dumps(test_evidence, indent=2))

    try:
        after, before = _time_window_from_evidence(test_evidence)
        print(f"[DEBUG] AIS time window {after} → {before}")
        # warm caches to show helpful debug
        _load_ports_cache()
        _load_region_polys_main()

        result = enrich_with_AIS(test_evidence)
        print("\n[RESULT]")
        print(json.dumps(result, indent=2))
    except Exception as e:
        print("\n[ERROR]")
        print(str(e))
