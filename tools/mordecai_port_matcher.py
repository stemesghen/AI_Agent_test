from __future__ import annotations

import os
import json
import math
import unicodedata
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple

import pandas as pd
from difflib import SequenceMatcher


# ---------------- paths & config ----------------
BASE_DIR = Path(__file__).resolve().parents[1]

MORD_HITS_CSV = Path(
    os.getenv("MORDECAI_HITS_CSV", BASE_DIR / "data" / "extracted" / "mordecai_ims_hits.csv")
)
PORTS_KB_FINAL = Path(
    os.getenv("PORTS_KB_FINAL", BASE_DIR / "data" / "ports_kb_final.json")
)
OUT_MATCHES_CSV = Path(
    os.getenv("MORDECAI_PORT_MATCHES_CSV", BASE_DIR / "data" / "extracted" / "mordecai_ports_matches.csv")
)

SCORE_MIN = float(os.getenv("MORD_MIN_TOTAL_SCORE", "0.55"))
MAX_MATCHES_PER_HIT = int(os.getenv("MORD_MAX_MATCHES_PER_HIT", "10"))
MAX_DIST_KM = float(os.getenv("MORD_MAX_DIST_KM", "500"))  # beyond this, drop candidates


# ---------------- text + similarity helpers ----------------
def _norm_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def _token_set(s: str) -> set[str]:
    return {t for t in _norm_text(s).split() if t}


def name_similarity(a: str, b: str) -> float:
    """
    Combined character + token-based similarity.
    """
    a_n = _norm_text(a)
    b_n = _norm_text(b)
    if not a_n or not b_n:
        return 0.0

    char_sim = SequenceMatcher(None, a_n, b_n).ratio()
    ta, tb = _token_set(a_n), _token_set(b_n)
    if not ta or not tb:
        jacc = 0.0
    else:
        jacc = len(ta & tb) / len(ta | tb)
    return max(char_sim, jacc)


# ---------------- distance helpers ----------------
def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def dist_score_km(d: float) -> float:
    """
    Turn distance (km) into [0,1] score.
    0 km → 1.0
    >= MAX_DIST_KM → 0.0
    linear in between.
    """
    if d is None or d >= MAX_DIST_KM:
        return 0.0
    return max(0.0, min(1.0, 1.0 - d / MAX_DIST_KM))


# ---------------- ports KB loading ----------------
def load_ports_kb(path: Path) -> Dict[str, List[Dict[str, Any]]]:
    """
    Load ports_kb_final.json and index by country_iso2.

    Expected fields per record:
      ims_facility_id, country_iso2, unlocode, location_name, city,
      lat, lon, admin1, is_seaport, aliases (list).
    """
    if not path.exists():
        raise FileNotFoundError(f"ports_kb_final.json not found at {path}")

    with path.open("r", encoding="utf-8") as f:
        items = json.load(f)

    by_country: Dict[str, List[Dict[str, Any]]] = {}
    for rec in items:
        c = (rec.get("country_iso2") or "").upper()
        if not c:
            continue
        by_country.setdefault(c, []).append(rec)

    print(f"[MORDECAI-PORTS] Loaded {len(items)} ports; {len(by_country)} countries")
    return by_country


# ---------------- scoring per hit ----------------
def score_port_for_hit(hit: Dict[str, Any], port: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
    """
    Compute name_score, dist_score, admin_score and total_score for a given
    Mordecai hit vs a single port.
    """
    # --- name similarity ---
    hit_name = hit.get("name", "") or ""
    loc_name = port.get("location_name", "") or ""
    city = port.get("city", "") or ""
    aliases = port.get("aliases") or []
    if not isinstance(aliases, list):
        aliases = []

    best_name = 0.0
    # location_name
    if loc_name:
        best_name = max(best_name, name_similarity(hit_name, loc_name))
    # city
    if city:
        best_name = max(best_name, name_similarity(hit_name, city))
    # aliases
    for a in aliases:
        if a:
            best_name = max(best_name, name_similarity(hit_name, a))

    name_score_val = best_name

    # --- distance score ---
    hit_lat = hit.get("lat")
    hit_lon = hit.get("lon")
    port_lat = port.get("lat")
    port_lon = port.get("lon")

    dist_km_val = None
    dist_score_val = 0.0
    try:
        if hit_lat is not None and hit_lon is not None and port_lat is not None and port_lon is not None:
            dist_km_val = haversine_km(float(hit_lat), float(hit_lon), float(port_lat), float(port_lon))
            dist_score_val = dist_score_km(dist_km_val)
    except Exception:
        dist_km_val = None
        dist_score_val = 0.0

    # --- admin1 score ---
    admin_score_val = 0.0
    hit_admin1_name = _norm_text(hit.get("admin1_name", "") or "")
    hit_admin1_code = _norm_text(hit.get("admin1_code", "") or "")
    port_admin1 = _norm_text(port.get("admin1", "") or "")
    # simple equality or substring match
    if hit_admin1_name and port_admin1 and (hit_admin1_name == port_admin1 or hit_admin1_name in port_admin1 or port_admin1 in hit_admin1_name):
        admin_score_val = 1.0
    elif hit_admin1_code and port_admin1 and hit_admin1_code == port_admin1:
        admin_score_val = 1.0

    # --- combined ---
    total = 0.6 * name_score_val + 0.3 * dist_score_val + 0.1 * admin_score_val

    return total, {
        "name_score": name_score_val,
        "dist_km": dist_km_val if dist_km_val is not None else float("nan"),
        "dist_score": dist_score_val,
        "admin_score": admin_score_val,
        "total_score": total,
    }


# ---------------- main runner ----------------
def run():
    print("[MORDECAI-PORTS] Running mordecai_port_matcher.py")
    if not MORD_HITS_CSV.exists():
        raise FileNotFoundError(f"mordecai_ims_hits.csv not found at {MORD_HITS_CSV}")

    df_hits = pd.read_csv(MORD_HITS_CSV)
    print(f"[MORDECAI-PORTS] Loaded {len(df_hits)} Mordecai hits from {MORD_HITS_CSV}")

    ports_by_country = load_ports_kb(PORTS_KB_FINAL)

    out_rows: List[Dict[str, Any]] = []

    for idx, row in df_hits.iterrows():
        try:
            doc_id = row.get("doc_id", "")
            name = row.get("name", "")
            country_iso2 = (row.get("country_iso2", "") or "").upper()
            lat = row.get("lat")
            lon = row.get("lon")
            admin1_name = row.get("admin1_name", "")
            admin1_code = row.get("admin1_code", "")
            geonameid = row.get("geonameid", "")

            if not country_iso2:
                continue

            candidates = ports_by_country.get(country_iso2, [])
            if not candidates:
                continue

            best_matches: List[Tuple[Dict[str, Any], Dict[str, float]]] = []

            for port in candidates:
                # Only consider seaports (if field present)
                is_seaport = port.get("is_seaport", False)
                if isinstance(is_seaport, str):
                    is_seaport = str(is_seaport).lower() in {"true", "1", "yes", "y"}
                if not is_seaport:
                    continue

                total_score, comps = score_port_for_hit(row.to_dict(), port)

                # drop if distance too large
                dist_km_val = comps.get("dist_km", float("nan"))
                if not math.isnan(dist_km_val) and dist_km_val > MAX_DIST_KM:
                    continue

                if total_score < SCORE_MIN:
                    continue

                best_matches.append((port, comps))

            if not best_matches:
                continue

            # sort and trim
            best_matches.sort(key=lambda x: x[1]["total_score"], reverse=True)
            best_matches = best_matches[:MAX_MATCHES_PER_HIT]

            for port, comps in best_matches:
                out_rows.append(
                    {
                        "doc_id": doc_id,
                        "place_name": name,
                        "geonameid": geonameid,
                        "hit_lat": lat,
                        "hit_lon": lon,
                        "hit_country_iso2": country_iso2,
                        "hit_admin1_name": admin1_name,
                        "hit_admin1_code": admin1_code,
                        "ims_facility_id": port.get("ims_facility_id", ""),
                        "unlocode": port.get("unlocode", ""),
                        "facility_name": port.get("location_name", ""),
                        "facility_city": port.get("city", ""),
                        "facility_country_iso2": (port.get("country_iso2", "") or "").upper(),
                        "facility_admin1": port.get("admin1", ""),
                        "facility_lat": port.get("lat", ""),
                        "facility_lon": port.get("lon", ""),
                        "distance_km": comps.get("dist_km", float("nan")),
                        "name_score": comps.get("name_score", 0.0),
                        "admin_score": comps.get("admin_score", 0.0),
                        "dist_score": comps.get("dist_score", 0.0),
                        "total_score": comps.get("total_score", 0.0),
                    }
                )
        except Exception as e:
            print(f"[MORDECAI-PORTS] Error on hit index={idx}: {e}")

    out_df = pd.DataFrame(out_rows)
    OUT_MATCHES_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_MATCHES_CSV, index=False)
    print(f"[MORDECAI-PORTS] Wrote {len(out_rows)} rows → {OUT_MATCHES_CSV}")


if __name__ == "__main__":
    run()

