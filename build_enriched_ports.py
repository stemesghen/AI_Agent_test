#Read the IMS facilities list - in ports_lookup

#Read ports.csv - combined from un/locode and geonames 

#Match each facility to the correct port in ports.csv

#Copy coordinates + official UN/LOCODE


#It reads ports_lookup.json

#It reads ports.csv

#It normalizes name/lat/lon

#It builds official_unlocode like "AE" + "JAZ" -> "AEJAZ"

#It does per-country fuzzy matching

#It writes ports_lookup_enriched.json with match_confidence and needs_manual_review

import json
import math
import pandas as pd
from difflib import SequenceMatcher
from pathlib import Path

IMS_PATH = Path("data/ports_lookup.json")      # IMS facilities
UNLOCODE_PATH = Path("data/ports.csv")         # merged UN/LOCODE + geonames
OUT_PATH = Path("data/ports_lookup_enriched.json")

# -----------------------
# helpers
# -----------------------

def fuzzy_sim(a: str, b: str) -> float:
    """Return similarity 0..1 between two strings."""
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()

def coalesce_lat_lon(row):
    """
    Use decimal lat/lon if present.
    Fallback to lat_un/lon_un if lat/lon are NaN but lat_un/lon_un exist.
    Returns (lat, lon) as floats or (None, None).
    """
    lat = row.get("lat")
    lon = row.get("lon")

    # pandas may give NaN as float('nan'), which is not None but is not usable
    def is_bad(x):
        return x is None or (isinstance(x, float) and math.isnan(x))

    if is_bad(lat) and not is_bad(row.get("lat_un")):
        lat = row.get("lat_un")
    if is_bad(lon) and not is_bad(row.get("lon_un")):
        lon = row.get("lon_un")

    # if still bad, set to None
    if is_bad(lat):
        lat = None
    if is_bad(lon):
        lon = None

    return lat, lon

def normalize_unlocode(country_iso2, locode):
    """
    Build something like 'AEJAZ' from country='AE', locode='JAZ'.
    If locode is missing (NaN), return None.
    """
    if not isinstance(country_iso2, str):
        return None
    if not isinstance(locode, str):
        return None
    country_iso2 = country_iso2.strip()
    locode = locode.strip()
    if not country_iso2 or not locode:
        return None
    return country_iso2 + locode

def pick_best_match_for_facility(ims_row, un_rows, threshold=0.8):
    """
    Given one IMS facility row and a list of UN rows
    (already filtered to same country),
    return best UN row + score + needs_manual_review.
    """
    country = ims_row.get("country_iso2", "").strip()
    loc_name = ims_row.get("location_name", "")
    city     = ims_row.get("city", "")

    # only compare to same-country UN rows
    same_ctry_rows = [r for r in un_rows if str(r.get("country_iso2","")).strip() == country]

    best_row = None
    best_score = 0.0

    for cand in same_ctry_rows:
        place_name = cand.get("place_name", "")
        score_name = fuzzy_sim(loc_name, place_name)
        score_city = fuzzy_sim(city, place_name)
        score = max(score_name, score_city)

        if score > best_score:
            best_score = score
            best_row = cand

    if best_row is None:
        return None, 0.0, True

    needs_review = best_score < threshold
    return best_row, best_score, needs_review

# -----------------------
# load IMS facilities
# -----------------------

with open(IMS_PATH, "r", encoding="utf-8") as f:
    ims_facilities = json.load(f)

# ims_facilities is expected to be a list[dict] like:
# {
#   "ims_facility_id": "AAR",
#   "unlocode": "AAR",
#   "country_iso2": "DK",
#   "location_name": "Aarhus",
#   "city": "Midtjylland",
#   "aliases": []
# }

# -----------------------
# load UN/LOCODE+GeoNames table
# -----------------------

raw_df = pd.read_csv(UNLOCODE_PATH)

# We'll build a cleaned list of dicts with consistent keys:
unlocode_rows = []
for _, row in raw_df.iterrows():
    country_iso2 = str(row.get("country", "")).strip()  # e.g. "AE"
    locode_part  = row.get("locode")
    if isinstance(locode_part, float) and math.isnan(locode_part):
        locode_part = None
    if isinstance(locode_part, str):
        locode_part = locode_part.strip()

    official_unlocode = normalize_unlocode(country_iso2, locode_part)

    # choose best name we have
    place_name = row.get("name_ascii")
    if isinstance(place_name, float) and math.isnan(place_name):
        place_name = None
    if not place_name:
        place_name = row.get("name")
    if isinstance(place_name, float) and math.isnan(place_name):
        place_name = None
    if not place_name:
        place_name = ""

    lat_val, lon_val = coalesce_lat_lon({
        "lat": row.get("lat"),
        "lon": row.get("lon"),
        "lat_un": row.get("lat_un"),
        "lon_un": row.get("lon_un"),
    })

    # we don't have region auto yet (Baltic Sea, etc.), that's a later manual fill
    unlocode_rows.append({
        "country_iso2": country_iso2,
        "place_name": place_name,
        "official_unlocode": official_unlocode,
        "lat": lat_val,
        "lon": lon_val,
        "region": None,
    })

# -----------------------
# match IMS -> UN rows
# -----------------------

enriched = []

for ims in ims_facilities:
    best_row, score, needs_review = pick_best_match_for_facility(
        ims, unlocode_rows, threshold=0.8
    )

    if best_row is None:
        enriched.append({
            "ims_facility_id": ims.get("ims_facility_id"),
            "country_iso2": ims.get("country_iso2"),
            "location_name": ims.get("location_name"),
            "city": ims.get("city"),
            "aliases": ims.get("aliases", []),

            "official_unlocode": None,
            "lat": None,
            "lon": None,
            "region": None,

            "match_confidence": 0.0,
            "needs_manual_review": True
        })
        continue

    enriched.append({
        "ims_facility_id": ims.get("ims_facility_id"),
        "country_iso2": ims.get("country_iso2"),
        "location_name": ims.get("location_name"),
        "city": ims.get("city"),
        "aliases": ims.get("aliases", []),

        # pulled from UN/LOCODE reference
        "official_unlocode": best_row.get("official_unlocode"),
        "lat": best_row.get("lat"),
        "lon": best_row.get("lon"),
        "region": best_row.get("region"),  # still None for now

        "match_confidence": round(score, 4),
        "needs_manual_review": needs_review
    })

# -----------------------
# write to JSON
# -----------------------

with open(OUT_PATH, "w", encoding="utf-8") as out:
    json.dump(enriched, out, ensure_ascii=False, indent=2)

print(f"[OK] Enriched {len(enriched)} facilities → {OUT_PATH}")
