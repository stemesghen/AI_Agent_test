# geo/mordecai_integration.py

from __future__ import annotations
from typing import List, Optional, TypedDict
from pathlib import Path

import pycountry
from mordecai3 import Geoparser

from nlp.country_codes import load_country_codes


# -------------------------------------------------------------------
# Country code resources
# -------------------------------------------------------------------

COUNTRY_CSV = Path("data/country_ISO.csv")
ALIAS_CSV = Path("data/country_aliases.csv")

# If you don't have country_aliases.csv yet, this keeps it from crashing
if not ALIAS_CSV.exists():
    ALIAS_CSV = None  # load_country_codes will handle None

NAME2ISO, ISO2NAME, ALIAS2ISO = load_country_codes(COUNTRY_CSV, ALIAS_CSV)


# -------------------------------------------------------------------
# Global Mordecai client
# -------------------------------------------------------------------

_geo = Geoparser()


class MordecaiPlace(TypedDict, total=False):
    name: str
    lat: Optional[float]
    lon: Optional[float]
    country_iso3: Optional[str]
    country_iso2: Optional[str]
    admin1_name: Optional[str]
    admin1_code: Optional[str]
    feature_class: Optional[str]
    feature_code: Optional[str]
    start_char: int
    end_char: int
    geonameid: str


# -------------------------------------------------------------------
# ISO3 → ISO2 conversion
# -------------------------------------------------------------------

def _convert_iso3_to_iso2(code3: str | None) -> Optional[str]:
    if not code3:
        return None

    # First, try pycountry (best for RUS/YEM/CHN/etc.)
    try:
        country = pycountry.countries.get(alpha_3=code3.upper())
        if country:
            return country.alpha_2
    except Exception:
        pass

    # Optional fallback: treat the value as a name and look in NAME2ISO
    norm = (code3 or "").strip().lower()
    for cname, iso2 in NAME2ISO.items():
        if cname.lower() == norm:
            return iso2

    return None


# -------------------------------------------------------------------
# Main extraction function
# -------------------------------------------------------------------

def extract_places_with_mordecai(text: str) -> List[MordecaiPlace]:
    text = text or ""
    if not text.strip():
        return []

    result = _geo.geoparse_doc(text)
    ents = result.get("geolocated_ents", [])
    places: List[MordecaiPlace] = []

    for ent in ents:
        iso3 = ent.get("country_code3")
        iso2 = _convert_iso3_to_iso2(iso3)

        place: MordecaiPlace = {
            "name": ent.get("name"),
            "lat": ent.get("lat"),
            "lon": ent.get("lon"),
            "country_iso3": iso3,
            "country_iso2": iso2,
            "admin1_name": ent.get("admin1_name"),
            "admin1_code": ent.get("admin1_code"),
            "feature_class": ent.get("feature_class"),
            "feature_code": ent.get("feature_code"),
            "start_char": ent.get("start_char"),
            "end_char": ent.get("end_char"),
            "geonameid": ent.get("geonameid"),
        }

        if place.get("name"):
            places.append(place)

    return places

