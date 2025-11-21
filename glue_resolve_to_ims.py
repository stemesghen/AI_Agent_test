# glue_resolve_to_ims.py

from __future__ import annotations
from typing import Any, Dict, List, Optional
import os

from dotenv import load_dotenv

# For debugging (keeps your old pattern)
DEBUG_GLUE = os.getenv("DEBUG_GLUE", "0") == "1"


def _dbg_glue(msg: str) -> None:
    if DEBUG_GLUE:
        print(f"[GLUE] {msg}")


# Ensure .env values override stale env vars
load_dotenv(override=True)

# --- Local imports ---
from geo.mordecai_integration import extract_places_with_mordecai, MordecaiPlace
from geo.ims_kb import lookup_alias, get_facility_by_id, facilities_by_country
from ims.ims_client import IMSClient
from ims.facility_resolver import resolve_facility_id  # optional fallback


# -------------------------------------------------------------------
# Global IMS client (for fallback live calls, if needed)
# -------------------------------------------------------------------

IMS_BASE_URL = (os.environ.get("IMS_BASE_URL") or "").rstrip("/")
IMS_TOKEN = os.environ.get("IMS_TOKEN", "")

# Normalize base URL, and keep env consistent for other code paths
if IMS_BASE_URL:
    os.environ["IMS_BASE_URL"] = IMS_BASE_URL
if IMS_TOKEN:
    os.environ["IMS_TOKEN"] = IMS_TOKEN

ims_client: Optional[IMSClient] = None
if IMS_BASE_URL:
    ims_client = IMSClient(IMS_BASE_URL, token=IMS_TOKEN)


def _priority(place: MordecaiPlace) -> int:
    """
    Priority for Mordecai places: cities > spots > countries/regions > other.
    """
    fc = place.get("feature_class")
    if fc == "P":   # populated place
        return 0
    if fc == "S":   # spot, building, farm, etc.
        return 1
    if fc == "A":   # country / region
        return 2
    return 3


def resolve_text_to_ims_facilities(
    text: str,
    country_hint: str | None = None,
    top_k_offline: int = 5,
    max_ims_results: int = 5,
) -> List[Dict[str, Any]]:
    """
    NEW VERSION – Mordecai + alias KB + IMS facilities.

    1) Use Mordecai3 to extract + geocode places from the text.
    2) For each place:
       2a) Try alias match in port_kb_ims_only.json.
           If matched, pull full facility record from ims_facilities_json.
       2b) If alias match fails and we only have country info, fall back
           to IMS facilities in that country (and optionally live IMS API).
    3) Return a list of candidate facility hits with rich provenance.

    Returned rows look like:
    {
      "mordecai_place": <MordecaiPlace dict>,
      "kb_entry": <port_kb entry or None>,
      "ims_facility": <IMS facility dict or None>,
      "ims_via": "kb_alias" | "kb_country" | "live_ims" | "none",
      "ims_reason": "...",
    }

    Existing callers that only look at 'ims_facility' can keep doing that.
    """
    text = text or ""
    _dbg_glue(f"text[:160]={text[:160]!r}")
    _dbg_glue(
        f"country_hint={country_hint!r} top_k_offline={top_k_offline} "
        f"max_ims_results={max_ims_results}"
    )

    # --- Step 1: Mordecai (NER + geocoding + disambiguation) ---
    places: List[MordecaiPlace] = extract_places_with_mordecai(text)
    _dbg_glue(f"mordecai places count={len(places)}")

    # Optional: restrict places if a country hint is provided
    if country_hint:
        hint_upper = country_hint.upper()
        places = [
            p for p in places
            if (p.get("country_iso2") or "").upper() == hint_upper
               or (p.get("country_iso3") or "").upper() == hint_upper
               or (p.get("name") or "").lower() == country_hint.lower()
        ]
        _dbg_glue(f"filtered places by country_hint -> {len(places)}")

    # Prioritize and truncate
    places = sorted(places, key=_priority)[:top_k_offline]

    results: List[Dict[str, Any]] = []
    seen_fac_ids: set[str] = set()

    # --- Step 2: per-place resolution ---
    for place in places:
        name = place.get("name")
        if not name:
            continue

        iso2 = place.get("country_iso2")
        _dbg_glue(f"place={name!r} iso2={iso2!r}")

        # 2a) Alias KB match
        kb_matches = lookup_alias(name, country_iso2=iso2)
        _dbg_glue(f"alias matches={len(kb_matches)} for {name!r}")

        if kb_matches:
            for kb_entry in kb_matches:
                fac_id = kb_entry.get("ims_facility_id")
                fac = get_facility_by_id(fac_id) if fac_id else None

                if not fac or fac_id in seen_fac_ids:
                    continue

                seen_fac_ids.add(fac_id)

                results.append(
                    {
                        "mordecai_place": place,
                        "kb_entry": kb_entry,
                        "ims_facility": fac,
                        "ims_via": "kb_alias",
                        "ims_reason": "matched_alias_in_port_kb",
                    }
                )

                if len(results) >= max_ims_results:
                    return results

            # if alias matches exist, we don't need country fallback for this place
            continue

        # 2b) No alias match – fallback: country-based scan in IMS facilities JSON
        if iso2:
            facs_in_country = facilities_by_country(iso2)
            _dbg_glue(
                f"country fallback for {name!r} iso2={iso2!r}, "
                f"fac_count={len(facs_in_country)}"
            )

            # Simple heuristic: just take a few facilities in that country.
            # You can refine this later with text similarity to locationName/city.
            for fac in facs_in_country[: max_ims_results - len(results)]:
                fac_id = fac.get("id")
                if not fac_id or fac_id in seen_fac_ids:
                    continue

                seen_fac_ids.add(fac_id)
                results.append(
                    {
                        "mordecai_place": place,
                        "kb_entry": None,
                        "ims_facility": fac,
                        "ims_via": "kb_country",
                        "ims_reason": "same_country_as_mordecai_place",
                    }
                )

                if len(results) >= max_ims_results:
                    return results

        # 2c) Optional: live IMS API fallback if we still have no results for this place
        if ims_client and len(results) < max_ims_results:
            _dbg_glue(f"live IMS fallback for place={name!r}")

            try:
                fac, reason = resolve_facility_id(
                    ims_client=ims_client,
                    country_iso2=iso2,
                    location_name=name,
                    city=place.get("admin1_name"),  # weak, but sometimes useful
                    unlocode=None,
                )
            except Exception as e:  # keep pipeline robust
                _dbg_glue(f"live IMS error for {name!r}: {e!r}")
                fac, reason = None, "error"

            if isinstance(fac, dict):
                fac_id = fac.get("id")
                if fac_id and fac_id not in seen_fac_ids:
                    seen_fac_ids.add(fac_id)
                    results.append(
                        {
                            "mordecai_place": place,
                            "kb_entry": None,
                            "ims_facility": fac,
                            "ims_via": "live_ims",
                            "ims_reason": reason,
                        }
                    )
                    if len(results) >= max_ims_results:
                        return results

    _dbg_glue(f"final ims_hits count={len(results)}")
    return results


if __name__ == "__main__":
    # Quick manual test
    example_text = "Explosion near Port of Aarhus, Denmark, with tankers diverted to Ust-Luga and Primorsk."
    hits = resolve_text_to_ims_facilities(example_text, country_hint="Denmark")

    for r in hits:
        place = r["mordecai_place"]
        fac = r["ims_facility"]
        attrs = fac.get("attributes", {}) if isinstance(fac, dict) else {}

        print(
            f"{r['ims_via']:>9} | {r['ims_reason']:<30} | "
            f"{place.get('name'):<15} | "
            f"{attrs.get('countryISOCode')} | "
            f"{attrs.get('locationISOCode')} | "
            f"{attrs.get('locationName')} | "
            f"IMS ID: {fac.get('id')}"
        )

