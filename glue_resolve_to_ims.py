# glue_resolve_to_ims.py
import os
from typing import List, Dict, Any
from dotenv import load_dotenv

DEBUG_GLUE = os.getenv("DEBUG_GLUE", "0") == "1"


def _dbg_glue(msg: str) -> None:
    if DEBUG_GLUE:
        print(f"[GLUE] {msg}")


# Ensure .env values override any existing OS env vars (even if they were blank/stale)
load_dotenv(override=True)

from place_resolver import PlaceResolver
from ims.ims_client import IMSClient
from ims.facility_resolver import resolve_facility_id


def resolve_text_to_ims_facilities(
    text: str,
    country_hint: str | None = None,
    top_k_offline: int = 5,
    max_ims_results: int = 3,
) -> List[Dict[str, Any]]:
    """
    1) Offline: text -> candidate place keys (UN/LOCODE/GeoNames/Region)
    2) Online: query IMS using best filters
    3) Return IMS-only results with provenance and scores
    """
    text = text or ""
    _dbg_glue(f"text[:160]={text[:160]!r}")
    _dbg_glue(
        f"country_hint={country_hint!r} top_k_offline={top_k_offline} "
        f"max_ims_results={max_ims_results}"
    )

    # --- Offline step: local lookup using UN/LOCODE + GeoNames ---
    pr = PlaceResolver(
        unlocode_csv="data/raw/unlocode_labeled.csv",
        geonames_csv="data/raw/geonames_labeled.csv",
        country_csv="data/country_ISO.csv",
        regions_json="data/region_lookup.json",
    )
    candidates = pr.resolve(text, country_hint=country_hint, top_k=top_k_offline)

    _dbg_glue(f"offline candidates count={len(candidates)}")
    if DEBUG_GLUE:
        preview = [
            {
                "source": c.get("source"),
                "countryISOCode": c.get("countryISOCode"),
                "locationISOCode": c.get("locationISOCode"),
                "locationName": c.get("locationName"),
                "unlocode": c.get("unlocode"),
                "score": c.get("score"),
            }
            for c in candidates[:10]
        ]
        _dbg_glue(f"offline candidates preview={preview}")

    # --- Online step: IMS API calls ---
    ims_base = (os.environ.get("IMS_BASE_URL") or "https://owic-imp.ims.insurity.com/WebAPI").rstrip("/")
    ims_token = os.environ.get("IMS_TOKEN", "")

    # Keep environment consistent for downstream calls if needed
    os.environ["IMS_BASE_URL"] = ims_base
    if ims_token:
        os.environ["IMS_TOKEN"] = ims_token

    client = IMSClient(ims_base, token=ims_token)

    ims_hits: List[Dict[str, Any]] = []
    seen_ids = set()

    for c in candidates:
        unlocode = c.get("unlocode") or ""
        country  = c.get("countryISOCode") or ""
        locname  = c.get("locationName") or ""
        city     = c.get("subdivision") or None

        _dbg_glue(
            f"IMS-CALL "
            f"country={country!r} locname={locname!r} city={city!r} unlocode={unlocode!r}"
        )

        fac, reason = resolve_facility_id(
            ims_client=client,
            country_iso2=country,
            location_name=locname,
            unlocode=unlocode if len(unlocode) == 5 else None,
            city=city,
        )

        fac_id = fac.get("id") if isinstance(fac, dict) else None
        _dbg_glue(f"IMS-RESP reason={reason!r} fac_id={fac_id!r}")

        if fac and fac_id not in seen_ids:
            seen_ids.add(fac_id)
            ims_hits.append({
                "candidate": c,
                "ims_reason": reason,
                "ims_facility": fac,
            })
            if len(ims_hits) >= max_ims_results:
                break

    _dbg_glue(f"ims_hits count={len(ims_hits)}")
    if DEBUG_GLUE and ims_hits:
        prev = []
        for r in ims_hits[:5]:
            fac = r["ims_facility"]
            attrs = fac.get("attributes", {}) if isinstance(fac, dict) else {}
            prev.append({
                "ims_reason": r["ims_reason"],
                "countryISOCode": attrs.get("countryISOCode"),
                "locationISOCode": attrs.get("locationISOCode"),
                "locationName": attrs.get("locationName"),
                "id": fac.get("id"),
            })
        _dbg_glue(f"ims_hits preview={prev}")

    return ims_hits


if __name__ == "__main__":
    # Example test
    example_text = "Explosion near Port of Aarhus, Denmark"
    results = resolve_text_to_ims_facilities(example_text, country_hint="Denmark")

    for r in results:
        fac = r["ims_facility"]
        attrs = fac.get("attributes", {})
        print(
            f" {r['ims_reason']} | "
            f"{attrs.get('countryISOCode')} | "
            f"{attrs.get('locationISOCode')} | "
            f"{attrs.get('locationName')} | "
            f"IMS ID: {fac.get('id')}"
        )

