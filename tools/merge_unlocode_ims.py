#!/usr/bin/env python
"""
Merge improved UN/LOCODE data + IMS facilities + existing ports_kb_ims_only
into a single unified ports knowledge base: ports_kb_final.json

Inputs (adjust paths as needed):
  - data/unlocode_improved/code-list-improved.csv
  - data/unlocode_improved/aliases-improved.csv
  - data/unlocode_improved/parents.csv
  - data/ims_facilities_lookup.json          (raw IMS JSON export)
  - data/ports_kb_ims_only.json              (your current KB)

Output:
  - data/ports_kb_final.json

IMS remains the anchor ID: every record in ports_kb_final.json is keyed
by ims_facility_id, with UN/LOCODE + old aliases used for enrichment.
"""

from __future__ import annotations

import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd

# ---------- CONFIG: adjust these paths to your repo layout ----------

BASE_DIR = Path(__file__).resolve().parents[1]

UNLOCODE_DIR = BASE_DIR / "data" / "unlocode_improved"
IMS_DIR = BASE_DIR / "data"

CODELIST_CSV = UNLOCODE_DIR / "code-list-improved.csv"
ALIASES_CSV = UNLOCODE_DIR / "aliases-improved.csv"
PARENTS_CSV = UNLOCODE_DIR / "parents.csv"

IMS_FACILITIES_JSON = IMS_DIR / "ims_facilities_lookup.json"
PORTS_KB_IMS_ONLY_JSON = IMS_DIR / "ports_kb_ims_only.json"

OUTPUT_JSON = IMS_DIR / "ports_kb_final.json"


# ---------- helpers ----------

def norm(s: str) -> str:
    if not isinstance(s, str):
        return ""
    return s.strip().lower()


def load_unlocode_codelist(path: Path) -> Dict[str, Dict[str, Any]]:
    """
    Load code-list-improved.csv into a dict keyed by full UNLOCODE, e.g. 'ZWNAG'.

    Your example rows look like:

      (no Nominatim),UN/LOCODE
      ,ZW,NAG,Norton,Norton,MW,RL,-----6--,1601,,1753S 03042E,,"-17.88333,30.70000",1,UN/LOCODE
      X,CN,TAG,Taicang Pt,Taicang Pt,JS,XX,1-------,1407,,3127N 12106E,Use CNTAC,"31.45000,121.10000",N/A,CNTAC

    We map columns by position:

      c0: flag / change marker (ignored)
      c1: Country (ISO2)
      c2: Locode-3
      c3: Name
      c4: NameAlt (often ascii)
      c5: Subdivision (admin1)
      c6: Status (RL, AI, XX, etc.)
      c7: Function (UN/LOCODE function string: '1-------', '-----6--', etc.)
      c8: Date
      c9: IATA
      c10: Coordinates (UN/LOCODE style: 1753S 03042E)
      c11: Remarks / extra
      c12: CoordinatesDecimal (e.g. '-17.88333,30.70000')
      c13: Quality / Nominatim flag
      c14: Source

    We build full_unlocode as Country + Locode-3, e.g. 'ZW' + 'NAG' -> 'ZWNAG'.
    """
    # First line is "(no Nominatim),UN/LOCODE", so skip 1 row and read raw.
    df = pd.read_csv(path, header=None, skiprows=1, dtype=str).fillna("")

    data: Dict[str, Dict[str, Any]] = {}
    for _, row in df.iterrows():
        country = row[1].strip().upper()
        loc3 = row[2].strip().upper()
        if not country or not loc3:
            continue

        full_code = country + loc3  # e.g. 'ZW' + 'NAG' -> 'ZWNAG'

        name = row[3].strip()
        subdiv = row[5].strip()
        status = row[6].strip()
        function = row[7].strip()
        coords_dec = row[12].strip()  # "-17.88333,30.70000"

        data[full_code] = {
            "unlocode": full_code,
            "country_iso2": country,
            "name": name,
            "subdivision": subdiv,
            "function": function,
            "status": status,
            "coordinates_decimal": coords_dec,
        }

    return data


def load_unlocode_aliases(path: Path) -> Dict[str, List[str]]:
    """
    Load aliases-improved.csv into dict: full_unlocode -> list of aliases.

    Format:

      Unlocode,Alias
      ZWNAG,ノートン
      ZWNAG,諾頓
      ZWVFA,"Victoria Falls, Zimbabwe"
      ...
    """
    COLUMN_UNLOCODE = "Unlocode"
    COLUMN_ALIAS = "Alias"

    df = pd.read_csv(path, dtype=str).fillna("")

    aliases: Dict[str, List[str]] = defaultdict(list)
    for _, row in df.iterrows():
        code = row[COLUMN_UNLOCODE].strip().upper()
        alias = row[COLUMN_ALIAS].strip()
        if not code or not alias:
            continue
        aliases[code].append(alias)
    return aliases


def load_unlocode_parents(path: Path) -> Dict[str, str]:
    """
    parents.csv: child_unlocode -> parent_unlocode

      Unlocode,Parent
      SMFCO,SMSRV
      THLKR,THBKK
      USAAP,USBAL
      ...
    """
    COLUMN_UNLOCODE = "Unlocode"
    COLUMN_PARENT = "Parent"

    df = pd.read_csv(path, dtype=str).fillna("")

    parents: Dict[str, str] = {}
    for _, row in df.iterrows():
        child = row[COLUMN_UNLOCODE].strip().upper()
        parent = row[COLUMN_PARENT].strip().upper()
        if child and parent:
            parents[child] = parent
    return parents


def load_ims_facilities(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    # raw is like {"data": [ {...}, {...} ]} from IMS export
    if isinstance(raw, dict) and "data" in raw:
        return raw["data"]
    elif isinstance(raw, list):
        return raw
    else:
        raise ValueError(f"Unexpected ims_facilities JSON structure: {type(raw)}")


def load_ports_kb_ims_only(path: Path) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """
    Load your existing ports_kb_ims_only.json into dict keyed by (ims_facility_id, country_iso2).

    This avoids collisions when IMS reuses IDs across countries (e.g. ZWN in BE and NL).
    """
    with path.open("r", encoding="utf-8") as f:
        items = json.load(f)

    by_key: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for item in items:
        fid = item.get("ims_facility_id")
        c2 = (item.get("country_iso2") or "").strip().upper()
        if fid and c2:
            by_key[(fid, c2)] = item
    return by_key


def parse_lat_lon_from_decimal(coord_str: str) -> (Optional[float], Optional[float]):
    if not coord_str:
        return None, None
    if "," in coord_str:
        lat_str, lon_str = coord_str.split(",", 1)
    elif " " in coord_str:
        lat_str, lon_str = coord_str.split(None, 1)
    else:
        return None, None
    try:
        return float(lat_str), float(lon_str)
    except ValueError:
        return None, None


def is_seaport(function_str: str) -> bool:
    """
    UN/LOCODE function string: '1' indicates seaport.
    We keep it simple and just check for '1' anywhere in the string.
    """
    if not function_str:
        return False
    return "1" in function_str


# ---------- main merge ----------

def build_ports_kb_final() -> List[Dict[str, Any]]:
    print(f"[MERGE] Loading improved UN/LOCODE from {CODELIST_CSV}")
    unlocode_data = load_unlocode_codelist(CODELIST_CSV)

    print(f"[MERGE] Loading aliases from {ALIASES_CSV}")
    unlocode_aliases = load_unlocode_aliases(ALIASES_CSV)

    print(f"[MERGE] Loading parents from {PARENTS_CSV}")
    parent_map = load_unlocode_parents(PARENTS_CSV)

    print(f"[MERGE] Loading IMS facilities from {IMS_FACILITIES_JSON}")
    ims_facilities = load_ims_facilities(IMS_FACILITIES_JSON)

    print(f"[MERGE] Loading existing ports_kb_ims_only from {PORTS_KB_IMS_ONLY_JSON}")
    old_kb_by_key = load_ports_kb_ims_only(PORTS_KB_IMS_ONLY_JSON)

    merged: List[Dict[str, Any]] = []

    for fac in ims_facilities:
        attrs = fac.get("attributes", {})
        ims_id = fac.get("id") or attrs.get("locationISOCode")
        if not ims_id:
            continue

        country_iso2 = (attrs.get("countryISOCode") or "").strip().upper()
        loc3 = (attrs.get("locationISOCode") or fac.get("id") or "").strip().upper()

        full_unlocode = ""
        if len(country_iso2) == 2 and len(loc3) == 3:
            full_unlocode = country_iso2 + loc3

        ims_name = (attrs.get("locationName") or "").strip()
        ims_city = (attrs.get("city") or "").strip()

        # base record: IMS is the anchor ID
        rec: Dict[str, Any] = {
            "ims_facility_id": ims_id,
            "source": "ims",
            "country_iso2": country_iso2,
            "unlocode": full_unlocode,
            "location_name": ims_name,
            "city": ims_city,
            "lat": None,
            "lon": None,
            "admin1": "",
            "unlocode_function": "",
            "unlocode_status": "",
            "is_seaport": False,
            "parent_unlocode": "",
            "aliases": [],  # fill below
        }

        # enrich from improved UN/LOCODE
        if full_unlocode and full_unlocode in unlocode_data:
            ud = unlocode_data[full_unlocode]
            rec["admin1"] = ud.get("subdivision", "")
            rec["unlocode_function"] = ud.get("function", "")
            rec["unlocode_status"] = ud.get("status", "")
            rec["is_seaport"] = is_seaport(ud.get("function", ""))

            lat, lon = parse_lat_lon_from_decimal(ud.get("coordinates_decimal", ""))
            rec["lat"] = lat
            rec["lon"] = lon

        # parent relationship (hierarchy)
        if full_unlocode and full_unlocode in parent_map:
            rec["parent_unlocode"] = parent_map[full_unlocode]

        # aliases from IMS + old KB + unlocode aliases
        alias_set: Dict[str, str] = {}

        def add_alias(a: str):
            a = a.strip()
            if not a:
                return
            key = norm(a)
            if key and key not in alias_set:
                alias_set[key] = a

        # IMS fields
        add_alias(ims_name)
        add_alias(ims_city)

        # old KB aliases (country-safe via composite key)
        old = old_kb_by_key.get((ims_id, country_iso2))
        if old:
            for a in old.get("aliases", []):
                add_alias(a)
            # also maintain backward compatibility: any extra fields we don't overwrite
            for k, v in old.items():
                if k not in rec and k != "aliases":
                    rec[k] = v

        # UN/LOCODE aliases
        if full_unlocode and full_unlocode in unlocode_aliases:
            for a in unlocode_aliases[full_unlocode]:
                add_alias(a)

        # store final alias list (original casing)
        rec["aliases"] = sorted(alias_set.values())

        merged.append(rec)

    print(f"[MERGE] Built {len(merged)} unified port/facility records")
    return merged


def main():
    kb = build_ports_kb_final()
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(kb, f, ensure_ascii=False, indent=2)
    print(f"[MERGE] Wrote {len(kb)} records to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()

