#!/usr/bin/env python3
# merge_ports.py
#
# Safely merge UN/LOCODE + GeoNames into a deterministic port.csv (no live API calls).
# Optionally, if you explicitly pass an IMS Facilities JSON cache or a mapping CSV,
# it will backfill ims_facility_id using strict, auditable rules.
#
# Usage examples:
#   python merge_ports.py --unlocode data/UNLOCODE.csv --geonames data/geonames.csv -o data/port.csv
#   python merge_ports.py --unlocode data/UNLOCODE.csv --geonames data/geonames.csv \
#       --ims data/facilities_YYYY-MM-DD.json -o data/port.csv
#   python merge_ports.py --unlocode data/UNLOCODE.csv --geonames data/geonames.csv \
#       --ims-map data/ims_map.csv -o data/port.csv
#   python merge_ports.py --unlocode data/UNLOCODE.csv --geonames data/geonames.csv \
#       --ims data/facilities_YYYY-MM-DD.json --allow-name-match --fuzzy-threshold 0.92 -o data/port.csv
#
# Output columns (front block) then every other column preserved:
#   source, ims_facility_id, country_iso2, unlocode, geonameid, location_name, name_ascii, city,
#   admin1, admin2, feature_class, feature_code, population, timezone, coords, lat, lon,
#   match_rule, match_score, match_candidate_count, ims_source_file, ims_last_seen, ...
#
# Notes:
# - We NEVER drop rows. Duplicates across sources are kept.
# - We NEVER “invent” ims_facility_id. We only assign if a rule passes strictly.
# - Ambiguous candidates => leave ims_facility_id blank and record match metadata.

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, List

import pandas as pd
import re
import unicodedata
from difflib import SequenceMatcher


# ---------------------------
# Helpers
# ---------------------------

def to_ascii(s: Optional[str]) -> str:
    if s is None:
        return ""
    return unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode("ascii")


def parse_unlocode_coords(coord: str):
    """
    Parse UN/LOCODE 'DDMMN DDDMME' (or variant) into decimal lat, lon.
    Examples: '4230N 00131E' -> (42.5, 1.5167)
    """
    if not isinstance(coord, str) or not coord.strip():
        return None, None
    m = re.match(r"^\s*(\d{2})(\d{2})([NS])\s+(\d{3})(\d{2})([EW])\s*$", coord)
    if not m:
        m2 = re.match(r"^\s*(\d{2})(\d{2})([NS])\s*(\d{2,3})(\d{2})([EW])\s*$", coord)
        if not m2:
            return None, None
        g = m2.groups()
    else:
        g = m.groups()
    lat_deg, lat_min, lat_hem, lon_deg, lon_min, lon_hem = g
    lat = int(lat_deg) + int(lat_min) / 60.0
    lon = int(lon_deg) + int(lon_min) / 60.0
    if lat_hem == "S":
        lat = -lat
    if lon_hem == "W":
        lon = -lon
    return round(lat, 6), round(lon, 6)


def round_or_none(x, nd=3):
    try:
        return round(float(x), nd)
    except Exception:
        return None


def seq_ratio(a: str, b: str) -> float:
    a = (a or "").strip().lower()
    b = (b or "").strip().lower()
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def file_mtime_iso(path: Optional[str]) -> str:
    if not path:
        return ""
    try:
        ts = os.path.getmtime(path)
        return datetime.utcfromtimestamp(ts).strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        return ""


# ---------------------------
# Loaders
# ---------------------------

def load_unlocode(path: str) -> pd.DataFrame:
    p = Path(path)

    # Try Excel first if extension matches
    if p.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(p, dtype=str)
    else:
        # CSV reader with fallback encodings and separators
        encodings = ["utf-8", "utf-8-sig", "cp1252", "latin1"]
        seps = [",", ";", "\t"]
        last_err = None
        df = None
        for enc in encodings:
            for sep in seps:
                try:
                    df = pd.read_csv(
                        p,
                        dtype=str,
                        encoding=enc,
                        sep=sep,
                        engine="python",          # more forgiving
                        keep_default_na=False     # keep empty strings, not NaN
                    )
                    # Heuristic: if only one column but contains separators, try next sep
                    if df.shape[1] == 1 and any(sep_ in str(df.columns[0]) for sep_ in [",",";","\t"]):
                        continue
                    # If the header row looks smashed (very long single col), try next sep
                    if df.shape[1] == 1 and len(df.columns[0]) > 200:
                        continue
                    last_err = None
                    break
                except Exception as e:
                    last_err = e
            if last_err is None:
                break
        if df is None:
            raise last_err or ValueError(f"Failed to read {path} with tried encodings={encodings} and seps={seps}")

    # Normalize whitespace
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    # Lowercase headers for flexible mapping
    cols_lower = [str(c).strip().lower() for c in df.columns]
    df.columns = cols_lower

    # Map typical UN/LOCODE columns
    mapping = {}
    candidates = {
        "country": ["country", "cc", "ctry", "iso2", "iso_2", "a"],
        "locode": ["locode", "location", "code", "b"],
        "name": ["name", "location name", "locationname", "c", "e"],
        "name_ascii": ["namewodiacritics", "name_ascii", "normalized", "d"],
        "subdiv": ["subdiv", "subdivision", "adm1", "fips", "h"],
        "function": ["function", "func", "g"],
        "status": ["status", "stat", "i"],
        "date": ["date", "updated", "j"],
        "iata": ["iata", "airport", "air", "iataa", "m"],
        "coords": ["coordinates", "coord", "k"],
        "remarks": ["remarks", "remark", "n", "note"],
    }
    for key, names in candidates.items():
        for n in names:
            if n in cols_lower:
                mapping[key] = n
                break

    # Reasonable fallbacks if headers are unusual
    if "country" not in mapping and cols_lower:
        mapping["country"] = cols_lower[0]
    if "locode" not in mapping and len(cols_lower) > 1:
        mapping["locode"] = cols_lower[1]
    if "name" not in mapping and len(cols_lower) > 2:
        mapping["name"] = cols_lower[2]
    if "name_ascii" not in mapping and "name" in mapping:
        mapping["name_ascii"] = mapping["name"]
    # coords fallback often sits later in the sheet
    if "coords" not in mapping:
        for guess in cols_lower[::-1]:
            if "coord" in guess or "lat" in guess or "long" in guess:
                mapping["coords"] = guess
                break

    # Build normalized output
    out = pd.DataFrame()
    out["source"] = "unlocode"
    out["country_iso2"] = df.get(mapping.get("country", ""), "")
    loc = df.get(mapping.get("locode", ""), "").fillna("")
    cc = out["country_iso2"].fillna("")
    # Ensure UN/LOCODE has country prefix (e.g., "DKAAR")
    out["unlocode"] = [
        (c.upper() + l.upper() if l and not str(l).upper().startswith(c.upper()) else str(l).upper())
        for c, l in zip(cc, loc)
    ]
    out["location_name"] = df.get(mapping.get("name", ""), "")
    out["name_ascii"] = df.get(mapping.get("name_ascii", ""), "").map(to_ascii)
    out["city"] = df.get(mapping.get("name", ""), "")
    out["coords"] = df.get(mapping.get("coords", ""), "")

    # Try to parse lat/lon from typical UN/LOCODE coord format; fall back to splitable coords
    lat, lon = [], []
    for c in out["coords"].fillna(""):
        la, lo = parse_unlocode_coords(c)
        if la is None and isinstance(c, str) and ("," in c or "\t" in c or ";" in c):
            # common "lat,lon" fallback
            parts = re.split(r"[,\t;]\s*", c.strip())
            if len(parts) >= 2:
                la = round_or_none(parts[0], 6)
                lo = round_or_none(parts[1], 6)
        lat.append(la); lon.append(lo)
    out["lat"] = lat; out["lon"] = lon

    # keep extra fields if present
    for extra in ["subdiv", "function", "status", "date", "iata", "remarks"]:
        if extra in mapping:
            out[extra] = df[mapping[extra]]

    return out


def load_geonames(path: str) -> pd.DataFrame:
    p = Path(path)
    # Try standard GeoNames TSV first
    try:
        df = pd.read_csv(p, sep="\t", header=None, dtype=str, low_memory=False)
        if df.shape[1] >= 19:
            df = df.rename(columns={
                0: "geonameid", 1: "name", 2: "name_ascii", 3: "alt_names", 4: "lat", 5: "lon",
                6: "feature_class", 7: "feature_code", 8: "country_iso2", 9: "cc2", 10: "admin1",
                11: "admin2", 12: "admin3", 13: "admin4", 14: "population", 15: "elevation",
                16: "dem", 17: "timezone", 18: "moddate"
            })
        else:
            # Fallback to CSV with headers
            df2 = pd.read_csv(p, dtype=str)
            if df2 is not None:
                df = df2
    except Exception:
        df = pd.read_csv(p, dtype=str)

    # Normalize expected fields
    cols = df.columns.str.lower().tolist()

    def col(*names):
        for n in names:
            if n.lower() in cols:
                return n
        return None

    name_col = col("name")
    asci_col = col("asciiname", "name_ascii")
    lat_col = col("latitude", "lat")
    lon_col = col("longitude", "lon")
    ctry_col = col("country code", "country", "country_iso2", "country code 2", "cc2")
    admin1 = col("admin1", "admin1 code", "state", "admin1code")
    admin2 = col("admin2", "admin2 code", "county", "admin2code")
    geonameid = col("geonameid", "id")
    feature_class = col("feature class", "feature_class")
    feature_code = col("feature code", "feature_code")
    population = col("population")
    timezone = col("timezone")

    out = pd.DataFrame()
    out["source"] = "geonames"
    out["geonameid"] = df[geonameid] if geonameid else ""
    out["country_iso2"] = df[ctry_col] if ctry_col else ""
    out["location_name"] = df[name_col] if name_col else ""
    out["name_ascii"] = df[asci_col] if asci_col else (df[name_col].map(to_ascii) if name_col else "")
    out["city"] = df[name_col] if name_col else ""

    out["lat"] = pd.to_numeric(df[lat_col], errors="coerce") if lat_col else None
    out["lon"] = pd.to_numeric(df[lon_col], errors="coerce") if lon_col else None
    out["coords"] = out.apply(lambda r: f"{r['lat']},{r['lon']}" if pd.notnull(r["lat"]) and pd.notnull(r["lon"]) else "", axis=1)

    if admin1: out["admin1"] = df[admin1]
    if admin2: out["admin2"] = df[admin2]
    if feature_class: out["feature_class"] = df[feature_class]
    if feature_code: out["feature_code"] = df[feature_code]
    if population: out["population"] = df[population]
    if timezone: out["timezone"] = df[timezone]

    # carry along any extra columns too
    for c in df.columns:
        if c not in out.columns:
            out[c] = df[c]

    return out


def load_ims_from_json(path: str) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    items = payload.get("data", payload if isinstance(payload, list) else [])
    rows = []
    for it in items:
        attr = it.get("attributes", {}) if isinstance(it, dict) else {}
        rows.append({
            "ims_facility_id": (it.get("id") if isinstance(it, dict) else ""),
            "unlocode": (attr.get("locationISOCode") or ""),
            "country_iso2": (attr.get("countryISOCode") or ""),
            "location_name": (attr.get("locationName") or ""),
            "name_ascii": to_ascii(attr.get("locationName") or ""),
            "city": (attr.get("city") or ""),
        })
    return pd.DataFrame(rows)


def load_ims_mapping_csv(path: str) -> pd.DataFrame:
    # Expected at least: unlocode, ims_facility_id
    df = pd.read_csv(path, dtype=str)
    for col in ["unlocode", "ims_facility_id"]:
        if col not in df.columns:
            df[col] = ""
    return df


# ---------------------------
# Merge & backfill logic
# ---------------------------

def union_frames(frames: List[pd.DataFrame]) -> pd.DataFrame:
    all_cols = sorted(set().union(*[set(df.columns) for df in frames]))
    frames = [df.reindex(columns=all_cols) for df in frames]
    merged = pd.concat(frames, ignore_index=True)
    # Helpers
    merged["lat_r3"] = merged["lat"].apply(round_or_none)
    merged["lon_r3"] = merged["lon"].apply(round_or_none)
    merged["coords"] = merged.apply(
        lambda r: f"{r['lat']},{r['lon']}" if pd.notnull(r['lat']) and pd.notnull(r['lon']) else (r.get("coords", "") or ""),
        axis=1
    )
    return merged


def is_port_like(row: pd.Series) -> bool:
    # Prefer UN/LOCODE rows. For GeoNames, restrict to likely port/harbor features.
    if str(row.get("source", "")).lower() == "unlocode":
        return True
    fcode = str(row.get("feature_code", "") or "").upper()
    portish = {"HBR", "MRNA", "STNB"}  # Harbor, Marina, (rare) Submarine ballast station
    return fcode in portish


def backfill_ims_ids(
    ports: pd.DataFrame,
    ims_df: Optional[pd.DataFrame],
    ims_map_df: Optional[pd.DataFrame],
    ims_source_file: Optional[str],
    allow_name_match: bool,
    fuzzy_threshold: float,
    port_only: bool,
    country_strict: bool,
    fuzzy_margin: float = 0.02,
) -> pd.DataFrame:
    """
    ports: merged ports df (UN/LOCODE + GeoNames union)
    ims_df: extracted from IMS JSON cache (optional)
    ims_map_df: user mapping csv (optional) for exact unlocode mapping
    """

    # Ensure provenance columns exist
    for c in ["ims_facility_id", "match_rule", "match_score", "match_candidate_count", "ims_source_file", "ims_last_seen"]:
        if c not in ports.columns:
            ports[c] = ""

    ims_last_seen = file_mtime_iso(ims_source_file) if ims_source_file else ""
    if ims_source_file:
        ports["ims_source_file"] = ports["ims_source_file"].replace("", ims_source_file)
        ports["ims_last_seen"] = ports["ims_last_seen"].replace("", ims_last_seen)

    # ---- Rule 0: mapping CSV (unlocode exact) ----
    if ims_map_df is not None and not ims_map_df.empty:
        map_simple = ims_map_df[["unlocode", "ims_facility_id"]].dropna()
        map_simple["unlocode"] = map_simple["unlocode"].astype(str).str.strip().str.upper()
        mp = dict(zip(map_simple["unlocode"], map_simple["ims_facility_id"]))
        mask_empty = (ports["ims_facility_id"].isna()) | (ports["ims_facility_id"] == "")
        is_in = ports["unlocode"].astype(str).str.strip().str.upper().isin(mp.keys())
        fill_idx = ports.index[mask_empty & is_in]
        ports.loc[fill_idx, "ims_facility_id"] = ports.loc[fill_idx, "unlocode"].astype(str).str.strip().str.upper().map(mp)
        ports.loc[fill_idx, "match_rule"] = "unlocode_map"
        ports.loc[fill_idx, "match_score"] = "1.0"
        ports.loc[fill_idx, "match_candidate_count"] = "1"

    # ---- IMS JSON-based rules ----
    if ims_df is not None and not ims_df.empty:
        ims_df = ims_df.copy()
        ims_df["unlocode"] = ims_df["unlocode"].astype(str).str.strip().str.upper()
        ims_df["country_iso2"] = ims_df["country_iso2"].astype(str).str.strip().str.upper()
        ims_df["name_ascii"] = ims_df["name_ascii"].astype(str).str.strip().str.lower()

        # Rule 1: exact UN/LOCODE match
        mask_empty = (ports["ims_facility_id"].isna()) | (ports["ims_facility_id"] == "")
        left = ports.loc[mask_empty].copy()
        left["unlocode_norm"] = left["unlocode"].astype(str).str.strip().str.upper()

        exact = left.merge(
            ims_df[["unlocode", "ims_facility_id"]],
            how="left", left_on="unlocode_norm", right_on="unlocode"
        )

        fill_mask = exact["ims_facility_id"].fillna("") != ""
        if fill_mask.any():
            idx = exact.index[fill_mask]
            ports.loc[idx, "ims_facility_id"] = exact.loc[fill_mask, "ims_facility_id"].values
            ports.loc[idx, "match_rule"] = "unlocode_exact"
            ports.loc[idx, "match_score"] = "1.0"
            ports.loc[idx, "match_candidate_count"] = "1"
            if ims_source_file:
                ports.loc[idx, "ims_source_file"] = ims_source_file
                ports.loc[idx, "ims_last_seen"] = ims_last_seen

        # Optional: name+country rules
        if allow_name_match:
            # pre-index IMS by country for faster filtering
            ims_by_ctry = {ctry: g for ctry, g in ims_df.groupby("country_iso2")}

            remaining_mask = ((ports["ims_facility_id"].isna()) | (ports["ims_facility_id"] == ""))
            if port_only:
                remaining_mask = remaining_mask & ports.apply(is_port_like, axis=1)

            rem = ports.loc[remaining_mask].copy()
            rem["country_norm"] = rem["country_iso2"].astype(str).str.strip().str.upper()
            rem["name_norm"] = rem["name_ascii"].astype(str).str.strip().str.lower()

            to_assign = []
            for rid, r in rem.iterrows():
                nm = r["name_norm"]
                if not nm:
                    continue

                # Country selection (strict by default)
                if country_strict:
                    ctry = r["country_norm"]
                    ims_c = ims_by_ctry.get(ctry)
                    if ims_c is None or ims_c.empty:
                        continue
                else:
                    # Not strict: search across all
                    ims_c = ims_df

                # Exact name_ascii + country
                # (use equality check; if multiple exact, skip for ambiguity)
                eq = ims_c[ims_c["name_ascii"] == nm]
                if len(eq) == 1:
                    to_assign.append((rid, eq["ims_facility_id"].iloc[0], "name_ascii_country_exact", 1, 0.95, 1))
                    continue
                elif len(eq) > 1:
                    continue  # ambiguous

                # Fuzzy name + country with threshold
                scores = ims_c["name_ascii"].apply(lambda s: seq_ratio(s, nm))
                ims_scored = ims_c.assign(ratio=scores).sort_values("ratio", ascending=False)
                if ims_scored.empty:
                    continue
                top = ims_scored.iloc[0]
                if top["ratio"] >= float(fuzzy_threshold):
                    # ensure uniqueness within a margin
                    close = ims_scored[ims_scored["ratio"] >= (float(top["ratio"]) - float(fuzzy_margin))]
                    if len(close) == 1:
                        to_assign.append((rid, top["ims_facility_id"], "name_ascii_country_fuzzy", min(3, len(ims_scored)), float(top["ratio"]), 1))
                    # else ambiguous -> skip

            for rid, ims_id, rule, cand_total, score, unique_count in to_assign:
                ports.at[rid, "ims_facility_id"] = ims_id
                ports.at[rid, "match_rule"] = rule
                ports.at[rid, "match_score"] = f"{score:.4f}"
                ports.at[rid, "match_candidate_count"] = str(cand_total if rule != "name_ascii_country_exact" else 1)
                if ims_source_file:
                    ports.at[rid, "ims_source_file"] = ims_source_file
                    ports.at[rid, "ims_last_seen"] = ims_last_seen

    return ports


def finalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    preferred = [
        "source", "ims_facility_id", "country_iso2", "unlocode", "geonameid",
        "location_name", "name_ascii", "city",
        "admin1", "admin2", "feature_class", "feature_code",
        "population", "timezone", "coords", "lat", "lon",
        "match_rule", "match_score", "match_candidate_count",
        "ims_source_file", "ims_last_seen",
    ]
    # Ensure all preferred columns exist
    for c in preferred:
        if c not in df.columns:
            df[c] = ""
    # Drop helper cols if present
    df = df.drop(columns=[c for c in ["lat_r3", "lon_r3"] if c in df.columns])
    # Keep the original order: preferred first, then the rest
    rest = [c for c in df.columns if c not in preferred]
    return df[preferred + rest]



# ---------------------------
# Main
# ---------------------------

def main():
    ap = argparse.ArgumentParser(description="Merge UN/LOCODE + GeoNames (+ optional IMS cache/mapping) into port.csv with safe backfilling.")
    ap.add_argument("--unlocode", required=True, help="Path to UN/LOCODE CSV/XLSX")
    ap.add_argument("--geonames", required=True, help="Path to GeoNames TSV/CSV (e.g., cities1000.txt)")
    ap.add_argument("--ims", help="Optional IMS Facilities JSON cache (e.g., saved from Postman)")
    ap.add_argument("--ims-map", help="Optional CSV mapping with columns: unlocode, ims_facility_id")
    ap.add_argument("--allow-name-match", action="store_true", help="Enable name+country matching when UN/LOCODE exact match fails")
    ap.add_argument("--fuzzy-threshold", type=float, default=0.92, help="Fuzzy threshold for name matching (default 0.92)")
    ap.add_argument("--no-port-only", action="store_true", help="Allow GeoNames non-port features to be matched (NOT recommended)")
    ap.add_argument("--no-country-strict", action="store_true", help="Allow cross-country name matching (NOT recommended)")
    ap.add_argument("-o", "--out", default="port.csv", help="Output CSV path")

    args = ap.parse_args()

    # Load sources
    u = load_unlocode(args.unlocode)
    g = load_geonames(args.geonames)

    merged = union_frames([u, g])

    # Optional IMS inputs
    ims_df = load_ims_from_json(args.ims) if args.ims else None
    ims_map_df = load_ims_mapping_csv(args.ims_map) if args.ims_map else None
    ims_source_file = args.ims if args.ims else None

    # Safe backfill
    merged = backfill_ims_ids(
        ports=merged,
        ims_df=ims_df,
        ims_map_df=ims_map_df,
        ims_source_file=ims_source_file,
        allow_name_match=bool(args.allow_name_match),
        fuzzy_threshold=float(args.fuzzy_threshold),
        port_only=(not args.no_port_only),
        country_strict=(not args.no_country_strict),
    )

    out_df = finalize_columns(merged)
    out_df.to_csv(args.out, index=False, encoding="utf-8")
    print(f"[OK] Wrote {len(out_df):,} rows -> {args.out}")


if __name__ == "__main__":
    main()
