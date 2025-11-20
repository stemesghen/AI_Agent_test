#!/usr/bin/env python3
"""
backfill_ports.py

Goal
----
1) Read your IMS-enriched ports file (may have null lat/lon/region).
2) Read UN/LOCODE+GeoNames CSV (your merged "ports.csv").
3) Backfill missing lat/lon using, in order of preference:
   - GeoNames decimal lat/lon (columns like lat/lon)
   - Parsed UN/LOCODE coords (columns like coords or lat_un/lon_un in DM format)
4) (Re)assign region using IHO World Seas shapefile.
5) Write an updated JSON with provenance flags.

Usage (PowerShell)
------------------
python backfill_ports.py `
  --ims-json data/ports_lookup_enriched_with_region.json `
  --unlocode-csv data/ports.csv `
  --seas-shp data/world_seas/World_Seas_IHO_v3.shp `
  --out-json data/ports_lookup_enriched_with_region.backfilled.json
"""

import argparse
import json
import re
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from difflib import SequenceMatcher

# geopandas/shapely are optional; if missing, region step is skipped gracefully
try:
    import geopandas as gpd
    from shapely.geometry import Point
except Exception:
    gpd = None
    Point = None


# -------------------------------------------------
# String normalization & fuzzy helpers
# -------------------------------------------------
def norm_text(s: str) -> str:
    """ASCII fold, lowercase, strip punctuation, collapse spaces."""
    import unicodedata, re
    s = str(s or "")
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]+", " ", s)
    s = s.replace(" port of ", " ")
    return " ".join(s.split())


def jaccard_tokens(a: str, b: str) -> float:
    A, B = set(a.split()), set(b.split())
    return len(A & B) / max(1, len(A | B))


def fuzzy_ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


# -------------------------------------------------
# UN/LOCODE DM coordinate parsing (e.g., "2559N 05533E")
# -------------------------------------------------

def parse_dm_token(tok: str) -> float:
    """
    Parse a UN/LOCODE DM/DMM token into signed decimal degrees.
    Examples:
      '548S'      -> 5°48' S
      '313N'      -> 3°13' N
      '04318W'    -> 43°18' W
      '2559N'     -> 25°59' N
    Rule: last two digits before the hemisphere letter are minutes (if len>=3).
    """
    tok = str(tok or "").strip().upper()
    if not tok or tok[-1] not in "NSEW":
        raise ValueError(f"Unrecognized DM token: {tok}")

    hemi = tok[-1]
    digits = tok[:-1]  # everything before N/S/E/W, e.g. '04318'

    if not digits.isdigit():
        raise ValueError(f"Unrecognized DM token: {tok}")

    if len(digits) <= 3:
        # e.g. '548' -> 5°48'
        deg = int(digits[:-2]) if len(digits) > 2 else int(digits or 0)
        mins = int(digits[-2:]) if len(digits) >= 2 else 0
    else:
        # e.g. '04318' -> 43°18'
        deg = int(digits[:-2])
        mins = int(digits[-2:])

    val = deg + mins / 60.0
    if hemi in ("S", "W"):
        val = -val
    return val


def parse_unlocode_coords(coords: Optional[str],
                          lat_un: Optional[str],
                          lon_un: Optional[str]) -> Optional[Tuple[float, float]]:
    """
    Extract decimal (lat, lon) from either:
      - 'coords' combined string like '2559N 05533E' or '548S 04318W'
      - 'lat_un','lon_un' separate DM tokens
    """
    try:
        if isinstance(coords, str) and coords.strip():
            parts = coords.replace(",", " ").split()
            # handle things like "548S 04318W" or "2559N 05533E"
            if len(parts) >= 2:
                lat = parse_dm_token(parts[0])
                lon = parse_dm_token(parts[1])
                return (lat, lon)

        if pd.notna(lat_un) and pd.notna(lon_un):
            lat = parse_dm_token(str(lat_un))
            lon = parse_dm_token(str(lon_un))
            return (lat, lon)
    except Exception:
        return None

    return None



def pick_best_latlon(cand: dict) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    """
    Choose best lat/lon from GeoNames first, else UN/LOCODE coords.
    Returns (lat, lon, source_tag)
    """
    lat_g = cand.get("lat")
    lon_g = cand.get("lon")
    try:
        if pd.notnull(lat_g) and pd.notnull(lon_g):
            return float(lat_g), float(lon_g), "geonames"
    except Exception:
        pass

    latlon = parse_unlocode_coords(cand.get("coords"), cand.get("lat_un"), cand.get("lon_un"))
    if latlon:
        return latlon[0], latlon[1], "unlocode"

    return None, None, None


# -------------------------------------------------
# Region helper (optional if GeoPandas available)
# -------------------------------------------------
def region_from_point(seas_gdf, point, name_field: str = "NAME") -> Optional[str]:
    try:
        hit = seas_gdf[seas_gdf.contains(point)]
        if hit.empty:
            return None
        return hit.iloc[0].get(name_field) or None
    except Exception:
        return None


# -------------------------------------------------
# Column aliasing for ports.csv (be liberal)
# -------------------------------------------------
ALIAS_MAP = {
    "country": ["country", "country_iso2", "ctry", "iso2", "Country"],
    "locode":  ["locode", "locationisocode", "unlocode", "locode3", "code", "LOCODE"],
    "name":    ["name", "location_name", "place", "port_name", "Name"],
    "city":    ["city", "municipality", "town", "City"],
    "lat":     ["lat", "latitude", "lat_geonames", "Latitude"],
    "lon":     ["lon", "longitude", "lng", "long", "lon_geonames", "Longitude"],
    "coords":  ["coords", "coord", "unlocode_coords"],
    "lat_un":  ["lat_un", "un_lat", "lat_dm"],
    "lon_un":  ["lon_un", "un_lon", "lon_dm"],
    "source":  ["source", "src"],
}


def normalize_ports_csv(df_raw: pd.DataFrame) -> pd.DataFrame:
    lower = {c.lower(): c for c in df_raw.columns}

    def first_present(keys):
        for k in keys:
            if k.lower() in lower:
                return lower[k.lower()]
        return None

    rename_map = {}
    for std, variants in ALIAS_MAP.items():
        hit = first_present(variants)
        if hit:
            rename_map[hit] = std

    df = df_raw.rename(columns=rename_map).copy()

    # Ensure all expected columns exist
    expected = ["country", "locode", "name", "city", "lat", "lon",
                "coords", "lat_un", "lon_un", "source"]
    for k in expected:
        if k not in df.columns:
            df[k] = None

    # Build join key CC+LOCODE (upper, no spaces)
    def key_from_country_locode(country, locode):
        if pd.isna(country) or pd.isna(locode):
            return None
        c = str(country).strip().upper()
        l = str(locode).strip().upper()
        return (c + l) if (c and l) else None

    df["join_key"] = df.apply(lambda r: key_from_country_locode(r.get("country"), r.get("locode")), axis=1)

    # Normalized name/city for exact & fuzzy matching
    df["name_norm"] = df["name"].map(norm_text)
    df["city_norm"] = df["city"].map(norm_text)

    return df


# -------------------------------------------------
# Index building WITH de-duplication of join_key
# -------------------------------------------------
def _row_quality_score(r: pd.Series) -> int:
    """
    Higher = better. Prioritize:
      1) GeoNames decimal lat/lon present
      2) UN/LOCODE coords (coords OR lat_un+lon_un)
      3) Having a name/source at all
    """
    score = 0
    if pd.notna(r.get("lat")) and pd.notna(r.get("lon")):
        score += 3
    has_unlocode_coords = pd.notna(r.get("coords")) or (
        pd.notna(r.get("lat_un")) and pd.notna(r.get("lon_un"))
    )
    if has_unlocode_coords:
        score += 2
    if pd.notna(r.get("name")):
        score += 1
    if pd.notna(r.get("source")):
        score += 1
    return score


def build_indexes(df: pd.DataFrame):
    # Ensure lowercase cols
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]

    # Keep rows that have a join_key at all
    has_key = df["join_key"].notna() & (df["join_key"].astype(str).str.len() > 0)
    df_keyed = df.loc[has_key].copy()

    # Score each row; pick best per join_key
    df_keyed["__score__"] = df_keyed.apply(_row_quality_score, axis=1)

    # Sort by (join_key asc, score desc) and drop duplicates so we keep the best
    df_best = (
        df_keyed
        .sort_values(["join_key", "__score__"], ascending=[True, False])
        .drop_duplicates(subset=["join_key"], keep="first")
    )

    # Build by_join dict from the deduped frame
    by_join = {
        k: row
        for k, row in df_best.set_index("join_key").to_dict(orient="index").items()
    }

    # Also build (country,name)->list and (country,city)->list for fallback
    name_idx = {}
    city_idx = {}

    for _, r in df.iterrows():
        c = (str(r.get("country") or "")).upper()
        nm = (str(r.get("name_norm") or "")).strip()
        ct = (str(r.get("city_norm") or "")).strip()
        if c and nm:
            name_idx.setdefault((c, nm), []).append(r)
        if c and ct:
            city_idx.setdefault((c, ct), []).append(r)

    # Optional debug: how many duplicates were collapsed
    dup_count = (len(df_keyed) - len(df_best))
    print(f"[INFO] join_key duplicates collapsed: {dup_count}")

    return by_join, name_idx, city_idx


# -------------------------------------------------
# Candidate pickers
# -------------------------------------------------
def pick_best_in_country(df: pd.DataFrame, country_iso2: str, name_norm: str) -> Optional[dict]:
    cand = df[df["country"].astype(str).str.upper() == (country_iso2 or "").upper()].copy()
    if cand.empty:
        cand = df.copy()
    # score by hybrid metric (Jaccard + fuzzy)
    cand["score"] = cand["name_norm"].apply(
        lambda x: 0.6 * jaccard_tokens(x, name_norm) + 0.4 * fuzzy_ratio(x, name_norm)
    )
    cand = cand.sort_values("score", ascending=False)
    if len(cand):
        top = cand.iloc[0]
        return top.to_dict() if top["score"] >= 0.80 else None
    return None


# -------------------------------------------------
# Main
# -------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ims-json", required=True, help="IMS JSON: ports_lookup_enriched_with_region.json")
    ap.add_argument("--unlocode-csv", required=True, help="UN/LOCODE+GeoNames CSV (your merged ports.csv)")
    ap.add_argument("--seas-shp", required=True, help="World_Seas_IHO_v3.shp")
    ap.add_argument("--out-json", required=True, help="Output JSON path")
    args = ap.parse_args()

    ims_path = Path(args.ims_json)
    csv_path = Path(args.unlocode_csv)
    shp_path = Path(args.seas_shp)
    out_path = Path(args.out_json)

    print(f"[INFO] Loading IMS JSON → {ims_path}")
    ims = json.loads(ims_path.read_text(encoding="utf-8"))

    print(f"[INFO] Loading UN/LOCODE CSV → {csv_path}")
    df_raw = pd.read_csv(csv_path)
    df = normalize_ports_csv(df_raw)
    by_join, name_idx, city_idx = build_indexes(df)
    print(f"[INFO] Loaded {len(df)} UN/LOCODE rows (with join_key for direct matches).")

    seas = None
    if gpd is not None and Point is not None:
        try:
            print(f"[INFO] Loading seas shapefile → {shp_path}")
            seas = gpd.read_file(shp_path)
            if seas.crs is None or (hasattr(seas.crs, "to_epsg") and seas.crs.to_epsg() != 4326):
                seas = seas.to_crs(4326)
        except Exception as e:
            print(f"[WARN] Could not load seas shapefile: {e}")
            seas = None
    else:
        print("[WARN] geopandas/shapely not available; region assignment will be skipped.")

    updated_coords = 0
    updated_region = 0
    total = len(ims)

    for row in ims:
        cc = (row.get("country_iso2") or "").upper().strip()
        official_unl = (row.get("official_unlocode") or "")
        official_unl = str(official_unl or "").upper().strip()  # e.g., "GBIOP"

        locname = row.get("location_name") or ""
        locname_norm = norm_text(locname)
        city_norm = norm_text(row.get("city") or "")

        lat = row.get("lat")
        lon = row.get("lon")

        # ---------------------------------
        # 1) Fill lat/lon if missing
        # ---------------------------------
        if lat is None or lon is None:
            candidate = None

            # A) Direct key from official UN/LOCODE "CC + LOC"
            if official_unl and len(official_unl) >= 5:
                key = official_unl[:2] + official_unl[2:]
                candidate = by_join.get(key)

            # B) Fallback: derive key from IMS cc + ims_facility_id if it looks like a 2–5 char code
            if candidate is None:
                ims_code = (row.get("ims_facility_id") or "").upper().strip()
                if cc and ims_code and 2 <= len(ims_code) <= 5:
                    key = cc + ims_code
                    candidate = by_join.get(key)

            # C) Exact (country, normalized name)
            if candidate is None and cc and locname_norm:
                exacts = name_idx.get((cc, locname_norm))
                if exacts:
                    candidate = exacts[0].to_dict()

            # D) Exact (country, normalized city)
            if candidate is None and cc and city_norm:
                exacts = city_idx.get((cc, city_norm))
                if exacts:
                    candidate = exacts[0].to_dict()

            # E) Fuzzy best in-country by name
            if candidate is None and locname_norm:
                candidate = pick_best_in_country(df, cc, locname_norm)

            # If we got a candidate, choose best lat/lon (geonames → unlocode DM)
            if candidate is not None:
                plat, plon, prov = pick_best_latlon(candidate)
                if plat is not None and plon is not None and abs(plat) <= 90 and abs(plon) <= 180:
                    row["lat"] = float(plat)
                    row["lon"] = float(plon)
                    row["geom_source"] = prov  # 'geonames' or 'unlocode'
                    updated_coords += 1

                    # small nudge: if official UN/LOCODE aligned the join, lift confidence
                    if official_unl and candidate.get("join_key") == (official_unl[:2] + official_unl[2:]):
                        row["match_confidence"] = max(row.get("match_confidence", 0.0), 0.80)

        # ---------------------------------
        # 2) (Re)assign region using seas
        # ---------------------------------
        lat = row.get("lat")
        lon = row.get("lon")
        if seas is not None and lat is not None and lon is not None:
            try:
                pt = Point(float(lon), float(lat))
                region = region_from_point(seas, pt, name_field="NAME")
                if region:
                    if row.get("region") != region:
                        row["region"] = region
                        updated_region += 1
            except Exception:
                pass

        # Keep a consistent review flag
        if "needs_manual_review" not in row:
            row["needs_manual_review"] = False if (row.get("lat") and row.get("lon")) else True

    print(f"[OK] Backfilled lat/lon for {updated_coords} / {total} facilities.")
    print(f"[OK] Assigned/updated region for {updated_region} facilities.")
    print(f"[OK] Writing → {out_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(ims, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[DONE]")


if __name__ == "__main__":
    main()

