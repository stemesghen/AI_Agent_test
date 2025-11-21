# nlp/port_index.py
"""
Build a canonical in-memory port index.

Sources:
- UN/LOCODE (official or labeled CSV)
- GeoNames (cities + harbors; brings 'alternatenames' as aliases)
- Optional alias CSV: data/port_aliases.csv (columns: country_iso2,location_name,alias)
- Optional importance CSV: data/port_importance.csv (columns: country_iso2,location_name,importance [0..1])

Env:
- LOAD_GEONAMES=1 to include GN (default 1)

Output records (per facility):
{
  country_iso2, location_name, city, unlocode, lat, lon,
  aliases: List[str], subdivision, iata, source, importance (0..1)
}
"""

import os
import json
import math
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

ALIAS_CSV = Path(os.getenv("PORT_ALIAS_CSV", "data/port_aliases.csv"))
IMPORTANCE_CSV = Path(os.getenv("PORT_IMPORTANCE_CSV", "data/port_importance.csv"))


def _safe_str(x):
    return ("" if (x is None or (isinstance(x, float) and math.isnan(x))) else str(x)).strip()


def _to_float(x) -> Optional[float]:
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return None
        f = float(x)
        return f if math.isfinite(f) else None
    except Exception:
        return None


def load_region_centroids(regions_json: Path) -> Dict[str, Tuple[float, float]]:
    """
    Lightweight loader for region centroids from region_lookup.json.

    regions_json: Path to JSON file like data/region_lookup.json with
      [
        {"name": "...", "lat": float, "lon": float, "countries": [...]},
        ...
      ]

    Returns:
      { name_lower: (lat, lon), ... }
    """
    if not regions_json or not regions_json.exists():
        return {}
    data = json.loads(Path(regions_json).read_text(encoding="utf-8"))
    out: Dict[str, Tuple[float, float]] = {}
    for r in data:
        name = _safe_str(r.get("name")).lower()
        lat  = _to_float(r.get("lat"))
        lon  = _to_float(r.get("lon"))
        if name and lat is not None and lon is not None:
            out[name] = (lat, lon)
    return out


def _load_alias_table() -> pd.DataFrame:
    """
    Optional alias table
    columns: country_iso2, location_name, alias
    """
    if ALIAS_CSV and ALIAS_CSV.exists():
        df = pd.read_csv(ALIAS_CSV, dtype=str)
        df = df.rename(columns={c: c.strip() for c in df.columns})
        need = {"country_iso2", "location_name", "alias"}
        if need.issubset(set(df.columns)):
            return df[list(need)].fillna("")
    return pd.DataFrame(columns=["country_iso2", "location_name", "alias"])


def _load_importance_table() -> pd.DataFrame:
    """
    Optional importance table
    columns: country_iso2, location_name, importance [0..1]
    """
    if IMPORTANCE_CSV and IMPORTANCE_CSV.exists():
        df = pd.read_csv(IMPORTANCE_CSV, dtype=str)
        df = df.rename(columns={c: c.strip() for c in df.columns})
        need = {"country_iso2", "location_name", "importance"}
        if need.issubset(set(df.columns)):
            # normalize importance to [0,1]
            def clamp(v):
                try:
                    x = float(v)
                    if x < 0:
                        return 0.0
                    if x > 1:
                        return 1.0
                    return x
                except Exception:
                    return 0.0

            df["importance"] = df["importance"].map(clamp)
            return df[list(need)]
    # Empty table with zero default
    return pd.DataFrame(columns=["country_iso2", "location_name", "importance"])


def load_ports_index_from_sources(
    unlocode_csv: Path,
    geonames_csv: Path,
) -> List[Dict[str, Any]]:
    """
    Build a mixed port index from UN/LOCODE + GeoNames, then augment with
    alias and importance tables.

    Returns a list of dicts:
      {
        "country_iso2": str,
        "location_name": str,
        "city": str,
        "unlocode": str,
        "lat": float | None,
        "lon": float | None,
        "aliases": List[str],
        "subdivision": str,
        "iata": str,
        "source": "unlocode" | "geonames",
        "importance": float in [0,1],
      }
    """
    out: List[Dict[str, Any]] = []

    # ---------- UN/LOCODE ----------
    if unlocode_csv and unlocode_csv.exists():
        u = pd.read_csv(unlocode_csv, dtype=str, low_memory=False)
        for row in u.itertuples(index=False):
            iso2   = _safe_str(getattr(row, "Country", None)).upper()
            loc3   = _safe_str(getattr(row, "Location", None)).upper()
            name   = _safe_str(getattr(row, "NameWoDiacritics", None)) or _safe_str(getattr(row, "Name", None))
            subdiv = _safe_str(getattr(row, "Subdivision", None) or getattr(row, "SubDiv", None))
            iata   = _safe_str(getattr(row, "IATA", None))
            unloc  = f"{iso2}{loc3}" if iso2 and loc3 else ""

            if not (iso2 or name or unloc):
                continue

            out.append({
                "country_iso2": iso2,
                "location_name": name or "",
                "city": name or "",
                "unlocode": unloc,
                "lat": None,
                "lon": None,
                "aliases": [],
                "subdivision": subdiv,
                "iata": iata,
                "source": "unlocode",
                "importance": 0.0,
            })

    # ---------- GeoNames (optional; gate by env) ----------
    if (
        geonames_csv and geonames_csv.exists()
        and os.getenv("LOAD_GEONAMES", "1") == "1"
    ):
        g = pd.read_csv(geonames_csv, dtype=str, low_memory=False)

        name_col = "asciiname" if "asciiname" in g.columns else ("name" if "name" in g.columns else None)
        lat_col  = "latitude" if "latitude" in g.columns else ("lat" if "lat" in g.columns else None)
        lon_col  = "longitude" if "longitude" in g.columns else ("lon" if "lon" in g.columns else None)

        # Prefer harbors and populated places
        if {"feature_class", "feature_code"} <= set(g.columns):
            fclass = g["feature_class"].fillna("")
            fcode  = g["feature_code"].fillna("").str.upper()
            mask_keep = (fclass.eq("P")) | (fcode.isin({"H.HBR", "H.PRTL", "H.PRT", "H.MOOR"}))
            g = g[mask_keep]

        keep_cols = [c for c in [
            "country_code", name_col, "alternatenames", lat_col, lon_col, "admin1_code"
        ] if c is not None and c in g.columns]
        g = g.loc[:, keep_cols].copy()

        for row in g.itertuples(index=False, name=None):
            vals = {keep_cols[i]: row[i] for i in range(len(keep_cols))}
            city = _safe_str(vals.get(name_col, "")) if name_col else ""
            iso2 = _safe_str(vals.get("country_code", "")).upper()
            lat  = _to_float(vals.get(lat_col)) if lat_col else None
            lon  = _to_float(vals.get(lon_col)) if lon_col else None

            aliases = []
            alts = _safe_str(vals.get("alternatenames", ""))
            if alts:
                aliases = [a.strip() for a in alts.split(",") if a.strip()]

            out.append({
                "country_iso2": iso2,
                "location_name": city,
                "city": city,
                "unlocode": "",
                "lat": lat,
                "lon": lon,
                "aliases": aliases[:40],   # plenty of surface forms
                "subdivision": _safe_str(vals.get("admin1_code", "")),
                "iata": "",
                "source": "geonames",
                "importance": 0.0,
            })

    # ---------- Merge optional alias & importance tables ----------
    if not out:
        return out

    df = pd.DataFrame(out)

    # Merge aliases from alias CSV (append into list)
    alias_df = _load_alias_table()
    if not alias_df.empty:
        # join on country+location_name (case-insensitive safe join)
        alias_df["_key"] = (
            alias_df["country_iso2"].str.upper()
            + "///"
            + alias_df["location_name"].str.strip().str.lower()
        )
        df["_key"] = (
            df["country_iso2"].str.upper()
            + "///"
            + df["location_name"].str.strip().str.lower()
        )
        merged = df.merge(alias_df, on="_key", how="left", suffixes=("", "_al"))

        # fold aliases
        def fold_alias(row):
            aliases = list(row["aliases"]) if isinstance(row["aliases"], list) else []
            extra = _safe_str(row.get("alias"))
            if extra and extra not in aliases:
                aliases.append(extra)
            return aliases

        merged["aliases"] = merged.apply(fold_alias, axis=1)
        df = merged.drop(columns=["country_iso2_al", "location_name_al", "alias"])

    # Merge importance
    imp_df = _load_importance_table()
    if not imp_df.empty:
        imp_df["_key"] = (
            imp_df["country_iso2"].str.upper()
            + "///"
            + imp_df["location_name"].str.strip().str.lower()
        )
        if "_key" not in df.columns:
            df["_key"] = (
                df["country_iso2"].str.upper()
                + "///"
                + df["location_name"].str.strip().str.lower()
            )
        df = df.merge(imp_df[["_key", "importance"]], on="_key", how="left")
        df["importance_x"] = df.get("importance_x", 0.0)
        df["importance_y"] = df.get("importance_y", 0.0)
        df["importance"] = (
            df["importance_y"]
            .fillna(df["importance_x"])
            .fillna(0.0)
            .astype(float)
        )
        df = df.drop(columns=[c for c in ["importance_x", "importance_y"] if c in df.columns])

    # Convert back to list of dicts
    # Make sure 'aliases' is list for all rows
    def ensure_list(x):
        if isinstance(x, list):
            return x
        if isinstance(x, str) and x.strip():
            # If it came as "['a','b']" accidentally, try to split safely
            if x.strip().startswith("[") and x.strip().endswith("]"):
                try:
                    import ast
                    y = ast.literal_eval(x)
                    return [str(s) for s in y if s]
                except Exception:
                    pass
            return [x]
        return []

    df["aliases"] = df["aliases"].map(ensure_list)

    return df.to_dict(orient="records")
