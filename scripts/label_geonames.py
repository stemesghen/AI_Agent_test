#!/usr/bin/env python3
# scripts/label_geonames.py
from __future__ import annotations
import argparse
import pandas as pd
import unicodedata

def to_ascii(s: str) -> str:
    if s is None: return ""
    s = str(s)
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")

def norm_upper(s: str) -> str:
    return (s or "").strip().upper()

def norm_name(s: str) -> str:
    return to_ascii((s or "").strip())

def _series_or_blank(df: pd.DataFrame, variants, default_len=None):
    """Return Series for first matching column (case-insensitive). Else blank Series."""
    if default_len is None:
        default_len = len(df)
    lower = {c.lower().strip(): c for c in df.columns}
    for v in variants:
        key = v.lower().strip()
        if key in lower:
            return df[lower[key]].astype(str)
    return pd.Series([""] * default_len, index=df.index, dtype=str)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geonames", required=True, help="Path to raw GeoNames CSV")
    ap.add_argument("-o", "--out", default="data/labeled_geonames.csv")
    args = ap.parse_args()

    # Robust read
    df = pd.read_csv(args.geonames, dtype=str, encoding="utf-8", engine="python", on_bad_lines="skip")

    n = len(df)
    port_name     = _series_or_blank(df, ["name"])
    name_ascii    = _series_or_blank(df, ["asciiname", "name_ascii", "normalized"])
    lat           = _series_or_blank(df, ["latitude", "lat"])
    lon           = _series_or_blank(df, ["longitude", "lon", "lng"])
    country_iso2  = _series_or_blank(df, ["country code", "country_code", "country_iso2", "iso2"]).map(norm_upper)
    admin1        = _series_or_blank(df, ["admin1 code", "admin1_code", "admin1"])
    admin2        = _series_or_blank(df, ["admin2 code", "admin2_code", "admin2"])
    feature_class = _series_or_blank(df, ["feature class", "feature_class"])
    feature_code  = _series_or_blank(df, ["feature code", "feature_code"])
    population    = _series_or_blank(df, ["population"])
    timezone      = _series_or_blank(df, ["timezone"])

    out = pd.DataFrame({
        "source":        pd.Series(["geonames"] * n, index=df.index),
        "port_name":     port_name.str.strip(),
        "name_ascii":    name_ascii.str.strip(),
        "city":          pd.Series([""] * n, index=df.index),  # GeoNames doesn’t have a distinct “city” for ports
        "country_iso2":  country_iso2,
        "unlocode":      pd.Series([""] * n, index=df.index),  # not in GeoNames
        "unloc_core":    pd.Series([""] * n, index=df.index),
        "lat":           lat.str.strip(),
        "lon":           lon.str.strip(),
        "coords":        pd.Series([""] * n, index=df.index),
        "feature_class": feature_class.str.strip(),
        "feature_code":  feature_code.str.strip(),
        "admin1":        admin1.str.strip(),
        "admin2":        admin2.str.strip(),
        "population":    population.str.strip(),
        "timezone":      timezone.str.strip(),
    })

    has_xy = (out["lat"] != "") & (out["lon"] != "")
    out.loc[has_xy, "coords"] = out.loc[has_xy, ["lat", "lon"]].agg(",".join, axis=1)

    empty_ascii = out["name_ascii"] == ""
    out.loc[empty_ascii, "name_ascii"] = out.loc[empty_ascii, "port_name"].map(norm_name)

    out.to_csv(args.out, index=False, encoding="utf-8")
    print(f"[OK] wrote {len(out):,} rows -> {args.out}")

if __name__ == "__main__":
    main()
