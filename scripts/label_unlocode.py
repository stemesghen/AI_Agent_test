#!/usr/bin/env python3
# scripts/label_unlocode.py
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
    ap.add_argument("--unlocode", required=True, help="Path to raw UNLOCODE CSV")
    ap.add_argument("-o", "--out", default="data/labeled_unlocode.csv")
    args = ap.parse_args()

    # Robust read (skip bad lines to survive odd rows)
    df = pd.read_csv(args.unlocode, dtype=str, encoding="utf-8", engine="python", on_bad_lines="skip")

    country   = _series_or_blank(df, ["country", "country_code", "iso2"]).map(norm_upper)
    loc       = _series_or_blank(df, ["location", "locode", "loc"])
    name      = _series_or_blank(df, ["name"])
    name_wo   = _series_or_blank(df, ["namewodiacritics", "name_ascii"])
    subdiv    = _series_or_blank(df, ["subdiv", "subdivision", "admin1", "admin1_code"])
    status    = _series_or_blank(df, ["status"])
    function  = _series_or_blank(df, ["function"])
    coords    = _series_or_blank(df, ["coordinates", "coords"])
    iata      = _series_or_blank(df, ["iata"])
    remarks   = _series_or_blank(df, ["remarks"])

    # Build codes
    unlocode = (country.fillna("") + loc.fillna("")).astype(str).str.upper().str.strip()
    unloc_core = unlocode.str[-3:]

    # Names
    port_name  = name.astype(str)
    name_ascii = name_wo.astype(str)
    port_name  = port_name.where(port_name.str.strip() != "", name_ascii)
    name_ascii = name_ascii.where(name_ascii.str.strip() != "", port_name.map(norm_name))

    out = pd.DataFrame({
        "source":          "unlocode",
        "port_name":       port_name.str.strip(),
        "name_ascii":      name_ascii.map(norm_name),
        "city":            subdiv.str.strip(),   # SubDiv often acts like admin/city
        "country_iso2":    country,
        "unlocode":        unlocode,
        "unloc_core":      unloc_core,
        "lat":             "",                  # UN/LOCODE coords are not lat/lon; leave blank for now
        "lon":             "",
        "coords":          coords.str.strip(),  # raw like 0402N07001W
        "feature_class":   "",
        "feature_code":    "",
        "admin1":          subdiv.str.strip(),
        "admin2":          "",
        "population":      "",
        "timezone":        "",
        # useful extras kept
        "unloc_status":    status.str.strip(),
        "unloc_function":  function.str.strip(),
        "iata":            iata.str.strip(),
        "remarks":         remarks.str.strip(),
    }).fillna("")

    out.to_csv(args.out, index=False, encoding="utf-8")
    print(f"[OK] wrote {len(out):,} rows -> {args.out}")

if __name__ == "__main__":
    main()
