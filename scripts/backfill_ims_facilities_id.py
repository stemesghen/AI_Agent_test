#!/usr/bin/env python3
"""
Backfill IMS Facility IDs into ports.csv using a saved IMS Facilities JSON cache.
Usage examples:
  python scripts/backfill_ims_facilities_id.py --in data/port.csv --ims data/Facilities_Ports_lookup.json -o data/port_with_ims.csv
  python scripts/backfill_ims_facilities_id.py --in data/port.csv --ims data/facilities_2025-11-06.json --fuzzy-threshold 0.92 -o data/port_with_ims.csv
"""
from __future__ import annotations
import argparse, json, re, os
from datetime import datetime, UTC
from pathlib import Path
from difflib import SequenceMatcher
import pandas as pd

def ts_iso(path: str | None) -> str:
    if not path: return ""
    try:
        return datetime.fromtimestamp(os.path.getmtime(path), UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        return ""

def normalize(s: str | None) -> str:
    return str(s).strip().lower() if s is not None else ""

def fuzzy(a: str, b: str) -> float:
    return SequenceMatcher(None, normalize(a), normalize(b)).ratio()

def load_ports(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, low_memory=False).fillna("")
    # Ensure columns we’ll write exist
    for c in ["ims_facility_id","match_rule","match_score","match_candidate_count","ims_source_file","ims_last_seen"]:
        if c not in df.columns: df[c] = ""
    # Normalize case for join keys
    if "unlocode" in df.columns:
        df["unlocode"] = df["unlocode"].astype(str).str.strip().str.upper()
    if "country_iso2" in df.columns:
        df["country_iso2"] = df["country_iso2"].astype(str).str.strip().str.upper()
    # Common name field
    if "location_name" not in df.columns:
        df["location_name"] = df.get("name", "")
    return df

def load_ims(ims_json_path: str) -> pd.DataFrame:
    with open(ims_json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    items = payload.get("data") or payload.get("Data") or payload

    rows = []
    for it in items:
        fac_id = str(it.get("id", "") or "")
        attr = it.get("attributes", {}) or {}
        rows.append({
            "ims_facility_id":    fac_id,
            "ims_locationISOCode": str(attr.get("locationISOCode", "") or "").strip().upper(),
            "ims_countryISOCode":  str(attr.get("countryISOCode",  "") or "").strip().upper(),
            "ims_locationName":    str(attr.get("locationName",    "") or "").strip(),
            "ims_city":            str(attr.get("city",            "") or "").strip(),
        })

    df = pd.DataFrame(rows).fillna("")
    # Ensure columns exist to avoid KeyErrors
    for c in ["ims_facility_id","ims_locationISOCode","ims_countryISOCode","ims_locationName","ims_city"]:
        if c not in df.columns: df[c] = ""
    return df

def backfill(
    ports: pd.DataFrame,
    ims_df: pd.DataFrame,
    ims_source_file: str,
    fuzzy_threshold: float = 0.90,
    do_fuzzy: bool = True,
) -> pd.DataFrame:

    ims_last_seen = ts_iso(ims_source_file)
    if "ims_source_file" in ports.columns:
        ports.loc[ports["ims_source_file"] == "", "ims_source_file"] = ims_source_file
    if "ims_last_seen" in ports.columns:
        ports.loc[ports["ims_last_seen"] == "", "ims_last_seen"] = ims_last_seen

    # 1) Exact UN/LOCODE join first (safest)
    merged = ports.merge(
        ims_df[["ims_facility_id","ims_locationISOCode","ims_countryISOCode"]],
        how="left",
        left_on="unlocode",
        right_on="ims_locationISOCode",
        suffixes=("", "_ims"),
    )

    # if we have a hit, stamp rule/score
    mask_exact = (merged["ims_facility_id"].astype(str) != "")
    merged.loc[mask_exact, "match_rule"] = merged.loc[mask_exact, "match_rule"].replace("", "unlocode_exact")
    merged.loc[mask_exact, "match_score"] = merged.loc[mask_exact, "match_score"].replace("", "1.0")
    merged.loc[mask_exact, "match_candidate_count"] = merged.loc[mask_exact, "match_candidate_count"].replace("", "1")

    # 2) Optional fuzzy fallback: name + same country
    if do_fuzzy:
        # Preindex IMS by country for speed
        ims_by_ctry = {ctry: g.copy() for ctry, g in ims_df.groupby("ims_countryISOCode")}
        # Work on the rows still missing IDs
        todo = merged[merged["ims_facility_id"].eq("")].index
        for idx in todo:
            ctry = str(merged.at[idx, "country_iso2"] or "")
            name = str(merged.at[idx, "location_name"] or "")
            if not ctry or not name:
                continue
            ims_c = ims_by_ctry.get(ctry)
            if ims_c is None or ims_c.empty:
                continue
            # compute best match
            scores = ims_c["ims_locationName"].apply(lambda s: fuzzy(name, s))
            j = scores.idxmax()
            best_score = float(scores.loc[j]) if j is not None else 0.0
            if best_score >= fuzzy_threshold:
                merged.at[idx, "ims_facility_id"] = ims_c.at[j, "ims_facility_id"]
                merged.at[idx, "match_rule"] = "name_country_fuzzy"
                merged.at[idx, "match_score"] = f"{best_score:.3f}"
                merged.at[idx, "match_candidate_count"] = "1"
            else:
                if not merged.at[idx, "match_rule"]:
                    merged.at[idx, "match_rule"] = "no_match"
                if not merged.at[idx, "match_score"]:
                    merged.at[idx, "match_score"] = "0"

    return merged

def main():
    ap = argparse.ArgumentParser(description="Backfill ims_facility_id into port.csv using IMS JSON")
    ap.add_argument("--in", dest="in_path", required=True, help="Input ports CSV (from merge step)")
    ap.add_argument("--ims", dest="ims_json", required=True, help="IMS Facilities JSON (has top-level 'id')")
    ap.add_argument("-o", "--out", dest="out_path", required=False, help="Output CSV; default data/port_with_ims.csv")
    ap.add_argument("--fuzzy-threshold", type=float, default=0.90, help="Fuzzy name+country threshold (default 0.90)")
    ap.add_argument("--no-fuzzy", action="store_true", help="Disable fuzzy fallback")
    args = ap.parse_args()

    ports = load_ports(args.in_path)
    ims_df = load_ims(args.ims_json)

    amended = backfill(
        ports=ports,
        ims_df=ims_df,
        ims_source_file=args.ims_json,
        fuzzy_threshold=float(args.fuzzy_threshold),
        do_fuzzy=not args.no_fuzzy,
    )

    out_path = args.out_path or str(Path(args.in_path).with_name("port_with_ims.csv"))
    amended.to_csv(out_path, index=False, encoding="utf-8")

    # quick summary
    total = len(amended)
    filled = (amended["ims_facility_id"] != "").sum()
    print(f"[OK] wrote {total:,} rows → {out_path} | filled ims_facility_id: {filled:,} ({filled/total:.1%})")

if __name__ == "__main__":
    main()

