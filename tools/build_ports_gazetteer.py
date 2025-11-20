# tools/build_ports_gazetteer.py
import argparse, re
from pathlib import Path
import pandas as pd

UN_PATH_DEFAULT = Path("data/raw/unlocode.txt")
GEONAMES_PATH_DEFAULT = Path("data/raw/geonames_allcountries.txt")
OUT_DIR = Path("data/gazetteer")
OUT_CSV = OUT_DIR / "ports.csv"

# -----------------------------
# UN/LOCODE fixed-width layout
# -----------------------------
COLSPECS = [
    (0, 2),    # Country
    (3, 6),    # Location
    (7, 42),   # Name
    (43, 78),  # NameWoDiacritics
    (78, 81),  # SubDiv
    (81, 89),  # Function (8 flags)
    (89, 91),  # Status
    (91, 95),  # Date (YYMM)
    (95, 99),  # IATA
    (99, 111), # Coordinates "2515N 05516E"
    (111, 999) # Remarks
]
COLNAMES = [
    "Country","Location","Name","NameWoDiacritics","SubDiv",
    "Function","Status","Date","IATA","Coordinates","Remarks"
]

# Alternative fixed-width layout with explicit 1-char separators
FWF_WIDTHS = [3, 2,1, 3,1, 35,1, 35,1, 5,1, 8,1, 2,1, 4,1, 3,1, 13]
FWF_NAMES  = [
    "pad", "Country","sep1","Location","sep2","Name","sep3","NameWoDiacritics","sep4",
    "SubDiv","sep5","Function","sep6","Status","sep7","Date","sep8","IATA","sep9","Coordinates"
]


SECTION_HEADER = re.compile(r'^\s*[A-Z]{2}\s+\.[A-Z].*')
ALIAS_LINE     = re.compile(r'^\s*(\=|\|)\s')

PORTY_WORDS = (
    " port", " puerto", " harbour", " harbor",
    " terminal", " anchorage", " quay", " wharf", " jetty"
)


def read_unlocode_txt(path: Path, encoding: str | None) -> pd.DataFrame:
    """Parse the UN/LOCODE fixed-width .txt using the proven widths."""
    df = pd.read_fwf(
        path, widths=FWF_WIDTHS, names=FWF_NAMES, dtype=str,
        encoding=encoding or "cp1252", on_bad_lines="skip"
    ).fillna("")
    # strip whitespace
    for c in df.columns:
        df[c] = df[c].astype(str).str.strip()

    # drop padding + 1-char separators
    sep_cols = ["pad"] + [c for c in df.columns if c.startswith("sep")]
    df = df.drop(columns=sep_cols, errors="ignore")

    # keep plausible rows (2-letter country + 3-char location)
    valid_country = df["Country"].str.fullmatch(r"[A-Z]{2}", na=False)
    valid_loc     = df["Location"].str.fullmatch(r"[A-Z0-9]{3}", na=False)
    df = df[valid_country & valid_loc].copy()

    # normalize function (sometimes spaces are present)
    df["Function"] = df["Function"].str.replace(" ", "-", regex=False)

    # Some dumps include section headers as rows—filter any leftover
    df = df[~df["Name"].str.startswith(".")].copy()

    return df.reset_index(drop=True)

def select_unlocode_ports(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Return rows that are real seaports, depending on mode."""
    if mode == "strict":
        # UN/LOCODE function position 1 == '1' (seaport)
        keep = df["Function"].str.len().gt(0) & df["Function"].str[0].eq("1")
    else:
        # 'broad': function contains '1' OR name looks port-ish
        fn_has_seaport = df["Function"].str.contains("1", na=False)
        name_porty = (
            df["Name"].str.lower().str.contains("|".join(PORTY_WORDS), regex=True, na=False) |
            df["NameWoDiacritics"].str.lower().str.contains("|".join(PORTY_WORDS), regex=True, na=False)
        )
        keep = fn_has_seaport | name_porty

    df = df[keep].copy()

    # Extract UN lat/lon tokens if present (e.g., "4230N 00131E")
    df[["lat_un","lon_un"]] = df["Coordinates"].str.extract(
        r'([0-9]{4}[NS])\s+([0-9]{5}[EW])', expand=True
    )

    # display_name preferring ASCII fallback
    df["display_name"] = df["NameWoDiacritics"].where(
        df["NameWoDiacritics"].ne(""), df["Name"]
    )

    out = df.rename(columns={
        "Country":"country",
        "Location":"locode",
        "SubDiv":"subdiv",
        "display_name":"name",
        "Name":"name_raw",
        "NameWoDiacritics":"name_ascii",
        "Function":"function",
        "Status":"status",
        "Date":"date",
        "IATA":"iata",
        "Coordinates":"coords",
    })
    out["source"] = "unlocode"

    cols = [
        "country","locode","subdiv","name","name_raw","name_ascii",
        "function","status","date","iata","coords","lat_un","lon_un",
        "source"
    ]
    for c in cols:
        if c not in out.columns:
            out[c] = ""
    return out[cols].reset_index(drop=True)

# --------------------------------
# GeoNames: allCountries.txt (TSV)
# Columns (standard; see geonames):
# geonameid name asciiname alternatenames latitude longitude feature class feature code country code ...
# --------------------------------
GEONAMES_COLS = [
    "geonameid","name","asciiname","alternatenames","latitude","longitude",
    "feature_class","feature_code","country_code","cc2","admin1","admin2","admin3","admin4",
    "population","elevation","dem","timezone","moddate"
]

HARBOR_CODES = {"H.HBR", "H.MRNA"}  # harbor, marina (you can add more if you like, e.g., "H.MOLE")

def read_geonames_ports(path: Path) -> pd.DataFrame:
    # GeoNames is tab-separated, UTF-8
    usecols = list(range(len(GEONAMES_COLS)))
    df = pd.read_csv(
        path, sep="\t", header=None, names=GEONAMES_COLS,
        usecols=usecols, dtype=str, low_memory=False, encoding="utf-8"
    ).fillna("")
    # keep only harbors/marinas
    df["fc_fc"] = df["feature_class"].astype(str).str.strip() + "." + df["feature_code"].astype(str).str.strip()
    df = df[df["fc_fc"].isin(HARBOR_CODES)].copy()

    # Build output shape
    out = pd.DataFrame({
        "country": df["country_code"].str.upper(),
        "locode": "",  # none for GeoNames
        "subdiv": df["admin1"],
        "name": df["asciiname"].where(df["asciiname"].ne(""), df["name"]),
        "name_raw": df["name"],
        "name_ascii": df["asciiname"],
        "function": "", "status": "", "date": "", "iata": "",
        "coords": "",
        "lat_un": "", "lon_un": "",
        "lat": df["latitude"], "lon": df["longitude"],
        "source": "geonames"
    })
    return out

def dedupe_merge(un_df: pd.DataFrame, gn_df: pd.DataFrame) -> pd.DataFrame:
    # unify columns
    for col in ["lat","lon"]:
        if col not in un_df.columns:
            un_df[col] = ""
    # concat
    all_df = pd.concat([un_df, gn_df], ignore_index=True)

    # Simple de-duplication by (country, lower(name))
    key_name = all_df["name"].astype(str).str.lower().str.strip()
    key_ctry = all_df["country"].astype(str).str.upper().str.strip()
    all_df["_key"] = key_ctry + "||" + key_name

    # prefer UN/LOCODE row when duplicates exist
    all_df["prio"] = all_df["source"].map({"unlocode": 1, "geonames": 2}).fillna(3)

    all_df = all_df.sort_values(["_key","prio"]).drop_duplicates("_key", keep="first").drop(columns=["_key","prio"])
    return all_df.reset_index(drop=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unlocode", default=str(UN_PATH_DEFAULT), help="UN/LOCODE fixed-width .txt")
    ap.add_argument("--geonames", default=str(GEONAMES_PATH_DEFAULT), help="GeoNames allCountries.txt (UTF-8)")
    ap.add_argument("--encoding", default="cp1252", help="Encoding for UN/LOCODE txt (cp1252 recommended)")
    ap.add_argument("--mode", choices=["strict","broad"], default="broad", help="strict=seaport flag only; broad=seaport OR name contains port-ish words + GeoNames harbors")
    ap.add_argument("--no-geonames", action="store_true", help="Skip GeoNames merge")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # UN/LOCODE
    un_df_raw = read_unlocode_txt(Path(args.unlocode), args.encoding)
    un_ports = select_unlocode_ports(un_df_raw, args.mode)

    # GeoNames (optional)
    if args.no_geonames:
        final_df = un_ports
    else:
        gn_df = read_geonames_ports(Path(args.geonames))
        final_df = dedupe_merge(un_ports, gn_df)

    final_df.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(f"[OK] UN/LOCODE rows parsed: {len(un_df_raw)}  →  kept as ports: {len(un_ports)}")
    if not args.no_geonames:
        print(f"[OK] GeoNames harbors/marinas: {len(gn_df)}")
    print(f"[OK] Wrote {OUT_CSV}  ({len(final_df)} total rows)")

if __name__ == "__main__":
    main()

