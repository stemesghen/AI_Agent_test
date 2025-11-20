#!/usr/bin/env python
"""
Build SIMMO-style country alias list.

Steps (mirrors the journal’s “flags” logic):

1. Load canonical country list from country_ISO.csv
   - auto-detects the column for country name (e.g. Name, Country, country_name)
   - auto-detects the column for ISO2 code (e.g. Code, iso2)

2. Scan your corpus (e.g. data/extracted/*.csv) for all columns
   that contain 'country' in their name and collect distinct raw strings.

3. Fuzzy map raw strings → canonical country_name + iso2 using rapidfuzz.

4. For each canonical country_name, query DBpedia to pull extra
   English surface forms (lexicalization dataset style) and filter junk
   (demonym-like adjectives, “X’s”, “citizens of X”, etc.).

5. Apply manual overrides for tricky cases (DPRK, ROK, UAE, Ivory Coast, etc.).

6. Write a single CSV:
      alias, canonical_name, iso2, source
   where `source` ∈ {corpus-fuzzy, dbpedia, manual}
"""

import argparse
import glob
import logging
import re
import sys
import time
from typing import Dict, Iterable, List, Set, Tuple

import pandas as pd

try:
    # pip install rapidfuzz
    from rapidfuzz import fuzz, process
except ImportError as e:
    raise SystemExit(
        "This script requires rapidfuzz. Install it with:\n"
        "    pip install rapidfuzz\n"
    ) from e

try:
    # Used for DBpedia SPARQL requests
    import requests
except ImportError as e:
    raise SystemExit(
        "This script requires requests. Install it with:\n"
        "    pip install requests\n"
    ) from e


# ---------------------------- CONFIG ---------------------------------

# Candidate column names for country name and ISO2 in the input CSV
CANDIDATE_NAME_COLS = [
    "country_name",
    "CountryName",
    "country",
    "Country",
    "name",
    "Name",
]

CANDIDATE_ISO2_COLS = [
    "iso2",
    "ISO2",
    "code",
    "Code",
    "alpha2",
    "Alpha2",
]

# Fuzzy matching threshold for raw corpus strings → canonical country_name
FUZZY_THRESHOLD = 85  # you can tune this

# DBpedia SPARQL endpoint
DBPEDIA_SPARQL_URL = "https://dbpedia.org/sparql"

# How long to sleep between DBpedia queries (be polite)
DBPEDIA_SLEEP_SEC = 0.2

# Max results per canonical country from DBpedia
DBPEDIA_MAX_ALIASES = 200

# Junk/demonym suffixes to drop from DBpedia surface forms
DEMONYM_SUFFIXES = ("ish", "ese", "ian", "ean", "iote", "born")

# Surface forms matching these patterns will be dropped
JUNK_PATTERNS = [
    r".+’s$",         # e.g. "Italy’s"
    r".+'s$",         # "Mexico's"
    r".+ citizens$",  # "French citizens"
]

# Manual overrides: raw alias (normalized) → ISO2
MANUAL_OVERRIDES: Dict[str, str] = {
    # Common sanctions / geopolitical variants
    "occupied palestinian territory": "PS",
    "palestinian territory": "PS",
    "palestinian territories": "PS",
    "west bank": "PS",
    "gaza strip": "PS",

    "dprk": "KP",
    "democratic people’s republic of korea": "KP",
    "democratic peoples republic of korea": "KP",
    "north korea": "KP",

    "rok": "KR",
    "republic of korea": "KR",
    "south korea": "KR",

    "ivory coast": "CI",
    "cote d’ivoire": "CI",
    "cote d'ivoire": "CI",

    "uae": "AE",
    "u.a.e.": "AE",
    "emirates": "AE",
    "the emirates": "AE",

    "holland": "NL",
    "the netherlands": "NL",

    # Common abbreviations / noisy but frequent
    "uk": "GB",
    "u.k.": "GB",
    "great britain": "GB",

    "usa": "US",
    "u.s.a.": "US",
    "united states": "US",
    "united states of america": "US",
}


# ---------------------------------------------------------------------


def norm(s: str) -> str:
    """Simple normalization: lowercase + strip + collapse spaces."""
    if not isinstance(s, str):
        return ""
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def load_countries(country_csv: str) -> pd.DataFrame:
    """
    Load canonical country list and auto-detect the name and iso2 columns.

    Works with files that have e.g.:
      - country_name / iso2
      - Name / Code
      - Country / Code
      - etc.
    """
    df = pd.read_csv(country_csv)
    cols = list(df.columns)

    # Auto-detect name column
    name_col = None
    for c in CANDIDATE_NAME_COLS:
        if c in df.columns:
            name_col = c
            break

    # Auto-detect iso2 column
    iso_col = None
    for c in CANDIDATE_ISO2_COLS:
        if c in df.columns:
            iso_col = c
            break

    if name_col is None or iso_col is None:
        raise SystemExit(
            f"Could not auto-detect country name/iso2 columns in {country_csv}.\n"
            f"Found columns: {cols}\n"
            f"Tried name candidates: {CANDIDATE_NAME_COLS}\n"
            f"Tried iso2 candidates: {CANDIDATE_ISO2_COLS}"
        )

    logging.info(
        "Detected country name column=%r, iso2 column=%r in %s",
        name_col,
        iso_col,
        country_csv,
    )

    # Normalize to internal standard column names
    df = df.rename(columns={name_col: "country_name", iso_col: "iso2"})
    df["country_name"] = df["country_name"].astype(str)
    df["iso2"] = df["iso2"].astype(str).str.upper()

    return df


def collect_raw_country_strings(paths: Iterable[str]) -> Set[str]:
    """
    Scan all given CSV paths, look for columns that contain 'country'
    in their name, and collect distinct non-null strings.
    """
    raw: Set[str] = set()
    for path in paths:
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            logging.warning("Failed to read %s: %s", path, exc)
            continue

        country_cols = [c for c in df.columns if "country" in c.lower()]
        if not country_cols:
            continue

        for col in country_cols:
            vals = df[col].dropna().astype(str)
            for v in vals:
                v_norm = norm(v)
                if v_norm:
                    raw.add(v_norm)

    logging.info("Collected %d distinct raw country strings from corpus.", len(raw))
    return raw


def fuzzy_map_to_iso(
    raw_strings: Iterable[str],
    canonical_df: pd.DataFrame,
    threshold: int = FUZZY_THRESHOLD,
) -> List[Tuple[str, str, str]]:
    """
    Map raw corpus strings to (alias, canonical_name, iso2) using fuzzy match
    against canonical country_name list.
    """

    # canonical normalized name -> iso2
    canon_to_iso: Dict[str, str] = {
        norm(row["country_name"]): row["iso2"]
        for _, row in canonical_df.iterrows()
    }
    canon_names = list(canon_to_iso.keys())

    mappings: List[Tuple[str, str, str]] = []

    for raw in raw_strings:
        # If raw is already exactly a canonical normalized name, accept directly
        if raw in canon_to_iso:
            mappings.append((raw, raw, canon_to_iso[raw]))
            continue

        # Fuzzy match
        best = process.extractOne(
            raw,
            canon_names,
            scorer=fuzz.WRatio,
        )
        if best is None:
            continue
        candidate, score, _ = best
        if score < threshold:
            continue

        iso2 = canon_to_iso[candidate]
        mappings.append((raw, candidate, iso2))

    logging.info(
        "Fuzzy-mapped %d raw country strings to canonical names (threshold=%d).",
        len(mappings),
        threshold,
    )
    return mappings


def sparql_query(query: str) -> dict:
    """Send a SPARQL query to DBpedia; return JSON results."""
    params = {
        "query": query,
        "format": "application/sparql-results+json",
    }
    resp = requests.get(DBPEDIA_SPARQL_URL, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def find_dbpedia_country_resource(canonical_name: str) -> str:
    """
    Try to find a DBpedia resource URI for a given canonical country name.
    We follow the SIMMO idea: ensure it's typed as dbo:Country.
    Returns resource URI or "" if not found.
    """
    # 1. Try direct label match for dbo:Country
    q = f"""
    PREFIX dbo: <http://dbpedia.org/ontology/>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    SELECT DISTINCT ?s WHERE {{
      ?s a dbo:Country ;
         rdfs:label ?label .
      FILTER (lang(?label) = "en" && lcase(str(?label)) = "{canonical_name.lower()}")
    }}
    LIMIT 5
    """
    data = sparql_query(q)
    bindings = data.get("results", {}).get("bindings", [])
    if bindings:
        return bindings[0]["s"]["value"]

    # 2. Fallback: look for lexvo labels attached to dbo:Country
    q2 = f"""
    PREFIX dbo: <http://dbpedia.org/ontology/>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX lvont: <http://lexvo.org/ontology#>
    SELECT DISTINCT ?s WHERE {{
      ?s a dbo:Country .
      ?s lvont:label ?label .
      FILTER (lang(?label) = "en" && lcase(str(?label)) = "{canonical_name.lower()}")
    }}
    LIMIT 5
    """
    data2 = sparql_query(q2)
    bindings2 = data2.get("results", {}).get("bindings", [])
    if bindings2:
        return bindings2[0]["s"]["value"]

    return ""


def get_dbpedia_aliases_for_country(canonical_name: str) -> Set[str]:
    """
    Get English lexicalization surface forms for a country concept from DBpedia,
    filtered to be useful as flag names (SIMMO style).
    """
    aliases: Set[str] = set()

    try:
        resource = find_dbpedia_country_resource(canonical_name)
    except Exception as exc:
        logging.warning("DBpedia lookup failed for %s: %s", canonical_name, exc)
        return aliases

    if not resource:
        return aliases

    # Get all lexvo:label surface forms for this resource
    q = f"""
    PREFIX lvont: <http://lexvo.org/ontology#>
    SELECT DISTINCT ?label WHERE {{
      <{resource}> lvont:label ?label .
      FILTER (lang(?label) = "en")
    }}
    LIMIT {DBPEDIA_MAX_ALIASES}
    """
    try:
        data = sparql_query(q)
    except Exception as exc:
        logging.warning("DBpedia lexicalization query failed for %s: %s", canonical_name, exc)
        return aliases

    bindings = data.get("results", {}).get("bindings", [])
    for b in bindings:
        label = b["label"]["value"]
        clean = norm(label)
        if not clean:
            continue
        if _is_junk_surface_form(clean):
            continue
        aliases.add(clean)

    # Also, add the canonical name itself
    aliases.add(norm(canonical_name))

    return aliases


def _is_junk_surface_form(s: str) -> bool:
    """Apply SIMMO-like filters to drop demonyms and noisy forms."""
    # Very short or long things are suspicious
    if len(s) < 3 or len(s) > 80:
        return True

    for suf in DEMONYM_SUFFIXES:
        if s.endswith(suf):
            return True

    for pat in JUNK_PATTERNS:
        if re.fullmatch(pat, s):
            return True

    # Things like "citizens of X", "people of X"
    if " citizens" in s or " people" in s:
        return True

    return False


def apply_manual_overrides(
    aliases: Dict[Tuple[str, str, str], str],
    canonical_df: pd.DataFrame,
) -> Dict[Tuple[str, str, str], str]:
    """
    Add manual override aliases on top of existing alias mapping.

    `aliases` is a dict mapping (alias, canonical_name, iso2) → source.
    """
    iso2_to_canon: Dict[str, str] = {
        row["iso2"]: row["country_name"]
        for _, row in canonical_df.iterrows()
    }

    for raw_norm, iso2 in MANUAL_OVERRIDES.items():
        alias_norm = norm(raw_norm)
        iso2 = iso2.upper()
        canon = iso2_to_canon.get(iso2, iso2)  # best effort
        key = (alias_norm, norm(canon), iso2)
        # Don’t overwrite if already present
        if key not in aliases:
            aliases[key] = "manual"

    return aliases


def build_alias_table(
    country_csv: str,
    corpus_glob: str,
    out_csv: str,
    skip_dbpedia: bool = False,
):
    # 1) canonical country list
    canonical_df = load_countries(country_csv)

    # 2) collect raw corpus strings
    paths = glob.glob(corpus_glob)
    logging.info("Scanning corpus files matching %r (%d found).", corpus_glob, len(paths))
    raw_strings = collect_raw_country_strings(paths)

    # 3) fuzzy map to canonical
    fuzzy_mappings = fuzzy_map_to_iso(raw_strings, canonical_df, threshold=FUZZY_THRESHOLD)

    # alias dict: (alias, canonical_name, iso2) -> source
    alias_sources: Dict[Tuple[str, str, str], str] = {}

    # a) from corpus/fuzzy
    for alias, canon, iso2 in fuzzy_mappings:
        key = (alias, canon, iso2)
        alias_sources[key] = "corpus-fuzzy"

    # 4) DBpedia lexicalization, per canonical country
    if not skip_dbpedia:
        logging.info("Fetching DBpedia lexicalizations (this may take a while)...")
        seen_canon: Set[str] = set()
        for _, row in canonical_df.iterrows():
            canon = norm(row["country_name"])
            if not canon or canon in seen_canon:
                continue
            seen_canon.add(canon)
            iso2 = row["iso2"]
            try:
                aliases = get_dbpedia_aliases_for_country(canon)
            except Exception as exc:
                logging.warning("DBpedia aliases failed for %s: %s", canon, exc)
                aliases = set()

            for alias in aliases:
                key = (alias, canon, iso2)
                # Preserve any existing “corpus-fuzzy” source and only add dbpedia if new
                if key not in alias_sources:
                    alias_sources[key] = "dbpedia"

            time.sleep(DBPEDIA_SLEEP_SEC)

    # 5) manual overrides
    alias_sources = apply_manual_overrides(alias_sources, canonical_df)

    # 6) Write output CSV
    rows = []
    for (alias, canon, iso2), src in sorted(alias_sources.items()):
        rows.append(
            {
                "alias": alias,
                "canonical_name": canon,
                "iso2": iso2,
                "source": src,
            }
        )

    out_df = pd.DataFrame(rows)
    # Deduplicate by alias+iso2, keep first source type encountered
    out_df = out_df.drop_duplicates(subset=["alias", "iso2"]).reset_index(drop=True)
    out_df.to_csv(out_csv, index=False)
    logging.info("Wrote %d alias rows to %s", len(out_df), out_csv)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Build SIMMO-style country alias table (flags)."
    )
    p.add_argument(
        "--country-csv",
        required=True,
        help="Path to canonical country list (e.g. data/country_ISO.csv).",
    )
    p.add_argument(
        "--corpus-glob",
        default="data/extracted/*.csv",
        help="Glob for CSVs to mine raw country strings from "
             "(default: data/extracted/*.csv).",
    )
    p.add_argument(
        "-o",
        "--out",
        required=True,
        help="Output CSV for alias table (e.g. data/country_aliases.csv).",
    )
    p.add_argument(
        "--skip-dbpedia",
        action="store_true",
        help="Skip DBpedia lexicalization step (only fuzzy + manual).",
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose logging.",
    )
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    build_alias_table(
        country_csv=args.country_csv,
        corpus_glob=args.corpus_glob,
        out_csv=args.out,
        skip_dbpedia=args.skip_dbpedia,
    )


if __name__ == "__main__":
    main()
