#nlp/country_codes.py
from __future__ import annotations
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Optional
from unidecode import unidecode
from rapidfuzz import fuzz


# ---------------------------
# Normalization helper
# ---------------------------
def _norm(s: str) -> str:
    """
    Normalize country or demonym words:
      - ASCII fold
      - lowercase
      - collapse whitespace
    """
    return " ".join(unidecode((s or "").strip()).lower().split())


# ---------------------------
# Load ISO CSV → name2iso
# ---------------------------
def _load_canonical_country_codes(csv_path: Path) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Internal helper:
      Returns:
        name2iso:  normalized country name -> ISO2
        iso2name:  ISO2 -> canonical country name

    Works with your Name / Code CSV.
    """
    if not csv_path.exists():
        return {}, {}

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        cols = {c.lower(): c for c in (reader.fieldnames or [])}

        # Your CSV uses "Name" and "Code"
        name_col = cols.get("name") or cols.get("country") or cols.get("country_name")
        iso_col = cols.get("code") or cols.get("iso2") or cols.get("alpha-2")

        if not name_col or not iso_col:
            raise RuntimeError(
                "country_ISO.csv must have columns like 'Name' and 'Code' or similar."
            )

    name2iso: Dict[str, str] = {}
    iso2name: Dict[str, str] = {}

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = (row.get(name_col) or "").strip()
            iso2 = (row.get(iso_col) or "").strip().upper()
            if not name or len(iso2) != 2:
                continue
            name2iso[_norm(name)] = iso2
            # first one wins; that's fine for canonical mapping
            if iso2 not in iso2name:
                iso2name[iso2] = name

    return name2iso, iso2name


# ---------------------------
# Load alias CSV → alias2iso
# ---------------------------
def _load_country_aliases(alias_path: Optional[Path]) -> Dict[str, str]:
    """
    Load alias table (country_aliases.csv) and return:
        alias2iso: normalized alias/demonym/variant -> ISO2

    If the file does not exist, returns {} so code still works.
    """
    if alias_path is None:
        return {}

    if not alias_path.exists():
        # allow running before aliases are built
        return {}

    alias2iso: Dict[str, str] = {}
    with alias_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        cols = {c.lower(): c for c in (reader.fieldnames or [])}

        alias_col = cols.get("alias")
        iso_col = cols.get("iso2") or cols.get("code") or cols.get("alpha-2")

        if not alias_col or not iso_col:
            raise RuntimeError(
                f"{alias_path} must have at least 'alias' and 'iso2' columns. "
                f"Found: {list(reader.fieldnames or [])}"
            )

        for row in reader:
            alias = (row.get(alias_col) or "").strip()
            iso2 = (row.get(iso_col) or "").strip().upper()
            if not alias or len(iso2) != 2:
                continue
            alias2iso[_norm(alias)] = iso2

    return alias2iso


# ---------------------------
# Public loader
# ---------------------------
def load_country_codes(
    csv_path: Path,
    alias_csv_path: Optional[Path] = None,
) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]:
    """
    Returns:
      name2iso:   normalized canonical country name -> ISO2
      iso2name:   ISO2 -> canonical country name
      alias2iso:  normalized alias/demonym/variant -> ISO2

    - csv_path:       your country_ISO.csv (Name, Code)
    - alias_csv_path: data/country_aliases.csv (alias, iso2, ...)
    """
    name2iso, iso2name = _load_canonical_country_codes(csv_path)
    alias2iso = _load_country_aliases(alias_csv_path)
    return name2iso, iso2name, alias2iso


# ---------------------------
# Smart fuzzy + alias resolver
# ---------------------------
def map_names_to_iso2(
    names: List[str],
    name2iso: Dict[str, str],
    alias2iso: Optional[Dict[str, str]] = None,
) -> List[str]:
    """
    Map arbitrary country-ish strings to ISO2 codes using:
      1) alias2iso (from country_aliases.csv: demonyms, DBpedia forms, manual overrides)
      2) exact matches from name2iso (canonical names)
      3) fuzzy match + demonym stripping against canonical names

    Handles:
      • Exact matches (Russia, China, Finland)
      • Demonyms (Russian, Chinese, Venezuelan, Finnish, Greek) via alias2iso
      • Partial roots (russ -> russia, venezu -> venezuela)
      • Filters junk phrases (“Chinese Bulker Sinks”)
    """
    if not names or not name2iso:
        return []

    alias2iso = alias2iso or {}
    seen = set()
    results: List[str] = []

    # List of (normalized_country_name, iso2) for fuzzy matching
    country_items = list(name2iso.items())

    for raw in names:
        if not raw:
            continue

        txt = _norm(raw)
        if not txt:
            continue

        # Filter junk the model extracts sometimes
        if "bulker sinks" in txt:
            continue
        if "cargo" in txt and "sinks" in txt:
            continue

        iso = None

        # -----------------------
        # 0) Alias table first
        # -----------------------
        iso = alias2iso.get(txt)
        if iso and iso not in seen:
            seen.add(iso)
            results.append(iso)
            continue

        # -----------------------
        # 1) Exact canonical match
        # -----------------------
        iso = name2iso.get(txt)
        if iso and iso not in seen:
            seen.add(iso)
            results.append(iso)
            continue

        # -----------------------
        # 2) Straight fuzzy match
        # -----------------------
        best_iso = None
        best_score = 0

        for cname, ciso in country_items:
            # partial_ratio handles "russian federation" vs "russia"
            score = fuzz.partial_ratio(txt, cname)
            if score > best_score:
                best_score = score
                best_iso = ciso

        if best_iso and best_score >= 90:
            if best_iso not in seen:
                seen.add(best_iso)
                results.append(best_iso)
            continue

        # -----------------------
        # 3) Demonym stripping + fuzzy
        # -----------------------
        root = txt
        for suf in ("ian", "ean", "an", "ish", "ese", "i"):
            if root.endswith(suf) and len(root) > len(suf) + 2:
                root = root[: -len(suf)]
                break

        if root != txt:
            best2_iso = None
            best2_score = 0

            for cname, ciso in country_items:
                score2 = fuzz.partial_ratio(root, cname)
                if score2 > best2_score:
                    best2_score = score2
                    best2_iso = ciso

            if best2_iso and best2_score >= 85:
                if best2_iso not in seen:
                    seen.add(best2_iso)
                    results.append(best2_iso)
                continue

    return results


