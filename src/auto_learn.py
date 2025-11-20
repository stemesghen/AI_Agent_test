# src/alias_store.py
from __future__ import annotations
import csv
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Optional

ALIASES_DIR = Path("data/aliases")
ALIASES_DIR.mkdir(parents=True, exist_ok=True)

PORT_ALIASES_CSV   = ALIASES_DIR / "port_aliases.csv"
VESSEL_ALIASES_CSV = ALIASES_DIR / "vessel_aliases.csv"

# CSV headers we’ll use
# port_aliases.csv   -> alias, canonical_name, facility_id, locode, notes
# vessel_aliases.csv -> alias, canonical_name, imo, notes

@dataclass
class PortAlias:
    alias: str
    canonical_name: str | None
    facility_id: str | None
    locode: str | None
    notes: str | None = None

@dataclass
class VesselAlias:
    alias: str
    canonical_name: str | None
    imo: str | None
    notes: str | None = None

def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))

def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

def load_port_aliases() -> Dict[str, PortAlias]:
    rows = _read_csv(PORT_ALIASES_CSV)
    out: Dict[str, PortAlias] = {}
    for r in rows:
        alias = (r.get("alias") or "").strip().lower()
        if not alias:
            continue
        out[alias] = PortAlias(
            alias=alias,
            canonical_name=(r.get("canonical_name") or "").strip() or None,
            facility_id=(r.get("facility_id") or "").strip() or None,
            locode=(r.get("locode") or "").strip() or None,
            notes=(r.get("notes") or "").strip() or None,
        )
    return out

def load_vessel_aliases() -> Dict[str, VesselAlias]:
    rows = _read_csv(VESSEL_ALIASES_CSV)
    out: Dict[str, VesselAlias] = {}
    for r in rows:
        alias = (r.get("alias") or "").strip().lower()
        if not alias:
            continue
        out[alias] = VesselAlias(
            alias=alias,
            canonical_name=(r.get("canonical_name") or "").strip() or None,
            imo=(r.get("imo") or "").strip() or None,
            notes=(r.get("notes") or "").strip() or None,
        )
    return out

def upsert_port_alias(alias: str, canonical_name: str | None, facility_id: str | None, locode: str | None, notes: str | None = None):
    alias_lc = (alias or "").strip().lower()
    rows = _read_csv(PORT_ALIASES_CSV)
    fieldnames = ["alias", "canonical_name", "facility_id", "locode", "notes"]
    found = False
    for r in rows:
        if (r.get("alias") or "").strip().lower() == alias_lc:
            r["canonical_name"] = canonical_name or ""
            r["facility_id"] = facility_id or ""
            r["locode"] = locode or ""
            r["notes"] = notes or ""
            found = True
            break
    if not found:
        rows.append({
            "alias": alias, "canonical_name": canonical_name or "",
            "facility_id": facility_id or "", "locode": locode or "", "notes": notes or ""
        })
    _write_csv(PORT_ALIASES_CSV, rows, fieldnames)

def upsert_vessel_alias(alias: str, canonical_name: str | None, imo: str | None, notes: str | None = None):
    alias_lc = (alias or "").strip().lower()
    rows = _read_csv(VESSEL_ALIASES_CSV)
    fieldnames = ["alias", "canonical_name", "imo", "notes"]
    found = False
    for r in rows:
        if (r.get("alias") or "").strip().lower() == alias_lc:
            r["canonical_name"] = canonical_name or ""
            r["imo"] = imo or ""
            r["notes"] = notes or ""
            found = True
            break
    if not found:
        rows.append({
            "alias": alias, "canonical_name": canonical_name or "", "imo": imo or "", "notes": notes or ""
        })
    _write_csv(VESSEL_ALIASES_CSV, rows, fieldnames)
