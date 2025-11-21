from __future__ import annotations

import json
import math
import unicodedata
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from difflib import SequenceMatcher
from collections import defaultdict

# ----------------------------- paths -----------------------------
BASE_DIR = Path(__file__).resolve().parents[0]

PORTS_KB_FINAL = BASE_DIR / "data" / "ports_kb_final.json"
PORTS_KB_FALLBACK = BASE_DIR / "data" / "ports_kb_ims_only.json"


# ----------------------------- utilities -----------------------------

def norm(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def jaccard_tokens(a: str, b: str) -> float:
    ta = set(norm(a).split())
    tb = set(norm(b).split())
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return inter / union


def seq_ratio(a: str, b: str) -> float:
    a_n = norm(a)
    b_n = norm(b)
    if not a_n or not b_n:
        return 0.0
    return SequenceMatcher(None, a_n, b_n).ratio()


def geo_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Haversine distance in km.
    """
    if None in (lat1, lon1, lat2, lon2):
        return 1e9
    r = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return r * c


# ----------------------------- main class -----------------------------

class PlaceResolver:
    """
    Offline resolver over unified ports_kb_final.json.

    Core ideas:
      - Given context countries (from Mordecai) -> filter candidate ports.
      - Given a span text + countries -> fuzzy match over aliases.
      - Always return IMS IDs as the canonical identifier.
    """

    def __init__(self, kb_path: Path | None = None):
        kb_path = kb_path or (PORTS_KB_FINAL if PORTS_KB_FINAL.exists() else PORTS_KB_FALLBACK)
        if not kb_path.exists():
            raise FileNotFoundError(f"Ports KB not found at {kb_path}")

        self.kb_path = kb_path
        with kb_path.open("r", encoding="utf-8") as f:
            self.items: List[Dict[str, Any]] = json.load(f)

        # Indexes
        self.by_country: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.by_ims_id: Dict[str, Dict[str, Any]] = {}
        self.alias_index: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        for rec in self.items:
            country = (rec.get("country_iso2") or "").upper()
            if country:
                self.by_country[country].append(rec)

            ims_id = rec.get("ims_facility_id")
            if ims_id:
                self.by_ims_id[ims_id] = rec

            for alias in rec.get("aliases", []):
                key = norm(alias)
                if key:
                    self.alias_index[key].append(rec)

    # ------------- public helpers -------------

    def get_ports_by_countries(
        self,
        countries_iso2: List[str],
        only_seaports: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        All ports in the specified countries, optionally only seaports.
        """
        out: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for c in countries_iso2:
            c = (c or "").upper()
            for rec in self.by_country.get(c, []):
                if only_seaports and not rec.get("is_seaport", False):
                    continue
                key = rec.get("ims_facility_id") or rec.get("unlocode")
                if key and key not in seen:
                    seen.add(key)
                    out.append(rec)
        return out

    def match_span_to_ports(
        self,
        span_text: str,
        countries_iso2: Optional[List[str]] = None,
        min_score: float = 0.8,
        top_k: int = 5,
    ) -> List[Tuple[Dict[str, Any], float, str]]:
        """
        Given span_text and optional list of context countries, return list of
        (port_record, score, match_alias) sorted by descending score.

        Score is a combination of SequenceMatcher + Jaccard.
        """
        span_norm = norm(span_text)
        if not span_norm:
            return []

        # 1) Candidate set
        if countries_iso2:
            candidates = self.get_ports_by_countries(countries_iso2, only_seaports=True)
        else:
            candidates = self.items

        results: List[Tuple[Dict[str, Any], float, str]] = []

        for rec in candidates:
            best_score = 0.0
            best_alias = ""
            for alias in rec.get("aliases", []):
                s1 = seq_ratio(span_text, alias)
                s2 = jaccard_tokens(span_text, alias)
                score = 0.7 * s1 + 0.3 * s2
                if score > best_score:
                    best_score = score
                    best_alias = alias

            if best_score >= min_score:
                results.append((rec, best_score, best_alias))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def get_by_ims_id(self, ims_facility_id: str) -> Optional[Dict[str, Any]]:
        return self.by_ims_id.get(ims_facility_id)

    # ------------- high-level convenience -------------

    def resolve_port_with_context(
        self,
        span_text: str,
        context_countries: Optional[List[str]] = None,
        min_score: float = 0.8,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Thin wrapper that returns just the port dicts (plus score metadata) for convenience.
        """
        matches = self.match_span_to_ports(
            span_text=span_text,
            countries_iso2=context_countries,
            min_score=min_score,
            top_k=top_k,
        )
        out: List[Dict[str, Any]] = []
        for rec, score, alias in matches:
            rec_out = dict(rec)
            rec_out["_match_score"] = score
            rec_out["_match_alias"] = alias
            out.append(rec_out)
        return out

