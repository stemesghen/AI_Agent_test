# ims/ims_lookup.py

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from rapidfuzz import fuzz
from ims.ims_client import IMSClient
from utils_atomic import to_json_atomic  # atomic writes

CACHE_PATH = Path("data/vessels_cache.json")

def _norm_name(s: Optional[str]) -> str:
    return (s or "").strip().upper()

def _cache_key_imo(imo: str) -> str:
    return f"IMO:{str(imo).strip()}"

def _cache_key_name(name: str) -> str:
    return f"NAME:{_norm_name(name)}"

class IMSLookup:
    def __init__(self, client: IMSClient):
        self.client = client
        self.cache: Dict[str, Any] = self._load_cache()

    def _load_cache(self) -> Dict[str, Any]:
        if CACHE_PATH.exists():
            try:
                with open(CACHE_PATH, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_cache(self) -> None:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        to_json_atomic(self.cache, str(CACHE_PATH))  # <-- atomic

    def _cache_get(self, key: str) -> Optional[List[Dict[str, Any]]]:
        v = self.cache.get(key)
        return v if isinstance(v, list) else None

    def _cache_put(self, key: str, vessels: List[Dict[str, Any]]) -> None:
        if not isinstance(vessels, list):
            return
        self.cache[key] = vessels
        self._save_cache()

    # ---------------------------------------------------------------------
    # Vessel lookup: IMO → cache → API; else NAME → cache → API (fuzzy >=90)
    # entities may include:
    #   {"imo": "1234567", "vessels": ["PACIFIC GAS", ...], "flag_iso": "PA", "carrier": "CMA CGM"}
    # ---------------------------------------------------------------------
    def find_vessels(self, entities: Dict[str, Any]) -> Dict[str, Any]:
        used, query = "NONE", ""
        vessels: List[Dict[str, Any]] = []

        # 1) IMO first (exact)
        imo = (entities.get("imo") or "").strip()
        if imo:
            ck = _cache_key_imo(imo)
            cached = self._cache_get(ck)
            if cached:
                return {"used": "CACHE_IMO", "query": imo, "vessels": cached}

            try:
                resp = self.client.lookup_vessel(imo)
                data = resp.get("vessels", []) or []
                if data:
                    self._cache_put(ck, data)
                    return {"used": "IMO", "query": imo, "vessels": data}
            except Exception as e:
                print(f"[WARN] IMS IMO lookup failed for {imo}: {e}")

        # 2) Name search (optionally with filters)
        vnames = entities.get("vessels", []) or []
        flag_iso = (entities.get("flag_iso") or "").strip() or None
        carrier  = (entities.get("carrier") or "").strip()  or None

        best_name, best_score, best_rows = "", 0, []

        for vname in vnames:
            vn = vname.strip()
            if len(vn) < 3:
                continue

            ck = _cache_key_name(vn)
            cached = self._cache_get(ck)
            if cached:
                return {"used": "CACHE_NAME", "query": vn, "vessels": cached}

            try:
                resp = self.client.lookup_vessel(
                    vn,
                    contains=True,          # substring match for names (if API supports it)
                    flag_iso=flag_iso,
                    carrier_name=carrier,
                )
                rows = resp.get("vessels", []) or []
                if not rows:
                    continue

                # defensive fuzzy score: only compute if we have names
                names = [r.get("VesselName", "") for r in rows if r.get("VesselName")]
                score = max([fuzz.WRatio(vn, n) for n in names], default=0)
                if score >= 90 and score > best_score:
                    best_name, best_score, best_rows = vn, score, rows

            except Exception as e:
                print(f"[WARN] IMS vessel name lookup failed for {vn}: {e}")

        if best_rows:
            self._cache_put(_cache_key_name(best_name), best_rows)
            return {"used": "NAME", "query": best_name, "vessels": best_rows}

        return {"used": used, "query": query, "vessels": vessels}

    # ---------------------------------------------------------------------
    # Facilities: accept items with score >= threshold; dedupe best per id
    # matched_ports: [{ "facility": {...}, "score": float, "phrase": "..." }, ...]
    # ---------------------------------------------------------------------
    def find_facilities(self, matched_ports: List[Dict[str, Any]], threshold: float = 0.75) -> List[Dict[str, Any]]:
        confirmed: List[Dict[str, Any]] = []
        for m in matched_ports or []:
            try:
                if float(m.get("score", 0.0)) >= float(threshold):
                    confirmed.append(m)
            except Exception:
                continue

        best_by_id: Dict[str, Dict[str, Any]] = {}
        for m in confirmed:
            fac = m.get("facility") or {}
            fid = fac.get("ims_facility_id")
            if not fid:
                continue
            prev = best_by_id.get(fid)
            if (prev is None) or (m.get("score", 0.0) > prev.get("score", 0.0)):
                best_by_id[fid] = m

        return list(best_by_id.values())

    # ---------------------------------------------------------------------
    # Shipments: for each confirmed facility, hit shipments endpoint
    # ---------------------------------------------------------------------
    def find_shipments(self, confirmed_facilities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        shipments: List[Dict[str, Any]] = []
        for m in confirmed_facilities or []:
            fac = m.get("facility") or {}
            fid = fac.get("ims_facility_id")
            if not fid:
                continue
            try:
                rows = self.client.query_shipments_by_facility(fid)
                if isinstance(rows, list):
                    shipments.extend(rows)
            except Exception as e:
                print(f"[WARN] Failed to fetch shipments for facility {fid}: {e}")
        return shipments

