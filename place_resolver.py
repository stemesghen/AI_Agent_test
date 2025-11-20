from __future__ import annotations
import pandas as pd
import numpy as np
import json, re, math, unicodedata, os
from typing import List, Dict, Optional, Tuple

DEBUG_PLACES = os.getenv("DEBUG_PLACES", "0") == "1"


def _dbg_places(msg: str) -> None:
    if DEBUG_PLACES:
        print(f"[PLACES] {msg}")


# ----------------------------- paths -----------------------------
UNLOCODE_CSV        = "data/raw/unlocode_labeled.csv"
GEONAMES_CSV        = "data/raw/geonames_labeled.csv"
COUNTRY_CSV         = "data/country_ISO.csv"
REGIONS_JSON        = "data/region_lookup.json"
IMS_FACILITIES_JSON = "data/ports_kb_ims_only.json"  # local IMS cache


# ----------------------------- utilities -----------------------------
def norm(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def tokenize(s: str) -> set:
    s = norm(s)
    if not s:
        return set()
    return set(t for t in re.split(r"[ ,;:/()\-]+", s) if t)


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def parse_degmin(dm: str) -> Tuple[str, str]:
    if not isinstance(dm, str):
        return "", ""
    s = dm.strip().upper()
    m = re.match(r"^\s*(\d{2})(\d{2})([NS])\s+(\d{3})(\d{2})([EW])\s*$", s)
    if not m:
        return "", ""
    la, lb, lh, oa, ob, oh = m.groups()
    lat = int(la) + int(lb) / 60
    lon = int(oa) + int(ob) / 60
    if lh == "S":
        lat = -lat
    if oh == "W":
        lon = -lon
    return f"{lat:.6f}", f"{lon:.6f}"


def haversine_m(lat1, lon1, lat2, lon2) -> float:
    try:
        lat1, lon1, lat2, lon2 = map(float, (lat1, lon1, lat2, lon2))
    except Exception:
        return float("nan")
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl   = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2 * R * math.asin(math.sqrt(a))


def km(meters: float) -> float:
    return meters / 1000.0 if pd.notna(meters) else float("nan")


def read_csv_any(path: str) -> pd.DataFrame:
    for enc in ("utf-8-sig", "latin1"):
        try:
            return pd.read_csv(path, dtype=str, low_memory=False, encoding=enc).fillna("")
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, dtype=str, low_memory=False).fillna("")


def load_json_any(path: str):
    for enc in ("utf-8-sig", "utf-8", "cp1252"):
        try:
            with open(path, "r", encoding=enc) as f:
                return json.load(f)
        except UnicodeDecodeError:
            continue
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return json.loads(f.read())


# --------------------------- loaders ---------------------------------
def load_unlocode(path: str = UNLOCODE_CSV) -> pd.DataFrame:
    df = read_csv_any(path)
    cols = [c.strip() for c in df.columns]
    df.columns = cols
    official = {
        "Country", "Location", "Name", "NameWoDiacritics", "SubDiv",
        "Function", "Status", "Date", "IATA", "Coordinates", "Remarks"
    }
    if official.issubset(set(cols)):
        lat, lon = zip(*[parse_degmin(x) for x in df["Coordinates"]])
        out = pd.DataFrame({
            "unlocode":            (df["Country"].str.upper().str.strip() + df["Location"].str.upper().str.strip()),
            "country_iso2":        df["Country"].str.upper().str.strip(),
            "location_code3":      df["Location"].str.upper().str.strip(),
            "location_name":       df["Name"].str.strip(),
            "name_ascii_unlocode": df["NameWoDiacritics"].replace("", pd.NA).fillna(df["Name"]).str.strip(),
            "subdivision":         df["SubDiv"].str.strip(),
            "lat_unlocode":        list(lat),
            "lon_unlocode":        list(lon),
            "function_code":       df["Function"].astype(str).str.strip(),
        }).fillna("")
    else:
        out = pd.DataFrame({
            "unlocode":            df.get("unlocode", "").str.upper().str.strip(),
            "country_iso2":        df.get("country_iso2", "").str.upper().str.strip(),
            "location_code3":      df.get("unlocode", "").str.upper().str.strip().str[-3:],
            "location_name":       df.get("location_name", "").str.strip(),
            "name_ascii_unlocode": df.get("name_ascii", "").str.strip(),
            "subdivision":         df.get("subdivision/admin", "").str.strip(),
            "lat_unlocode":        df.get("latitude", ""),
            "lon_unlocode":        df.get("longitude", ""),
            "function_code":       df.get("function_code", "").astype(str),
        }).fillna("")
    out["name_ascii_unlocode"] = out["name_ascii_unlocode"].map(norm)
    out["name_tokens"]         = out["name_ascii_unlocode"].map(tokenize)
    return out


def load_geonames(path: str = GEONAMES_CSV) -> pd.DataFrame:
    need = [
        "geonameid", "name", "asciiname", "alternatenames",
        "latitude", "longitude", "feature_class", "feature_code",
        "country_code", "admin1_code", "admin2_code"
    ]
    df = read_csv_any(path)
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"[GeoNames] Missing columns: {miss}")
    g = df[need].copy().fillna("")
    g["name_ascii_geonames"] = g["asciiname"].replace("", pd.NA).fillna(g["name"]).map(norm)
    g["name_tokens"]         = g["name_ascii_geonames"].map(tokenize)
    g["country_code"]        = g["country_code"].str.upper()
    return g


def load_countries(path: str = COUNTRY_CSV) -> pd.DataFrame:
    df = read_csv_any(path)
    cols = {c.lower(): c for c in df.columns}
    iso2_col = cols.get("iso2") or cols.get("alpha2") or cols.get("code") or cols.get("country_iso2") or list(df.columns)[0]
    name_col = cols.get("name") or cols.get("country") or cols.get("country_name") or list(df.columns)[1]
    out = pd.DataFrame({
        "iso2": df[iso2_col].str.upper().str.strip(),
        "name_norm": df[name_col].map(norm),
    }).drop_duplicates().fillna("")
    return out


def load_regions(path: str = REGIONS_JSON) -> pd.DataFrame:
    """
    Lightweight loader used for region centroids; full region→countries mapping
    is handled separately via region_to_countries in PlaceResolver.__init__.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        data = []
    rows = []
    for it in data:
        rows.append({
            "region_name": it.get("name", ""),
            "lat": str(it.get("lat", "")),
            "lon": str(it.get("lon", "")),
            "name_norm": norm(it.get("name", "")),
        })
    return pd.DataFrame(rows).fillna("")


def load_ims_facilities(path: str = IMS_FACILITIES_JSON) -> pd.DataFrame:
    """
    Load local IMS facilities cache (ports_kb_ims_only.json or ims_facilities_lookup.json) and normalize.
    """
    if not os.path.exists(path):
        _dbg_places(f"IMS_FACILITIES_JSON not found at {path}")
        return pd.DataFrame(columns=[
            "ims_facility_id", "country_iso2", "location_code3",
            "locationName", "city", "unlocode",
            "lat_ims", "lon_ims", "aliases",
            "name_ascii_ims", "name_tokens", "alias_tokens"
        ])

    data = load_json_any(path)
    _dbg_places(f"Loaded IMS facilities JSON with type={type(data).__name__}")

    # New style: list of flat dicts
    if isinstance(data, list) and data and "attributes" not in (data[0] or {}):
        rows = []
        for item in data:
            country_iso2 = (item.get("country_iso2") or item.get("countryISOCode") or "").upper().strip()
            loc_code3    = (item.get("location_code3") or item.get("locationISOCode") or "").upper().strip()
            unlocode     = (item.get("unlocode") or "").upper().strip()
            locationName = (item.get("locationName") or "").strip()
            city         = (item.get("city") or "").strip()
            lat          = item.get("lat_ims") or item.get("latitude", "")
            lon          = item.get("lon_ims") or item.get("longitude", "")
            aliases      = item.get("aliases") or []

            if isinstance(aliases, str):
                aliases_list = [a.strip() for a in re.split(r"[|,]", aliases) if a.strip()]
            else:
                aliases_list = [str(a).strip() for a in aliases if str(a).strip()]

            name_ascii = norm(locationName)
            alias_tokens = set()
            for a in aliases_list:
                alias_tokens |= tokenize(a)

            rows.append({
                "ims_facility_id": item.get("ims_facility_id") or item.get("id", ""),
                "country_iso2": country_iso2,
                "location_code3": loc_code3,
                "locationName": locationName,
                "city": city,
                "unlocode": unlocode,
                "lat_ims": lat,
                "lon_ims": lon,
                "aliases": "|".join(aliases_list),
                "name_ascii_ims": name_ascii,
                "name_tokens": tokenize(locationName),
                "alias_tokens": alias_tokens,
            })

        _dbg_places(f"IMS facilities (flat) rows={len(rows)}")

        if not rows:
            return pd.DataFrame(columns=[
                "ims_facility_id", "country_iso2", "location_code3",
                "locationName", "city", "unlocode",
                "lat_ims", "lon_ims", "aliases",
                "name_ascii_ims", "name_tokens", "alias_tokens"
            ])
        return pd.DataFrame(rows).fillna("")

    # Old style: [{"id": "...", "attributes": {...}}]
    if not isinstance(data, list):
        data = data.get("data", []) if isinstance(data, dict) else []

    rows = []
    for item in data:
        attr = item.get("attributes", {}) or {}
        country_iso2 = (attr.get("countryISOCode") or "").upper().strip()
        loc_code3    = (attr.get("locationISOCode") or "").upper().strip()
        unlocode     = (attr.get("unlocode") or "").upper().strip()
        locationName = (attr.get("locationName") or "").strip()
        city         = (attr.get("city") or "").strip()
        lat          = attr.get("latitude", "")
        lon          = attr.get("longitude", "")
        aliases      = attr.get("aliases") or []

        if isinstance(aliases, str):
            aliases_list = [a.strip() for a in re.split(r"[|,]", aliases) if a.strip()]
        else:
            aliases_list = [str(a).strip() for a in aliases if str(a).strip()]

        name_ascii = norm(locationName)
        alias_tokens = set()
        for a in aliases_list:
            alias_tokens |= tokenize(a)

        rows.append({
            "ims_facility_id": item.get("id", ""),
            "country_iso2": country_iso2,
            "location_code3": loc_code3,
            "locationName": locationName,
            "city": city,
            "unlocode": unlocode,
            "lat_ims": lat,
            "lon_ims": lon,
            "aliases": "|".join(aliases_list),
            "name_ascii_ims": name_ascii,
            "name_tokens": tokenize(locationName),
            "alias_tokens": alias_tokens,
        })

    _dbg_places(f"IMS facilities (old style) rows={len(rows)}")

    if not rows:
        return pd.DataFrame(columns=[
            "ims_facility_id", "country_iso2", "location_code3",
            "locationName", "city", "unlocode",
            "lat_ims", "lon_ims", "aliases",
            "name_ascii_ims", "name_tokens", "alias_tokens"
        ])

    df = pd.DataFrame(rows).fillna("")
    return df


# --------------------------- helpers ---------------------------------
class CountryIndex:
    def __init__(self, df: pd.DataFrame):
        self.by_iso2 = {str(r["iso2"]).upper(): r for _, r in df.iterrows()}
        self.name_to_iso2 = {}
        for _, r in df.iterrows():
            iso2 = str(r["iso2"]).upper()
            nm   = str(r["name_norm"])
            if nm:
                self.name_to_iso2[nm] = iso2

    def to_iso2(self, name_or_code: str) -> str:
        if not isinstance(name_or_code, str):
            return ""
        s = name_or_code.strip()
        if len(s) == 2 and s.isalpha():
            return s.upper()
        return self.name_to_iso2.get(norm(s), "")


def name_score(left_tokens: set, right_tokens: set, left_str: str, right_str: str) -> float:
    s = jaccard(left_tokens, right_tokens)
    if left_str == right_str:
        s += 0.40
    elif left_str.startswith(right_str) or right_str.startswith(left_str):
        s += 0.20
    elif left_str in right_str or right_str in left_str:
        s += 0.10
    return float(min(s, 1.0))


# --------- span plausibility helper (strict-ish) ---------
_BAD_SPAN_SINGLE = {
    "the", "since", "repairs", "tugs", "salvage",
    "news", "north", "south", "east", "west"
}


def _is_plausible_span(s: str) -> bool:
    """
    Heuristic filter for place/port spans to kill obvious junk like 'The'.
    """
    if not isinstance(s, str):
        return False
    t = s.strip()
    if not t:
        return False
    if len(t) < 3 or len(t) > 80:
        return False
    # basic word constraints
    words = t.split()
    if len(words) > 5:
        return False
    if t.lower() in _BAD_SPAN_SINGLE:
        return False
    # must contain at least one letter
    if not any(ch.isalpha() for ch in t):
        return False
    # don't allow all-caps single general words
    if len(words) == 1 and words[0].isupper() and words[0].lower() in _BAD_SPAN_SINGLE:
        return False
    return True


# --------------------------- resolver --------------------------------
class PlaceResolver:
    """
    Unified place resolver with mixed-source candidates.

    Option B behavior (for this project):

      • build_candidates_for_article(ports_text, country_iso2s, region_names, top_k_span)
        - For each country ISO2: IMS → UNLOCODE (ports only).
        - For each region name: region → countries → same as above.
        - For each explicit port span: span-based match (IMS/UNLOCODE/GeoNames).
        - NO "guessing" a single port when there are no port spans; that logic
          is enforced in run.py (no port_span chosen when ports_text == []).
    """

    def __init__(
        self,
        unlocode_csv: str = UNLOCODE_CSV,
        geonames_csv: str = GEONAMES_CSV,
        country_csv: str = COUNTRY_CSV,
        regions_json: str = REGIONS_JSON,
    ):
        self.U = load_unlocode(unlocode_csv)
        self.G = load_geonames(geonames_csv)
        self.C = CountryIndex(load_countries(country_csv))
        self.R = load_regions(regions_json)
        self.I = load_ims_facilities(IMS_FACILITIES_JSON)

        _dbg_places(
            f"init: UNLOCODE rows={len(self.U)}, GEONAMES rows={len(self.G)}, "
            f"IMS rows={len(self.I)}, REGIONS rows={len(self.R)}"
        )

        # Filter UNLOCODE to port-like entries: Function contains "1" (sea port)
        if "function_code" in self.U.columns:
            port_mask = self.U["function_code"].astype(str).str.contains("1", na=False)
            self.U_ports = self.U[port_mask].reset_index(drop=True)
        else:
            self.U_ports = self.U.copy().reset_index(drop=True)

        # per-country slices
        self.G_by_ctry = {k: v.reset_index(drop=True) for k, v in self.G.groupby("country_code")}
        self.U_by_ctry = {k: v.reset_index(drop=True) for k, v in self.U_ports.groupby("country_iso2")}
        if not self.I.empty and "country_iso2" in self.I.columns:
            self.I_by_ctry = {k: v.reset_index(drop=True) for k, v in self.I.groupby("country_iso2")}
        else:
            self.I_by_ctry = {}

        # Known place names for optional filtering
        known = set()
        if "name_ascii_unlocode" in self.U.columns:
            known.update(self.U["name_ascii_unlocode"].dropna().tolist())
        if "name_ascii_geonames" in self.G.columns:
            known.update(self.G["name_ascii_geonames"].dropna().tolist())
        if not self.I.empty and "name_ascii_ims" in self.I.columns:
            known.update(self.I["name_ascii_ims"].dropna().tolist())
        self.known_place_names = {norm(x) for x in known if isinstance(x, str) and x.strip()}
        _dbg_places(f"known_place_names size={len(self.known_place_names)}")

        # region → countries mapping (for Option B: region-based country hints)
        self.region_to_countries: Dict[str, List[str]] = {}
        try:
            reg_raw = load_json_any(regions_json)
            if isinstance(reg_raw, list):
                for entry in reg_raw:
                    nm = norm(entry.get("name", ""))
                    if not nm:
                        continue
                    countries = entry.get("countries", []) or []
                    iso_list = []
                    for c in countries:
                        if not isinstance(c, str):
                            continue
                        iso_list.append(c.upper())
                    if iso_list:
                        self.region_to_countries[nm] = iso_list
            _dbg_places(f"region_to_countries keys={len(self.region_to_countries)}")
        except Exception:
            self.region_to_countries = {}

    # ---------- small utilities ----------
    def nearest_ports(self, lat: float, lon: float, radius_km: float = 80, top_k: int = 5) -> pd.DataFrame:
        U = self.U_ports.copy()
        U["dist_m"] = [haversine_m(lat, lon, a, b) for a, b in zip(U["lat_unlocode"], U["lon_unlocode"])]
        U = U[U["dist_m"].notna()]
        U["dist_km"] = U["dist_m"].map(km)
        in_radius = U[U["dist_km"] <= radius_km] if radius_km > 0 else U
        return in_radius.sort_values("dist_km").head(top_k)

    def _extract_place_candidates(self, text: str) -> List[str]:
        """
        Extract explicit port/city-like spans from text, using simple patterns
        plus a strict plausibility filter.
        """
        cands: List[str] = []
        t = text or ""

        # Port of X
        for m in re.finditer(r"\b(?:Port|Harbo[u]?r)\s+of\s+([A-Z][A-Za-z0-9 .,'’\-]{2,})\b", t, flags=re.I):
            cands.append(m.group(1).strip(" ,.;:"))

        # X Port / X Terminal / etc
        for m in re.finditer(
            r"\b([A-Z][A-Za-z0-9 .,'’\-]{2,})\s+(?:Port|Harbo[u]?r|Terminal|Anchorage|Jetty|Quay|Dock)\b",
            t,
            flags=re.I,
        ):
            cands.append(m.group(1).strip(" ,.;:"))

        # Generic capitalized spans (potential cities / ports)
        caps = re.findall(r"\b([A-Z][A-Za-z'\-]*(?:\s+[A-Z][A-Za-z'\-]*){0,3})\b", t)
        cands.extend(caps)

        seen = set()
        out = []
        for c in cands:
            n = norm(c)
            if not n or n in seen:
                continue
            if not _is_plausible_span(c):
                continue
            seen.add(n)
            out.append(c)

        _dbg_places(f"_extract_place_candidates found={len(out)} names={out[:8]}")
        return out[:12]

    def _emit_ims(self, r: pd.Series, score: float) -> Dict:
        return {
            "source": "IMS",
            "score": float(score),
            "distance_m": "",
            "countryISOCode": r["country_iso2"],
            "locationISOCode": r["location_code3"],
            "locationName": r["locationName"],
            "unlocode": r["unlocode"],
            "name_ascii_unlocode": "",
            "subdivision": r["city"],
            "lat_ims": r["lat_ims"],
            "lon_ims": r["lon_ims"],
            "lat_unlocode": "",
            "lon_unlocode": "",
            "lat_geonames": "",
            "lon_geonames": "",
            "geonameid": "",
            "alternatenames": r["aliases"],
            "ims_facility_id": r["ims_facility_id"],
            "row_kind": "ims_only",
        }

    def _emit_un(self, r: pd.Series, score: float) -> Dict:
        return {
            "source": "UNLOCODE",
            "score": float(score),
            "distance_m": "",
            "countryISOCode": r["country_iso2"],
            "locationISOCode": r["location_code3"],
            "locationName": r["location_name"],
            "unlocode": r["unlocode"],
            "name_ascii_unlocode": r["name_ascii_unlocode"],
            "subdivision": r["subdivision"],
            "lat_ims": "",
            "lon_ims": "",
            "lat_unlocode": r["lat_unlocode"],
            "lon_unlocode": r["lon_unlocode"],
            "lat_geonames": "",
            "lon_geonames": "",
            "geonameid": "",
            "alternatenames": "",
            "ims_facility_id": "",
            "row_kind": "unlocode_only",
        }

    def _emit_gn(self, r: pd.Series, score: float) -> Dict:
        return {
            "source": "GEONAMES",
            "score": float(score),
            "distance_m": "",
            "countryISOCode": r["country_code"],
            "locationISOCode": "",
            "locationName": r["name"],
            "unlocode": "",
            "name_ascii_unlocode": "",
            "subdivision": r["admin1_code"],
            "lat_ims": "",
            "lon_ims": "",
            "lat_unlocode": "",
            "lon_unlocode": "",
            "lat_geonames": r["latitude"],
            "lon_geonames": r["longitude"],
            "geonameid": r["geonameid"],
            "alternatenames": r["alternatenames"],
            "ims_facility_id": "",
            "row_kind": "geonames_only",
        }

    # ---------- geo-driven helper methods ----------
    def candidates_for_country(self, iso2: str) -> List[Dict]:
        """
        IMS → UNLOCODE (ports-only) for a single country ISO2.

        NO GeoNames here: we rely on UNLOCODE + IMS for ports, to avoid
        millions of city candidates.
        """
        iso2 = (iso2 or "").upper()
        if not iso2:
            return []
        out: List[Dict] = []

        I = self.I_by_ctry.get(iso2, pd.DataFrame())
        U = self.U_by_ctry.get(iso2, pd.DataFrame())

        _dbg_places(
            f"candidates_for_country({iso2}) I={len(I)} U={len(U)}"
        )

        if not I.empty:
            for _, r in I.iterrows():
                out.append(self._emit_ims(r, score=0.7))

        if not U.empty:
            for _, r in U.iterrows():
                out.append(self._emit_un(r, score=0.5))

        return out

    def candidates_for_region(self, region_name: str) -> List[Dict]:
        """
        Use region_lookup to map region → countries, then reuse candidates_for_country.
        """
        key = norm(region_name)
        iso_list = self.region_to_countries.get(key, [])
        _dbg_places(f"candidates_for_region({region_name!r}) → iso_list={iso_list}")
        out: List[Dict] = []
        for iso in iso_list:
            out.extend(self.candidates_for_country(iso))
        return out

    def candidates_for_span(self, span: str, country_hint: Optional[str] = None, top_k: int = 8) -> List[Dict]:
        """
        Span-based matching across IMS / UNLOCODE / GeoNames.
        """
        span = (span or "").strip()
        if not span:
            return []
        return self.resolve(span, country_hint=country_hint, top_k=top_k)

    # ---------- main span-based resolve ----------
    def resolve(self, text: str, country_hint: Optional[str] = None, top_k: int = 3) -> List[Dict]:
        """
        Mixed-source candidate builder, keeping CSV/JSON sources separate.
        """
        text = text or ""
        iso2 = self.C.to_iso2(country_hint) if country_hint else ""
        _dbg_places(
            f"resolve country_hint={country_hint!r} → iso2={iso2} text[:120]={text[:120]!r}"
        )

        Uc = self.U_by_ctry.get(iso2, self.U_ports) if iso2 else self.U_ports
        Gc = self.G_by_ctry.get(iso2, self.G) if iso2 else self.G
        Ic = self.I_by_ctry.get(iso2, self.I) if iso2 and self.I_by_ctry else self.I

        names = self._extract_place_candidates(text)
        _dbg_places(f"resolve extracted names={names}")

        if not names:
            _dbg_places("resolve: no explicit place spans → []")
            return []

        out: List[Dict] = []

        # 1) IMS candidates
        ims_hits: List[Tuple[float, int]] = []
        if not Ic.empty:
            for nm in names:
                nm_norm = norm(nm)
                nm_toks = tokenize(nm)
                for j, row_t in enumerate(Ic.itertuples(index=False), start=0):
                    row = row_t._asdict()
                    s_name = name_score(nm_toks, row["name_tokens"], nm_norm, row["name_ascii_ims"])
                    s_alias = 0.0
                    if isinstance(row["alias_tokens"], set):
                        s_alias = jaccard(nm_toks, row["alias_tokens"])
                    s = max(s_name, s_alias)
                    ims_hits.append((s, j))

            ims_hits.sort(key=lambda x: x[0], reverse=True)
            seen_idx = set()
            ims_best = []
            for s, j in ims_hits:
                if j in seen_idx:
                    continue
                seen_idx.add(j)
                ims_best.append((s, j))
                if len(ims_best) >= max(top_k, 5):
                    break

            _dbg_places(f"resolve IMS candidate count={len(ims_best)}")

            for s, j in ims_best:
                if s <= 0.0:
                    continue
                r = Ic.loc[j]
                out.append(self._emit_ims(r, score=round(float(s), 3)))

        # 2) UNLOCODE candidates
        un_hits: List[Tuple[float, int]] = []
        for nm in names:
            nm_norm = norm(nm)
            nm_toks = tokenize(nm)
            for j, row_t in enumerate(Uc.itertuples(index=False), start=0):
                row = row_t._asdict()
                s = name_score(nm_toks, row["name_tokens"], nm_norm, row["name_ascii_unlocode"])
                un_hits.append((s, j))

        un_hits.sort(key=lambda x: x[0], reverse=True)
        un_best, seen_idx = [], set()
        for s, j in un_hits:
            if j in seen_idx:
                continue
            seen_idx.add(j)
            un_best.append((s, j))
            if len(un_best) >= max(top_k, 5):
                break

        _dbg_places(f"resolve UNLOCODE candidate count={len(un_best)}")

        for s, j in un_best:
            if s <= 0.0:
                continue
            r = Uc.loc[j]
            out.append(self._emit_un(r, score=round(float(s), 3)))

        # 3) GeoNames candidates (limited count; only for span-based)
        gn_hits: List[Tuple[float, int]] = []
        for nm in names:
            nm_norm = norm(nm)
            nm_toks = tokenize(nm)
            for j, row_t in enumerate(Gc.itertuples(index=False), start=0):
                row = row_t._asdict()
                s = name_score(nm_toks, row["name_tokens"], nm_norm, row["name_ascii_geonames"])
                if row["feature_class"] == "H" and str(row["feature_code"]).upper().startswith("H."):
                    s += 0.05
                gn_hits.append((s, j))

        gn_hits.sort(key=lambda x: x[0], reverse=True)
        gn_best, seen_idx = [], set()
        for s, j in gn_hits:
            if j in seen_idx:
                continue
            seen_idx.add(j)
            gn_best.append((s, j))
            if len(gn_best) >= max(top_k, 5):
                break

        _dbg_places(f"resolve GEONAMES candidate count={len(gn_best)}")

        for s, j in gn_best:
            if s <= 0.0:
                continue
            r = Gc.loc[j]
            out.append(self._emit_gn(r, score=round(float(s), 3)))

        if not out:
            _dbg_places("resolve: out is empty after all sources")
            return []

        # Deduplicate and sort
        def sort_key(c: Dict) -> Tuple[int, float]:
            src = c.get("source", "")
            if src == "IMS":
                prio = 0
            elif src == "UNLOCODE":
                prio = 1
            else:
                prio = 2
            return (prio, -float(c.get("score") or 0.0))

        seen_keys = set()
        deduped: List[Dict] = []
        for c in sorted(out, key=sort_key):
            key = (
                c.get("source", ""),
                c.get("ims_facility_id", ""),
                c.get("unlocode", ""),
                c.get("geonameid", ""),
                c.get("locationName", ""),
            )
            if key in seen_keys:
                continue
            seen_keys.add(key)
            deduped.append(c)
            if len(deduped) >= top_k:
                break

        _dbg_places(f"resolve: returning {len(deduped)} candidates (top_k={top_k})")
        if DEBUG_PLACES:
            preview = [
                {
                    "source": c.get("source"),
                    "countryISOCode": c.get("countryISOCode"),
                    "locationISOCode": c.get("locationISOCode"),
                    "locationName": c.get("locationName"),
                    "unlocode": c.get("unlocode"),
                    "score": c.get("score"),
                }
                for c in deduped
            ]
            _dbg_places(f"resolve preview={preview}")

        return deduped

    # ---------- NEW: geo-driven candidate builder ----------
    def build_candidates_for_article(
        self,
        ports_text: List[str],
        country_iso2s: Optional[List[str]] = None,
        region_names: Optional[List[str]] = None,
        top_k_span: int = 8,
    ) -> List[Dict]:
        """
        Option B candidate builder:

        - For each country ISO2: IMS → UNLOCODE ports.
        - For each region name: region → countries → same as above.
        - For each explicit port span: span-based match (IMS/UNLOCODE/GeoNames).
        - We DO NOT decide whether to pick a final port here; run.py handles that.
        """
        country_iso2s = country_iso2s or []
        region_names = region_names or []
        ports_text = ports_text or []

        # dedupe spans a bit
        seen_span_norm = set()
        valid_spans: List[str] = []
        for p in ports_text:
            n = norm(p)
            if not n or n in seen_span_norm:
                continue
            seen_span_norm.add(n)
            valid_spans.append(p)

        _dbg_places(
            f"build_candidates_for_article country_iso2s={country_iso2s} "
            f"region_names={region_names} ports_text={ports_text} "
            f"valid_spans={valid_spans} top_k_span={top_k_span}"
        )

        candidates: List[Dict] = []
        seen = set()

        def add_cand(c: Dict):
            key = (
                c.get("source", ""),
                c.get("countryISOCode", ""),
                c.get("locationISOCode", ""),
                c.get("unlocode", ""),
                c.get("geonameid", ""),
                c.get("locationName", ""),
            )
            if key in seen:
                return
            seen.add(key)
            candidates.append(c)

        # 1) by country (from iso2 hints)
        for iso in country_iso2s:
            for c in self.candidates_for_country(iso):
                add_cand(c)

        # 2) by region name (map region→countries, then reuse candidates_for_country)
        for r in region_names:
            for c in self.candidates_for_region(r):
                add_cand(c)

        # 3) explicit port spans from NER (if any)
        for span in valid_spans:
            for c in self.candidates_for_span(span, country_hint=None, top_k=top_k_span):
                add_cand(c)

        _dbg_places(f"build_candidates_for_article returning {len(candidates)} candidates")
        if DEBUG_PLACES:
            preview = [
                {
                    "source": c.get("source"),
                    "countryISOCode": c.get("countryISOCode"),
                    "locationISOCode": c.get("locationISOCode"),
                    "locationName": c.get("locationName"),
                    "unlocode": c.get("unlocode"),
                    "score": c.get("score"),
                }
                for c in candidates[:15]
            ]
            _dbg_places(f"build_candidates_for_article preview={preview}")

        return candidates

