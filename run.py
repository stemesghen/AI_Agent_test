from __future__ import annotations
import os, json, glob, re, csv, sys, math
from pathlib import Path
from typing import Dict, Any, List, Tuple, Set, Optional
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from ftfy import fix_text

# NEW: similarity + normalization helpers
from difflib import SequenceMatcher
import unicodedata
from collections import defaultdict

# --- local imports ---
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import safe_fname
from utils_atomic import to_csv_atomic
from ims.ims_client import IMSClient
from ims.facility_resolver import resolve_facility_id, list_facilities_by_country
from ims.ais_client import AISClient
from place_resolver import PlaceResolver
from disambiguation.features import assemble_feature_vector
from disambiguation.model import PortDisambModel, vectorize
from nlp.ner_model import EntityExtractor
from nlp.country_codes import load_country_codes, map_names_to_iso2
from nlp.port_index import load_region_centroids
from tools.mordecai_context import load_mordecai_hits, build_doc_context

# ---------------- paths ----------------
IN_DIR = os.getenv("EXTRACT_IN_DIR", "data/is_incident")
NORM_DIR = "data/normalized"
OUT_DIR = "data/extracted"
AIS_DIR = Path("data/ais")

UNLOCODE_CSV = Path("data/raw/unlocode_labeled.csv")
GEONAMES_CSV = Path("data/raw/geonames_labeled.csv")
REGIONS_JSON = Path("data/region_lookup.json")
COUNTRY_CSV = Path("data/country_ISO.csv")
COUNTRY_ALIASES_CSV = Path("data/country_aliases.csv")

SUMMARY_CSV = Path(OUT_DIR) / "summary_out.csv"

# NEW: training candidate dump
TRAIN_CANDIDATES_CSV = Path(OUT_DIR) / "port_disamb_candidates.csv"
DUMP_TRAINING_CANDIDATES = bool(int(os.getenv("DUMP_TRAINING_CANDIDATES", "0")))

CONF_THRESHOLD = float(os.getenv("PORT_CONF_THRESHOLD", "0.72"))
TOP_K_CANDS = int(os.getenv("TOP_K_PORT_CANDIDATES", "8"))
TOP_K_OFFLINE = int(os.getenv("TOP_K_OFFLINE", "1600"))

IMS_SUGGEST_LIMIT = int(os.getenv("IMS_SUGGEST_LIMIT", "50"))
LIVE_WRITE_EVERY = int(os.getenv("LIVE_WRITE_EVERY", "2"))
IMS_ALWAYS_QUERY_TOP = int(os.getenv("IMS_ALWAYS_QUERY_TOP", "1"))
AIS_PROXIMITY_BONUS_KM = float(os.getenv("AIS_PROXIMITY_BONUS_KM", "75"))
AIS_SEARCH_RADIUS_KM = float(os.getenv("AIS_SEARCH_RADIUS_KM", "100"))
AIS_LOOKBACK_HOURS = int(os.getenv("AIS_LOOKBACK_HOURS", "72"))

SKIP_AIS = True  # hard-disable AIS for now
DEBUG_NER = os.getenv("DEBUG_NER", "0") == "1"
DEBUG_PIPE = os.getenv("DEBUG_PIPE", "0") == "1"

# NEW: Mordecai port matches integration
MORD_PORT_MATCHES_CSV = Path(OUT_DIR) / "mordecai_ports_matches.csv"
MORDECAI_MIN_SCORE = float(os.getenv("MORDECAI_MIN_SCORE", "0.45"))
MORDECAI_MAX_PORTS_PER_DOC = int(os.getenv("MORDECAI_MAX_PORTS_PER_DOC", "8"))


def _dbg_pipe(msg: str) -> None:
    if DEBUG_PIPE:
        print(f"[PIPE] {msg}")


# ---------------- encoding-safe I/O ----------------
def load_json_any(path: Path) -> Any:
    for enc in ("utf-8-sig", "utf-8", "cp1252"):
        try:
            with open(path, "r", encoding=enc) as f:
                return json.load(f)
        except UnicodeDecodeError:
            continue
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return json.loads(f.read())


# ---------------- cleaning helpers ----------------
INLINE_NOISE = [
    (re.compile(r"<[^>]+>"), " "),
    (re.compile(r"https?://\S+"), " "),
    (re.compile(r"\s+•\s+"), " "),
    (re.compile(r"\s{2,}"), " "),
]


def _looks_like_masthead_line(s: str) -> bool:
    s = s.strip()
    if not s or len(s) > 40:
        return False
    if any(ch in s for ch in ".:;!?@#$/\\"):
        return False
    toks = s.split()
    if not (1 <= len(toks) <= 3):
        return False

    def ok(w: str) -> bool:
        return (w.isupper() and w.isalpha()) or (w[:1].isupper() and w[1:].islower())

    return all(ok(w) for w in toks)


def _looks_like_boilerplate_line(s: str) -> bool:
    t = s.strip().lower()
    if not t or len(t) <= 2:
        return True
    if re.match(r"^(share|subscribe|login|log in|sign in|sign up|print|email|comments?\b)", t):
        return True
    if re.match(r"^(total\s+views\s*:\s*\d+|views\s*:\s*\d+|\d+\s+views)$", t):
        return True
    if re.match(r"^(photo|image|video|source|credit)\s*[:\-–]", t):
        return True
    if re.match(r"^(privacy|cookie|cookies|terms|consent)\b", t):
        return True
    if re.match(r"^by\s+[A-Z][A-Za-z.'\-]+(?:\s+[A-Z][A-Za-z.'\-]+){0,2}\s*$", s):
        return True
    return False


def _noise_ratio(s: str) -> float:
    if not s:
        return 1.0
    letters = sum(ch.isalpha() for ch in s)
    return 1.0 - (letters / max(1, len(s)))


def _strip_inline_wire_tags(text: str) -> str:
    text = re.sub(r"\(\s*[A-Z][A-Za-z .&]{2,30}\s*\)\s*[—\-–]\s*", " ", text)
    text = re.sub(
        r"(?im)^\s*by\s+[^,\n]{0,120}?\(\s*[A-Z][A-Za-z .&]{2,30}\s*\)\s*[—\-–]?\s*",
        "",
        text,
    )
    return text


def clean_article_text(title: str, body: str) -> str:
    try:
        if isinstance(body, str) and body.lstrip().startswith("{") and '"content_text"' in body:
            j = json.loads(body)
            body = j.get("content_text", body)
    except Exception:
        pass
    raw = f"{title or ''}\n{body or ''}"
    for rx, repl in INLINE_NOISE:
        raw = rx.sub(repl, raw)
    raw = _strip_inline_wire_tags(raw)
    lines_in = [ln.strip() for ln in raw.splitlines()]
    lines_out = []
    for s in lines_in:
        if not s:
            continue
        if _looks_like_masthead_line(s):
            continue
        if _looks_like_boilerplate_line(s):
            continue
        if len(s) <= 2:
            continue
        if _noise_ratio(s) > 0.65:
            continue
        lines_out.append(s)
    return re.sub(r"\s{2,}", " ", "\n".join(lines_out)).strip()


# ---------------- NEW: name similarity + pruning helpers ----------------
def _norm_simple(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    return " ".join(s.lower().strip().split())


def _token_set(s: str) -> set[str]:
    return {t for t in _norm_simple(s).split() if t}


def _name_similarity(a: str, b: str) -> float:
    a_n = _norm_simple(a)
    b_n = _norm_simple(b)
    if not a_n or not b_n:
        return 0.0

    char_sim = SequenceMatcher(None, a_n, b_n).ratio()
    ta, tb = _token_set(a_n), _token_set(b_n)
    if not ta or not tb:
        jacc = 0.0
    else:
        jacc = len(ta & tb) / len(ta | tb)

    return max(char_sim, jacc)


def prune_candidates(
    pr_hits: List[Dict[str, Any]],
    ports_text: List[str],
    name_hints: List[str],
    max_total: int = 5000,
    max_per_source: Dict[str, int] | None = None,
) -> List[Dict[str, Any]]:
    """
    Aggressively prune PlaceResolver candidates using the actual port text spans.
    """
    if not pr_hits:
        return []

    if max_per_source is None:
        max_per_source = {
            "IMS": 800,
            "UNLOCODE": 800,
            "GEONAMES": 200,  # keep small; GeoNames is noisy
        }

    ports_text = ports_text or []
    has_ports = bool(ports_text)
    name_hints = [iso.upper() for iso in (name_hints or [])]

    # If we have no explicit ports, we still allow candidates (Option B),
    # but we will NEVER choose a final port_span later. Here we just
    # downsample a bit, keeping IMS + UNLOCODE.
    if not has_ports:
        ims = [h for h in pr_hits if (h.get("source") or "").upper() == "IMS"]
        un = [h for h in pr_hits if (h.get("source") or "").upper() == "UNLOCODE"]
        ims.sort(key=lambda c: c.get("score", 0.0), reverse=True)
        un.sort(key=lambda c: c.get("score", 0.0), reverse=True)
        out = ims[: max_per_source.get("IMS", max_total)] + un[: max_per_source.get("UNLOCODE", max_total)]
        return out[:max_total]

    # With explicit ports, use name similarity.
    sims_cache: Dict[str, float] = {}

    def best_sim_to_ports(cand_name: str) -> float:
        cand_name = cand_name or ""
        if not cand_name:
            return 0.0
        if cand_name in sims_cache:
            return sims_cache[cand_name]
        best = 0.0
        for p in ports_text:
            s = _name_similarity(cand_name, p)
            if s > best:
                best = s
        sims_cache[cand_name] = best
        return best

    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for h in pr_hits:
        src = (h.get("source") or "").upper()
        name = h.get("locationName") or h.get("unlocode") or ""
        sim = best_sim_to_ports(name)
        h["_name_sim"] = sim

        if src == "IMS":
            if sim < 0.10:
                continue
        elif src == "UNLOCODE":
            if sim < 0.20:
                continue
        elif src == "GEONAMES":
            if sim < 0.35:
                continue

        buckets[src].append(h)

    out: List[Dict[str, Any]] = []
    for src, cand_list in buckets.items():
        limit = max_per_source.get(src, max_total)
        cand_list.sort(
            key=lambda c: (c.get("_name_sim", 0.0), c.get("score", 0.0)),
            reverse=True,
        )
        out.extend(cand_list[:limit])

    if len(out) > max_total:
        out.sort(
            key=lambda c: (c.get("_name_sim", 0.0), c.get("score", 0.0)),
            reverse=True,
        )
        out = out[:max_total]

    for c in out:
        c.pop("_name_sim", None)
    return out


# ---------------- scoring helpers ----------------
def portish_context_bonus(text: str) -> float:
    _PORTY_RX = re.compile(r"\b(port|harbo[u]?r|terminal|jetty|anchorage|berth|dock|pier|quay)\b", re.I)
    bonus = 0.0
    if _PORTY_RX.search(text):
        bonus += 0.06
    if any(p in text.lower() for p in ["ust-luga", "primorsk", "kaliningrad"]):
        bonus += 0.04
    return min(bonus, 0.10)


def _haversine_km(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    R = 6371.0
    la1, lo1 = math.radians(a[0]), math.radians(a[1])
    la2, lo2 = math.radians(b[0]), math.radians(b[1])
    dlat = la2 - la1
    dlon = lo2 - lo1
    h = (math.sin(dlat / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin(dlon / 2) ** 2)
    return 2 * R * math.asin(min(1.0, math.sqrt(h)))


def ais_bonus_km(candidate: Dict[str, Any], ais_points: List[Tuple[float, float]], within_km: float) -> float:
    try:
        lat = (
            candidate.get("lat")
            or candidate.get("latitude")
            or candidate.get("attributes", {}).get("latitude")
            or candidate.get("lat_unlocode")
            or candidate.get("lat_geonames")
        )
        lon = (
            candidate.get("lon")
            or candidate.get("longitude")
            or candidate.get("attributes", {}).get("longitude")
            or candidate.get("lon_unlocode")
            or candidate.get("lon_geonames")
        )
        if lat is None or lon is None or not ais_points:
            return 0.0
        lat, lon = float(lat), float(lon)
        best = min((_haversine_km((lat, lon), pt) for pt in ais_points), default=1e9)
        if best <= within_km:
            return 0.08
        if best <= within_km * 2:
            return 0.04
        return 0.0
    except Exception:
        return 0.0


# ---------------- region → countries helper ----------------
def region_to_iso2_candidates(regions_text: List[str], region_centroids: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    for r in regions_text or []:
        entry = region_centroids.get(r) or {}
        for iso2 in entry.get("countries", []):
            if iso2 and iso2 not in seen:
                out.append(iso2)
                seen.add(iso2)
    return out


# region centroid from region names
def infer_region_centroid(regions_text: List[str], region_centroids: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    coords: List[Tuple[float, float]] = []
    for r in regions_text or []:
        entry = region_centroids.get(r) or {}
        lat = entry.get("lat")
        lon = entry.get("lon")
        if lat is not None and lon is not None:
            coords.append((float(lat), float(lon)))
    if not coords:
        return None
    la = sum(c[0] for c in coords) / len(coords)
    lo = sum(c[1] for c in coords) / len(coords)
    return (la, lo)


# ---------------- country ISO2 cleaning ----------------
def clean_iso2_list(raw_iso2: List[str]) -> List[str]:
    iso2 = set(raw_iso2 or [])
    junk = {
        "IO",  # Indian Ocean (we don't treat as a "country")
        "CF",
        "CD",
    }
    iso2 -= junk
    return sorted(iso2)


# ---------------- IMS helper ----------------
def ims_try_for_top_candidate(
    ims_client: IMSClient,
    ports_text: List[str],
    top_cand: Dict[str, Any] | None,
    top_score: float,
    conf_threshold: float,
    own_iso: str | None,
    search_iso2s: List[str],
) -> Tuple[Optional[dict], str, str]:
    """
    Decide whether to call IMS, and if so, how.
    Returns (facility_obj_or_none, ims_status, ims_reason)

    IMPORTANT: If there are NO explicit ports_text, we never call IMS to
    "guess" a port. We only suggest facilities via list_facilities_by_country.
    """
    if not ports_text or top_cand is None or top_score < conf_threshold:
        return None, "skipped", "no_explicit_port_or_low_conf"

    def _one(country_iso2: str) -> Tuple[Optional[dict], str, str]:
        print(
            f"[IMS][call] name={top_cand.get('locationName','')} "
            f"| city={top_cand.get('subdivision','')} "
            f"| country={country_iso2} | unlocode={top_cand.get('unlocode')}"
        )
        fac, why = resolve_facility_id(
            ims_client,
            country_iso2=country_iso2 or top_cand.get("countryISOCode", ""),
            location_name=top_cand.get("locationName", ""),
            unlocode=top_cand.get("unlocode") or None,
            city=top_cand.get("subdivision") or None,
        )
        return fac, ("hit" if fac else "miss"), why

    # 1) Try own_iso (from candidate itself)
    if own_iso:
        fac, st, why = _one(own_iso)
        if fac:
            return fac, st, why

    return None, "miss", "no_match"


# ---------------- strict port span cleanup ----------------
_BAD_PORT_SPANS = {
    "port",
    "ports",
    "harbour",
    "harbor",
    "since",
    "tugs",
    "salvage",
}


def _clean_single_port_span(s: str) -> str:
    if not isinstance(s, str):
        return ""
    t = s.strip()
    if not t:
        return ""
    # strip trailing punctuation
    t = re.sub(r"[\s,;:]+$", "", t)
    return t


def filter_port_spans(spans: List[str]) -> List[str]:
    """
    Clean and filter NER 'ports' spans so junk like
    'since the 1990s. The l' does not survive.
    """
    out: List[str] = []
    seen: Set[str] = set()
    for raw in spans or []:
        t = _clean_single_port_span(raw)
        if not t:
            continue
        low = t.lower()
        if low in _BAD_PORT_SPANS:
            continue
        # length constraints
        if len(t) < 3 or len(t) > 80:
            continue
        words = t.split()
        if len(words) > 6:
            continue
        # must contain at least one letter, and start with a capital
        if not any(ch.isalpha() for ch in t):
            continue
        if not words[0][0].isupper():
            continue
        # don't keep obvious fragments
        if words[0].lower() in {"since", "the", "in"} and len(words) <= 2:
            continue

        if t not in seen:
            seen.add(t)
            out.append(t)

    return out


# unify lat/lon on candidates when possible
def cand_latlon(c: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    for la_key, lo_key in [
        ("lat", "lon"),
        ("lat_ims", "lon_ims"),
        ("lat_unlocode", "lon_unlocode"),
        ("lat_geonames", "lon_geonames"),
        ("latitude", "longitude"),
    ]:
        la, lo = c.get(la_key), c.get(lo_key)
        try:
            if la is not None and lo is not None:
                return float(la), float(lo)
        except Exception:
            continue
    return None, None


# ---------------- main runner ----------------
def run():
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    print("[RUN] extract.run v3.6 — NER + PortDisamb + Mordecai matches + unified ports_kb_final")

    ims_base = (os.getenv("IMS_BASE_URL") or "").rstrip("/")
    ims_token = os.getenv("IMS_TOKEN", "")
    ims_client = IMSClient(ims_base, token=ims_token)
    IMS_ENABLED = bool(ims_token)
    if not IMS_ENABLED:
        print("[IMS] disabled: missing IMS_TOKEN")

    # NEW unified ports KB resolver (ports_kb_final.json)
    place_resolver = PlaceResolver()

    region_centroids = load_region_centroids(REGIONS_JSON)
    extractor = EntityExtractor()

    # country_ISO + country_aliases (demonyms etc.)
    name2iso, iso2name, alias2iso = load_country_codes(
        COUNTRY_CSV,
        COUNTRY_ALIASES_CSV,
    )
    disamb = PortDisambModel("data/models/port_disamb.joblib")

    # Mordecai doc-level context (countries/admin1 per doc_id)
    try:
        mordecai_df = load_mordecai_hits()
        doc_ctx = build_doc_context(mordecai_df)
        print(f"[MORDECAI] Built context for {len(doc_ctx)} docs")
    except FileNotFoundError:
        doc_ctx = {}
        print("[MORDECAI] mordecai_ims_hits.csv not found — continuing without Mordecai context")

    # NEW: Mordecai → IMS port matches per doc_id
    mord_matches_by_doc: Dict[str, List[Dict[str, Any]]] = {}
    if MORD_PORT_MATCHES_CSV.exists():
        try:
            mm_df = pd.read_csv(MORD_PORT_MATCHES_CSV)
            print(f"[MORDECAI-MATCH] Loaded {len(mm_df)} rows from {MORD_PORT_MATCHES_CSV}")
            grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
            for _, row in mm_df.iterrows():
                doc_id = str(row.get("doc_id", ""))
                total_score = float(row.get("total_score", 0.0) or 0.0)
                if not doc_id:
                    continue
                if total_score < MORDECAI_MIN_SCORE:
                    continue
                grouped[doc_id].append(
                    {
                        "ims_facility_id": str(row.get("ims_facility_id", "") or ""),
                        "facility_unlocode": str(row.get("facility_unlocode", "") or ""),
                        "facility_name": str(row.get("facility_name", "") or ""),
                        "facility_lat": row.get("facility_lat"),
                        "facility_lon": row.get("facility_lon"),
                        "distance_km": row.get("distance_km"),
                        "total_score": total_score,
                    }
                )
            # sort per-doc by score desc
            for d, lst in grouped.items():
                lst.sort(key=lambda r: r.get("total_score", 0.0), reverse=True)
            mord_matches_by_doc = dict(grouped)
            print(f"[MORDECAI-MATCH] Aggregated matches for {len(mord_matches_by_doc)} docs")
        except Exception as e:
            print(f"[MORDECAI-MATCH] Error loading {MORD_PORT_MATCHES_CSV}: {e}")
            mord_matches_by_doc = {}
    else:
        print(f"[MORDECAI-MATCH] {MORD_PORT_MATCHES_CSV} not found — Mordecai matches disabled")

    def aggregate_mordecai_ports(doc_id: str) -> Dict[str, str]:
        """
        For a given doc_id, return pipe-joined strings of Mordecai-matched ports.
        """
        matches = mord_matches_by_doc.get(doc_id) or []
        if not matches:
            return {
                "mord_ports_ims_ids": "",
                "mord_ports_unlocodes": "",
                "mord_ports_names": "",
                "mord_ports_scores": "",
                "mord_ports_distance_km": "",
            }

        subset = matches[:MORDECAI_MAX_PORTS_PER_DOC]
        ims_ids: List[str] = []
        unlocs: List[str] = []
        names: List[str] = []
        scores: List[str] = []
        dists: List[str] = []

        for m in subset:
            ims_ids.append(m.get("ims_facility_id", ""))
            unlocs.append(m.get("facility_unlocode", ""))
            names.append(m.get("facility_name", ""))
            scores.append(f"{m.get('total_score', 0.0):.3f}")
            d = m.get("distance_km", "")
            try:
                if d is None or (isinstance(d, float) and math.isnan(d)):
                    dists.append("")
                else:
                    dists.append(f"{float(d):.1f}")
            except Exception:
                dists.append("")
        return {
            "mord_ports_ims_ids": "|".join(ims_ids),
            "mord_ports_unlocodes": "|".join(unlocs),
            "mord_ports_names": "|".join(names),
            "mord_ports_scores": "|".join(scores),
            "mord_ports_distance_km": "|".join(dists),
        }

    ais_live = AISClient()
    if SKIP_AIS:
        print("[AIS] skipped via SKIP_AIS=True")
    else:
        print("[AIS] live client configured" if ais_live.base_url else "[AIS] live client not configured")

    cls_files = glob.glob(f"{IN_DIR}/*.classify.json")
    incident_files: List[str] = []
    for cf in cls_files:
        try:
            data = load_json_any(Path(cf))
            if str(data.get("is_incident", "")).lower() in {"true", "1", "yes"}:
                incident_files.append(cf)
        except Exception:
            continue

    print(f"[RUN] Incidents to process: {len(incident_files)}")

    summary_rows: List[Dict[str, Any]] = []
    train_rows: List[Dict[str, Any]] = []
    processed = 0

    for cf in incident_files:
        processed += 1
        cls = load_json_any(Path(cf))
        doc_id = cls["doc_id"]
        safe_id = safe_fname(doc_id)

        norm_path = Path(NORM_DIR) / f"{safe_id}.json"
        if not norm_path.exists():
            continue

        norm = load_json_any(norm_path)

        title = fix_text(norm.get("title", ""))[:500]
        text_raw = fix_text(norm.get("content_text", ""))
        cleaned = clean_article_text(title, text_raw)

        # ---- NER
        ents = extractor.extract(cleaned)
        vessels = ents.get("vessels", [])
        imos = ents.get("imos", [])
        countries_text = ents.get("countries", [])
        regions_text = ents.get("regions", [])
        ports_raw = ents.get("ports", [])

        # NEW: strict cleanup of port spans before using them anywhere
        ports_text = filter_port_spans(ports_raw)

        # ---- Country candidates (from country_ISO.csv + country_aliases.csv)
        name_hints_raw = map_names_to_iso2(
            countries_text,
            name2iso,
            alias2iso=alias2iso,
        )
        name_hints = clean_iso2_list(name_hints_raw)
        region_iso2s = region_to_iso2_candidates(regions_text, region_centroids)

        # NEW: Mordecai country context for this doc
        m_ctx = doc_ctx.get(doc_id, {})
        mord_countries = m_ctx.get("countries", [])
        mord_admin1 = m_ctx.get("admin1_names", [])

        # Merge Mordecai countries into our ISO2 hints
        name_hints = clean_iso2_list(name_hints + mord_countries)

        if DEBUG_NER or DEBUG_PIPE:
            print("\n================ NER DEBUG ================")
            print(f"doc_id: {cls.get('doc_id')}")
            print(f"title: {title[:120]!r}")
            print("------------------------------------------")
            print(f"ports (raw): {ports_raw}")
            print(f"ports (clean): {ports_text}")
            print(f"countries (raw): {countries_text}")
            print(f"regions (raw): {regions_text}")
            print(f"ISO2 from names+Mordecai: {name_hints}")
            print(f"Mordecai countries: {mord_countries}")
            print(f"Mordecai admin1: {mord_admin1}")
            print("==========================================\n")

        _dbg_pipe(f"doc_id={cls.get('doc_id')} title={title[:80]!r}")
        _dbg_pipe(f"ports_text={ports_text}")
        _dbg_pipe(f"countries_text={countries_text} → iso={name_hints}")
        _dbg_pipe(f"regions_text={regions_text} → region_iso2s={region_iso2s}")

        # search_iso2s used only for IMS suggestions / fallback
        search_iso2s: List[str] = []
        seen_iso: Set[str] = set()
        for iso in region_iso2s + name_hints:
            if iso and iso not in seen_iso:
                search_iso2s.append(iso)
                seen_iso.add(iso)

        _dbg_pipe(f"search_iso2s (for IMS fallback)={search_iso2s}")

        # ---- Port candidates: for each cleaned port span, use unified KB
        _dbg_pipe("[PIPE] building candidates via PlaceResolver.resolve_port_with_context(...)")

        pr_hits_raw: List[Dict[str, Any]] = []
        span_by_id: Dict[int, str] = {}

        # Context countries: prefer name_hints (includes Mordecai) then search_iso2s
        context_countries = name_hints or search_iso2s

        for span_text in ports_text:
            matches = place_resolver.resolve_port_with_context(
                span_text=span_text,
                context_countries=context_countries,
                min_score=0.55,  # a bit lower; ML model will re-rank
                top_k=TOP_K_CANDS,
            )
            for rec in matches:
                cand = {
                    "source": rec.get("source", "IMS"),
                    "row_kind": "span_port",
                    "countryISOCode": rec.get("country_iso2", ""),
                    "locationISOCode": (rec.get("unlocode") or "")[-3:],
                    "unlocode": rec.get("unlocode", ""),
                    "locationName": rec.get("location_name", ""),
                    "subdivision": rec.get("admin1", ""),
                    "city": rec.get("city", ""),
                    "lat": rec.get("lat"),
                    "lon": rec.get("lon"),
                    "alternatenames": rec.get("aliases", []),
                }
                pr_hits_raw.append(cand)
                span_by_id[id(cand)] = span_text

        _dbg_pipe(f"PlaceResolver initial candidates: {len(pr_hits_raw)}")

        pr_hits = prune_candidates(
            pr_hits_raw,
            ports_text=ports_text,
            name_hints=name_hints,
            max_total=TOP_K_OFFLINE,
        )

        _dbg_pipe(f"After prune_candidates → {len(pr_hits)} candidates")

        # region center for distance features
        region_center = infer_region_centroid(regions_text, region_centroids)
        _dbg_pipe(f"region_center={region_center}")

        feats_list: List[Dict[str, float]] = []
        X_rows: List[Any] = []
        spans: List[str] = []
        cands_meta: List[Dict[str, Any]] = []

        for h in pr_hits:
            span = span_by_id.get(id(h)) or h.get("locationName", "") or h.get("unlocode", "")
            spans.append(span)

            la, lo = cand_latlon(h)
            if la is not None and lo is not None:
                h["lat"], h["lon"] = la, lo

            if region_center and la is not None and lo is not None:
                dist = _haversine_km((la, lo), region_center)
            else:
                dist = 0.0
            h["dist_km"] = dist

            aliases_raw = h.get("alternatenames", "")
            if isinstance(aliases_raw, str) and aliases_raw:
                aliases = [a.strip() for a in aliases_raw.split(",") if a.strip()]
            else:
                aliases = h.get("alternatenames", []) or []

            feats = assemble_feature_vector(
                span,
                cleaned,
                name_hints,
                vessels,
                regions_text,
                {
                    "location_name": h.get("locationName", ""),
                    "city": h.get("subdivision", "") or h.get("city", ""),
                    "aliases": aliases,
                    "country_iso2": h.get("countryISOCode", ""),
                    "lat": h.get("lat"),
                    "lon": h.get("lon"),
                    "dist_km": h.get("dist_km", 0.0),
                },
            )
            feats_list.append(feats)
            X_rows.append(vectorize(feats))
            cands_meta.append(h)

        # ---- Track unmatched raw port spans
        all_ports_raw = ports_text or []
        cand_names_norm = {
            (c.get("locationName") or "").strip().lower()
            for c in cands_meta
        }
        unmatched_ports = []
        for p in all_ports_raw:
            if not p:
                continue
            if p.strip().lower() not in cand_names_norm:
                unmatched_ports.append(p)

        all_ports_raw_str = "|".join(all_ports_raw) if all_ports_raw else ""
        unmatched_ports_str = "|".join(unmatched_ports) if unmatched_ports else ""

        _dbg_pipe(f"Built feature vectors for {len(cands_meta)} candidates")

        # ---- dump candidates for training (optional)
        if DUMP_TRAINING_CANDIDATES and pr_hits:
            for idx, (h, span, feats) in enumerate(zip(cands_meta, spans, feats_list)):
                train_rows.append(
                    {
                        "doc_id": cls["doc_id"],
                        "candidate_idx": idx,
                        "span_text": span,
                        "source": h.get("source", ""),
                        "row_kind": h.get("row_kind", ""),
                        "countryISOCode": h.get("countryISOCode", ""),
                        "locationISOCode": h.get("locationISOCode", ""),
                        "locationName": h.get("locationName", ""),
                        "city": h.get("subdivision", "") or h.get("city", ""),
                        "title": title[:200],
                        "regions_text": "|".join(regions_text) if regions_text else "",
                        "countries_text": "|".join(countries_text) if countries_text else "",
                        "sim_name": feats.get("sim_name", 0.0),
                        "sim_city": feats.get("sim_city", 0.0),
                        "sim_alias": feats.get("sim_alias", 0.0),
                        "tok_jacc_name": feats.get("tok_jacc_name", 0.0),
                        "tok_jacc_city": feats.get("tok_jacc_city", 0.0),
                        "sim_best": feats.get("sim_best", 0.0),
                        "country_match_any": feats.get("country_match_any", 0.0),
                        "country_has_hint": feats.get("country_has_hint", 0.0),
                        "dist_km": feats.get("dist_km", 0.0),
                        "dist_inv": feats.get("dist_inv", 0.0),
                        "ctx_vessel": feats.get("ctx_vessel", 0.0),
                        "ctx_region": feats.get("ctx_region", 0.0),
                        "ctx_harbor_terms": feats.get("ctx_harbor_terms", 0.0),
                        "sim_best_x_country": feats.get("sim_best_x_country", 0.0),
                        "sim_best_x_ctx": feats.get("sim_best_x_ctx", 0.0),
                        "label": "",
                    }
                )

        # ---- Rank candidates & run AIS
        ranked: List[Tuple[float, int]] = []
        top_idx, top_score = None, 0.0
        ais_points: List[Tuple[float, float]] = []

        if X_rows:
            X = np.vstack(X_rows)
            probs = disamb.predict_proba(X)
            base_scores = [float(p) + portish_context_bonus(cleaned) for p in probs]

            # tiny bonus for IMS-sourced candidates
            for i, cand in enumerate(cands_meta):
                if (cand.get("source") or "").upper() == "IMS":
                    base_scores[i] += 0.03

            provisional = sorted([(s, i) for i, s in enumerate(base_scores)], reverse=True)
            cand_center: Optional[Tuple[float, float]] = None
            if provisional:
                pi = provisional[0][1]
                la, lo = cand_latlon(cands_meta[pi])
                if la is not None and lo is not None:
                    cand_center = (la, lo)
            if not cand_center and region_center:
                cand_center = region_center

            # AIS skipped anyway
            for i, base in enumerate(base_scores):
                bonus = ais_bonus_km(cands_meta[i], ais_points, AIS_PROXIMITY_BONUS_KM)
                ranked.append((base + bonus, i))
            ranked.sort(key=lambda x: x[0], reverse=True)
            if ranked:
                top_idx = ranked[0][1]
                top_score = ranked[0][0]

        _dbg_pipe(f"X_rows={len(X_rows)} top_idx={top_idx} top_score={top_score:.3f}")

        # Precompute country/region fields
        country_name_detected = countries_text[0] if countries_text else ""
        country_iso2_detected = (name_hints[0] if name_hints else "") or ""
        countries_all = "|".join(countries_text) if countries_text else ""
        countries_iso2_all = "|".join(name_hints) if name_hints else ""
        all_regions_all = "|".join(regions_text) if regions_text else ""
        all_ports_all_candidates = (
            "|".join(
                [
                    (c.get("locationName") or c.get("unlocode") or "")
                    for c in cands_meta
                ]
            )
            if cands_meta
            else ""
        )

        # NEW: aggregate Mordecai ports for this doc_id
        mord_cols = aggregate_mordecai_ports(doc_id)

        # ---- Option B: if NO explicit port spans → always "no confident port"
        if not ports_text:
            low_conf_or_none = True
        else:
            low_conf_or_none = (
                not X_rows
                or top_idx is None
                or (not IMS_ALWAYS_QUERY_TOP and top_score < CONF_THRESHOLD)
            )
        _dbg_pipe(
            f"low_conf_or_none={low_conf_or_none} "
            f"(CONF_THRESHOLD={CONF_THRESHOLD} ports_text_present={bool(ports_text)})"
        )

        if low_conf_or_none:
            lat_out, lon_out = ("", "")
            if region_center:
                lat_out, lon_out = f"{region_center[0]}", f"{region_center[1]}"

            ims_labels: List[str] = []
            ims_status = "skipped_no_candidates"
            ims_reason = "no_port_candidates_or_no_explicit_ports"

            # IMS used only to suggest ports by country here
            if IMS_ENABLED and (search_iso2s or name_hints):
                try:
                    for iso2 in (search_iso2s or name_hints)[:3]:
                        _dbg_pipe(f"[IMS-SUGGEST] list_facilities_by_country iso2={iso2}")
                        facs = list_facilities_by_country(
                            ims_client, iso2, limit=IMS_SUGGEST_LIMIT
                        )
                        _dbg_pipe(f"[IMS-SUGGEST] got {len(facs)} facilities for {iso2}")
                        for fac in facs:
                            attr_f = fac.get("attributes", {}) or {}
                            label = (
                                f"{attr_f.get("ountryISOCode","")}:"  # country
                                f"{attr_f.get("locationName","")} "    # name
                                f"[{attr_f.get("locationISOCode","")}]"  # code
                            )
                            ims_labels.append(label)
                    if ims_labels:
                        ims_status = "suggest_only"
                        ims_reason = "countries_only_or_low_conf"
                except Exception as e:
                    ims_status = "ims_error_suggestions"
                    ims_reason = str(e)[:200]

            row_out = {
                "doc_id": cls["doc_id"],
                "title": title[:150],
                "is_incident": str(cls.get("is_incident", "")).lower()
                in {"true", "1", "yes"},
                "port_span": "",
                "port_local_name": "",
                "port_city_local": "",
                "port_source": "",
                "port_score": round(top_score, 3),
                "port_matched_field": "none",
                "lat_used": lat_out,
                "lon_used": lon_out,
                "country_name_detected": country_name_detected,
                "country_iso2_detected": country_iso2_detected,
                "country_iso2_facility": "",
                "region_in_text": (regions_text[0] if regions_text else ""),
                "countries_all": countries_all,
                "countries_iso2_all": countries_iso2_all,
                "all_regions_all": all_regions_all,
                "all_ports_all_candidates": all_ports_all_candidates,
                "all_ports_raw": all_ports_raw_str,
                "unmatched_ports": unmatched_ports_str,
                "ims_lookup_status": ims_status,
                "ims_lookup_reason": ims_reason,
                "ims_facility_id": "",
                "ims_locationISOCode": "",
                "ims_locationName": "",
                "ims_countryISOCode": "",
                "ims_city": "",
                "ims_facility_suggestions": "|".join(ims_labels),
            }
            row_out.update(mord_cols)
            summary_rows.append(row_out)

            if LIVE_WRITE_EVERY and processed % LIVE_WRITE_EVERY == 0:
                to_csv_atomic(pd.DataFrame(summary_rows), str(SUMMARY_CSV))
                print(f"[LIVE] wrote interim CSVs @{processed}")
                if DUMP_TRAINING_CANDIDATES and train_rows:
                    to_csv_atomic(pd.DataFrame(train_rows), str(TRAIN_CANDIDATES_CSV))
                    print(f"[LIVE] wrote interim training candidates @{processed}")
            continue

        # ---- IMS resolve for best candidate (only if explicit ports present)
        ims_facility, ims_status, ims_reason = None, "skipped", ""
        ims_labels = []

        if IMS_ENABLED and top_idx is not None:
            cand = cands_meta[top_idx]
            own_iso = (cand.get("countryISOCode") or "").upper()
            _dbg_pipe(
                f"IMS_ENABLED=True, querying IMS for top candidate "
                f"(own_iso={own_iso}, top_score={top_score:.3f})"
            )

            ims_facility, ims_status, ims_reason = ims_try_for_top_candidate(
                ims_client=ims_client,
                ports_text=ports_text,
                top_cand=cand,
                top_score=top_score,
                conf_threshold=CONF_THRESHOLD,
                own_iso=own_iso,
                search_iso2s=search_iso2s,
            )

        # Country fallback suggestions for UI/review (does NOT override ims_facility)
        fetched: List[Dict[str, Any]] = []
        if IMS_ENABLED and (search_iso2s or name_hints):
            try:
                for iso2 in (search_iso2s or name_hints)[:3]:
                    _dbg_pipe(f"[IMS-LIST] list_facilities_by_country iso2={iso2}")
                    fetched.extend(
                        list_facilities_by_country(
                            ims_client, iso2, limit=IMS_SUGGEST_LIMIT
                        )
                    )
                _dbg_pipe(f"[IMS-LIST] fetched={len(fetched)} combined facilities")
            except Exception as e:
                if "401" in str(e).lower() or "unauthorized" in str(e).lower():
                    print("[IMS] 401 during list_by_country; disabling for this run.")
                    IMS_ENABLED = False

        for fac in fetched:
            attr_f = fac.get("attributes", {}) or {}
            label = (
                f"{attr_f.get('countryISOCode','')}:"  # country
                f"{attr_f.get('locationName','')} "    # name
                f"[{attr_f.get('locationISOCode','')}]"  # code
            )
            ims_labels.append(label)

        # ---- Prefer lat/lon for AIS column output
        def pick_coords() -> Tuple[str, str]:
            if top_idx is not None:
                la = cands_meta[top_idx].get("lat")
                lo = cands_meta[top_idx].get("lon")
                if la is not None and lo is not None:
                    return f"{la}", f"{lo}"
            if ims_facility:
                attr_f2 = ims_facility.get("attributes", {}) or {}
                la = attr_f2.get("latitude")
                lo = attr_f2.get("longitude")
                if la not in ("", None) and lo not in ("", None):
                    return str(la), str(lo)
            if region_center:
                return f"{region_center[0]}", f"{region_center[1]}"
            return "", ""

        lat_out, lon_out = pick_coords()

        attr = ims_facility.get("attributes", {}) or {} if ims_facility else {}
        country_iso2_facility = attr.get("countryISOCode", "")

        _dbg_pipe(
            f"FINAL port_span={spans[top_idx] if top_idx is not None else ''} "
            f"ims_status={ims_status} ims_reason={ims_reason} "
            f"ims_locationISOCode={attr.get('locationISOCode', '')}"
        )

        row_out = {
            "doc_id": cls["doc_id"],
            "title": title[:150],
            "is_incident": str(cls.get("is_incident", "")).lower()
            in {"true", "1", "yes"},
            "port_span": spans[top_idx] if top_idx is not None else "",
            "port_local_name": (
                cands_meta[top_idx].get("locationName", spans[top_idx])
                if top_idx is not None
                else ""
            ),
            "port_city_local": (
                cands_meta[top_idx].get("subdivision", "")
                or cands_meta[top_idx].get("city", "")
                if top_idx is not None
                else ""
            ),
            "port_source": (
                cands_meta[top_idx].get("source", "") if top_idx is not None else ""
            ),
            "port_score": round(top_score, 3),
            "port_matched_field": "ml_prob",
            "lat_used": lat_out,
            "lon_used": lon_out,
            "country_name_detected": country_name_detected,
            "country_iso2_detected": country_iso2_detected,
            "country_iso2_facility": country_iso2_facility,
            "region_in_text": (regions_text[0] if regions_text else ""),
            "countries_all": countries_all,
            "countries_iso2_all": countries_iso2_all,
            "all_regions_all": all_regions_all,
            "all_ports_all_candidates": all_ports_all_candidates,
            "all_ports_raw": all_ports_raw_str,
            "unmatched_ports": unmatched_ports_str,
            "ims_lookup_status": ims_status,
            "ims_lookup_reason": ims_reason,
            "ims_facility_id": ims_facility.get("id") if ims_facility else "",
            "ims_locationISOCode": attr.get("locationISOCode", ""),
            "ims_locationName": attr.get("locationName", ""),
            "ims_countryISOCode": attr.get("countryISOCode", ""),
            "ims_city": attr.get("city", ""),
            "ims_facility_suggestions": "|".join(ims_labels),
        }
        row_out.update(mord_cols)
        summary_rows.append(row_out)

        if LIVE_WRITE_EVERY and processed % LIVE_WRITE_EVERY == 0:
            to_csv_atomic(pd.DataFrame(summary_rows), str(SUMMARY_CSV))
            print(f"[LIVE] wrote interim CSVs @{processed}")
            if DUMP_TRAINING_CANDIDATES and train_rows:
                to_csv_atomic(pd.DataFrame(train_rows), str(TRAIN_CANDIDATES_CSV))
                print(f"[LIVE] wrote interim training candidates @{processed}")

    # ---- Final write (atomic)
    df_out = pd.DataFrame(summary_rows)
    desired_cols = [
        "doc_id",
        "title",
        "is_incident",
        "port_span",
        "port_local_name",
        "port_city_local",
        "port_source",
        "port_score",
        "port_matched_field",
        "lat_used",
        "lon_used",
        "country_name_detected",
        "country_iso2_detected",
        "country_iso2_facility",
        "region_in_text",
        "countries_all",
        "countries_iso2_all",
        "all_regions_all",
        "all_ports_all_candidates",
        "all_ports_raw",
        "unmatched_ports",
        "ims_lookup_status",
        "ims_lookup_reason",
        "ims_facility_id",
        "ims_locationISOCode",
        "ims_locationName",
        "ims_countryISOCode",
        "ims_city",
        "ims_facility_suggestions",
        # NEW Mordecai columns:
        "mord_ports_ims_ids",
        "mord_ports_unlocodes",
        "mord_ports_names",
        "mord_ports_scores",
        "mord_ports_distance_km",
    ]
    extra_cols = [c for c in df_out.columns if c not in desired_cols]
    df_out = df_out[[c for c in desired_cols if c in df_out.columns] + extra_cols]

    to_csv_atomic(df_out, str(SUMMARY_CSV))
    print(f"[OK] wrote {len(summary_rows)} rows → {SUMMARY_CSV}")

    if DUMP_TRAINING_CANDIDATES and train_rows:
        train_df = pd.DataFrame(train_rows)
        to_csv_atomic(train_df, str(TRAIN_CANDIDATES_CSV))
        print(f"[OK] wrote {len(train_rows)} candidate rows → {TRAIN_CANDIDATES_CSV}")


if __name__ == "__main__":
    run()

