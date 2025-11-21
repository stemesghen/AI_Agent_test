import os
import re
import json
from pathlib import Path
from typing import List, Dict, Any

import spacy
from transformers import pipeline as hf_pipeline
from transformers.utils import logging as hf_logging

# country gazetteer loader
from nlp.country_codes import load_country_codes

hf_logging.set_verbosity_error()

# =========================
# Model selection
# =========================
MARITIME_MODELS = [
    
"dslim/bert-base-NER"
#	"wjallen/maritime-ner",
 #   "wchen1021/ship-ner-base",
]
GENERIC_MODEL = "dslim/bert-base-NER"

USE_MARITIME = os.getenv("MARITIME_NER", "1") == "1"
MODEL_LIST = (MARITIME_MODELS + [GENERIC_MODEL]) if USE_MARITIME else [GENERIC_MODEL]

# =========================
# Regex + lightweight heuristics
# =========================
IMO_RE = re.compile(r"\bIMO\s*(?:No\.?|number|#|:)?\s*([1-9]\d{6})\b", re.I)

VESSEL_HINTS = re.compile(
    r"\b(?:MV|M\/V|MT|M\/T|LPG|LNG|Bulk(?:\s+Carrier)?|Tanker|Container(?:\s+Ship)?|Ro-?Ro|PCTC|Car(?:go)?\s+Ship|"
    r"Drillship|Dredger|VLCC|ULCC|FPSO)\b",
    re.I,
)

MARITIME_CONTEXT = re.compile(
    r"\b(berth|anch(or|ing|age)|moored?|port|harbou?r|vessel|ship|cargo|crew|captain|pilot|tug|tow|AIS|IMO|spill|"
    r"ground(?:ed|ing)|collision|allision|piracy|hijack|rescue|distress|terminal|jetty|quay)\b",
    re.I,
)

VESSEL_NAME_RE = re.compile(r"\b(?:(?:MV|M\/V|MT|M\/T|LPG|LNG)\s+)?([A-Z][A-Za-z0-9\-’' ]{2,})\b")

PORT_OF_RE = re.compile(r"\b(?:Port|Harbor|Harbour)\s+of\s+([A-Z][A-Za-z0-9 .,'’\-]{2,})\b", re.I)
PORT_TAIL_RE = re.compile(
    r"\b([A-Z][A-Za-z0-9 .,'’\-]{2,})\s+(?:Port|Harbor|Harbour|Terminal|Anchorage|Jetty|Quay)\b", re.I
)
PORT_HEAD_RE = re.compile(r"\b(?:Port)\s+([A-Z][A-Za-z0-9 .,'’\-]{2,})\b", re.I)

USE_STRICT_VESSEL = os.getenv("STRICT_VESSEL_FILTER", "1") == "1"
VESSEL_KB_PATH = Path(os.getenv("VESSEL_KB_PATH", ""))

REGION_FILE = Path(os.getenv("REGION_LOOKUP_PATH", "data/region_lookup.json"))

# --------------------------
# Configurable stopwords (generic, not publisher-specific)
# --------------------------

# country gazetteer CSVs
COUNTRY_CSV = Path(os.getenv("COUNTRY_CSV_PATH", "data/country_ISO.csv"))
COUNTRY_ALIASES_CSV = Path(os.getenv("COUNTRY_ALIASES_PATH", "data/country_aliases.csv"))

PORT_STOPWORDS_FILE = Path(os.getenv("PORT_STOPWORDS_FILE", "data/config/port_stopwords.txt"))


def _load_list(path: Path) -> set[str]:
    try:
        if path.exists():
            return {
                ln.strip().lower()
                for ln in path.read_text(encoding="utf-8").splitlines()
                if ln.strip()
            }
    except Exception:
        pass
    return set()


PORT_CANDIDATE_STOPWORDS = _load_list(PORT_STOPWORDS_FILE) or {
    "facebook",
    "twitter",
    "linkedin",
    "sharethis",
    "share",
    "subscribe",
    "login",
    "sign in",
    "sign up",
    "cookie",
    "privacy",
    "terms",
    "advertisement",
    "sponsored",
    "views",
    "comments",
    "photo",
    "image",
    "video",
    "source",
}


def _is_bad_port_token(s: str) -> bool:
    if not s:
        return False
    t = " ".join(s.split()).strip().lower()
    if t in PORT_CANDIDATE_STOPWORDS:
        return True
    # generic single tokens that are never ports
    if len(t) <= 2:
        return True
    # URLs, emails, or most-non-alpha tokens
    if re.search(r"https?://|\S+@\S+|\.com\b|\.net\b|\.org\b", t):
        return True
    non_alpha_ratio = 1.0 - (sum(ch.isalpha() for ch in s) / max(1, len(s)))
    if non_alpha_ratio > 0.6:
        return True
    return False


def _dedupe(seq):
    seen, out = set(), []
    for s in seq:
        s2 = (s or "").strip()
        if not s2:
            continue
        if s2 not in seen:
            seen.add(s2)
            out.append(s2)
    return out


def _load_region_names():
    if REGION_FILE.exists():
        try:
            with open(REGION_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            return [str(r.get("name", "")).strip().lower() for r in data if r.get("name")]
        except Exception:
            return []
    return []


REGION_NAMES_LC = _load_region_names()


def _extract_regions(text: str):
    if not REGION_NAMES_LC:
        return []
    low = (text or "").lower()
    found = []
    for r in REGION_NAMES_LC:
        if r and r in low:
            found.append(r.title())
    return _dedupe(found)


def _load_country_gazetteer():
    """
    Load country + alias names from the same CSVs used in run.py via load_country_codes.
    We only use them to *detect* names; mapping to ISO2 is still done later by map_names_to_iso2.
    """
    try:
        if not COUNTRY_CSV.exists():
            return set()
        name2iso, iso2name, alias2iso = load_country_codes(COUNTRY_CSV, COUNTRY_ALIASES_CSV)
        names = set()

        # canonical country names (e.g. "China", "Russian Federation")
        for nm in (name2iso or {}).keys():
            if nm:
                names.add(nm.strip().lower())

        # alias forms (e.g. "People's Republic of China", "PRC", demonyms etc. if present)
        for al in (alias2iso or {}).keys():
            if al:
                names.add(al.strip().lower())

        return names
    except Exception:
        return set()


COUNTRY_GAZ = _load_country_gazetteer()


def _build_country_patterns():
    """
    Build regex patterns with word boundaries for each country/alias name.
    We skip very short names (<3 chars) to avoid 'us' matching in 'business'.
    """
    patterns = {}
    for nm in COUNTRY_GAZ:
        if not nm:
            continue
        if len(nm) < 3:
            continue
        try:
            patterns[nm] = re.compile(r"\b" + re.escape(nm) + r"\b", re.I)
        except re.error:
            # in case some alias has weird chars
            continue
    return patterns


COUNTRY_PATTERNS = _build_country_patterns()


def _extract_countries_gazetteer(text: str):
    """
    Use simple regex matches against the country gazetteer to find explicit
    country/alias mentions in the raw text.
    """
    if not text or not COUNTRY_PATTERNS:
        return []
    low_text = text.lower()
    found = []
    for nm, rx in COUNTRY_PATTERNS.items():
        if rx.search(low_text):
            # Title-case as a reasonable display form; the later map_names_to_iso2()
            # will handle mapping to ISO2 codes.
            found.append(nm.title())
    return _dedupe(found)


def _looks_like_ship_name(span: str, text: str) -> bool:
    s = (span or "").strip()
    if not s or len(s) < 3:
        return False
    if len(s.split()) > 6:
        return False
    if VESSEL_NAME_RE.search(s):
        return True
    has_alpha = any(ch.isalpha() for ch in s)
    has_digit = any(ch.isdigit() for ch in s)
    if has_alpha and has_digit and any(w.isupper() for w in s.split()):
        return True
    if text:
        low = text.lower()
        s_low = s.lower()
        idx = low.find(s_low)
        if idx >= 0:
            L = max(0, idx - 80)
            R = min(len(low), idx + len(s_low) + 80)
            window = low[L:R]
            if VESSEL_HINTS.search(window) or MARITIME_CONTEXT.search(window):
                return True
    return False


def _filter_vessel_candidates(cands, text):
    if not USE_STRICT_VESSEL:
        return _dedupe(cands)
    kept = []
    for c in cands:
        if _looks_like_ship_name(c, text):
            kept.append(c)
    return _dedupe(kept)


def _load_vessel_kb():
    if VESSEL_KB_PATH and VESSEL_KB_PATH.exists():
        try:
            with open(VESSEL_KB_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            return set(s.strip().lower() for s in data if s and isinstance(s, str))
        except Exception:
            return None
    return None


VESSEL_KB = _load_vessel_kb()


def _gate_with_kb(vessels):
    if not VESSEL_KB:
        return vessels
    out = []
    for v in vessels:
        if v.strip().lower() in VESSEL_KB:
            out.append(v)
    return _dedupe(out)


def _normalize_port_phrase(s: str) -> str:
    s = (s or "").strip(" ,.;:()[]{}")
    s = re.sub(r"\s+(?:area|region)$", "", s, flags=re.I).strip()
    return s


def _extract_ports_by_regex(text: str):
    ports = []
    for m in PORT_OF_RE.finditer(text or ""):
        ports.append(_normalize_port_phrase(m.group(1)))
    for m in PORT_TAIL_RE.finditer(text or ""):
        ports.append(_normalize_port_phrase(m.group(1)))
    for m in PORT_HEAD_RE.finditer(text or ""):
        ports.append(_normalize_port_phrase(m.group(1)))
    ports = [p for p in ports if p and len(p) >= 3 and not p.lower().startswith(("the ", "a ", "an "))]
    # generic stopword guard
    ports = [p for p in ports if not _is_bad_port_token(p)]
    return _dedupe(ports)


class EntityExtractor:
    def __init__(self):
        print(f"[NER] Using model list: {MODEL_LIST}")
        self.hf_ner = None
        last_err = None
        for name in MODEL_LIST:
            try:
                self.hf_ner = hf_pipeline(
                    "ner",
                    model=name,
                    tokenizer=name,
                    aggregation_strategy="simple",
                    #device_map="auto",  # GPU if available -- comment out because conflicts with the torch version needed for Mordecai
                    device=-1, #force cpu
		)
                break
            except Exception as e:
                last_err = e
        if self.hf_ner is None:
            raise RuntimeError(f"NER load failed. Last error: {last_err}")

        try:
            self.spacy_nlp = spacy.load("en_core_web_sm")
        except OSError:
            raise RuntimeError("spaCy model missing. Run: python -m spacy download en_core_web_sm")

        # spaCy batching knobs
        self.spacy_batch_size = int(os.getenv("SPACY_BATCH_SIZE", "16"))
        self.spacy_n_process = int(os.getenv("SPACY_N_PROCESS", "1"))

    # ------------ single ------------
    def extract(self, text: str) -> Dict[str, Any]:
        try:
            hf_entities = self.hf_ner(text, truncation=True, max_length=512)
        except Exception:
            hf_entities = []
        try:
            spacy_doc = self.spacy_nlp(text)
        except Exception:
            spacy_doc = None
        return self._assemble_entities(text, hf_entities, spacy_doc)

    # ------------ batched ------------
    def extract_many(self, texts: List[str]) -> List[Dict[str, Any]]:
        if not texts:
            return []
        try:
            hf_batches = self.hf_ner(texts, truncation=True, max_length=512)
        except Exception:
            hf_batches = [[] for _ in texts]

        docs = []
        try:
            for d in self.spacy_nlp.pipe(
                texts,
                batch_size=self.spacy_batch_size,
                n_process=self.spacy_n_process,
            ):
                docs.append(d)
        except Exception:
            docs = [None] * len(texts)

        out = []
        for t, hf_ents, doc in zip(texts, hf_batches, docs):
            out.append(self._assemble_entities(t, hf_ents, doc))
        return out

    # ------------ assembly ------------
    def _assemble_entities(
        self,
        text: str,
        hf_entities: List[Dict[str, Any]],
        spacy_doc,
    ) -> Dict[str, Any]:
        ports: List[str] = []
        vessels: List[str] = []
        incidents: List[str] = []
        countries: List[str] = []

        # ---------- HuggingFace NER ----------
        for ent in hf_entities or []:
            label_raw = (ent.get("entity_group") or ent.get("entity") or "").upper()
            value = " ".join((ent.get("word") or "").split())
            if not value:
                continue

            # Vessel-like
            if "VESSEL" in label_raw:
                if _looks_like_ship_name(value, text):
                    vessels.append(value)

            # Location-like labels: treat as both generic ports/places AND country candidates
            elif label_raw in {"PORT", "HARBOR", "HARBOUR", "LOC", "LOCATION", "GPE"}:
                if not _is_bad_port_token(value):
                    ports.append(value)
                    # Also surface them as country candidates; map_names_to_iso2 will
                    # later decide which ones are real countries.
                    countries.append(value)

            # Organizations that might actually be vessel names
            elif label_raw in {"ORG", "ORGANIZATION"}:
                if _looks_like_ship_name(value, text):
                    vessels.append(value)

            elif label_raw in {"EVENT"}:
                incidents.append(value)

        # ---------- spaCy NER ----------
        if spacy_doc is not None:
            try:
                for ent in spacy_doc.ents:
                    txt = ent.text
                    if not txt:
                        continue

                    # Geo-political entities, locations, facilities:
                    # keep existing behavior (ports) AND ALSO treat as country candidates.
                    if ent.label_ in ("GPE", "LOC", "FAC"):
                        if not _is_bad_port_token(txt):
                            ports.append(txt)
                        countries.append(txt)

                    # ORG/PRODUCT/WORK_OF_ART as vessel names when they look like ships
                    elif ent.label_ in ("ORG", "PRODUCT", "WORK_OF_ART"):
                        if _looks_like_ship_name(txt, text):
                            vessels.append(txt)

                    elif ent.label_ == "EVENT":
                        incidents.append(txt)

                    # NORP = demonyms, nationalities (e.g., "Chinese", "Russian")
                    # surface as country-like; map_names_to_iso2 will resolve.
                    elif ent.label_ == "NORP":
                        countries.append(txt)
            except Exception:
                pass

        # ---------- Regex booster for ports ----------
        ports += _extract_ports_by_regex(text)

        # ---------- IMO numbers ----------
        imos = list({m.group(1) for m in IMO_RE.finditer(text)})

        # Gazetteer booster for countries (on top of spaCy NORP)
        gaz_countries = _extract_countries_gazetteer(text)
        for c in gaz_countries:
            countries.append(c)

        # ---------- Vessel fallbacks ----------
        if not vessels and (VESSEL_HINTS.search(text) or MARITIME_CONTEXT.search(text)):
            for m in VESSEL_NAME_RE.finditer(text):
                name = m.group(0).strip()
                if len(name) >= 3 and not name.lower().startswith(("the ", "a ", "an ")):
                    if _looks_like_ship_name(name, text):
                        vessels.append(name)

        # ---------- Regions from your region_lookup.json ----------
        regions = _extract_regions(text)

        # ---------- Final vessel gating ----------
        vessels = _gate_with_kb(_filter_vessel_candidates(_dedupe(vessels), text))
        ports = _dedupe(ports)
        countries = _dedupe(countries)

        # ---------- Remove pure country names from ports ----------
        country_lc = {c.lower() for c in countries}
        filtered_ports = []
        for p in ports:
            pl = p.lower()
            # drop anything that is exactly a country name / alias
            if pl in COUNTRY_GAZ or pl in country_lc:
                continue
            filtered_ports.append(p)
        ports = _dedupe(filtered_ports)


        return {
            "ports": ports,
            "vessels": vessels,
            "incidents": incidents,
            "countries": countries,
            "regions": _dedupe(regions),
            "imos": _dedupe(imos),
        }

