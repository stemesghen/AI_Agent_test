import re
import unicodedata
from typing import Iterable

GENERIC_PORT_WORDS = {
    "port", "porto", "harbor", "harbour", "terminal", "anchorage", "anchorage area",
    "port-of", "port-of-", "portarea", "port area"
}

_ws_re = re.compile(r"\s+")
_punct_re = re.compile(r"[^\w\s'-]", flags=re.UNICODE)

def strip_accents(s: str) -> str:
    # ASCII fold via NFD
    s = unicodedata.normalize("NFD", s or "")
    return "".join(ch for ch in s if unicodedata.category(ch) != "Mn")

def normalize_text(s: str) -> str:
    """
    Lowercase, ASCII-fold, trim punctuation, collapse spaces.
    """
    s = (s or "").strip()
    s = strip_accents(s)
    s = s.casefold()
    s = _punct_re.sub(" ", s)
    s = _ws_re.sub(" ", s).strip()
    return s

def normalize_unlocode(code: str) -> str:
    code = (code or "").replace(" ", "").upper()
    return code if code else ""

def remove_generic_port_words(tokens: Iterable[str]) -> str:
    out = [t for t in tokens if t and t not in GENERIC_PORT_WORDS]
    return " ".join(out)

def normalize_for_match(s: str) -> str:
    """
    Full normalization for fuzzy match:
      - normalize_text
      - remove generic port words
      - collapse spaces again
    """
    base = normalize_text(s)
    tokens = base.split(" ")
    cleaned = remove_generic_port_words(tokens)
    return _ws_re.sub(" ", cleaned).strip()
