from __future__ import annotations
import re
import unicodedata
from typing import List, Iterable, Set

# Very conservative stopword list for location spans
STOPWORDS: Set[str] = {
    "the", "a", "an", "and", "or", "of", "in", "on", "at", "to", "for",
    "by", "with", "from", "as", "is", "are", "was", "were", "this", "that",
    "these", "those", "it", "its", "his", "her", "their", "our", "your",
    "since", "during", "after", "before", "about", "over", "under",
    "into", "out", "up", "down"
}

# Allow demonyms like "Chinese", "Greek" to pass for country mapping
ALLOW_DEMONYM_SUFFIXES = ("ese", "ish", "ian", "i", "ic")  # Chinese, Turkish, Russian, Somali, Nordic


def _strip_punct(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    # remove leading/trailing punctuation
    return re.sub(r"^[^\w]+|[^\w]+$", "", s)


def _looks_like_word_token(s: str) -> bool:
    # reject anything with digits or internal punctuation (commas, periods, etc)
    if not s:
        return False
    if any(ch.isdigit() for ch in s):
        return False
    if re.search(r"[.,;:/\\]", s):
        return False
    return True


def _is_all_caps(s: str) -> bool:
    # treat "US", "UK" etc as valid 2-letter codes
    if len(s) <= 3 and s.isupper():
        return True
    return False


def is_valid_location_span(
    span: str,
    *,
    allow_demonyms: bool = True,
    min_len: int = 3,
) -> bool:
    """
    Very strict filter for NER / heuristic spans before geocoding.
    """
    if not isinstance(span, str):
        return False

    span = span.strip()
    if not span:
        return False

    # Kill obviously bad long fragments / sentences
    if len(span) > 80:
        return False
    if " " in span and span.count(" ") > 6:
        # too many words → probably a sentence
        return False

    # Normalize and check each token
    tokens = [t for t in span.split() if t]
    if not tokens:
        return False

    # Single word rules
    if len(tokens) == 1:
        tok = _strip_punct(tokens[0])
        if not tok:
            return False

        # Lowercase single words are almost never locations
        if tok.islower():
            return False

        low = tok.lower()

        if len(low) < min_len and not _is_all_caps(tok):
            return False

        if low in STOPWORDS:
            return False

        if not _looks_like_word_token(tok):
            return False

        # allow short all-caps like "US", "UK"
        if _is_all_caps(tok):
            return True

        # allow demonyms for country mapping ("Chinese", "Greek", etc.)
        if allow_demonyms and low.endswith(ALLOW_DEMONYM_SUFFIXES):
            return True

        # Capitalized word like "Shanghai", "Baltic", etc.
        return tok[0].isupper()

    # Multi-word rules
    # Require at least one capitalized token
    caps = [t for t in tokens if _strip_punct(t) and _strip_punct(t)[0].isupper()]
    if not caps:
        return False

    # Reject if majority of tokens are stopwords
    non_stop = [t for t in tokens if _strip_punct(t).lower() not in STOPWORDS]
    if not non_stop:
        return False

    # Reject if span clearly looks like part of a sentence
    if any(tok.endswith(".") for tok in tokens[:-1]):
        return False

    # Reject if contains digits or weird punctuation
    joined = " ".join(tokens)
    if not _looks_like_word_token(joined):
        return False

    return True


def clean_spans(spans: Iterable[str], *, allow_demonyms: bool = True) -> List[str]:
    """
    Deduplicate + aggressively filter spans.
    """
    seen = set()
    out: List[str] = []

    for raw in spans:
        if not isinstance(raw, str):
            continue
        s = raw.strip()
        if not s:
            continue

        # Quick collapse of whitespace
        s = re.sub(r"\s+", " ", s)

        if not is_valid_location_span(s, allow_demonyms=allow_demonyms):
            continue

        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(s)

    return out
