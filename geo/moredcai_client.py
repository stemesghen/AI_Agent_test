# geo/mordecai_client.py

from __future__ import annotations
from typing import Any, Dict
import os

from mordecai3 import Geoparser


# -------------------------------------------------------------------
# Global Geoparser instance
# -------------------------------------------------------------------
# Expensive to initialize (loads spaCy + transformer + model),
# so we create ONE and reuse it.
# -------------------------------------------------------------------

# Allow overriding ES host via env, but default to localhost
ES_HOST = os.getenv("MORDECAI_ES_HOST", "http://localhost:9200")


def _build_geoparser() -> Geoparser:
    """
    Create and configure the Geoparser. In the current Mordecai3
    version, you usually just need default settings, but we keep
    this separated in case we want to tweak config later.
    """
    # Mordecai3 reads ES endpoint from env / config internally.
    # If you ever need to pass options, you can adjust here.
    return Geoparser()


# Single shared instance
_GEO = _build_geoparser()


def geoparse_text(text: str) -> Dict[str, Any]:
    """
    Run Mordecai3 on raw article text and return the full result dict.

    Returns a dict of the form:
    {
        "doc_text": str,
        "event_location_raw": str,
        "geolocated_ents": [ { ... place dicts ... } ]
    }
    """
    return _GEO.geoparse_doc(text)

