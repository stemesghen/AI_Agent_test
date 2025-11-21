#!/usr/bin/env python
"""
Helpers to use Mordecai output (mordecai_ims_hits.csv) as context
for port resolution.

- Group hits by doc_id
- Extract country ISO2 per doc
- Optionally admin1 names per doc
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Any, Set

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
MORDECAI_CSV = BASE_DIR / "data" / "extracted" / "mordecai_ims_hits.csv"


def load_mordecai_hits(path: Path | None = None) -> pd.DataFrame:
    path = path or MORDECAI_CSV
    df = pd.read_csv(path, dtype={"start_char": int, "end_char": int})
    return df


def build_doc_context(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Returns:
      {
        doc_id: {
          "countries": ["CN","MY","RU"],
          "admin1_names": ["Irkutsk Oblast", ...],
        },
        ...
      }
    """
    ctx: Dict[str, Dict[str, Any]] = {}
    for doc_id, grp in df.groupby("doc_id"):
        countries: Set[str] = set()
        admin1: Set[str] = set()

        for _, row in grp.iterrows():
            # feature_class 'A' covers countries and admin regions
            if row.get("feature_class") == "A":
                c2 = (row.get("country_iso2") or "").strip().upper()
                if c2:
                    countries.add(c2)

                a1 = (row.get("admin1_name") or "").strip()
                if a1:
                    admin1.add(a1)

        ctx[doc_id] = {
            "countries": sorted(countries),
            "admin1_names": sorted(admin1),
        }

    return ctx


def main():
    df = load_mordecai_hits()
    ctx = build_doc_context(df)
    print(f"[MORDECAI] Built context for {len(ctx)} docs")
    # Example: print first few
    for i, (doc_id, info) in enumerate(ctx.items()):
        print(doc_id, "countries=", info["countries"], "admin1=", info["admin1_names"])
        if i >= 4:
            break


if __name__ == "__main__":
    main()

