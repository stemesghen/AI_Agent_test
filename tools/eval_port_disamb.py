# tools/eval_port_disamb.py
from __future__ import annotations
import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score


FEATURE_COLS: List[str] = [
    "sim_name",
    "sim_city",
    "sim_alias",
    "tok_jacc_name",
    "tok_jacc_city",
    "sim_best",
    "country_match_any",
    "country_has_hint",
    "dist_km",
    "dist_inv",
    "ctx_vessel",
    "ctx_region",
    "ctx_harbor_terms",
    "sim_best_x_country",
    "sim_best_x_ctx",
]


def main(path: str) -> None:
    df = pd.read_csv(path)

    # keep only rows where label is 0 or 1
    df = df[df["label"].isin([0, 1, "0", "1"])].copy()
    df["label"] = df["label"].astype(int)

    # drop docs that somehow have no positive label
    pos_counts = df.groupby("doc_id")["label"].sum()
    keep_docs = pos_counts[pos_counts > 0].index
    df = df[df["doc_id"].isin(keep_docs)].copy()

    X = df[FEATURE_COLS].fillna(0.0).to_numpy(dtype=float)
    y = df["label"].to_numpy(dtype=int)
    groups = df["doc_id"].to_numpy()

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    docs_test = groups[test_idx]

    clf = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="liblinear",
        max_iter=1000,
    )
    clf.fit(X_train, y_train)

    # predict probabilities for validation
    probs_test = clf.predict_proba(X_test)[:, 1]

    # per-doc top-1 accuracy: did we pick the candidate marked label=1?
    df_val = pd.DataFrame(
        {
            "doc_id": docs_test,
            "label": y_test,
            "prob": probs_test,
        }
    )

    # for each doc_id, find candidate with highest prob
    top_by_doc = df_val.sort_values("prob", ascending=False).groupby("doc_id").head(1)
    # success if that candidate is label=1
    doc_acc = top_by_doc["label"].mean()

    print(f"Per-candidate accuracy (baseline): {accuracy_score(y_test, (probs_test >= 0.5).astype(int)):.3f}")
    print(f"Per-document top-1 accuracy:       {doc_acc:.3f}")
    print(f"Validation docs: {top_by_doc['doc_id'].nunique()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--candidates",
        type=str,
        default="data/extracted/port_disamb_candidates.csv",
        help="Path to labeled candidate CSV.",
    )
    args = ap.parse_args()
    main(args.candidates)
