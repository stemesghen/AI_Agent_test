# tools/train_port_disamb.py
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# allow importing your project modules
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from disambiguation.model import PortDisambModel, FEATURE_ORDER

CANDIDATES_CSV = Path("data/extracted/port_disamb_candidates_2.csv")
MODEL_PATH     = "data/models/port_disamb.joblib"

def main():
    if not CANDIDATES_CSV.exists():
        raise SystemExit(f"Missing {CANDIDATES_CSV} – run extract.run with DUMP_TRAINING_CANDIDATES=1 first.")

    df = pd.read_csv(CANDIDATES_CSV)

    # keep only labeled rows (label must be 0 or 1)
    df = df[df["label"].isin([0, 1, "0", "1"])]
    if df.empty:
        raise SystemExit("No labeled rows found – fill the 'label' column first (0/1).")

    # convert label to int
    df["label"] = df["label"].astype(int)

    # build X matrix in the same order as FEATURE_ORDER
    missing = [c for c in FEATURE_ORDER if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing feature columns in candidates CSV: {missing}")

    X = df[FEATURE_ORDER].astype(float).to_numpy(dtype=np.float32)
    y = df["label"].to_numpy(dtype=np.int32)

    print(f"[TRAIN] examples: {X.shape[0]} | features: {X.shape[1]}")

    model = PortDisambModel(MODEL_PATH)
    model.fit_save(X, y)

    print(f"[TRAIN] Saved model to {MODEL_PATH}")

if __name__ == "__main__":
    main()
