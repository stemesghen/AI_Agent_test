# disambiguation/model.py
from __future__ import annotations
from typing import List, Dict, Any, Tuple
from pathlib import Path
import joblib
import numpy as np

FEATURE_ORDER = [
    "sim_name","sim_city","sim_alias","tok_jacc_name","tok_jacc_city","sim_best",
    "country_match_any","country_has_hint",
    "dist_km","dist_inv",
    "ctx_vessel","ctx_region","ctx_harbor_terms",
    "sim_best_x_country","sim_best_x_ctx",
]

def vectorize(features: Dict[str, float]) -> np.ndarray:
    return np.array([float(features.get(k, 0.0)) for k in FEATURE_ORDER], dtype=np.float32)

class PortDisambModel:
    """
    Paper-style ML disambiguation: small linear model trained to pick the right candidate.
    - If no trained model is present, you can fallback to a rules score.
    """
    def __init__(self, model_path: str = "data/models/port_disamb.joblib"):
        self.path = Path(model_path)
        self.model = None
        if self.path.exists():
            self.model = joblib.load(self.path)

    def is_loaded(self) -> bool:
        return self.model is not None

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # Return P(correct) for each row
        if self.model is None:
            # no model — return zeros; caller will use rule-based score
            return np.zeros((X.shape[0],), dtype=np.float32)
        proba = self.model.predict_proba(X)
        # scikit returns 2 columns for binary: [:,1] = P(positive)
        return proba[:, 1]

    # Optional: lightweight trainer (logistic regression)
    def fit_save(self, X: np.ndarray, y: np.ndarray) -> None:
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(max_iter=200, class_weight="balanced")
        clf.fit(X, y)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(clf, self.path)
        self.model = clf
