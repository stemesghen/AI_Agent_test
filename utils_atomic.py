# utils_atomic.py
import os
import json
import tempfile
import pandas as pd

def to_csv_atomic(df: pd.DataFrame, path: str) -> None:
    """Write CSV atomically: write to a temp file then os.replace."""
    d = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", dir=d, text=True)
    os.close(fd)
    try:
        df.to_csv(tmp, index=False)
        os.replace(tmp, path)
    finally:
        try:
            os.remove(tmp)
        except Exception:
            pass

def to_json_atomic(obj, path: str, **kwargs) -> None:
    """Write JSON atomically: write to a temp file then os.replace."""
    d = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", dir=d, text=True)
    os.close(fd)
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2, **kwargs)
        os.replace(tmp, path)
    finally:
        try:
            os.remove(tmp)
        except Exception:
            pass
