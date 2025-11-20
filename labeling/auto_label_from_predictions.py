# labeling/auto_label_from_predictions.py
import json, glob, pandas as pd
from pathlib import Path

CLS_DIR = Path("data/classified")
LABELS = Path("data/labels/review.csv")
LABELS.parent.mkdir(parents=True, exist_ok=True)

rows = []
for f in glob.glob(str(CLS_DIR / "*.classify.json")):
    j = json.load(open(f))
    rows.append({
        "doc_id": j["doc_id"],
        "is_incident_true": bool(j.get("is_incident", False)),
        "incident_types_true": ",".join(j.get("incident_types", [])),
        "vessel_true": "",
        "imo_true": "",
        "port_true": "",
        "date_true": (j.get("published_at","")[:10] or ""),
        "notes": "seed-from-pred"
    })

df = pd.DataFrame(rows)
if LABELS.exists():
    cur = pd.read_csv(LABELS)
    df = pd.concat([cur[~cur.doc_id.isin(df.doc_id)], df], ignore_index=True)
df.to_csv(LABELS, index=False)
print(f"seeded {len(df)} labeled rows → {LABELS}")

