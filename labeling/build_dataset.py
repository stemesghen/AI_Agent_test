import json
import pandas as pd
from pathlib import Path

# input folders
NORM_DIR = Path("data/normalized")
LABELS = Path("data/labels/review.csv")

# output folder
OUT = Path("datasets")
OUT.mkdir(parents=True, exist_ok=True)

def main():
    if not LABELS.exists():
        print(" No labels found at data/labels/review.csv. Run the Streamlit app and save a few labels first.")
        return

    gold = pd.read_csv(LABELS)
    rows = []

    for _, r in gold.iterrows():
        # match doc_id from normalized folder
        norm_file = NORM_DIR / f"{r.doc_id}.json"
        if not norm_file.exists():
            continue

        doc = json.load(open(norm_file))
        text = (doc.get("title", "") or "") + "\n" + (doc.get("content_text", "") or "")[:1200]

        rows.append({
            "doc_id": r.doc_id,
            "text": text,
            "is_incident": int(bool(r.is_incident_true)),
            "incident_types": r.incident_types_true or ""
        })

    if not rows:
        print("No matching docs found.")
        return

    df = pd.DataFrame(rows).sample(frac=1.0, random_state=42).reset_index(drop=True)
    n = len(df)
    n_train = int(0.8 * n)
    n_dev = int(0.1 * n)

    df.iloc[:n_train].to_csv(OUT / "train.csv", index=False)
    df.iloc[n_train:n_train + n_dev].to_csv(OUT / "dev.csv", index=False)
    df.iloc[n_train + n_dev:].to_csv(OUT / "test.csv", index=False)

    print(f"Wrote {len(df)} rows → datasets/train.csv, dev.csv, test.csv")

if __name__ == "__main__":
    main()

