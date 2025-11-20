# labeling/auto_label_rules.py
import json, glob, re, pandas as pd
from pathlib import Path

CLS_DIR = Path("data/classified")
NORM_DIR = Path("data/normalized")
LABELS = Path("data/labels/review.csv")
LABELS.parent.mkdir(parents=True, exist_ok=True)

INCIDENT_RE = re.compile(r"\b(ground(?:ed|ing)|collision|collided|allision|fire|blaze|on fire|piracy|pirate|hijack(?:ed|ing)|storm|hurricane|typhoon|cyclone|gale|rough seas|port\s+closure|strike|walkout|industrial action|spill|leak)\b", re.I)
NON_INCIDENT_RE = re.compile(r"\b(sanction|share[s]?\s+hit|tariff|fee|forecast|earning|market|profit|deal|acquisition)\b", re.I)

def norm_for(cf_path: Path) -> Path:
    return NORM_DIR / cf_path.name.replace(".classify.json", ".json")

rows = []
for cf in glob.glob(str(CLS_DIR / "*.classify.json")):
    cf = Path(cf)
    cls = json.load(open(cf))
    title = cls.get("title", "") or ""
    # pull full text from normalized file
    nf = norm_for(cf)
    content = ""
    if nf.exists():
        norm = json.load(open(nf))
        content = (norm.get("content_text", "") or "")[:2000]
    text_all = f"{title}\n{content}"

    if INCIDENT_RE.search(text_all) and not NON_INCIDENT_RE.search(text_all):
        is_incident_true = True
    elif NON_INCIDENT_RE.search(text_all) and not INCIDENT_RE.search(text_all):
        is_incident_true = False
    else:
        # uncertain -> skip (you’ll label these in the UI)
        continue

    # quick incident type heuristics (optional)
    incident_types = []
    for k, pat in {
        "grounding": r"\bground(?:ed|ing)\b",
        "collision": r"\b(collid(?:ed|e|ing)|collision|allision)\b",
        "fire": r"\b(fire|blaze|on fire)\b",
        "piracy": r"\b(piracy|pirate|hijack(?:ed|ing))\b",
        "weather": r"\b(storm|hurricane|typhoon|cyclone|gale|rough seas)\b",
        "port_closure": r"\bport\s+closure\b",
        "strike": r"\b(strike|walkout|industrial action)\b",
        "spill": r"\b(spill|leak)\b",
    }.items():
        if re.search(pat, text_all, re.I):
            incident_types.append(k)

    rows.append({
        "doc_id": cls["doc_id"],
        "is_incident_true": is_incident_true,
        "incident_types_true": ",".join(incident_types),
        "vessel_true": "",
        "imo_true": "",
        "port_true": "",
        "date_true": "",
        "notes": "auto-rule",
    })

if rows:
    df = pd.DataFrame(rows)
    # If review.csv already exists, append without dupes on doc_id
    if LABELS.exists():
        cur = pd.read_csv(LABELS)
        df = pd.concat([cur[~cur.doc_id.isin(df.doc_id)], df], ignore_index=True)
    df.to_csv(LABELS, index=False)
    print(f" wrote {len(df)} labels → {LABELS}")
else:
    print("no matches found – check regex or that normalized files exist.")

