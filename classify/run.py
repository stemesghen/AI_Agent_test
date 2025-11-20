# classify/run.py
from dotenv import load_dotenv
load_dotenv()

import os, json, shutil
from pathlib import Path
from utils import safe_fname

def get_provider():
    provider = os.getenv("LLM_PROVIDER", "mock").lower()
    if provider == "azure":
        from .providers.azure_provider import AzureOpenAIClassifier
        return AzureOpenAIClassifier()
    elif provider == "inhouse":
        from .providers.inhouse_provider import InHouseClassifier  # if/when you add it
        return InHouseClassifier()
    else:
        from .providers.mock_provider import MockClassifier
        return MockClassifier()

# Resolve paths off repo root (parent of this folder)
ROOT            = Path(__file__).resolve().parents[1]
IN_DIR          = ROOT / "data" / "normalized"
OUT_DIR         = ROOT / "data" / "classified"
INCIDENT_DIR    = ROOT / "data" / "is_incident"   # <-- incidents snapshot folder

OUT_DIR.mkdir(parents=True, exist_ok=True)
INCIDENT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    # RECURSIVE: handles nested files (e.g., data/normalized/sha256/*.json)
    files = list(IN_DIR.rglob("*.json"))
    print(f"[classify] IN_DIR={IN_DIR} | OUT_DIR={OUT_DIR} | inputs={len(files)}")

    clf = get_provider()
    total = 0
    incidents = 0

    for fp in files:
        with open(fp, "r", encoding="utf-8") as fh:
            doc = json.load(fh)

        title = doc.get("title", "")
        text  = doc.get("content_text", "") or ""

        safe_id = safe_fname(doc["doc_id"])
        out_path = OUT_DIR / f"{safe_id}.classify.json"

        if out_path.exists():
            # skip if already classified (keeps your current behavior fast)
            continue

        res = clf.classify(title, text)

        out = {
            "doc_id": doc["doc_id"],
            "url": doc.get("url", ""),
            "title": title,
            "published_at": doc.get("published_at", ""),
            "is_incident": bool(res.get("is_incident", False)),
            "incident_types": res.get("incident_types", []),
            "near_miss": bool(res.get("near_miss", False)),
            "confidence": float(res.get("confidence", 0.5)),
            "rationale": res.get("rationale", "azure-llm"),
        }

        with open(out_path, "w", encoding="utf-8") as oh:
            json.dump(out, oh, ensure_ascii=False, indent=2)

        total += 1
        incidents += int(bool(out["is_incident"]))

    print(f"[LLM_PROVIDER={os.getenv('LLM_PROVIDER','mock')}] Classified {total} new docs → {OUT_DIR} | new incidents: {incidents}")

    # ---- Build/refresh incidents snapshot folder ----
    # We rebuild from OUT_DIR so this works even when we skip reclassifying existing items.
    snapshot_written = 0
    keep_names = set()

    for cf in OUT_DIR.glob("*.classify.json"):
        try:
            with open(cf, "r", encoding="utf-8") as fh:
                j = json.load(fh)
        except Exception:
            continue

        if bool(j.get("is_incident", False)):
            dest = INCIDENT_DIR / cf.name
            # Write a fresh copy (atomic replace semantics via temp write is overkill here,
            # but shutil.copyfile is fine since source is stable on disk now)
            with open(dest, "w", encoding="utf-8") as oh:
                json.dump(j, oh, ensure_ascii=False, indent=2)
            keep_names.add(dest.name)
            snapshot_written += 1

    # Remove stale files in INCIDENT_DIR that are no longer incidents in OUT_DIR
    for existing in INCIDENT_DIR.glob("*.classify.json"):
        if existing.name not in keep_names:
            try:
                existing.unlink()
            except Exception:
                pass

    print(f"[classify] Incident snapshot refreshed → {INCIDENT_DIR} | count: {snapshot_written}")

if __name__ == "__main__":
    main()

