# inspect_batch.py
import json, glob, os
from datetime import datetime
def show(doc):
    print("—"*60)
    print("title:", doc["title"][:120])
    print("src:", doc["source_id"], "| published_at:", doc["published_at"])
    print("url:", doc["url"])
    print("len(content):", len(doc["content_text"]))
files = sorted(glob.glob("data/normalized/*.json"), key=os.path.getmtime)[-20:]
for f in files:
    with open(f, "r", encoding="utf-8") as fh:
        doc = json.load(fh)
        show(doc)

