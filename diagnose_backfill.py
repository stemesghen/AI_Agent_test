# diagnose_backfill.py
import json, pandas as pd
from pathlib import Path
from difflib import SequenceMatcher

IMS = Path("data/ports_lookup_enriched_with_region.json")
CSV = Path("data/ports.csv")

def norm_text(s: str) -> str:
    import unicodedata, re
    s = str(s or "")
    s = unicodedata.normalize("NFKD", s).encode("ascii","ignore").decode("ascii")
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]+"," ", s)
    s = s.replace(" port of ", " ")
    return " ".join(s.split())

def fuzzy(a,b): return SequenceMatcher(None,a,b).ratio()
def jacc(a,b):
    A,B=set(a.split()),set(b.split())
    return len(A & B)/max(1,len(A|B))

# Load
ims = json.loads(IMS.read_text(encoding="utf-8"))
df  = pd.read_csv(CSV)

# Try to infer/normalize columns in CSV
def first(df, names):
    cols = {c.lower():c for c in df.columns}
    for n in names:
        if n.lower() in cols: return cols[n.lower()]
    return None

c_country = first(df, ["country","country_iso2","iso2"])
c_unloc   = first(df, ["locode","unlocode","locationisocode"])
c_name    = first(df, ["name","port_name","location_name"])
c_city    = first(df, ["city","municipality"])
for need, col in [("country",c_country),("locode",c_unloc),("name",c_name)]:
    if not col: print(f"[WARN] ports.csv missing a '{need}'-like column")

# Build helpers
df["_country"] = df[c_country].astype(str).str.upper() if c_country else ""
df["_locode"]  = df[c_unloc].astype(str).str.upper() if c_unloc else ""
df["_name"]    = df[c_name].astype(str) if c_name else ""
df["_name_n"]  = df["_name"].map(norm_text)
df["_city_n"]  = df[c_city].astype(str).map(norm_text) if c_city else ""

def best_in_country(country_iso2, name_norm):
    cand = df[df["_country"]==str(country_iso2 or "").upper()].copy()
    if cand.empty: cand = df.copy()
    if cand.empty: return None
    cand["score"] = 0.6*cand["_name_n"].map(lambda x: jacc(x,name_norm)) + \
                    0.4*cand["_name_n"].map(lambda x: fuzzy(x,name_norm))
    cand = cand.sort_values("score", ascending=False)
    return cand.iloc[0][["_country","_locode","_name","score"]].to_dict()

# Diagnose first 10 IMS rows that need filling
todo = [r for r in ims if r.get("lat") is None or r.get("lon") is None][:10]
for i, r in enumerate(todo, 1):
    cc   = (r.get("country_iso2") or "").upper()
    name = r.get("location_name") or ""
    off  = (r.get("official_unlocode") or "").upper()
    ims_id = r.get("ims_facility_id")
    name_n = norm_text(name)
    print(f"\n[{i}] IMS: id={ims_id} cc={cc} name='{name}' offUN='{off}' name_n='{name_n}'")
    # Try derive locode from official_unlocode
    if off and len(off)>=5:
        key_cc, key_loc = off[:2], off[2:]
        hit = df[(df["_country"]==key_cc) & (df["_locode"]==key_loc)]
        print("  direct_official:", "HIT" if not hit.empty else "MISS", 
              f"({key_cc}+{key_loc}) rows={len(hit)}")
    # Show best fuzzy in-country
    best = best_in_country(cc, name_n)
    print("  best_in_country:", best)
