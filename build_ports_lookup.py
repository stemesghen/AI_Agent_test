# build_prts.py
import pandas as pd
import numpy as np
import math, re
from unicodedata import normalize as ucn

pd.set_option("future.no_silent_downcasting", True)

# -----------------------------
# Config / file paths
# -----------------------------
GEONAMES_CSV = "data/raw/geonames_labeled.csv"     # your labeled GeoNames
UNLOCODE_CSV = "data/raw/unlocode_labeled.csv"     # official or labeled UN/LOCODE
OUT_CSV      = "data/port.csv"

# -----------------------------
# Helpers
# -----------------------------
def to_ascii_lower(s: str) -> str:
    if s is None or (isinstance(s, float) and np.isnan(s)): return ""
    s = ucn("NFKD", str(s)).encode("ascii", "ignore").decode("ascii")
    s = s.lower().strip()
    s = re.sub(r"[\s\-_]+", " ", s)
    return s

def make_coords(lat, lon):
    try: return f"{float(lat):.6f},{float(lon):.6f}"
    except: return ""

def haversine(lat1, lon1, lat2, lon2):
    try: lat1, lon1, lat2, lon2 = map(float, (lat1, lon1, lat2, lon2))
    except: return np.nan
    R=6371000.0
    p1,p2 = math.radians(lat1), math.radians(lat2)
    dphi  = math.radians(lat2-lat1)
    dl    = math.radians(lon2-lon1)
    a = math.sin(dphi/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2*R*math.asin(math.sqrt(a))

def name_tokens(s):
    s = s or ""
    return set(t for t in re.split(r"[ ,/()]+", s) if t)

def jaccard(a, b):
    if not a or not b: return 0.0
    return len(a & b) / len(a | b)

def parse_degmin(s: str):
    """UN/LOCODE '4229N 00129E' -> ('42.483333','1.483333'); else ('','')"""
    if not isinstance(s, str): return "", ""
    s = s.strip().upper()
    m = re.match(r"^\s*(\d{2})(\d{2})([NS])\s+(\d{3})(\d{2})([EW])\s*$", s)
    if not m: return "", ""
    la, lb, lh, oa, ob, oh = m.groups()
    lat = int(la) + int(lb)/60
    lon = int(oa) + int(ob)/60
    if lh == "S": lat = -lat
    if oh == "W": lon = -lon
    return f"{lat:.6f}", f"{lon:.6f}"

# -----------------------------
# 1) Load GeoNames (labeled CSV)
# -----------------------------
geo_needed = [
    "geonameid","name","asciiname","alternatenames",
    "latitude","longitude","feature_class","feature_code",
    "country_code","admin1_code","admin2_code"
]
g_full = pd.read_csv(GEONAMES_CSV, dtype=str, low_memory=False).fillna("")
missing = [c for c in geo_needed if c not in g_full.columns]
if missing:
    raise ValueError(f"[GeoNames] Missing columns: {missing}\nHave: {list(g_full.columns)}")

g = g_full[geo_needed].copy()

# Keep cities & harbors
g = g[(g["feature_class"].eq("P")) |
      (g["feature_class"].eq("H") & g["feature_code"].eq("HBR"))].copy()

# Normalize
g["name_ascii"]     = g["asciiname"].replace("", np.nan).fillna(g["name"]).map(to_ascii_lower)
g["alt_names_norm"] = g["alternatenames"].map(to_ascii_lower)
g["coords"]         = [make_coords(a,b) for a,b in zip(g["latitude"], g["longitude"])]
g["name_tokens"]    = g["name_ascii"].map(name_tokens)
g["country_code"]   = g["country_code"].str.upper()

print(f"[GeoNames] rows kept: {len(g)} / {len(g_full)}")
print(g[["name","country_code","feature_class","feature_code"]].head(3).to_string(index=False))

# -----------------------------
# 2) Load UN/LOCODE (official or labeled)
# -----------------------------
def read_unlocode_any(path: str) -> pd.DataFrame:
    for enc in ("utf-8-sig", "latin1"):
        try:
            return pd.read_csv(path, dtype=str, low_memory=False, encoding=enc).fillna("")
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, dtype=str, low_memory=False).fillna("")

u_raw = read_unlocode_any(UNLOCODE_CSV)
cols = [c.strip() for c in u_raw.columns]
u_raw.columns = cols

official_cols = {"Country","Location","Name","NameWoDiacritics","SubDiv","Function","Status","Date","IATA","Coordinates","Remarks"}
pipeline_cols = {"unlocode","location_name","name_ascii","country_iso2","subdivision/admin",
                 "ims_facility_id","functions/flags","aliases","latitude","longitude"}

if official_cols.issubset(set(cols)):
    # Map official → pipeline; parse Coordinates to lat/lon
    lat_list, lon_list = zip(*[parse_degmin(v) for v in u_raw["Coordinates"]])
    u = pd.DataFrame({
        "unlocode":          (u_raw["Country"].str.upper().str.strip() + u_raw["Location"].str.upper().str.strip()),
        "location_name":     u_raw["Name"].str.strip(),
        "name_ascii":        u_raw["NameWoDiacritics"].replace("", pd.NA).fillna(u_raw["Name"]).str.strip(),
        "country_iso2":      u_raw["Country"].str.upper().str.strip(),
        "subdivision/admin": u_raw["SubDiv"].str.strip(),
        "ims_facility_id":   "",
        "functions/flags":   u_raw["Function"].str.strip(),
        "aliases":           "",
        "latitude":          list(lat_list),
        "longitude":         list(lon_list),
    }).fillna("")
elif pipeline_cols.issubset(set(cols)):
    u = u_raw[list(pipeline_cols)].copy()
else:
    raise ValueError(
        f"[UN/LOCODE] Unknown header schema. Have: {cols[:20]}\n"
        "Expected either official UN/LOCODE headers "
        "['Country','Location','Name','NameWoDiacritics','SubDiv','Function','Status','Date','IATA','Coordinates','Remarks'] "
        "or pipeline headers "
        "['unlocode','location_name','name_ascii','country_iso2','subdivision/admin','ims_facility_id','functions/flags','aliases','latitude','longitude']"
    )

# Normalize
u["name_ascii"]   = u["name_ascii"].replace("", pd.NA).fillna(u["location_name"]).map(to_ascii_lower)
u["country_iso2"] = u["country_iso2"].str.upper()
u["aliases_norm"] = u["aliases"].map(to_ascii_lower)
u["coords"]       = [make_coords(a,b) for a,b in zip(u["latitude"], u["longitude"])]
u["name_tokens"]  = u["name_ascii"].map(name_tokens)

n_nonempty_unloc = (u["unlocode"].str.len() > 0).sum()
if n_nonempty_unloc == 0:
    raise ValueError("[UN/LOCODE] All unlocode values are empty after mapping.")

print(f"[UN/LOCODE] rows: {len(u)} (with unlocode present: {n_nonempty_unloc})")
print(u[["unlocode","location_name","country_iso2"]].head(3).to_string(index=False))

# -----------------------------
# 3) Match by country + name (+ distance bonus)
# -----------------------------
g_by_ctry = {ctry: df.reset_index(drop=True) for ctry, df in g.groupby("country_code")}
matches = []

for _, row in u.iterrows():
    ctry = row["country_iso2"]
    cand_df = g_by_ctry.get(ctry)
    if cand_df is None or cand_df.empty:
        continue

    left_tokens = row["name_tokens"]
    left_name   = row["name_ascii"]

    scores = []
    for j, crow in cand_df.iterrows():
        right_tokens = crow["name_tokens"]
        s = jaccard(left_tokens, right_tokens)

        rn = crow["name_ascii"]
        if left_name == rn: s += 0.4
        elif left_name.startswith(rn) or rn.startswith(left_name): s += 0.2
        elif left_name in rn or rn in left_name: s += 0.1

        d = np.nan
        if row["coords"] and crow["coords"]:
            try:
                lat1, lon1 = map(float, row["coords"].split(","))
                lat2, lon2 = map(float, crow["coords"].split(","))
                d = haversine(lat1, lon1, lat2, lon2)
                if not np.isnan(d):
                    if d <= 5000:   s += 0.25
                    elif d <= 15000: s += 0.10
            except: pass

        scores.append((s, d, j))

    if not scores:
        continue

    scores.sort(key=lambda x: x[0], reverse=True)
    for s, d, j in scores[:3]:
        crow = cand_df.loc[j]
        matches.append({
            # UN/LOCODE side
            "unlocode": row["unlocode"],
            "ims_facility_id": row["ims_facility_id"],
            "location_name": row["location_name"],
            "name_ascii_unlocode": row["name_ascii"],
            "country_iso2": row["country_iso2"],
            "subdivision/admin": row["subdivision/admin"],
            "functions/flags": row["functions/flags"],
            "aliases_unlocode": row["aliases"],
            "lat_unlocode": row["latitude"],
            "lon_unlocode": row["longitude"],
            "coords_unlocode": row["coords"],
            # GeoNames side
            "geonameid": crow["geonameid"],
            "name_geonames": crow["name"],
            "asciiname_geonames": crow["asciiname"],
            "alternatenames": crow["alternatenames"],
            "feature_class": crow["feature_class"],
            "feature_code": crow["feature_code"],
            "admin1_code": crow["admin1_code"],
            "admin2_code": crow["admin2_code"],
            "lat_geonames": crow["latitude"],
            "lon_geonames": crow["longitude"],
            "coords_geonames": crow["coords"],
            # diagnostics
            "name_score": round(s, 3),
            "distance_m": np.nan if np.isnan(d) else int(d),
        })

match_df = pd.DataFrame(matches)

# -----------------------------
# 4) Union output (keep both sources' coords, no "preferred" lat/lon)
# -----------------------------
base = u.assign(
    name_ascii_unlocode=u["name_ascii"],
    aliases_unlocode=u["aliases"],
    lat_unlocode=u["latitude"],
    lon_unlocode=u["longitude"],
    coords_unlocode=u["coords"],
)[[
    "unlocode","ims_facility_id","location_name","name_ascii_unlocode",
    "country_iso2","subdivision/admin","functions/flags","aliases_unlocode",
    "lat_unlocode","lon_unlocode","coords_unlocode"
]]

if not match_df.empty:
    out = pd.concat([base, match_df], ignore_index=True, sort=False)
else:
    out = base.copy()
    for c in ["geonameid","name_geonames","asciiname_geonames","alternatenames",
              "feature_class","feature_code","admin1_code","admin2_code",
              "lat_geonames","lon_geonames","coords_geonames",
              "name_score","distance_m"]:
        out[c] = ""

# Optional: a quick provenance tag for coords presence
def coord_src(row):
    has_u = bool(str(row.get("lat_unlocode","")).strip()) and bool(str(row.get("lon_unlocode","")).strip())
    has_g = bool(str(row.get("lat_geonames","")).strip()) and bool(str(row.get("lon_geonames","")).strip())
    if has_u and has_g: return "both"
    if has_u: return "unlocode"
    if has_g: return "geonames"
    return "none"

out["coord_source"] = out.apply(coord_src, axis=1)

# Final columns (no preferred lat/lon)
final_cols = [
    # identity
    "unlocode","ims_facility_id","location_name","name_ascii_unlocode","country_iso2","subdivision/admin",
    # aliases
    "aliases_unlocode","alternatenames",
    # functions
    "functions/flags",
    # UN/LOCODE coords
    "lat_unlocode","lon_unlocode","coords_unlocode",
    # GeoNames coords
    "lat_geonames","lon_geonames","coords_geonames",
    # source detail
    "geonameid","name_geonames","asciiname_geonames","feature_class","feature_code",
    "admin1_code","admin2_code",
    # diagnostics
    "name_score","distance_m",
    # provenance
    "coord_source",
]
for c in final_cols:
    if c not in out.columns: out[c] = ""

out[final_cols].to_csv(OUT_CSV, index=False)
print(f"[OK] Wrote {len(out)} rows → {OUT_CSV}")
