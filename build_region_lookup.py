# build_region_lookup.py
import geopandas as gpd
import json
from pathlib import Path

SHAPE_FILE = Path("data/world_seas/World_Seas_IHO_v3.shp")
OUT_JSON = Path("data/region_lookup.json")

def build_region_lookup():
    gdf = gpd.read_file(SHAPE_FILE)
    regions = []
    for _, row in gdf.iterrows():
        name = str(row.get("NAME", "")).strip()
        if not name:
            continue
        geom = row.geometry
        centroid = geom.centroid
        regions.append({
            "name": name,
            "lat": round(centroid.y, 4),
            "lon": round(centroid.x, 4)
        })
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(regions, f, indent=2, ensure_ascii=False)
    print(f"[OK] Saved {len(regions)} regions → {OUT_JSON}")

if __name__ == "__main__":
    build_region_lookup()
