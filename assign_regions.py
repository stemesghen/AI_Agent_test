import json
import math
import geopandas as gpd
from shapely.geometry import Point
from pathlib import Path

# -----------------------------
# CONFIG PATHS
# -----------------------------
ENRICHED_IN = Path("data/ports_lookup_enriched.json")
SEAS_SHP    = Path("data/world_seas/World_Seas_IHO_v3.shp")
FINAL_OUT   = Path("data/ports_lookup_enriched_with_region.json")

# -----------------------------
# HELPERS
# -----------------------------
def is_bad_num(x):
    """Check if a numeric value is invalid (None, NaN)."""
    return (
        x is None or
        (isinstance(x, float) and math.isnan(x))
    )

def load_ports(enriched_path):
    """Load enriched JSON with IMS + UNLOCODE match results."""
    with open(enriched_path, "r", encoding="utf-8") as f:
        ports = json.load(f)
    print(f"[INFO] Loaded {len(ports)} enriched ports")
    return ports

def ports_to_gdf(ports):
    """Convert ports list to GeoDataFrame of points."""
    records = []
    for row in ports:
        lat = row.get("lat")
        lon = row.get("lon")

        if is_bad_num(lat) or is_bad_num(lon):
            geometry = None
        else:
            geometry = Point(float(lon), float(lat))  # shapely expects (x=lon, y=lat)

        records.append({
            "ims_facility_id": row.get("ims_facility_id"),
            "lat": lat,
            "lon": lon,
            "row_data": row,
            "geometry": geometry
        })

    gdf = gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326")
    print(f"[INFO] Created GeoDataFrame with {len(gdf)} rows")
    return gdf

def load_seas(seas_path):
    """Load the IHO Sea Areas shapefile and normalize columns."""
    seas = gpd.read_file(seas_path)
    print(f"[INFO] Loaded {len(seas)} sea polygons")

    # CRS check (make sure both layers use lat/lon)
    seas = seas.to_crs(epsg=4326)

    # For the IHO file, the sea name column is 'IHO_SEA'
    if "IHO_SEA" in seas.columns:
        seas = seas.rename(columns={"IHO_SEA": "region_name"})
    else:
        # Try fallback options if column name differs
        candidate_cols = ["NAME", "SEA_NAME", "name", "Region"]
        found = next((c for c in candidate_cols if c in seas.columns), None)
        if not found:
            raise ValueError(f"No sea name column found in shapefile. Columns: {list(seas.columns)}")
        seas = seas.rename(columns={found: "region_name"})

    return seas[["region_name", "geometry"]]

def spatial_join_ports_with_seas(ports_gdf, seas_gdf):
    """Spatially join ports (points) to seas (polygons)."""
    joined = gpd.sjoin(
        ports_gdf,
        seas_gdf,
        how="left",
        predicate="within"  # checks which polygon each point is inside
    )
    print(f"[INFO] Spatial join complete — matched {joined['region_name'].notna().sum()} ports to regions")
    return joined

def write_final(joined_gdf, final_path):
    """Write JSON with updated region field."""
    final_records = []
    for _, rec in joined_gdf.iterrows():
        row_data = rec["row_data"].copy()
        region_detected = rec.get("region_name", None)

        if isinstance(region_detected, float) and math.isnan(region_detected):
            region_detected = None

        if region_detected:
            row_data["region"] = region_detected  # e.g., "Baltic Sea"

        final_records.append(row_data)

    with open(final_path, "w", encoding="utf-8") as out:
        json.dump(final_records, out, ensure_ascii=False, indent=2)

    print(f"[OK] Wrote {len(final_records)} rows → {final_path}")

# -----------------------------
# MAIN
# -----------------------------
def main():
    ports_raw = load_ports(ENRICHED_IN)
    ports_gdf = ports_to_gdf(ports_raw)
    seas_gdf  = load_seas(SEAS_SHP)
    joined    = spatial_join_ports_with_seas(ports_gdf, seas_gdf)
    write_final(joined, FINAL_OUT)

if __name__ == "__main__":
    main()
