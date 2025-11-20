# inspect_hits.py
import pandas as pd

# ---- configure the CSV path
CSV_PATH = "data/ports.csv"

# Common header aliases we’ll accept
ALIASES = {
    "country": ["country", "country_iso2", "iso2", "COUNTRY"],
    "locode":  ["locode", "unlocode", "locationisocode", "LOCODE", "UNLOCODE"],
    # coordinate-ish columns we’ll display if present
    "lat_like": ["lat", "latitude", "lat_geonames", "LATITUDE", "Latitude (decimal)"],
    "lon_like": ["lon", "longitude", "lng", "lon_geonames", "LONGITUDE", "Longitude (decimal)"],
    "coords":   ["coords", "coord", "UN/LOCODE coords", "Coordinates", "UNLOCODE_COORDS"],
    "lat_un":   ["lat_un", "un_lat", "lat_dm", "LAT_DM"],
    "lon_un":   ["lon_un", "un_lon", "lon_dm", "LON_DM"],
    "name":     ["name","port_name","location_name","NAME"],
    "city":     ["city","municipality","CITY"],
}

def find_col(df, candidates):
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None

def show_row(df, unloc):
    cc = unloc[:2].upper()
    loc = unloc[2:].upper()  # usually 3 letters, sometimes more

    country_col = find_col(df, ALIASES["country"])
    locode_col  = find_col(df, ALIASES["locode"])

    if not country_col or not locode_col:
        print(f"\n== {unloc} ==")
        print("❌ Could not find country/locode columns in ports.csv.")
        print("   Present columns:", list(df.columns))
        return

    # Normalize comparison series (same index, no reindexing issues)
    cc_series  = df[country_col].astype(str).str.upper().str.strip()
    loc_series = df[locode_col].astype(str).str.upper().str.strip()

    # Some files store full 'CCLLL' (e.g., 'GBIOP') in the LOCODE column.
    # We match either exact 'LLL...' suffix or full 'CCLLL' equal to unloc.
    mask = (cc_series == cc) & (loc_series.str.endswith(loc) | (loc_series == unloc))

    hit = df[mask]
    print(f"\n== {unloc} ==")
    if hit.empty:
        print("No matching row. Try checking your country/locode columns or values.")
        # Show a few sample LOCODEs for the same country to sanity-check
        sample = df[cc_series == cc].head(5)[[country_col, locode_col]]
        if not sample.empty:
            print("Sample rows for same country:")
            print(sample.to_string(index=False))
        return

    row = hit.iloc[0]

    # Gather likely coordinate columns if present
    lat_col = find_col(df, ALIASES["lat_like"])
    lon_col = find_col(df, ALIASES["lon_like"])
    coords_col = find_col(df, ALIASES["coords"])
    latun_col = find_col(df, ALIASES["lat_un"])
    lonun_col = find_col(df, ALIASES["lon_un"])
    name_col  = find_col(df, ALIASES["name"])
    city_col  = find_col(df, ALIASES["city"])

    fields = {
        "country_col": country_col,
        "locode_col":  locode_col,
        "name_col":    name_col,
        "city_col":    city_col,
        "lat_col":     lat_col,
        "lon_col":     lon_col,
        "coords_col":  coords_col,
        "lat_un_col":  latun_col,
        "lon_un_col":  lonun_col,
    }

    print("Matched row (key fields):")
    print({
        "country": row.get(country_col, None),
        "locode":  row.get(locode_col, None),
        "name":    row.get(name_col, None) if name_col else None,
        "city":    row.get(city_col, None) if city_col else None,
        "lat":     row.get(lat_col, None) if lat_col else None,
        "lon":     row.get(lon_col, None) if lon_col else None,
        "coords":  row.get(coords_col, None) if coords_col else None,
        "lat_un":  row.get(latun_col, None) if latun_col else None,
        "lon_un":  row.get(lonun_col, None) if lonun_col else None,
    })
    print("Detected column mapping:", fields)

if __name__ == "__main__":
    df = pd.read_csv(CSV_PATH)

    # Test a few codes you saw in diagnose output
    for code in ["DKROR","BRNPN","USNRH","GBIOP","GBCQY","CIBAO"]:
        show_row(df, code)
