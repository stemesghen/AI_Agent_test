import requests
import json
from datetime import datetime, timedelta, timezone

# ====================================================
# CONFIGURATION
# ====================================================



BASE_URL = "https://api.lloydslistintelligence.com/v1"
AUTH_TOKEN = "eyJhbGciOiJSUzI1NiIsImtpZCI6ImEzck1VZ01Gdjl0UGNsTGE2eUYzekFrZnF1RSIsIng1dCI6ImEzck1VZ01Gdjl0UGNsTGE2eUYzekFrZnF1RSIsInR5cCI6ImF0K2p3dCJ9.eyJpc3MiOiJodHRwOi8vbGxveWRzbGlzdGludGVsbGlnZW5jZS5jb20iLCJuYmYiOjE3NjA0NzM3MTksImlhdCI6MTc2MDQ3MzcxOSwiZXhwIjoxNzYzMDY1NzE5LCJzY29wZSI6WyJsbGl3ZWJhcGkiXSwiYW1yIjpbImN1c3RvbWVyQXBpX2dyYW50Il0sImNsaWVudF9pZCI6IkN1c3RvbWVyQXBpIiwic3ViIjoiaW1zQGluc3VyaXR5LmNvbSIsImF1dGhfdGltZSI6MTc2MDQ3MzcxOSwiaWRwIjoic2FsZXNmb3JjZSIsImFjY2Vzc1Rva2VuIjoiMDBEOGQwMDAwMDlvaTM4IUFRRUFRSFlTLmRiVl9Za3UubThJMjBESHRyNXlIUG5Icks3QVNMelhSMEJHTHFQSHcwVzF0UG5hcXRuZ1hfUno4d2g2QUM3M01kVy5hempuNk9GS3FmemxRczFwM3dSTCIsInNlcnZpY2VJZCI6IiIsImVudGl0bGVtZW50VHlwZSI6IiIsImFjY291bnROYW1lIjoiIiwidXNlcm5hbWUiOiJpbXNAaW5zdXJpdHkuY29tIiwidXNlcklkIjoiMDA1TnowMDAwMEd0Z0RWSUFaIiwiY29udGFjdEFjY291bnRJZCI6IjAwMThkMDAwMDBrZ0I0cEFBRSIsInVzZXJUeXBlIjoiQ3NwTGl0ZVBvcnRhbCIsImVtYWlsIjoiaW1zQGluc3VyaXR5LmNvbSIsImdpdmVuX25hbWUiOiJJTVMiLCJmYW1pbHlfbmFtZSI6IkFQSSIsInNoaXBUbyI6IiIsImp0aSI6IkQyQjg2NEQ4MTM5QkQ3N0YxNDc3Qzg1RUExREZGNDU1In0.HmIGF4mRBM5M5hXR79LbC0ChzzZLlqj1lEXe-BtlT20PNN8XskQPOUoahC8FdHufB7FD7kcGiczRU7_iyIljMf6OTxZovJ7TTlT0kM2jX91yLZY8dVByOe-pJ_ViUr8CgCgRLDT-Hm5C47mbC6hRa2RyLf7SKPIVJhBQGDiIfNTD1aVTZ_rNTtok4o_xyEzjq-M63vPjLH5sL5WbInYe39_s0vPTl-my5XPCPQ_R5LEqgUAyDT7PV6JBS3i4LpTinrOz21kgNSUAYBn6qtIyLAu7lhSUfNJT5YpkN5sgaOvXrDQkJgnD4Hsp_C8qv8gFk_Z_EO7sm3JAZEE_SjpDNQ"

# Port coordinates (example: Aarhus)
LAT = 56.15716
LON = 10.22741

# Search radius and time window
DELTA_DEG = 1.80       # ~20 km box around the port
HOURS_BACK = 24 * 30   # last 7 days

# ====================================================
# BUILD TIME WINDOW
# ====================================================
end_time = datetime.now(timezone.utc)
start_time = end_time - timedelta(hours=HOURS_BACK)
received_after = start_time.strftime("%Y-%m-%dT%H:%M:%SZ")
received_before = end_time.strftime("%Y-%m-%dT%H:%M:%SZ")

# ====================================================
# BUILD POLYGON (RIGHT-HAND-RULE / ANTI-CLOCKWISE)
# ====================================================
polygon = {
    "type": "Polygon",
    "coordinates": [[
        [LON - DELTA_DEG, LAT - DELTA_DEG],  # bottom-left
        [LON + DELTA_DEG, LAT - DELTA_DEG],  # bottom-right
        [LON + DELTA_DEG, LAT + DELTA_DEG],  # top-right
        [LON - DELTA_DEG, LAT + DELTA_DEG],  # top-left
        [LON - DELTA_DEG, LAT - DELTA_DEG]   # close ring
    ]]
}

# ====================================================
# BUILD QUERY PARAMETERS
# ====================================================
params = {
    "messageFormat": "decoded",
    "receivedAfter": received_after,
    "receivedBefore": received_before,
    "position": json.dumps(polygon),
    "landFilter": "false",   # INCLUDE land positions
    "cleansed": "true"       # return deduplicated AIS messages
}

# ====================================================
# BUILD HEADERS
# ====================================================
headers = {
    "Authorization": "eyJhbGciOiJSUzI1NiIsImtpZCI6ImEzck1VZ01Gdjl0UGNsTGE2eUYzekFrZnF1RSIsIng1dCI6ImEzck1VZ01Gdjl0UGNsTGE2eUYzekFrZnF1RSIsInR5cCI6ImF0K2p3dCJ9.eyJpc3MiOiJodHRwOi8vbGxveWRzbGlzdGludGVsbGlnZW5jZS5jb20iLCJuYmYiOjE3NjA0NzM3MTksImlhdCI6MTc2MDQ3MzcxOSwiZXhwIjoxNzYzMDY1NzE5LCJzY29wZSI6WyJsbGl3ZWJhcGkiXSwiYW1yIjpbImN1c3RvbWVyQXBpX2dyYW50Il0sImNsaWVudF9pZCI6IkN1c3RvbWVyQXBpIiwic3ViIjoiaW1zQGluc3VyaXR5LmNvbSIsImF1dGhfdGltZSI6MTc2MDQ3MzcxOSwiaWRwIjoic2FsZXNmb3JjZSIsImFjY2Vzc1Rva2VuIjoiMDBEOGQwMDAwMDlvaTM4IUFRRUFRSFlTLmRiVl9Za3UubThJMjBESHRyNXlIUG5Icks3QVNMelhSMEJHTHFQSHcwVzF0UG5hcXRuZ1hfUno4d2g2QUM3M01kVy5hempuNk9GS3FmemxRczFwM3dSTCIsInNlcnZpY2VJZCI6IiIsImVudGl0bGVtZW50VHlwZSI6IiIsImFjY291bnROYW1lIjoiIiwidXNlcm5hbWUiOiJpbXNAaW5zdXJpdHkuY29tIiwidXNlcklkIjoiMDA1TnowMDAwMEd0Z0RWSUFaIiwiY29udGFjdEFjY291bnRJZCI6IjAwMThkMDAwMDBrZ0I0cEFBRSIsInVzZXJUeXBlIjoiQ3NwTGl0ZVBvcnRhbCIsImVtYWlsIjoiaW1zQGluc3VyaXR5LmNvbSIsImdpdmVuX25hbWUiOiJJTVMiLCJmYW1pbHlfbmFtZSI6IkFQSSIsInNoaXBUbyI6IiIsImp0aSI6IkQyQjg2NEQ4MTM5QkQ3N0YxNDc3Qzg1RUExREZGNDU1In0.HmIGF4mRBM5M5hXR79LbC0ChzzZLlqj1lEXe-BtlT20PNN8XskQPOUoahC8FdHufB7FD7kcGiczRU7_iyIljMf6OTxZovJ7TTlT0kM2jX91yLZY8dVByOe-pJ_ViUr8CgCgRLDT-Hm5C47mbC6hRa2RyLf7SKPIVJhBQGDiIfNTD1aVTZ_rNTtok4o_xyEzjq-M63vPjLH5sL5WbInYe39_s0vPTl-my5XPCPQ_R5LEqgUAyDT7PV6JBS3i4LpTinrOz21kgNSUAYBn6qtIyLAu7lhSUfNJT5YpkN5sgaOvXrDQkJgnD4Hsp_C8qv8gFk_Z_EO7sm3JAZEE_SjpDNQ",
    "Accept": "application/json"
}

# ====================================================
# EXECUTE API REQUEST
# ====================================================
print(f"Querying AIS data for {LAT:.2f}, {LON:.2f} (last {HOURS_BACK}h)...")
response = requests.get(f"{BASE_URL}/aislatestinformation", params=params, headers=headers)

print("STATUS:", response.status_code)
print("RAW RESPONSE (first 500 chars):", response.text[:500])

# ====================================================
# PARSE RESPONSE
# ====================================================
if response.status_code == 200:
    try:
        data = response.json()
        msgs = data.get("aisMessages", [])
        print(f"✅ Retrieved {len(msgs)} AIS messages.")
        if len(msgs) > 0:
            print("\nExample messages:")
            for m in msgs[:5]:
                print(f"- Vessel: {m.get('vesselName')} | Type: {m.get('shipType')} "
                      f"| Position: ({m.get('lat')}, {m.get('lon')}) | "
                      f"Time: {m.get('timestamp')}")
    except Exception as e:
        print(f"⚠️ Error parsing JSON: {e}")
else:
    print(f"❌ Error {response.status_code}: {response.text}")

