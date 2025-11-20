import os
from urllib.parse import urlencode
from .token_manager import get_session, login_for_token

BASE     = os.getenv("COMPANY_API_BASE", "").rstrip("/")
LOC_TYPE = os.getenv("COMPANY_API_LOC_TYPE", "Maritime")  # default to literal
TIMEOUT  = int(os.getenv("COMPANY_API_TIMEOUT", "15"))
COUNTRY_FILTER = os.getenv("COMPANY_API_COUNTRY_FILTER", "").strip()  # e.g., "US"

def get_facilities():
    base_url = f"{BASE}/api/{LOC_TYPE}/Facilities"
    url = f"{base_url}?{urlencode({'filter': COUNTRY_FILTER})}" if COUNTRY_FILTER else base_url
    print(f"[Company API] Requesting: {url}")

    s = get_session()
    r = s.get(url, timeout=TIMEOUT)
    if r.status_code == 401:
        login_for_token()
        s = get_session()
        r = s.get(url, timeout=TIMEOUT)

    if r.status_code >= 400:
        print(f"[Company API] Facilities error {r.status_code}: {r.text[:500]}")
        r.raise_for_status()

    js = r.json()
	
    print(f"[Company API] LOC_TYPE from env: {LOC_TYPE!r}")
    return js.get("data", js) if isinstance(js, dict) else js

