import os
import json
import time
import requests
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

# --- Env ---
BASE: str = os.getenv("COMPANY_API_BASE", "").rstrip("/")
USER: str = os.getenv("COMPANY_API_USER", "")
PASS: str = os.getenv("COMPANY_API_PASS", "")
LICENSEE: str = os.getenv("COMPANY_API_LICENSEE_ID", "")
TIMEOUT: int = int(os.getenv("COMPANY_API_TIMEOUT", "15"))
LOGIN_PATH: str = os.getenv("COMPANY_API_LOGIN_PATH", "/api/Authentication/Login")

# --- Cache file (token + cookies) ---
CACHE_FILE = Path("data/cache/company_token.json")
CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Cache helpers
# -----------------------------
def _load_cache() -> Optional[dict]:
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return None

def _save_cache(token: str, cookies: Dict[str, str], ttl_seconds: int = 5400) -> None:
    # refresh ~5 min early to be safe
    exp = time.time() + max(600, ttl_seconds - 300)
    CACHE_FILE.write_text(
        json.dumps({"token": token, "cookies": cookies, "expires_at": exp}),
        encoding="utf-8"
    )

# -----------------------------
# Response parsing
# -----------------------------
def _extract_token(js: Dict[str, Any]) -> Tuple[Optional[str], Optional[int]]:
    # JSON:API style (your API)
    try:
        tok = js["data"]["attributes"]["sessionToken"]
        ttl = js.get("data", {}).get("attributes", {}).get("expiresIn")
        return tok, ttl
    except Exception:
        pass

    # Other common shapes (fallback)
    for k in ("access_token", "token", "bearerToken", "sessionToken"):
        if k in js:
            return js[k], js.get("expires_in")

    if isinstance(js.get("data"), dict):
        inner = js["data"]
        for k in ("access_token", "token", "bearerToken", "sessionToken"):
            if k in inner:
                return inner[k], js.get("expires_in")

    return None, None

# -----------------------------
# Login / token acquisition
# -----------------------------
def login_for_token() -> dict:
    """Log in (JSON:API) and cache token + ASP.NET cookies."""
    if not BASE:
        raise RuntimeError("COMPANY_API_BASE is not set")
    if not LICENSEE:
        raise RuntimeError("COMPANY_API_LICENSEE_ID is not set in .env")

    url = f"{BASE}{LOGIN_PATH}"
    headers = {
        "Accept": "application/vnd.api+json",
        "Content-Type": "application/vnd.api+json",
    }
    body = {
        "data": {
            "type": "OMS-LoginRequest",
            "attributes": {
                "licenseeID": LICENSEE,
                "username": USER,
                "password": PASS,
            }
        }
    }

    r = requests.post(url, json=body, headers=headers, timeout=TIMEOUT)
    if r.status_code >= 400:
        raise RuntimeError(f"Login failed {r.status_code}: {r.text[:400]}")

    js = r.json()
    token, ttl = _extract_token(js)
    if not token:
        raise RuntimeError(f"Unexpected login response: {js}")

    cookies = requests.utils.dict_from_cookiejar(r.cookies)  # e.g., ASP.NET_SessionId
    _save_cache(token, cookies, int(ttl or 5400))
    print("[Company API] Token + cookies acquired and cached.")
    return {"token": token, "cookies": cookies, "expires_at": time.time() + int(ttl or 5400)}

# -----------------------------
# Sessions / headers
# -----------------------------
def get_session() -> requests.Session:
    """Return a session that includes Bearer auth + ASP.NET cookies + Accept header."""
    cached = _load_cache()
    if not cached or float(cached.get("expires_at", 0)) <= time.time():
        cached = login_for_token()

    s = requests.Session()
    # attach cookies
    for k, v in cached.get("cookies", {}).items():
        s.cookies.set(k, v)
    # headers similar to your Postman GET
    s.headers.update({
        "Authorization": f"Bearer {cached['token']}",
        "Accept": "application/json",
    })
    return s

# (Keep these if other modules use them)
def get_token() -> str:
    cached = _load_cache()
    return (cached and cached.get("token")) or login_for_token()["token"]

def auth_headers() -> Dict[str, str]:
    return {"Authorization": f"Bearer {get_token()}", "Accept": "application/json"}
