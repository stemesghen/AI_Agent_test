# auth_probe_plus.py
import os, requests
from dotenv import load_dotenv
load_dotenv()

BASE = (os.getenv("COMPANY_API_BASE") or os.getenv("IMS_BASE_URL") or "").rstrip("/")
USER = os.getenv("COMPANY_API_USER") or os.getenv("IMS_USER") or ""
PASS = os.getenv("COMPANY_API_PASS") or os.getenv("IMS_PASS") or ""
LIC  = os.getenv("COMPANY_API_LICENSEE_ID") or os.getenv("IMS_LICENSEE_ID") or "1"

if not BASE or not USER or not PASS:
    raise SystemExit("Set COMPANY_API_BASE, COMPANY_API_USER, COMPANY_API_PASS (.env)")

def show(r, url, note):
    t = (r.text or "")[:300].replace("\n", " ")
    print(f"[{r.status_code}] {note:<28} {url}")
    print("     ", t)

candidates = []

# IdentityServer / OAuth2 styles
candidates += [
    ("POST", "/connect/token", "form", {"grant_type":"password","username":USER,"password":PASS,"scope":"api"}),
    ("POST", "/token",         "form", {"grant_type":"password","username":USER,"password":PASS}),
    ("POST", "/Token",         "form", {"grant_type":"password","username":USER,"password":PASS}),
    ("POST", "/oauth/token",   "form", {"grant_type":"password","username":USER,"password":PASS}),
]

# Insurity “Authenticate” styles (body JSON & form)
for path in ["/api/Authentication/Authenticate", "/api/Authentication/LoginUser", "/Authentication/Authenticate"]:
    candidates += [
        ("POST", path, "json", {"UserName":USER, "Password":PASS, "LicenseeId":LIC}),
        ("POST", path, "form", {"UserName":USER, "Password":PASS, "LicenseeId":LIC}),
    ]

# Try licensee in header as well
HEADERS = {"Accept":"application/json","X-LicenseeId":str(LIC)}

sess = requests.Session()
for method, path, enc, body in candidates:
    url = BASE + path
    try:
        if method == "POST":
            if enc == "json":
                r = sess.post(url, json=body, headers=HEADERS, timeout=20)
                show(r, url, f"POST json  X-LicenseeId")
                if r.ok and ("Token" in (r.text or "") or "SessionToken" in (r.text or "")): break
            elif enc == "form":
                r = sess.post(url, data=body, headers=HEADERS, timeout=20)
                show(r, url, f"POST form X-LicenseeId")
                if r.ok and ("Token" in (r.text or "") or "SessionToken" in (r.text or "")): break
        else:
            pass
    except Exception as e:
        print(f"[ERR] {method} {url}: {e}")
