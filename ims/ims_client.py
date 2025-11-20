# ims/ims_client.py
# ims/ims_client.py
from __future__ import annotations
import os, requests
from urllib.parse import urljoin
from typing import Optional, Dict, Any, List

DEBUG_IMS = os.getenv("DEBUG_IMS", "0") == "1"


def _dbg_ims(msg: str) -> None:
    if DEBUG_IMS:
        print(f"[IMS-HTTP] {msg}")


class IMSClient:
    def __init__(self, base_url: Optional[str] = None, token: Optional[str] = None, timeout: int = 15):
        self.base_url = (base_url or os.getenv("IMS_BASE_URL", "")).rstrip("/")
        if not self.base_url:
            raise RuntimeError("IMS_BASE_URL missing")
        self.token = (token or os.getenv("IMS_TOKEN", "")).strip()
        if not self.token:
            raise RuntimeError("IMS_TOKEN missing")

        # e.g., Maritime, Aviation, Property, etc.
        self.location_type = (
            os.getenv("IMS_LOC_TYPE")
            or os.getenv("IMS_LOCATION_TYPE")
            or "Maritime"
        ).strip()

        # final example: https://.../WebAPI/api/Maritime/Facilities
        self.facilities_path = f"/api/{self.location_type}/Facilities"
        self.timeout = timeout

        if DEBUG_IMS:
            _dbg_ims(f"Initialized IMSClient base_url={self.base_url} location_type={self.location_type}")

    # ---- internals ----
    def _join(self, path: str) -> str:
        return urljoin(self.base_url.rstrip("/") + "/", path.lstrip("/"))

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/json",
            "User-Agent": "PythonIMSClient/1.0",
        }

    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        url = self._join(path)
        _dbg_ims(f"GET {url} params={params}")
        r = requests.get(url, headers=self._headers(), params=params, timeout=self.timeout)
        _dbg_ims(f"RESP status={r.status_code} content-type={r.headers.get('Content-Type')}")
        r.raise_for_status()
        try:
            data = r.json()
            if isinstance(data, dict):
                # small preview only
                if "items" in data and isinstance(data["items"], list):
                    _dbg_ims(f"JSON dict with items len={len(data['items'])}")
                elif "data" in data and isinstance(data["data"], list):
                    _dbg_ims(f"JSON dict with data len={len(data['data'])}")
                else:
                    _dbg_ims("JSON dict (no items/data keys)")
            elif isinstance(data, list):
                _dbg_ims(f"JSON list len={len(data)}")
            else:
                _dbg_ims(f"JSON type={type(data).__name__}")
            return data
        except Exception:
            _dbg_ims("Non-JSON response, returning text")
            return r.text

    # generic GET (needed for param-style filters)
    def get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        return self._get(path, params=params)

    # NEW: compatibility wrapper used by facility_resolver
    def request(self, method: str, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """
        Minimal wrapper so code that calls ims_client.request('GET', '/Facilities', ...)
        still works. We map '/Facilities' to the proper facilities path for this
        location type (Maritime, etc.).
        """
        method = method.upper()
        if method != "GET":
            raise ValueError(f"IMSClient.request only supports GET, got {method}")

        # If caller passes "/Facilities", rewrite to the proper API path
        if path.strip("/") == "Facilities":
            path = self.facilities_path

        return self.get(path, params=params)

    # ---- public: Facilities ----
    def get_facilities(self, filter_expr: Optional[str] = None, page_size: int = 50, page_number: int = 1) -> Any:
        """
        GET /api/:locationType/Facilities?filter=<expr>&page[size]=N&page[number]=M
        Example filter: "countryISOCode==CN"  or  "locationISOCode==AE"
        """
        params: Dict[str, Any] = {
            "page[size]": page_size,
            "page[number]": page_number,
        }
        if filter_expr:
            params["filter"] = filter_expr
        _dbg_ims(f"get_facilities filter={filter_expr} page_size={page_size} page_number={page_number}")
        return self._get(self.facilities_path, params=params)

    def list_facilities_by_country(self, country_iso2: str, page_size: int = 50) -> List[Dict[str, Any]]:
        """
        Convenience wrapper that fetches the first page filtered by country.
        """
        filt = f"countryISOCode=={country_iso2}"
        _dbg_ims(f"list_facilities_by_country({country_iso2})")
        data = self.get_facilities(filter_expr=filt, page_size=page_size, page_number=1)
        items: List[Dict[str, Any]] = []

        if isinstance(data, dict):
            if "items" in data and isinstance(data["items"], list):
                items = data["items"]
            elif "data" in data and isinstance(data["data"], list):
                items = data["data"]
        elif isinstance(data, list):
            items = data

        _dbg_ims(f"list_facilities_by_country({country_iso2}) -> {len(items)} facilities")
        return items

    def get_facility(self, ims_facility_id: str) -> Dict[str, Any]:
        """
        GET /api/:locationType/Facilities/{id}
        """
        path = f"{self.facilities_path}/{ims_facility_id}"
        _dbg_ims(f"get_facility id={ims_facility_id}")
        data = self._get(path)
        if isinstance(data, dict):
            _dbg_ims("get_facility returned dict")
            return data
        _dbg_ims("get_facility returned non-dict, forcing {}")
        return {}
