"""
AISClient — thin wrapper for Lloyd's List Intelligence AIS endpoints.

All calls are LIVE by default.

Env:
  AIS_BASE_URL
  AIS_AUTH_TOKEN
  AIS_API_KEY (optional)
  AIS_LATEST_PATH=/aislatestinformation
  AIS_BBOX_PATH=/aisbbox
  AIS_TRACK_PATH=/aistrack
  AIS_AUTH_SCHEME=Bearer
  AIS_FORCE_RAW_AUTH=0
  AIS_DEBUG=0
  AIS_MESSAGE_FORMAT=decoded
"""

from __future__ import annotations
import os
from typing import List, Tuple, Dict, Any, Optional
import requests

try:
    from dotenv import load_dotenv, find_dotenv  # type: ignore
    load_dotenv(find_dotenv(), override=True)
except Exception:
    pass


class AISClient:
    def __init__(
        self,
        base_url: Optional[str] = None,
        auth_token: Optional[str] = None,
        timeout: int = 12,
    ):
        self.base_url = (base_url or os.getenv("AIS_BASE_URL", "")).rstrip("/")
        self.auth_token = (auth_token or os.getenv("AIS_AUTH_TOKEN", "")).strip()
        self.timeout = timeout

        self.path_latest = os.getenv("AIS_LATEST_PATH", "/aislatestinformation")
        self.path_bbox   = os.getenv("AIS_BBOX_PATH",   "/aisbbox")
        self.path_track  = os.getenv("AIS_TRACK_PATH",  "/aistrack")

        self.api_key = os.getenv("AIS_API_KEY", "").strip()
        self.force_raw_auth = os.getenv("AIS_FORCE_RAW_AUTH", "0") in ("1", "true", "True")
        self.auth_scheme = os.getenv("AIS_AUTH_SCHEME", "Bearer").strip()
        self.debug = os.getenv("AIS_DEBUG", "0") in ("1", "true", "True")

        if not self.base_url:
            raise RuntimeError("AIS_BASE_URL missing (set it in .env or pass base_url=)")

    # ----------------- internals -----------------
    def _authorization_value(self) -> Optional[str]:
        tok = self.auth_token
        if not tok:
            return None
        if self.force_raw_auth:
            return tok
        if tok.lower().startswith((self.auth_scheme.lower() + " ").lower()):
            return tok
        return f"{self.auth_scheme} {tok}"

    def _headers(self) -> Dict[str, str]:
        h: Dict[str, str] = {
            "Accept": "application/json",
            "User-Agent": "PythonAISClient/1.1",
        }
        auth_val = self._authorization_value()
        if auth_val:
            h["Authorization"] = auth_val
        if self.api_key:
            h["x-api-key"] = self.api_key
        return h

    def _request_get(self, path: str, params: Optional[Dict[str, Any]] = None):
        url = f"{self.base_url}{path}"
        if self.debug:
            dbg_headers = {k: ("<redacted>" if k.lower() == "authorization" else v) for k, v in self._headers().items()}
            print(f"[AIS] GET {url} params={params or {}} headers={dbg_headers}")

        r = requests.get(url, headers=self._headers(), params=params, timeout=self.timeout)
        r.raise_for_status()

        try:
            j = r.json()
        except Exception:
            return r.text

        if isinstance(j, dict) and "data" in j and isinstance(j["data"], list):
            return j["data"]
        return j

    # ----------------- public API -----------------
    def latest_information(self, **query) -> List[Dict[str, Any]]:
        """
        GET /aislatestinformation
        """
        try:
            if "messageFormat" not in query:
                query["messageFormat"] = os.getenv("AIS_MESSAGE_FORMAT", "decoded")
            data = self._request_get(self.path_latest, params=query or None)
            return data if isinstance(data, list) else (data or [])
        except requests.HTTPError as e:
            status = getattr(e.response, "status_code", "?")
            text = getattr(e.response, "text", "")
            print(f"[AIS] latest_information HTTP {status}: {text[:300]}")
            return []
        except Exception as e:
            print(f"[AIS] latest_information error: {e}")
            return []

    def bbox(self, min_lat: float, min_lon: float, max_lat: float, max_lon: float) -> List[Dict[str, Any]]:
        params = {"minLat": min_lat, "minLon": min_lon, "maxLat": max_lat, "maxLon": max_lon}
        try:
            data = self._request_get(self.path_bbox, params=params)
            return data if isinstance(data, list) else (data or [])
        except requests.HTTPError as e:
            status = getattr(e.response, "status_code", "?")
            text = getattr(e.response, "text", "")
            print(f"[AIS] bbox HTTP {status}: {text[:300]}")
            return []
        except Exception as e:
            print(f"[AIS] bbox error: {e}")
            return []

    def track_by_mmsi(self, mmsi: str, hours: int = 24) -> List[Dict[str, Any]]:
        params = {"mmsi": mmsi, "hours": hours}
        try:
            data = self._request_get(self.path_track, params=params)
            return data if isinstance(data, list) else (data or [])
        except requests.HTTPError as e:
            status = getattr(e.response, "status_code", "?")
            text = getattr(e.response, "text", "")
            print(f"[AIS] track_by_mmsi HTTP {status}: {text[:300]}")
            return []
        except Exception as e:
            print(f"[AIS] track_by_mmsi error: {e}")
            return []

    # -------- helpers for scoring / geometry --------
    @staticmethod
    def to_points(items: List[Dict[str, Any]]) -> List[Tuple[float, float]]:
        pts: List[Tuple[float, float]] = []
        for it in items or []:
            lat = it.get("Latitude") or it.get("lat") or it.get("latitude")
            lon = it.get("Longitude") or it.get("lon") or it.get("longitude")
            try:
                if lat is not None and lon is not None:
                    pts.append((float(lat), float(lon)))
            except Exception:
                pass
        return pts
