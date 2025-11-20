# disambiguation/ais.py
from __future__ import annotations
from typing import List, Tuple, Optional
import math

def haversine_m(lat1, lon1, lat2, lon2) -> float:
    try:
        lat1, lon1, lat2, lon2 = map(float, (lat1, lon1, lat2, lon2))
    except Exception:
        return float("nan")
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi   = math.radians(lat2 - lat1)
    dl     = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2 * R * math.asin(math.sqrt(a))

def proximity_score_meters(d_m: float, soft_cap_km: float = 80.0) -> float:
    """
    Map distance→[0..1] with diminishing returns; 80km ≈ ~0.5.
    """
    if not (isinstance(d_m, (float, int))) or d_m <= 0:
        return 0.0
    d_km = d_m / 1000.0
    return max(0.0, min(1.0, 1.0 / (1.0 + (d_km / soft_cap_km))))

def best_proximity_for_candidate(ais_points: List[Tuple[float, float]],
                                 cand_lat: Optional[str],
                                 cand_lon: Optional[str]) -> float:
    if not ais_points or not cand_lat or not cand_lon:
        return 0.0
    try:
        clat = float(cand_lat); clon = float(cand_lon)
    except Exception:
        return 0.0
    best = 0.0
    for lat, lon in ais_points:
        d = haversine_m(lat, lon, clat, clon)
        best = max(best, proximity_score_meters(d))
    return best
