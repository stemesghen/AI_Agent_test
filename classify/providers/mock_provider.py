import re
from typing import Dict, List
from .base import Classifier

ALLOWED = {"grounding","collision","fire","piracy","weather","port_closure","strike","spill"}

R_INCIDENT = re.compile(r"\b(ground(?:ed|ing)|collid(?:ed|e|ing)|collision|allision|fire|blaze|on fire|piracy|pirate|hijack(?:ed|ing)|storm|hurricane|typhoon|cyclone|gale|rough seas|port\s+(?:closure|closed)|strike|walkout|industrial action|spill|leak)\b", re.I)
R_NEARMISS = re.compile(r"\b(prevented|averted|avoided)\b", re.I)

TYPES = {
    "grounding": r"\bground(?:ed|ing)\b",
    "collision": r"\b(collid(?:ed|e|ing)|collision|allision)\b",
    "fire": r"\b(fire|blaze|on fire)\b",
    "piracy": r"\b(piracy|pirate|hijack(?:ed|ing))\b",
    "weather": r"\b(storm|hurricane|typhoon|cyclone|gale|rough seas)\b",
    "port_closure": r"\bport\s+(closure|closed)\b",
    "strike": r"\b(strike|walkout|industrial action)\b",
    "spill": r"\b(spill|leak)\b",
}

# Non-incident gate—policy/market ops without incident cues
R_NONINCIDENT = re.compile(r"\b(sanction|share[s]? (?:hit|falls?)|fee|tariff|earnings|forecast|production|contract|order book|IPO)\b", re.I)

def _match_types(text: str) -> List[str]:
    labels = []
    for k, pat in TYPES.items():
        if re.search(pat, text, re.I):
            labels.append(k)
    # keep only allowed, keep stable order
    return [t for t in labels if t in ALLOWED]

class MockClassifier(Classifier):
    def classify(self, text: str) -> Dict:
        t = text or ""
        is_incident = bool(R_INCIDENT.search(t))
        near_miss = bool(R_NEARMISS.search(t)) and is_incident
        labels = _match_types(t) if is_incident else []

        # If it looks like policy/market and no incident keywords → force non-incident
        if not labels and R_NONINCIDENT.search(t):
            is_incident = False
            near_miss = False

        confidence = 0.75 if is_incident else 0.3
        rationale = (
            "Incident keywords found" if is_incident else
            "Policy/market news or no incident cues"
        )
        return {
            "is_incident": is_incident,
            "incident_types": labels,
            "near_miss": near_miss,
            "confidence": confidence,
            "rationale": rationale[:60]
        }

