
# classify/providers/azure_provider.py
import os, json, time
from .base import Classifier

try:
    from openai import AzureOpenAI
except Exception:
    AzureOpenAI = None

SYSTEM_PROMPT = """You are a cautious maritime and logistics incident analyst.
Classify whether the text describes a real vessel/port/logistics disruption.
Incident types ONLY: "grounding","collision","fire","explosion","piracy","weather","port_closure","strike","spill","engine_failure","canal_blockage","security_threat".
Rules:
- Policy/sanctions/market news ≠ incident.
- Forecasts/rumors without a concrete event ≠ incident.
- “Prevented/averted” counts as incident; set near_miss=true.
Return STRICT JSON only:
{ "is_incident": <bool>, "incident_types": <array>, "near_miss": <bool>, "confidence": <0..1>, "rationale": "<≤12 words>" }
"""

ALLOWED = {
    "grounding","collision","fire","explosion","piracy","weather",
    "port_closure","strike","spill","engine_failure","canal_blockage","security_threat"
}

def _need(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise RuntimeError(f"Missing environment variable: {name}. Set it before running.")
    return v

def _coerce_bool(x):
    if isinstance(x, bool): return x
    if isinstance(x, str): return x.strip().lower() in ("true","1","yes","y","t")
    return False

def _sanitize(d: dict) -> dict:
    types = [t for t in d.get("incident_types", []) if t in ALLOWED]
    out = {
        "is_incident": _coerce_bool(d.get("is_incident", False)),
        "incident_types": types,
        "near_miss": _coerce_bool(d.get("near_miss", False)),
        "confidence": float(d.get("confidence", 0.5)),
        "rationale": str(d.get("rationale","")).strip()[:60],
    }
    if not out["is_incident"]:
        out["incident_types"] = []
        out["near_miss"] = False
        out["confidence"] = min(out["confidence"], 0.5)
    return out

class AzureOpenAIClassifier(Classifier):
    def __init__(self):
        if AzureOpenAI is None:
            raise RuntimeError("openai client not installed; run: pip install openai==1.*")
        self.client = AzureOpenAI(
            azure_endpoint=_need("AZURE_OPENAI_ENDPOINT").rstrip("/"),
            api_key=_need("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
        )
        self.deployment = _need("AZURE_OPENAI_DEPLOYMENT")

    def classify(self, title: str, text: str):
        user_msg = f"Title:\n{(title or '').strip()}\n\nText:\n{(text or '').strip()}"
        body = {
            "model": self.deployment,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg[:3500]},
            ],
        }
        for attempt in range(2):
            try:
                resp = self.client.chat.completions.create(**body)
                content = resp.choices[0].message.content
                data = json.loads(content)
                return _sanitize(data)
            except Exception:
                if attempt == 0:
                    time.sleep(1.5)
                    continue
                raise
