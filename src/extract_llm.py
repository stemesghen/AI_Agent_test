# src/extract_llm.py
import os, json, re, requests
from typing import Any, Dict, Optional

print("[LLM] extract_llm.py v3 loaded")

SYSTEM = """You extract maritime entities from news.
- Output JSON only.
- Copy entity strings exactly as they appear in the article; do not invent or normalize.
- If an entity is not explicitly present, return null.
Keys required (lowercase): vessel, imo, port, spans
"""

USER_TMPL = """Article title:
{title}

Article text:
{body}

Extract:
- vessel: exact vessel name if explicitly named (e.g., "Hagland Captain", "Capana", "Falcon"). If generic like "a vessel", "warship", "tanker", return null.
- imo: 7-digit IMO if present; else null.
- port: exact port/harbor/terminal/city used as port context (e.g., "Halden", "Port of Long Beach", "Aden", "Sohar", "Djibouti"). If only regions like "Caribbean", "Eastern Pacific", "Gulf of Aden", return null.
- spans: provide character start/end offsets (0-based) for any non-null field taken from the main text.

Return JSON ONLY in this shape:
{"vessel": ..., "imo": ..., "port": ..., "spans": {"vessel":[start,end], "imo":[start,end], "port":[start,end]}}
"""

def _safe() -> Dict[str, Any]:
    return {"vessel": None, "imo": None, "port": None, "spans": {}}

def _build_payload(system: str, user: str) -> Dict[str, Any]:
    # Azure Responses API (works with gpt-5-mini)
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "response_format": {"type": "json_object"},
        "max_completion_tokens": 300,
        # do NOT send temperature for gpt-5-mini (only default=1 is supported)
    }

def _resp_text(j: dict) -> Optional[str]:
    # 1) responses api shape
    if isinstance(j, dict) and isinstance(j.get("output_text"), str):
        return j["output_text"]
    # 2) chat completions fallback shapes
    if "choices" in j and j["choices"]:
        msg = j["choices"][0].get("message", {})
        if isinstance(msg.get("content"), str):
            return msg["content"]
    # 3) nested "output" shapes
    out = j.get("output")
    if isinstance(out, dict):
        text_blocks = out.get("text")
        if isinstance(text_blocks, list) and text_blocks:
            content = text_blocks[0].get("content")
            if isinstance(content, list):
                for c in content:
                    if c.get("type") in ("output_text", "text") and isinstance(c.get("text"), str):
                        return c["text"]
    return None

def _coerce(parsed: Any) -> Dict[str, Any]:
    """Return dict with vessel/imo/port/spans; never raises."""
    res = _safe()
    try:
        if parsed is None:
            return res
        if isinstance(parsed, str):
            try:
                parsed = json.loads(parsed)
            except Exception:
                return res

        # normalize to dict
        if not isinstance(parsed, dict):
            # sometimes the model returns a list with one dict
            if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
                parsed = parsed[0]
            else:
                return res

        # merge common wrappers without assuming presence
        merged: Dict[str, Any] = {}
        for d in (
            parsed,
            parsed.get("entities") if isinstance(parsed.get("entities"), dict) else {},
            parsed.get("result")   if isinstance(parsed.get("result"), dict)   else {},
            parsed.get("data")     if isinstance(parsed.get("data"), dict)     else {},
        ):
            try:
                merged.update(d)
            except Exception:
                pass

        # helper to fetch a string field safely
        def sget(d: dict, keys):
            for k in keys:
                for kk in (k, k.lower(), k.upper(), k.capitalize()):
                    v = d.get(kk)
                    if v is None: 
                        continue
                    if isinstance(v, (dict, list)):
                        continue
                    vs = str(v).strip()
                    if vs:
                        return vs
            return None

        vessel = sget(merged, ["vessel", "ship", "name"])
        port   = sget(merged, ["port", "harbor", "harbour", "terminal", "city"])
        imo    = sget(merged, ["imo", "imo_number"])

        # normalize IMO to 7 digits
        if imo:
            imo = re.sub(r"\D+", "", imo)
            if not re.fullmatch(r"\d{7}", imo):
                imo = None

        spans = {}
        for sk in ("spans", "Spans", "SPANs"):
            val = merged.get(sk)
            if isinstance(val, dict):
                spans = val
                break

        res["vessel"] = vessel or None
        res["port"]   = port or None
        res["imo"]    = imo or None
        res["spans"]  = spans or {}
        return res
    except Exception as e:
        # absolute last resort: never propagate
        print(f"[LLM] _coerce error, returning safe: {e}")
        return res

def extract_with_llm(title: str, body: str) -> Optional[Dict[str, Any]]:
    endpoint = (os.getenv("AZURE_OPENAI_ENDPOINT") or "").rstrip("/")
    version  = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")
    deploy   = os.getenv("AZURE_OPENAI_DEPLOYMENT", "")
    key      = os.getenv("AZURE_OPENAI_KEY") or os.getenv("AZURE_OPENAI_API_KEY")

    if not (endpoint and version and deploy and key):
        print("[LLM] Missing Azure env vars (AZURE_OPENAI_ENDPOINT/API_VERSION/DEPLOYMENT/KEY).")
        return None

    url = f"{endpoint}/openai/deployments/{deploy}/responses?api-version={version}"
    headers = {"Content-Type": "application/json", "api-key": key}
    payload = _build_payload(SYSTEM, USER_TMPL.format(title=title or "", body=body or ""))

    try:
        r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60)
        if r.status_code != 200:
            print(f"[LLM] Azure call failed: {r.status_code} - {r.text[:300]}")
            return None

        j = r.json()
        txt = _resp_text(j)
        if not txt:
            print("[LLM] Could not parse response text.")
            return None

        # strip code fences if present
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", txt.strip()).strip()
        if raw:
            print("[LLM] raw:", raw[:240].replace("\n"," ") + ("…" if len(raw) > 240 else ""))

        # attempt to parse JSON; if it fails, try to grab the first {...} block
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            m = re.search(r"\{[\s\S]*\}", raw)
            parsed = json.loads(m.group(0)) if m else {}

        return _coerce(parsed)

    except Exception as e:
        print(f"[LLM] Error: {e}")
        return None
