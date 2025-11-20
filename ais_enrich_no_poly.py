#!/usr/bin/env python3
"""
aisenrichnopolytest.py
Minimal GET to Lloyd's AIS endpoint WITHOUT polygons — good for auth/debug.

Examples:
  python aisenrichnopolytest.py --base https://api.lloydslistintelligence.com/v1 \
    --token $LLOYDS_TOKEN --imo 9515802 --hours 6

  python aisenrichnopolytest.py --base https://api.lloydslistintelligence.com/v1 \
    --token $LLOYDS_TOKEN --mmsi 241486000 --after 2025-11-03T00:00:00Z --before 2025-11-03T06:00:00Z
"""

import os
import sys
import json
import argparse
from datetime import datetime, timedelta, timezone

import requests


def utc_now():
    return datetime.now(timezone.utc)


def clamp_to_last_7_days(start_dt, end_dt):
    now = utc_now()
    seven_days_ago = now - timedelta(days=7)
    start_dt = max(start_dt, seven_days_ago)
    end_dt = min(end_dt, now)
    if end_dt < start_dt:
        # nudge end_dt forward minimally to avoid inverted window
        end_dt = start_dt + timedelta(minutes=1)
    return start_dt, end_dt


def parse_time_window(args):
    if args.after and args.before:
        try:
            start = datetime.fromisoformat(args.after.replace("Z", "+00:00"))
            end = datetime.fromisoformat(args.before.replace("Z", "+00:00"))
        except ValueError:
            raise SystemExit("ERROR: after/before must be ISO-8601 (e.g., 2025-11-03T00:00:00Z)")
    else:
        # rolling window ending now
        hrs = max(1, int(args.hours))
        end = utc_now()
        start = end - timedelta(hours=hrs)
    start, end = clamp_to_last_7_days(start, end)
    return start.strftime("%Y-%m-%dT%H:%M:%SZ"), end.strftime("%Y-%m-%dT%H:%M:%SZ")


def build_headers(token):
    tok = (token or "").strip()
    if not tok:
        raise SystemExit("ERROR: token is required (use --token or LLOYDS_TOKEN env var).")
    # add Bearer if missing
    if not tok.lower().startswith("bearer "):
        tok = f"Bearer {tok}"
    return {
        "Authorization": tok,
        "Accept": "application/json",
    }


def build_params(args, received_after, received_before):
    params = {
        "messageFormat": "decoded",
        "receivedAfter": received_after,
        "receivedBefore": received_before,
        # Defaults you can tweak:
        "landFilter": "false",
        "cleansed": "true",
    }
    # Exactly one selector:
    selectors = [x for x in [args.imo, args.mmsi, args.vessel_id] if x]
    if len(selectors) != 1:
        raise SystemExit("ERROR: specify exactly ONE of --imo, --mmsi, or --vessel-id.")
    if args.imo:
        params["vesselImo"] = args.imo
    elif args.mmsi:
        params["mmsi"] = args.mmsi
    else:
        params["vesselId"] = args.vessel_id
    return params


def call_ais(base, headers, params, timeout=60):
    url = f"{base.rstrip('/')}/aislatestinformation"
    print("[INFO] GET", url)
    print("[INFO] params:", json.dumps(params))
    r = requests.get(url, headers=headers, params=params, timeout=timeout)
    print("[INFO] status:", r.status_code)
    if r.status_code in (401, 403):
        print("[HINT] 401/403 usually = expired/wrong token, wrong issuer, or duplicate Authorization headers.")
        print("[HINT] Ensure this is a **Lloyd's** token from /tokenprovider and still valid.")
        raise SystemExit(r.text)
    if r.status_code != 200:
        raise SystemExit(f"AIS error {r.status_code}: {r.text}")
    try:
        return r.json()
    except Exception:
        print("[WARN] Non-JSON response head:", r.text[:400])
        raise SystemExit("AIS returned non-JSON.")


def normalize_hits(raw_json):
    if "aisMessages" in raw_json:
        src = raw_json.get("aisMessages", [])
    elif isinstance(raw_json.get("Data"), dict):
        src = raw_json["Data"].get("aisMessages", [])
    else:
        src = []
    hits = []
    for m in src:
        hits.append({
            "vesselName": m.get("vesselName"),
            "imo": m.get("vesselImo") or m.get("imo"),
            "mmsi": m.get("mmsi"),
            "shipType": m.get("shipType"),
            "lat": m.get("lat"),
            "lon": m.get("lon"),
            "timestamp": m.get("timestamp") or m.get("receivedDateTimeUtc"),
        })
    return hits


def main():
    p = argparse.ArgumentParser(description="No-polygon AIS test")
    p.add_argument("--base", required=True, help="Base URL, e.g. https://api.lloydslistintelligence.com/v1")
    p.add_argument("--token", default=os.getenv("LLOYDS_TOKEN"), help="JWT or 'Bearer ...'")
    p.add_argument("--imo", help="Vessel IMO")
    p.add_argument("--mmsi", help="Vessel MMSI")
    p.add_argument("--vessel-id", help="Lloyd's vesselId")
    p.add_argument("--after", help="Start ISO time (UTC), e.g. 2025-11-03T00:00:00Z")
    p.add_argument("--before", help="End ISO time (UTC), e.g. 2025-11-03T06:00:00Z")
    p.add_argument("--hours", type=int, default=6, help="Rolling window size if after/before not provided (default 6)")
    args = p.parse_args()

    after, before = parse_time_window(args)
    headers = build_headers(args.token)
    params = build_params(args, after, before)

    raw = call_ais(args.base, headers, params)
    hits = normalize_hits(raw)

    print("\n[RESULT] hits:", len(hits))
    print(json.dumps(hits[:10], indent=2))  # show first few


if __name__ == "__main__":
    main()
