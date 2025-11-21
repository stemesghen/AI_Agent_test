#!/usr/bin/env python3
# scripts/run.py
from __future__ import annotations
import argparse, os, datetime, subprocess, sys
from pathlib import Path

# Defaults (override via env or CLI flags if you want)
UNLOCODE_PATH = os.getenv("UNLOCODE_PATH", "data/raw/UNLOCODE.csv")
GEONAMES_PATH = os.getenv("GEONAMES_PATH", "data/raw/geonames.csv")
PORTS_OUT     = os.getenv("PORTS_OUT", "data/port.csv")

IMS_BASE_URL  = os.getenv("IMS_BASE_URL", "")
IMS_TOKEN     = os.getenv("IMS_TOKEN", "")

DATA_DIR      = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

def _run(cmd: list[str]) -> int:
    print("[CMD]", " ".join(cmd)); return subprocess.call(cmd)

def refresh_ports():
    # Calls your merge script (no IMS here; deterministic)
    return _run([
        sys.executable, "scripts/merge_ports.py",
        "--unlocode", UNLOCODE_PATH,
        "--geonames", GEONAMES_PATH,
        "-o", PORTS_OUT
    ])

def refresh_ims():
    if not IMS_BASE_URL or not IMS_TOKEN:
        print("[WARN] IMS_BASE_URL / IMS_TOKEN not set — skipping IMS cache build")
        return 0
    date_str = datetime.date.today().isoformat()
    out_json = str(DATA_DIR / f"facilities_{date_str}.json")
    out_map  = str(DATA_DIR / f"ims_map_{date_str}.csv")
    env = os.environ.copy()
    env["IMS_BASE_URL"] = IMS_BASE_URL
    env["IMS_TOKEN"]    = IMS_TOKEN
    # Use the resolver CLI to build the cache
    return subprocess.call([
        sys.executable, "-m", "ims.facility_resolver", "build-cache",
        "--out-json", out_json,
        "--out-map", out_map
    ], env=env)

def _latest_facilities_json() -> str | None:
    files = sorted([p for p in DATA_DIR.glob("facilities_*.json")])
    return str(files[-1]) if files else None

def backfill_strict(inplace: bool = True):
    ims_json = _latest_facilities_json()
    if not ims_json:
        print("[WARN] No facilities_*.json found in data/. Run refresh_ims first.")
        return 1
    args = [
        sys.executable, "scripts/backfill_ims_facilities_id.py",
        "--in", PORTS_OUT,
        "--ims", ims_json,
    ]
    if inplace: args.append("--inplace")
    return _run(args)

def backfill_fuzzy(inplace: bool = True, threshold: float = 0.92):
    ims_json = _latest_facilities_json()
    if not ims_json:
        print("[WARN] No facilities_*.json found in data/. Run refresh_ims first.")
        return 1
    args = [
        sys.executable, "scripts/backfill_ims_id.py",
        "--in", PORTS_OUT,
        "--ims", ims_json,
        "--allow-name-match",
        "--fuzzy-threshold", str(threshold),
    ]
    if inplace: args.append("--inplace")
    return _run(args)

def main():
    ap = argparse.ArgumentParser(description="Pipeline runner")
    ap.add_argument("task", choices=["refresh_ports","refresh_ims","backfill_strict","backfill_fuzzy","full"])
    args = ap.parse_args()

    if args.task == "refresh_ports":
        sys.exit(refresh_ports())
    elif args.task == "refresh_ims":
        sys.exit(refresh_ims())
    elif args.task == "backfill_strict":
        sys.exit(backfill_strict(inplace=True))
    elif args.task == "backfill_fuzzy":
        sys.exit(backfill_fuzzy(inplace=True))
    elif args.task == "full":
        code = refresh_ports()
        if code != 0: sys.exit(code)
        code = refresh_ims()
        if code != 0: sys.exit(code)
        # choose strict by default; switch to fuzzy if/when you’re ready
        code = backfill_strict(inplace=True)
        sys.exit(code)

if __name__ == "__main__":
    main()
