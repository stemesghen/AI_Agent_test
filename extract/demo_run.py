#!/usr/bin/env python3
import os, shutil, subprocess, sys
from pathlib import Path

def resolve_repo_root():
    # If run as module (python -m extract.demo_run), __file__ is in extract/
    here = Path(__file__).resolve()
    if (here.parent / "data").exists() and (here.parent / "extract").exists():
        # Running from repo root already
        return here.parent
    # If inside extract/, repo root is parent of extract/
    if (here.parent.name == "extract") and (here.parent.parent / "data").exists():
        return here.parent.parent
    # Fallback: walk up to find a "data" folder
    p = here
    for _ in range(4):
        if (p / "data").exists():
            return p
        p = p.parent
    return here.parent  # last resort

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run full pipeline on a small subset.")
    parser.add_argument("--source", default="data/classified", help="Folder with classified JSONs")
    parser.add_argument("--limit", type=int, default=40, help="How many files to run")
    parser.add_argument("--ner", default="dslim/bert-base-NER", help="NER models (comma ok if your code supports)")
    parser.add_argument("--threshold", default="0.9", help="PORT_CONF_THRESHOLD override for demo")
    args = parser.parse_args()

    repo_root = resolve_repo_root()
    os.chdir(repo_root)  # <<-- move here

    source_dir     = (repo_root / args.source).resolve()
    demo_dir       = (repo_root / "data" / "_sample" / "classified").resolve()
    extracted_demo = (repo_root / "data" / "_sample" / "extracted").resolve()
    live_dir       = (repo_root / "data" / "classified").resolve()
    backup_dir     = (repo_root / "data" / "classified_full").resolve()

    print(f"[DEMO] repo_root      = {repo_root}")
    print(f"[DEMO] source_dir     = {source_dir}")
    print(f"[DEMO] live_dir       = {live_dir}")
    print(f"[DEMO] demo classified= {demo_dir}")
    print(f"[DEMO] demo extracted = {extracted_demo}")
    print(f"[DEMO] limit          = {args.limit}")

    if not source_dir.exists():
        print(f"[ERR] Source folder not found: {source_dir}")
        sys.exit(1)

    demo_dir.mkdir(parents=True, exist_ok=True)
    extracted_demo.mkdir(parents=True, exist_ok=True)

    # Clean previous demo copy
    for p in demo_dir.glob("*.json"):
        p.unlink()

    files = sorted(source_dir.glob("*.json"))[: args.limit]
    if not files:
        print(f"[ERR] No JSON files found in {source_dir}")
        sys.exit(1)
    for f in files:
        shutil.copy2(f, demo_dir / f.name)
    print(f"[DEMO] Copied {len(files)} files into {demo_dir}")

    # Swap: move live -> backup, demo -> live
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
    try:
        print("[DEMO] Swapping demo folder into place...")
        live_dir.rename(backup_dir)
        demo_dir.rename(live_dir)

        # Env overrides (fast but same pipeline)
        env = os.environ.copy()
        env["NER_MODELS"] = args.ner
        env["PORT_CONF_THRESHOLD"] = args.threshold
        env["IMS_OPTIONAL"] = env.get("IMS_OPTIONAL", "1")  # harmless if unused

        print("[DEMO] Running full pipeline (extract.run)...")
        subprocess.run([sys.executable, "-m", "extract.run"], check=True, env=env)
        print("[DEMO] Pipeline finished.")

    except Exception as e:
        print(f"[ERR] Pipeline failed: {e}")
    finally:
        # Restore original structure
        print("[DEMO] Restoring original folders...")
        if live_dir.exists():
            live_dir.rename(demo_dir)  # move current live (demo) back
        if backup_dir.exists():
            backup_dir.rename(live_dir)  # restore original live
        print("[DEMO] Done. Outputs will be wherever your run.py writes them (e.g., data/extracted).")

if __name__ == "__main__":
    main()


