import subprocess, sys

def run(cmd):
    print(f"→ {cmd}")
    proc = subprocess.run(cmd, shell=True)
    if proc.returncode != 0:
        sys.exit(proc.returncode)

def main():
    run("python -m ingest.run_ingest")  # if you want to re-pull feeds
    run("python -m classify.run")
    run("python -m extract.run")
    run("python -m src.report")

if __name__ == "__main__":
    main()
