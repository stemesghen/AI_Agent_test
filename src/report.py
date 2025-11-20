from pathlib import Path
import json, csv, datetime as dt

EXTRACT_DIR = Path("data/extracted")
REPORT_DIR  = Path("reports/daily")

def load_summary():
    sum_csv = EXTRACT_DIR / "_summary.csv"
    if not sum_csv.exists():
        return []

    rows = []
    with sum_csv.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows

def group_by_facility(rows):
    groups = {}
    for r in rows:
        fid = r.get("facility_id") or ""
        key = f"fac:{fid}" if fid else f"port:{r.get('port') or ''}"
        groups.setdefault(key, []).append(r)
    return groups

def write_report_md(groups, today):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    md_path = REPORT_DIR / f"{today}.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write(f"# Maritime Incidents — {today}\n\n")
        total = sum(len(v) for v in groups.values())
        f.write(f"- Total incidents: **{total}**\n\n")
        for key, items in sorted(groups.items(), key=lambda kv: -len(kv[1])):
            head = items[0]
            fac_name = head.get("facility_name") or ""
            locode   = head.get("locode") or ""
            label = fac_name or head.get("port") or key
            f.write(f"## {label}  \n")
            if locode:
                f.write(f"- UN/LOCODE: `{locode}`  \n")
            f.write(f"- Count: **{len(items)}**\n\n")
            for it in items[:10]:  # cap list, keep report tidy
                f.write(f"  - {it.get('date') or it.get('published_at','')} — {it.get('title','')[:120]}\n")
            if len(items) > 10:
                f.write(f"  - … and {len(items)-10} more.\n")
            f.write("\n")
    print(f"[OK] Wrote report → {md_path}")

def write_report_csv(rows, today):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = REPORT_DIR / f"{today}.csv"
    if rows:
        keys = list(rows[0].keys())
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow(r)
    print(f"[OK] Wrote CSV → {csv_path}")

def main():
    rows = load_summary()
    if not rows:
        print("[WARN] No _summary.csv found or no rows; run extraction first.")
        return
    today = dt.date.today().isoformat()
    groups = group_by_facility(rows)
    write_report_md(groups, today)
    write_report_csv(rows, today)

if __name__ == "__main__":
    main()
