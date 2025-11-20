# tools/eval_extractor.py
from __future__ import annotations
import argparse, csv, json
from pathlib import Path

def _norm(s: str | None) -> str:
    return (s or "").strip().casefold()

def _read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))

def _prf(tp, fp, fn):
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    f1   = (2 * prec * rec) / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1

def eval_field(gold_rows, pred_rows, gold_key, pred_key, alt_pred_key=None):
    tp = fp = fn = 0
    errors = []
    pred_map = {r["doc_id"]: r for r in pred_rows}
    for g in gold_rows:
        did = g["doc_id"]
        gold = _norm(g.get(gold_key))
        pred_rec = pred_map.get(did)
        pred = _norm(pred_rec.get(pred_key) if pred_rec else None)

        # Optional alternate prediction key (e.g., compare facility_id/locode instead of name)
        if alt_pred_key and pred_rec and not pred:
            pred = _norm(pred_rec.get(alt_pred_key))

        if gold and pred:
            if gold == pred:
                tp += 1
            else:
                fp += 1; fn += 1
                errors.append({"doc_id": did, "gold": g.get(gold_key), "pred": pred_rec.get(pred_key)})
        elif gold and not pred:
            fn += 1
            errors.append({"doc_id": did, "gold": g.get(gold_key), "pred": None})
        elif not gold and pred:
            fp += 1
            errors.append({"doc_id": did, "gold": None, "pred": pred_rec.get(pred_key)})
        # else both empty → neither tp/fp/fn

    prec, rec, f1 = _prf(tp, fp, fn)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": prec, "recall": rec, "f1": f1, "errors": errors}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True, help="CSV with columns: doc_id,vessel,imo,port (canonical port name or locode)")
    ap.add_argument("--pred", required=True, help="CSV produced by _summary.csv")
    ap.add_argument("--out",  required=True, help="Write JSON report here")
    ap.add_argument("--port_mode", choices=["name","locode","facility_id","auto"], default="auto",
                    help="How to compare ports: by name, locode, facility_id, or auto (prefer facility_id, then locode, else name)")
    args = ap.parse_args()

    gold_rows = _read_csv(Path(args.gold))
    pred_rows = _read_csv(Path(args.pred))

    # Vessel: exact name OR IMO if provided in gold (IMO wins)
    vessel_by_name = eval_field(gold_rows, pred_rows, "vessel", "vessel")
    vessel_by_imo  = eval_field(gold_rows, pred_rows, "imo", "imo")

    # Combine: count a TP if either IMO matches or name matches (we’ll approximate by max F1)
    vessel = max([vessel_by_name, vessel_by_imo], key=lambda x: x["f1"])

    # Port
    if args.port_mode == "facility_id":
        port = eval_field(gold_rows, pred_rows, "facility_id", "facility_id")
    elif args.port_mode == "locode":
        port = eval_field(gold_rows, pred_rows, "locode", "locode")
    elif args.port_mode == "name":
        port = eval_field(gold_rows, pred_rows, "port", "port")
    else:
        # auto: prefer facility_id, then locode, else name
        p_fac = eval_field(gold_rows, pred_rows, "facility_id", "facility_id")
        p_loc = eval_field(gold_rows, pred_rows, "locode", "locode")
        p_nam = eval_field(gold_rows, pred_rows, "port", "port")
        port  = max([p_fac, p_loc, p_nam], key=lambda x: x["f1"])

    report = {
        "counts": {"gold": len(gold_rows), "pred": len(pred_rows)},
        "vessel": {
            "by_name": vessel_by_name,
            "by_imo": vessel_by_imo,
            "best": {"precision": vessel["precision"], "recall": vessel["recall"], "f1": vessel["f1"]},
        },
        "port": port,
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
