"""How confident are the models when they are WRONG? Uses the saved internal test predictions only (no model run).

    python src/confidence_check.py            -> results/confidence_check.json

Confidence = the probability of the predicted class. Supports the demo's warning that a high percentage does not mean the
model is right. Internal (formal news) test set only; repeat with external_predictions.csv after the Taglish evaluation.
Each model gets the raw softmax numbers (as before) and, when results/calibration.json exists, a "calibrated" block with the
same numbers after temperature scaling (src/calibrate.py). Predictions are the same in both; only confidence changes.
"""
import argparse, csv, json, statistics
from pathlib import Path
from calibrate import calibrate_p_credible, load_temperature

ROOT = Path(__file__).resolve().parents[1]
MODELS = ["roberta-tagalog", "mbert", "xlmr"]


def check(path, T=1.0):
    rows = list(csv.DictReader(open(path, newline="")))
    pc = (lambda x: float(x["prob_credible"])) if T == 1.0 else (lambda x: calibrate_p_credible(x["prob_credible"], T))
    conf = lambda x: max(pc(x), 1 - pc(x))
    err = [conf(x) for x in rows if x["y_true"] != x["y_pred"]]; ok = [conf(x) for x in rows if x["y_true"] == x["y_pred"]]
    return {"n": len(rows), "errors": len(err),
            "median_confidence_when_wrong": round(statistics.median(err), 4) if err else None,
            "median_confidence_when_right": round(statistics.median(ok), 4) if ok else None,
            "errors_with_confidence_at_least_0.90": sum(c >= 0.90 for c in err),
            "errors_with_confidence_at_least_0.99": sum(c >= 0.99 for c in err),
            "share_of_all_predictions_at_least_0.99": round(sum(c >= 0.99 for c in err + ok) / len(rows), 4)}


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--file", default="test_predictions.csv", help="or external_predictions.csv")
    ap.add_argument("--results", default=str(ROOT / "results")); a = ap.parse_args(argv)
    out = {}
    for m in MODELS:
        f = Path(a.results) / m / a.file
        if not f.exists(): continue
        out[m] = check(f)
        T, ok = load_temperature(m, Path(a.results) / "calibration.json")
        if ok: out[m]["calibrated"] = {"T": T, **check(f, T)}
    name = "confidence_check.json" if a.file == "test_predictions.csv" else "external_confidence_check.json"
    (Path(a.results) / name).write_text(json.dumps({"source_file": a.file, "models": out}, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
