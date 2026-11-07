"""Post-hoc calibration by temperature scaling (Guo et al., 2017). No retraining: model weights are unchanged.

    python src/calibrate.py                 # Colab/GPU: logits on val + test for the 3 models, fit T, report
    python src/calibrate.py --from-logits   # re-fit and re-report from saved results/<model>/{val,test}_logits.csv

For each model one temperature T > 0 is fitted on the internal VALIDATION split (data/processed/val.csv) by minimising
the negative log-likelihood of softmax(logits / T). Dividing every logit by the same positive T never changes which
class is largest, so predictions (argmax) are identical before and after; only the confidence values change.
The Taglish/supplementary set is never used here (test only).

Outputs
  results/<model>/val_logits.csv, test_logits.csv   id, y_true, logit_not_credible, logit_credible (no text)
  results/calibration.json                          {"temperatures": {model: T}, n_val, seed, date, ...}
  results/calibration_report.json                   internal TEST split, before vs after: ECE (15 bins), NLL, Brier,
                                                    median confidence when wrong / right, reliability bins
Label convention: 1 = credible, 0 = not credible (positive class). Confidence = probability of the predicted class.
Brier score here = mean over posts of the squared error of P(credible) against the true label (binary Brier).

Other scripts import load_temperature() / calibrate_probs() from this file. Without results/calibration.json they fall
back to T = 1 (raw softmax) and say so.
"""
import argparse, csv, datetime, json, math, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
MODELS = ["roberta-tagalog", "mbert", "xlmr"]
N_BINS = 15


# ---------- helpers used by predict.py, confidence_check.py and the app backend (no torch needed) ----------
def load_temperature(model, path=None):
    """(T, calibrated?) for a model; (1.0, False) when results/calibration.json is missing or lacks the model."""
    p = Path(path) if path else RES / "calibration.json"
    try:
        t = json.load(open(p, encoding="utf-8"))["temperatures"][model]
        return float(t), True
    except (OSError, KeyError, ValueError, TypeError):
        return 1.0, False


def calibrate_p_credible(p_credible, T, eps=1e-7):
    """Rescale a 2-class softmax P(credible) with temperature T (same as softmax(logits / T) for 2 classes)."""
    p = min(max(float(p_credible), eps), 1 - eps)
    z = math.log(p / (1 - p)) / T  # logit(credible) - logit(not credible), divided by T
    return 1 / (1 + math.exp(-z)) if z >= 0 else math.exp(z) / (1 + math.exp(z))


# ---------- fitting and metrics (numpy only) ----------
def _np():
    import numpy as np
    return np


def softmax_T(logits, T):
    np = _np(); z = np.asarray(logits, float) / T; z = z - z.max(1, keepdims=True); e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def nll(logits, y, T):
    np = _np(); z = np.asarray(logits, float) / T; m = z.max(1, keepdims=True)
    lse = (m[:, 0] + np.log(np.exp(z - m).sum(1)))
    return float((lse - z[np.arange(len(y)), y]).mean())


def fit_temperature(logits, y):
    """Minimise NLL over log T: coarse grid on [0.05, 20], then golden-section refinement. NLL is convex in 1/T."""
    np = _np(); y = np.asarray(y, int)
    grid = np.linspace(math.log(0.05), math.log(20), 200)
    vals = [nll(logits, y, math.exp(g)) for g in grid]; i = int(np.argmin(vals))
    a, b = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
    g = (math.sqrt(5) - 1) / 2
    c, d = b - g * (b - a), a + g * (b - a)
    for _ in range(80):
        if nll(logits, y, math.exp(c)) < nll(logits, y, math.exp(d)): b = d
        else: a = c
        c, d = b - g * (b - a), a + g * (b - a)
    return math.exp((a + b) / 2)


def report(logits, y, T):
    """Calibration metrics for one temperature. Label ids: column 0 = not credible, column 1 = credible."""
    np = _np(); y = np.asarray(y, int); p = softmax_T(logits, T)
    pred = p.argmax(1); conf = p.max(1); right = pred == y
    edges = np.linspace(0, 1, N_BINS + 1); ece, bins = 0.0, []
    for k in range(N_BINS):
        m = (conf > edges[k]) & (conf <= edges[k + 1]) if k else (conf >= edges[0]) & (conf <= edges[1])
        if m.any():
            acc, mc = float(right[m].mean()), float(conf[m].mean()); ece += m.mean() * abs(acc - mc)
        else:
            acc, mc = None, None
        bins.append({"lo": round(float(edges[k]), 4), "hi": round(float(edges[k + 1]), 4), "n": int(m.sum()),
                     "accuracy": acc, "mean_confidence": mc})
    return {"T": T, "accuracy": float(right.mean()), "ece_15_bins": float(ece), "nll": nll(logits, y, T),
            "brier_p_credible": float(((p[:, 1] - (y == 1)) ** 2).mean()),
            "median_confidence_when_wrong": float(np.median(conf[~right])) if (~right).any() else None,
            "median_confidence_when_right": float(np.median(conf[right])),
            "share_predictions_conf_at_least_0.99": float((conf >= 0.99).mean()),
            "errors_conf_at_least_0.90": int((conf[~right] >= 0.90).sum()),
            "errors_conf_at_least_0.99": int((conf[~right] >= 0.99).sum()),
            "reliability_bins": bins}, pred


# ---------- logits I/O ----------
def write_logits(path, ids, y, logits):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["id", "y_true", "logit_not_credible", "logit_credible"])
        for i, t, (a, b) in zip(ids, y, logits): w.writerow([i, int(t), f"{a:.6f}", f"{b:.6f}"])


def read_logits(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8", newline="")))
    return ([r["id"] for r in rows], [int(r["y_true"]) for r in rows],
            [[float(r["logit_not_credible"]), float(r["logit_credible"])] for r in rows])


def compute_logits(name, split, source="auto"):
    """Run the fine-tuned model on data/processed/<split>.csv and return ids, labels and logits in label-id order."""
    import pandas as pd, torch
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import predict  # loads configs/config.yaml relative to the repo root
    f = ROOT / predict.CFG["data"]["splits_dir"] / f"{split}.csv"
    if not f.exists():
        raise SystemExit(f"missing {f}: run python src/clean_data.py --input data/raw/combined_dataset_clean.csv && python src/make_splits.py")
    df = pd.read_csv(f); tok, model = predict.load(name, source)
    nc = int(model.config.label2id.get("not_credible", 0)) if model.config.label2id else 0
    cr = 1 - nc
    out = []
    with torch.no_grad():
        for i in range(0, len(df), 32):
            b = tok(df.text.astype(str).tolist()[i:i + 32], truncation=True, max_length=predict.MAXLEN, padding=True,
                    return_tensors="pt").to(predict.DEV)
            lg = model(**b).logits.float().cpu()
            out += [[float(r[nc]), float(r[cr])] for r in lg]
    return df["id"].tolist(), df["label"].astype(int).tolist(), out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-logits", action="store_true", help="skip model runs; use saved results/<model>/*_logits.csv")
    ap.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    ap.add_argument("--source", default="auto", choices=["auto", "local", "hub"])
    a = ap.parse_args(argv)
    np = _np(); temps, rep = {}, {}
    for m in a.models:
        d = RES / m
        if not a.from_logits:
            for split in ("val", "test"):
                ids, y, lg = compute_logits(m, split, a.source); write_logits(d / f"{split}_logits.csv", ids, y, lg)
                print(f"[{m}] wrote results/{m}/{split}_logits.csv ({len(y)} rows)")
        _, yv, lv = read_logits(d / "val_logits.csv"); tid, yt, lt = read_logits(d / "test_logits.csv")
        T = fit_temperature(lv, yv); temps[m] = round(T, 4)
        before, pred_b = report(lt, yt, 1.0); after, pred_a = report(lt, yt, T)
        assert (pred_b == pred_a).all(), f"{m}: temperature scaling changed a prediction"
        saved = RES / m / "test_predictions.csv"; same = None
        if saved.exists():  # inference reproduces the predictions reported in Chapter IV?
            sp = {r["id"]: int(r["y_pred"]) for r in csv.DictReader(open(saved, encoding="utf-8", newline=""))}
            # label ids: argmax column 0 = not credible (0), column 1 = credible (1)
            same = sum(sp.get(str(i)) == int(p) for i, p in zip(tid, pred_a)) / len(tid)
        rep[m] = {"T": round(T, 4), "n_val": len(yv), "n_test": len(yt), "val_nll_before": nll(lv, yv, 1.0),
                  "val_nll_after": nll(lv, yv, T), "predictions_identical": True,
                  "share_test_predictions_matching_test_predictions_csv": same, "before": before, "after": after}
        print(f"[{m}] T = {T:.4f}  test ECE {before['ece_15_bins']:.4f} -> {after['ece_15_bins']:.4f}  "
              f"NLL {before['nll']:.4f} -> {after['nll']:.4f}  median conf. when wrong "
              f"{before['median_confidence_when_wrong']:.3f} -> {after['median_confidence_when_wrong']:.3f}")
    stamp = {"fitted_on": "internal validation split (data/processed/val.csv)", "objective": "negative log-likelihood",
             "seed": 42, "date": datetime.date.today().isoformat(), "n_val": {m: rep[m]["n_val"] for m in rep},
             "note": "Predictions are unchanged; only confidence values change. Never fitted on the Taglish set."}
    (RES / "calibration.json").write_text(json.dumps({"temperatures": temps, **stamp}, indent=2), encoding="utf-8")
    (RES / "calibration_report.json").write_text(json.dumps({"evaluated_on": "internal test split (data/processed/test.csv)",
        "ece_bins": N_BINS, "confidence": "probability of the predicted class", **stamp, "models": rep}, indent=2), encoding="utf-8")
    print("wrote results/calibration.json and results/calibration_report.json")


if __name__ == "__main__":
    main()
