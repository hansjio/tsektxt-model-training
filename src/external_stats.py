"""Statistics for the supplementary Taglish test set (TEST ONLY, never trained on).

    python src/external_stats.py
Reads results/<model>/external_predictions.csv for each model that has one, and writes
  results/external_bootstrap_ci.json   95% paired bootstrap CIs (accuracy, F1 of not-credible, weighted F1) + paired accuracy differences
  results/external_mcnemar_holm.json   pairwise McNemar tests; Holm correction over the 3 transformer pairs
Run after evaluate_external.py (x3) and `baseline.py --external ...`, then run compare.py to put it all in results/comparison.md.
Labels: 1 = credible, 0 = not credible; not credible is the positive class. No torch needed.
"""
import argparse, itertools, json
from pathlib import Path
import numpy as np, pandas as pd
from bootstrap_ci import wf1, ci
from compare import MODELS, TRANSFORMERS, mcnemar, holm


def f1_nc(y, p):  # F1 of the positive class (not credible = 0)
    tp = np.sum((y == 0) & (p == 0)); fp = np.sum((y != 0) & (p == 0)); fn = np.sum((y == 0) & (p != 0))
    return 0.0 if (2 * tp + fp + fn) == 0 else 2 * tp / (2 * tp + fp + fn)


def load(res):
    P = {m: pd.read_csv(res / m / "external_predictions.csv").sort_values("id").reset_index(drop=True) for m in MODELS if (res / m / "external_predictions.csv").exists()}
    if not P: raise SystemExit("No results/<model>/external_predictions.csv found. Run evaluate_external.py first.")
    ref = next(iter(P.values()))
    for m, d in P.items():
        if not (d.id.astype(str).values == ref.id.astype(str).values).all() or not (d.y_true.values == ref.y_true.values).all():
            raise SystemExit(f"{m}: ids/labels differ from the other models (all models must be scored on the same csv).")
    return P


def main(res="results", B=2000, seed=42):
    res = Path(res); P = load(res); y = next(iter(P.values())).y_true.values; n = len(y)
    idx = np.random.default_rng(seed).integers(0, n, size=(B, n))
    boot = {m: {"accuracy": np.array([np.mean(y[i] == d.y_pred.values[i]) for i in idx]),
                "f1_not_credible": np.array([f1_nc(y[i], d.y_pred.values[i]) for i in idx]),
                "weighted_f1": np.array([wf1(y[i], d.y_pred.values[i]) for i in idx])} for m, d in P.items()}
    point = {"accuracy": lambda p: float(np.mean(y == p)), "f1_not_credible": lambda p: float(f1_nc(y, p)), "weighted_f1": lambda p: float(wf1(y, p))}
    out = {"resamples": B, "seed": seed, "n": int(n), "note": "paired bootstrap over posts; one run, one seed; labels follow the fact-check source",
           "per_model": {}, "paired_accuracy_difference": {}}
    if n < 100: out["warning"] = f"only {n} posts: intervals will be wide"
    for m, d in P.items():
        out["per_model"][m] = {}
        for k, f in point.items(): out["per_model"][m].update({k: f(d.y_pred.values), k + "_ci95": ci(boot[m][k])})
    for a, b in itertools.combinations(P, 2):
        dd = boot[a]["accuracy"] - boot[b]["accuracy"]
        out["paired_accuracy_difference"][f"{a} minus {b}"] = {"mean": float(dd.mean()), "ci95": ci(dd), "ci_excludes_zero": bool(ci(dd)[0] > 0 or ci(dd)[1] < 0)}
    json.dump(out, open(res / "external_bootstrap_ci.json", "w"), indent=2)

    rows = []
    for a, b in itertools.combinations(P, 2):
        bb, cc, p = mcnemar(y, P[a].y_pred.values, P[b].y_pred.values)
        rows.append({"pair": f"{a} vs {b}", "b": bb, "c": cc, "p": p, "family": a in TRANSFORMERS and b in TRANSFORMERS})
    fam = [r for r in rows if r["family"]]
    for r, adj in zip(fam, holm([r["p"] for r in fam])): r["p_holm"] = adj; r["significant_holm_0.05"] = bool(adj < 0.05)
    for r in rows:
        if not r["family"]: r["p_holm"] = None; r["note"] = "descriptive, uncorrected"
    json.dump({"n": int(n), "alpha": 0.05, "correction": "Holm over the 3 pairwise transformer tests", "tests": rows}, open(res / "external_mcnemar_holm.json", "w"), indent=2)
    print(json.dumps(out["per_model"], indent=1)); print(json.dumps(rows, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--results-dir", default="results"); ap.add_argument("--resamples", type=int, default=2000)
    a = ap.parse_args(); main(a.results_dir, a.resamples)
