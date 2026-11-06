"""95% bootstrap confidence intervals on the internal test set (paired resampling, so model differences are comparable).

    python src/bootstrap_ci.py   -> results/bootstrap_ci.json
Needs results/<model>/test_predictions.csv for each model (run train.py and baseline.py first).
"""
import json
from pathlib import Path
import numpy as np, pandas as pd

MODELS = ["roberta-tagalog", "mbert", "xlmr", "baseline-tfidf"]


def wf1(y, p):  # weighted F1 for binary labels
    n, tot = len(y), 0.0
    for c in (0, 1):
        tp = np.sum((y == c) & (p == c)); fp = np.sum((y != c) & (p == c)); fn = np.sum((y == c) & (p != c))
        f = 0.0 if (2 * tp + fp + fn) == 0 else 2 * tp / (2 * tp + fp + fn)
        tot += f * np.sum(y == c)
    return tot / n


def ci(a): return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]


def main(B=2000, seed=42):
    P = {m: pd.read_csv(f"results/{m}/test_predictions.csv").sort_values("id").reset_index(drop=True) for m in MODELS if Path(f"results/{m}/test_predictions.csv").exists()}
    y = next(iter(P.values())).y_true.values; n = len(y); rng = np.random.default_rng(seed); idx = rng.integers(0, n, size=(B, n))
    acc = {m: np.array([np.mean(y[i] == p.y_pred.values[i]) for i in idx]) for m, p in P.items()}
    f1w = {m: np.array([wf1(y[i], p.y_pred.values[i]) for i in idx]) for m, p in P.items()}
    out = {"resamples": B, "n_test": int(n), "per_model": {}, "paired_accuracy_difference": {}}
    for m in P:
        out["per_model"][m] = {"accuracy": float(np.mean(y == P[m].y_pred.values)), "accuracy_ci95": ci(acc[m]),
                               "weighted_f1": float(wf1(y, P[m].y_pred.values)), "weighted_f1_ci95": ci(f1w[m])}
    ms = list(P)
    for i, a in enumerate(ms):
        for b in ms[i + 1:]:
            d = acc[a] - acc[b]; out["paired_accuracy_difference"][f"{a} minus {b}"] = {"mean": float(d.mean()), "ci95": ci(d), "ci_excludes_zero": bool(ci(d)[0] > 0 or ci(d)[1] < 0)}
    json.dump(out, open("results/bootstrap_ci.json", "w"), indent=2); print(json.dumps(out, indent=1)[:1800])


if __name__ == "__main__":
    main()
