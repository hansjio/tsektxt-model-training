"""Side-by-side metrics + pairwise McNemar tests (alpha=0.05) for internal and external test sets.

    python src/compare.py
Reads results/<model>/{test,external}_predictions.csv, writes results/comparison.md
"""
import itertools, json
from pathlib import Path
import pandas as pd
from scipy.stats import binomtest, chi2

MODELS = ["roberta-tagalog", "mbert", "xlmr", "baseline-tfidf"]


def mcnemar(y, p1, p2):
    c1, c2 = p1 == y, p2 == y
    b, c = int((c1 & ~c2).sum()), int((~c1 & c2).sum())
    if b + c == 0: return b, c, 1.0
    if b + c < 25: return b, c, float(binomtest(b, b + c, 0.5).pvalue)          # exact
    stat = (abs(b - c) - 1) ** 2 / (b + c); return b, c, float(1 - chi2.cdf(stat, 1))  # continuity-corrected


def section(kind, fname_pred, fname_met):
    lines = [f"## {kind}\n", "| Model | Acc | Precision (NC) | Recall (NC) | F1 (NC) | Macro F1 | Weighted F1 |", "|---|---|---|---|---|---|---|"]
    preds = {}
    for m in MODELS:
        pm, mm = Path("results") / m / fname_pred, Path("results") / m / fname_met
        if not (pm.exists() and mm.exists()): continue
        j = json.load(open(mm)); preds[m] = pd.read_csv(pm).sort_values("id").reset_index(drop=True)
        lines.append(f"| {m} | {j['accuracy']:.4f} | {j['precision_not_credible']:.4f} | {j['recall_not_credible']:.4f} | {j['f1_not_credible']:.4f} | {j['macro_f1']:.4f} | {j['weighted_f1']:.4f} |")
    if len(preds) < 2: return "\n".join(lines) + "\n\n(not enough models for McNemar)\n"
    lines += ["", "**McNemar (b = row-model right & col-model wrong; c = reverse)**", "", "| Pair | b | c | p-value | Significant (p<0.05) |", "|---|---|---|---|---|"]
    for m1, m2 in itertools.combinations(preds, 2):
        b, c, p = mcnemar(preds[m1].y_true.values, preds[m1].y_pred.values, preds[m2].y_pred.values)
        lines.append(f"| {m1} vs {m2} | {b} | {c} | {p:.4g} | {'yes' if p < 0.05 else 'no'} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    md = "# Model comparison (NC = not credible, the positive class)\n\n" + section("Internal formal test set", "test_predictions.csv", "metrics.json") \
         + "\n" + section("Supplementary Taglish election test set", "external_predictions.csv", "external_metrics.json")
    Path("results/comparison.md").write_text(md); print(md)
