"""Side-by-side metrics + pairwise McNemar tests (alpha=0.05) for internal and external test sets.

    python src/compare.py
Holm-corrects the 3 transformer pairs. Reads results/<model>/{test,external}_predictions.csv, writes results/comparison.md
"""
import itertools, json
from pathlib import Path
import pandas as pd
from scipy.stats import binomtest, chi2

MODELS = ["roberta-tagalog", "mbert", "xlmr", "baseline-tfidf"]
TRANSFORMERS = ["roberta-tagalog", "mbert", "xlmr"]  # the 3 pairwise tests of the main question get Holm correction


def mcnemar(y, p1, p2):
    c1, c2 = p1 == y, p2 == y
    b, c = int((c1 & ~c2).sum()), int((~c1 & c2).sum())
    if b + c == 0: return b, c, 1.0
    if b + c < 25: return b, c, float(binomtest(b, b + c, 0.5).pvalue)          # exact
    stat = (abs(b - c) - 1) ** 2 / (b + c); return b, c, float(1 - chi2.cdf(stat, 1))  # continuity-corrected


def holm(pvals):
    """Holm step-down adjusted p-values (same order as input). Controls family-wise error at alpha."""
    order = sorted(range(len(pvals)), key=lambda i: pvals[i]); adj = [0.0] * len(pvals); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(pvals) - rank) * pvals[i])); adj[i] = run
    return adj


def section(kind, fname_pred, fname_met, ci_file=None):
    lines = [f"## {kind}\n", "| Model | Acc | Precision (NC) | Recall (NC) | F1 (NC) | Macro F1 | Weighted F1 |", "|---|---|---|---|---|---|---|"]
    preds = {}
    for m in MODELS:
        pm, mm = Path("results") / m / fname_pred, Path("results") / m / fname_met
        if not (pm.exists() and mm.exists()): continue
        j = json.load(open(mm)); preds[m] = pd.read_csv(pm).sort_values("id").reset_index(drop=True)
        lines.append(f"| {m} | {j['accuracy']:.4f} | {j['precision_not_credible']:.4f} | {j['recall_not_credible']:.4f} | {j['f1_not_credible']:.4f} | {j['macro_f1']:.4f} | {j['weighted_f1']:.4f} |")
    if len(preds) < 2: return "\n".join(lines) + "\n\n(not enough models for McNemar)\n"
    pairs = [(m1, m2, *mcnemar(preds[m1].y_true.values, preds[m1].y_pred.values, preds[m2].y_pred.values)) for m1, m2 in itertools.combinations(preds, 2)]
    fam = [i for i, (m1, m2, *_r) in enumerate(pairs) if m1 in TRANSFORMERS and m2 in TRANSFORMERS]
    adj = dict(zip(fam, holm([pairs[i][4] for i in fam])))
    lines += ["", "**McNemar (b = row-model right & col-model wrong; c = reverse)**", "",
              "Holm correction is applied over the 3 pairwise tests between the transformers; pairs with the TF-IDF baseline are descriptive and uncorrected.", "",
              "| Pair | b | c | p-value | Holm-adjusted p | Significant (adj. p<0.05) |", "|---|---|---|---|---|---|"]
    for i, (m1, m2, b, c, p) in enumerate(pairs):
        if i in adj: lines.append(f"| {m1} vs {m2} | {b} | {c} | {p:.4g} | {adj[i]:.4g} | {'yes' if adj[i] < 0.05 else 'no'} |")
        else: lines.append(f"| {m1} vs {m2} | {b} | {c} | {p:.4g} | n/a (uncorrected) | {'yes' if p < 0.05 else 'no'} (uncorrected) |")
    if ci_file and Path(ci_file).exists():
        j = json.load(open(ci_file))
        lines += ["", f"**95% paired bootstrap CIs** ({j['resamples']} resamples, n = {j['n']}; single run, single seed)", "", "| Model | Accuracy [95% CI] | F1 (NC) [95% CI] | Weighted F1 [95% CI] |", "|---|---|---|---|"]
        f = lambda d, k: f"{d[k]:.4f} [{d[k + '_ci95'][0]:.4f}, {d[k + '_ci95'][1]:.4f}]"
        for m, d in j["per_model"].items(): lines.append(f"| {m} | {f(d, 'accuracy')} | {f(d, 'f1_not_credible')} | {f(d, 'weighted_f1')} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    md = "# Model comparison (NC = not credible, the positive class)\n\n" + section("Internal formal test set", "test_predictions.csv", "metrics.json") \
         + "\n" + section("Supplementary Taglish election test set", "external_predictions.csv", "external_metrics.json", "results/external_bootstrap_ci.json")
    Path("results/comparison.md").write_text(md); print(md)
