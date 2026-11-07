"""Paper figures, re-runnable. Reads results/ and writes PNG (300 dpi) + PDF to results/figures/.
Any figure whose input is missing is skipped (and printed), so re-run after new results exist.

    python src/make_figures.py

Numbers come only from results/*.json and results/<model>/*.csv. Label convention: 1 = credible, 0 = not credible (positive class).
Single seed, one run: figures show no ranking claim between RoBERTa-Tagalog and XLM-R (McNemar p = 0.55).
Also writes results/length_distribution.json (word counts only, no text) when data/raw and/or data/external/merged_for_annotation.csv exist.
"""
import csv, json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RES = Path(sys.argv[sys.argv.index("--results") + 1]) if "--results" in sys.argv else ROOT / "results"  # --results only for smoke tests
FIG = RES / "figures"
MODELS = ["roberta-tagalog", "mbert", "xlmr"]
ALL = MODELS + ["baseline-tfidf"]
NAME = {"roberta-tagalog": "RoBERTa-Tagalog", "mbert": "mBERT", "xlmr": "XLM-RoBERTa", "baseline-tfidf": "TF-IDF baseline"}
# Okabe-Ito (colour-blind safe)
COL = {"roberta-tagalog": "#0072B2", "mbert": "#E69F00", "xlmr": "#009E73", "baseline-tfidf": "#7F7F7F"}
C_RED, C_BLUE, C_GREY = "#D55E00", "#0072B2", "#999999"
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": .3, "axes.axisbelow": True})
W = 6.3  # A4 text width in inches (approx.)
done, skipped = [], []


def load(rel):
    p = RES / rel
    return json.load(open(p, encoding="utf-8")) if p.exists() else None


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.png", dpi=300, bbox_inches="tight"); fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig); done.append(name); print(f"saved results/figures/{name}.png/.pdf")


def skip(name, why):
    skipped.append(name); print(f"SKIP {name}: missing {why}")


def preds(rel):
    p = RES / rel
    if not p.exists(): return None
    return [{"y_true": int(r["y_true"]), "y_pred": int(r["y_pred"]), "p": float(r["prob_credible"]) if r.get("prob_credible") not in (None, "") else None}
            for r in csv.DictReader(open(p, encoding="utf-8", newline=""))]


def ci_bars(data, metrics, labels, title, name, models):
    """data = per_model dict with <metric> and <metric>_ci95."""
    models = [m for m in models if m in data]
    fig, ax = plt.subplots(figsize=(W, 3.2)); w = .8 / len(metrics); x = np.arange(len(models))
    hatches = ["", "//", "..", "xx"]
    for i, (k, lab) in enumerate(zip(metrics, labels)):
        v = np.array([data[m][k] for m in models]); ci = np.array([data[m][k + "_ci95"] for m in models])
        ax.bar(x + (i - (len(metrics) - 1) / 2) * w, v, w, yerr=[v - ci[:, 0], ci[:, 1] - v], capsize=3, label=lab, hatch=hatches[i],
               color=[COL[m] for m in models], edgecolor="black", linewidth=.6, error_kw={"lw": 1})
        for xi, vi, c_ in zip(x + (i - (len(metrics) - 1) / 2) * w, v, ci): ax.text(xi, c_[1] + .004, f"{vi:.3f}", ha="center", va="bottom", fontsize=6.5)
    lo = min(c[0] for m in models for k in metrics for c in [data[m][k + "_ci95"]])
    ax.set_ylim(max(0, lo - .1), 1.0); ax.set_yticks([t for t in ax.get_yticks() if t <= 1.0001]); ax.set_ylim(max(0, lo - .1), 1.0); ax.set_xticks(x); ax.set_xticklabels([NAME[m] for m in models]); ax.set_ylabel("score")
    h = [plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="black", hatch=hatches[i]) for i in range(len(metrics))]
    ax.legend(h, labels, loc="upper center", bbox_to_anchor=(.5, -.12), ncol=2, title="whiskers: 95% bootstrap CI", fontsize=7, title_fontsize=7)
    ax.set_title(title); fig.tight_layout(); save(fig, name)


def fig_a():
    d = load("bootstrap_ci.json")
    if not d: return skip("fig_a_internal_metrics", "bootstrap_ci.json")
    ci_bars(d["per_model"], ["accuracy", "weighted_f1"], ["Accuracy", "Weighted F1"],
            f"Internal test set (n = {d['n_test']})", "fig_a_internal_metrics", ALL)


def cm_axes(ax, y, p, title):
    m = np.zeros((2, 2), int)
    for t, q in zip(y, p): m[1 - t, 1 - q] += 1  # row/col 0 = not credible (positive), 1 = credible
    ax.imshow(m / m.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{m[i, j]}\n({m[i, j] / m[i].sum():.1%})", ha="center", va="center", fontsize=8, color="white" if m[i, j] / m[i].sum() > .5 else "black")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["not credible", "credible"]); ax.set_yticks([0, 1]); ax.set_yticklabels(["not credible", "credible"])
    ax.set_xlabel("predicted"); ax.set_title(title); ax.grid(False)


def fig_b():
    fig, axes = plt.subplots(1, 4, figsize=(W * 1.6, 2.9)); ok = 0
    for ax, m in zip(axes, ALL):
        r = preds(f"{m}/test_predictions.csv")
        if not r: ax.axis("off"); print(f"  (fig_b: no predictions for {m})"); continue
        cm_axes(ax, [x["y_true"] for x in r], [x["y_pred"] for x in r], NAME[m]); ok += 1
    axes[0].set_ylabel("true")
    if not ok: plt.close(fig); return skip("fig_b_confusion_matrices", "results/<model>/test_predictions.csv")
    fig.suptitle("Internal test set: confusion matrices (cell = count and share of true class)", y=1.02); fig.tight_layout(); save(fig, "fig_b_confusion_matrices")


def fig_c():
    d = load("leakage_probe.json")
    if not d: return skip("fig_c_leakage_probe", "leakage_probe.json")
    keys = [("full_text", "Full text"), ("first_12_words", "First 12 words"), ("last_12_words", "Last 12 words")]
    fig, ax = plt.subplots(figsize=(W * .8, 3)); x = np.arange(3)
    ax.bar(x - .2, [d[k]["accuracy"] for k, _ in keys], .4, color=C_BLUE, label="Accuracy", edgecolor="black", linewidth=.6)
    ax.bar(x + .2, [d[k]["macro_f1"] for k, _ in keys], .4, color=C_RED, hatch="//", label="Macro F1", edgecolor="black", linewidth=.6)
    for xi, k in zip(x, keys):
        ax.text(xi - .2, d[k[0]]["accuracy"] + .01, f"{d[k[0]]['accuracy']:.3f}", ha="center", fontsize=7)
        ax.text(xi + .2, d[k[0]]["macro_f1"] + .01, f"{d[k[0]]['macro_f1']:.3f}", ha="center", fontsize=7)
    ax.axhline(d["majority_class_accuracy"], color="black", ls="--", lw=1, label=f"majority-class accuracy ({d['majority_class_accuracy']:.3f})")
    ax.set_xticks(x); ax.set_xticklabels([l for _, l in keys]); ax.set_ylim(.5, 1.12); ax.set_ylabel("score"); ax.legend(loc="upper right", fontsize=7, ncol=1)
    ax.set_title("Leakage probe: TF-IDF on parts of the text"); fig.tight_layout(); save(fig, "fig_c_leakage_probe")


def conf_fig(models_data, pred_prefix, title, name, n_label):
    """Boxplot-free: median confidence when wrong vs right per model, plus shares of errors >= .90 / .99 confidence."""
    ms = [m for m in MODELS if m in models_data]
    fig, axes = plt.subplots(1, 2, figsize=(W * 1.15, 3.1)); x = np.arange(len(ms))
    a = axes[0]
    for off, key, lab, c, h in [(-.2, "median_confidence_when_wrong", "wrong", C_RED, "//"), (.2, "median_confidence_when_right", "right", C_BLUE, "")]:
        v = [models_data[m][key] for m in ms]
        a.bar(x + off, v, .4, color=c, hatch=h, label=lab, edgecolor="black", linewidth=.6)
        for xi, vi in zip(x + off, v): a.text(xi, vi + .005, f"{vi:.3f}", ha="center", fontsize=7)
    a.set_xticks(x); a.set_xticklabels([NAME[m] for m in ms], fontsize=7.5); a.set_ylim(.5, 1.2); a.set_yticks([.5, .6, .7, .8, .9, 1.0]); a.set_ylabel("median confidence in predicted class")
    a.legend(title="prediction", loc="upper center", ncol=2, fontsize=7, title_fontsize=7); a.set_title("Median confidence")
    b = axes[1]
    for off, key, lab, c, h in [(-.2, "errors_with_confidence_at_least_0.90", "conf. >= 0.90", C_BLUE, ""), (.2, "errors_with_confidence_at_least_0.99", "conf. >= 0.99", C_RED, "//")]:
        v = [models_data[m][key] / models_data[m]["errors"] for m in ms]
        b.bar(x + off, v, .4, color=c, hatch=h, label=lab, edgecolor="black", linewidth=.6)
        for xi, vi in zip(x + off, v): b.text(xi, vi + .01, f"{vi:.0%}", ha="center", fontsize=7)
    b.set_xticks(x); b.set_xticklabels([NAME[m] for m in ms], fontsize=7.5); b.set_ylim(0, 1.1); b.set_ylabel("share of errors")
    b.legend(loc="upper right", fontsize=7); b.set_title("Errors made with high confidence")
    fig.suptitle(title, y=1.02); fig.tight_layout(); save(fig, name)


def fig_d():
    d = load("confidence_check.json")
    if not d: return skip("fig_d_confidence_internal", "confidence_check.json")
    conf_fig(d["models"], "", "Internal test set: confidence when wrong vs right", "fig_d_confidence_internal", "")


def fig_j():
    d = load("external_confidence_check.json")
    if not d: return skip("fig_j_confidence_taglish", "external_confidence_check.json")
    conf_fig(d["models"], "", "Taglish test set: confidence when wrong vs right", "fig_j_confidence_taglish", "")


def fig_e():
    d = load("domain_shift.json")
    if not d: return skip("fig_e_accuracy_vs_words_kept", "domain_shift.json")
    order = ["first_15_words", "first_30_words", "first_60_words", "full"]
    fig, ax = plt.subplots(figsize=(W * .85, 3.3)); mk = {"roberta-tagalog": "o", "mbert": "s", "xlmr": "^"}
    for m, v in d["models"].items():
        ks = [k for k in order if k in v["length"]]
        xs = [int(k.split("_")[1]) if k != "full" else 120 for k in ks]  # 'full' drawn at the right edge, labelled
        ax.plot(range(len(ks)), [v["length"][k]["accuracy"] for k in ks], marker=mk.get(m, "o"), color=COL[m], label=NAME[m])
    ax.set_xticks(range(len(order))); ax.set_xticklabels(["first 15", "first 30", "first 60", "full text"]); ax.set_xlabel("words of the article kept")
    ax.set_ylabel("accuracy"); ax.set_title(f"Internal test articles cut short (n = {d['n_internal_rows']}, inference only)"); ax.legend(); fig.tight_layout()
    save(fig, "fig_e_accuracy_vs_words_kept")


def word_counts():
    """Counts-only distribution from data/raw (by class) and the collected posts. Writes results/length_distribution.json (no text)."""
    out = {}
    raw = ROOT / "data/raw/combined_dataset_clean.csv"
    if raw.exists():
        c = {"0": [], "1": []}
        for r in csv.DictReader(open(raw, encoding="utf-8-sig", newline="")):
            if r.get("Label") in c: c[r["Label"]].append(len((r.get("Content") or "").split()))
        out["training_articles"] = {("credible" if k == "1" else "not_credible"): v for k, v in c.items()}
    post = ROOT / "data/external/merged_for_annotation.csv"
    if post.exists():
        out["collected_posts"] = [len((r.get("text") or "").split()) for r in csv.DictReader(open(post, encoding="utf-8-sig", newline="")) if (r.get("text") or "").strip()]
    old = (load("length_distribution.json") or {}).get("counts", {})  # keep parts whose source files are absent (e.g. on Colab)
    for k, v in old.items(): out.setdefault(k, v)
    if not out: return None
    res = {"note": "word counts (whitespace split) only, no text", "counts": out}
    (RES / "length_distribution.json").write_text(json.dumps(res), encoding="utf-8")
    return out


def fig_f():
    d = word_counts() or (load("length_distribution.json") or {}).get("counts")
    if not d: return skip("fig_f_length_distribution", "data/raw or merged_for_annotation.csv, or results/length_distribution.json")
    bins = np.logspace(0, np.log10(5000), 40); fig, ax = plt.subplots(figsize=(W * .9, 3.3))
    for key, lab, c, ls in [("credible", "training articles, credible", C_BLUE, "-"), ("not_credible", "training articles, not credible", C_RED, "-")]:
        v = (d.get("training_articles") or {}).get(key)
        if v: ax.hist(np.clip(np.maximum(v, 1), 1, 5000), bins=bins, density=True, histtype="step", lw=1.6, color=c, ls=ls, label=f"{lab} (n={len(v)}, median {int(np.median(v))})")
    p = d.get("collected_posts")
    if p: ax.hist(np.clip(np.maximum(p, 1), 1, 5000), bins=bins, density=True, histtype="stepfilled", alpha=.45, color=C_GREY, edgecolor="black", label=f"collected posts (n={len(p)}, median {int(np.median(p))})")
    ax.set_xscale("log"); ax.set_xlabel("words per text (log scale)"); ax.set_ylabel("density"); ax.legend(fontsize=7); ax.set_title("Text length: training articles vs collected posts")
    fig.tight_layout(); save(fig, "fig_f_length_distribution")


def fig_g():
    d = load("external_bootstrap_ci.json")
    if not d: return skip("fig_g_taglish_metrics", "external_bootstrap_ci.json")
    ci_bars(d["per_model"], ["accuracy", "f1_not_credible", "weighted_f1"], ["Accuracy", "F1 (not credible)", "Weighted F1"],
            f"Taglish test set (n = {d['n']})", "fig_g_taglish_metrics", ALL)


def fig_h():
    d = load("external_length_bins.json")
    if not d: return skip("fig_h_taglish_by_length", "external_length_bins.json")
    bins = list(d["bins_words"]); labs = ["<20", "20-39", "40-99", "100+"]
    fig, ax = plt.subplots(figsize=(W * .9, 3.3)); mk = {"roberta-tagalog": "o", "mbert": "s", "xlmr": "^"}
    for m, v in d["models"].items():
        ax.plot(range(4), [v[b]["accuracy"] if v[b]["n"] else np.nan for b in bins], marker=mk.get(m, "o"), color=COL.get(m, "k"), label=NAME.get(m, m))
    first = next(iter(d["models"].values()))
    ax.set_xticks(range(4)); ax.set_xticklabels([f"{l} words\n(n={first[b]['n']})" for l, b in zip(labs, bins)])
    ax.set_ylabel("accuracy"); ax.set_ylim(0, 1); ax.legend(); ax.set_title("Taglish test set: accuracy by post length")
    fig.tight_layout(); save(fig, "fig_h_taglish_by_length")


def fig_i():
    d = load("ig_top_tokens.json")
    if not d: return skip("fig_i_ig_top_words", "ig_top_tokens.json")
    tops = d["top_tokens"]; ms = [m for m in MODELS if m in tops]; K = 10
    fig, axes = plt.subplots(1, len(ms), figsize=(W * 1.5, 3.8))
    axes = np.atleast_1d(axes)
    for ax, m in zip(axes, ms):
        rows = tops[m]["toward_not_credible"][:K][::-1]
        ax.barh([r["word"] for r in rows], [r["sum_attr"] for r in rows], color=COL[m], edgecolor="black", linewidth=.5)
        ax.set_title(f"{NAME[m]}\n(n = {tops[m]['n_posts']} posts)", fontsize=9); ax.set_xlabel("summed attribution"); ax.tick_params(axis="y", labelsize=7.5)
    fig.suptitle("Integrated Gradients: words pushing toward 'not credible' (descriptive, not causal)", y=1.02); fig.tight_layout(); save(fig, "fig_i_ig_top_words")


if __name__ == "__main__":
    for f in (fig_a, fig_b, fig_c, fig_d, fig_e, fig_f, fig_g, fig_h, fig_i, fig_j):
        try: f()
        except Exception as e:  # one bad input must not stop the others
            skipped.append(f.__name__); print(f"SKIP {f.__name__}: {type(e).__name__}: {e}", file=sys.stderr)
    print(f"\nmade {len(done)}: {', '.join(done)}\nskipped {len(skipped)}: {', '.join(skipped)}")
