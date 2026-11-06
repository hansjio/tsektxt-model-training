"""Over/underfitting diagnosis from the saved training logs (no GPU, no extra libs).

    python src/diagnose_fit.py

Reads results/<model>/training_log.json (+ metrics.json) and prints, per model:
  - per-epoch train loss vs validation loss, the gap, and the best epoch
  - a verdict (heuristic) and what it does / does not prove
Writes results/fit_diagnosis.md and results/fit_diagnosis.json.

Caveat: train loss is the running average DURING the epoch (dropout on, weights changing), so it
overstates train error a little. For a clean train-vs-val-vs-test accuracy gap, run
`python src/predict.py --split-gap` on a machine that has the checkpoints.
"""
import json
from pathlib import Path

MODELS = ["roberta-tagalog", "mbert", "xlmr"]


def load(model):
    log = json.load(open(f"results/{model}/training_log.json"))
    train, val = {}, {}
    for r in log:
        ep = round(r.get("epoch", 0))
        if "eval_loss" in r:
            val[ep] = r["eval_loss"]
        elif "loss" in r:
            train[ep] = r["loss"]
    return train, val


def verdict(train, val):
    eps = sorted(val)
    best = min(eps, key=lambda e: val[e])
    last = eps[-1]
    gap_best = val[best] - train[best]
    gap_last = val[last] - train[last]
    rise = val[last] - val[best]
    notes = []
    if train[last] > 0.45:
        notes.append("UNDERFITTING: training loss still high")
    if rise > 0.02 and train[last] < train[best]:
        notes.append(f"OVERFITTING after epoch {best}: val loss rose {rise:+.3f} while train loss kept falling")
    if gap_last > 0.1:
        notes.append(f"large generalisation gap at last epoch ({gap_last:.3f})")
    if not notes:
        notes.append("no clear over/underfitting signal")
    return best, gap_best, gap_last, notes


def main():
    out, md = {}, ["# Fit diagnosis (from training logs)\n"]
    for m in MODELS:
        try:
            tr, va = load(m)
        except FileNotFoundError:
            print(f"skip {m}: no log"); continue
        best, gb, gl, notes = verdict(tr, va)
        met = json.load(open(f"results/{m}/metrics.json"))
        md.append(f"## {m}\n\n| epoch | train loss | val loss | gap |\n|---|---|---|---|")
        for e in sorted(va):
            md.append(f"| {e}{' (best)' if e == best else ''} | {tr.get(e, float('nan')):.4f} | {va[e]:.4f} | {va[e]-tr.get(e, float('nan')):+.4f} |")
        md.append(f"\nTest accuracy {met['accuracy']:.4f}, weighted F1 {met['weighted_f1']:.4f}.\n")
        md += [f"- {n}" for n in notes] + [""]
        out[m] = {"best_epoch": best, "gap_at_best": gb, "gap_at_last": gl, "notes": notes,
                  "train_loss": tr, "val_loss": va}
        print(f"\n{m}: best epoch {best}, gap@best {gb:+.3f}, gap@last {gl:+.3f}")
        for n in notes: print("  -", n)
    md.append("## How to read this\n\n- Val loss bottoming at epoch 1-2 then rising is mild, normal overfitting; the saved model is the best-epoch checkpoint "
              "(early stopping), so the reported test numbers are NOT from the overfit epochs.\n"
              "- Val and test come from the same distribution (same source sites), so a small val-test gap does NOT rule out "
              "overfitting to source style (see leakage_probe.json). Only the Taglish set tests that.\n"
              "- Single seed per model: do not over-read differences smaller than seed noise. Run 3 seeds for mean +/- sd.\n")
    Path("results/fit_diagnosis.md").write_text("\n".join(md), encoding="utf-8")
    json.dump(out, open("results/fit_diagnosis.json", "w"), indent=2)
    print("\nwrote results/fit_diagnosis.md and .json")


if __name__ == "__main__":
    main()
