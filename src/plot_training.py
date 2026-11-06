"""Loss curves (train vs validation, per epoch) from results/<model>/training_log.json -> results/figures/loss_curves.png"""
import json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAMES = {"roberta-tagalog": "RoBERTa-Tagalog", "mbert": "mBERT", "xlmr": "XLM-RoBERTa"}
if __name__ == "__main__":
    Path("results/figures").mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8), sharey=True)
    for ax, (k, name) in zip(axes, NAMES.items()):
        log = json.load(open(f"results/{k}/training_log.json"))
        tr = [(e["epoch"], e["loss"]) for e in log if "loss" in e and "eval_loss" not in e]
        va = [(e["epoch"], e["eval_loss"]) for e in log if "eval_loss" in e]
        ax.plot(*zip(*tr), marker="o", label="train loss"); ax.plot(*zip(*va), marker="s", label="validation loss")
        b = min(va, key=lambda x: x[1]); ax.scatter([b[0]], [b[1]], s=160, facecolors="none", edgecolors="red", label=f"best (epoch {int(b[0])})")
        ax.set_title(name); ax.set_xlabel("epoch"); ax.set_xticks([int(x[0]) for x in va]); ax.grid(alpha=.3)
    axes[0].set_ylabel("loss"); axes[0].legend(); fig.suptitle("Training and validation loss (early stopping on validation loss)")
    fig.tight_layout(); fig.savefig("results/figures/loss_curves.png", dpi=160); print("saved results/figures/loss_curves.png")
