"""Taglish results split by post length (fixed bins), per model. Run AFTER evaluate_external.py (and before reading any scores).

    python src/length_bins.py --csv data/external/taglish_test.csv

Bins are fixed here in advance, NOT chosen from Taglish scores: under 20, 20-39, 40-99, 100+ words. They come from the word-count
spread of the collected posts (median 31; only 9 of 750 had 100+ words, so 100+ is mostly empty unless a long-form set is added)
and the training articles (about 90% had 100+ words, THESIS_INPUTS section 2). The primary result stays the full set. Needs results/<model>/external_predictions.csv (id, y_true, y_pred, prob_credible) and the test CSV
(id, text, label). Labels: 1 = credible, 0 = not credible (positive class). Output: results/external_length_bins.json / .md.
"""
import argparse, csv, json, math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ["roberta-tagalog", "mbert", "xlmr"]
BINS = [("under_20_words", 0, 20), ("20_to_39_words", 20, 40), ("40_to_99_words", 40, 100), ("100_plus_words", 100, 10**9)]


def wilson(k, n, z=1.96):
    if n == 0: return [None, None]
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [round((c - h) / d, 4), round((c + h) / d, 4)]


def stats(rows):
    n = len(rows); ok = sum(r["y_true"] == r["y_pred"] for r in rows)
    cred = [r for r in rows if r["y_true"] == 1]; nc = [r for r in rows if r["y_true"] == 0]
    f = lambda k, m: None if m == 0 else round(k / m, 4)
    return {"n": n, "n_credible": len(cred), "n_not_credible": len(nc),
            "accuracy": f(ok, n), "accuracy_95ci_wilson": wilson(ok, n),
            "credible_called_not_credible": f(sum(r["y_pred"] == 0 for r in cred), len(cred)),
            "not_credible_recall": f(sum(r["y_pred"] == 0 for r in nc), len(nc))}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(ROOT / "data/external/taglish_test.csv")); ap.add_argument("--results", default=str(ROOT / "results"))
    a = ap.parse_args(argv)
    words = {r["id"]: len(r["text"].split()) for r in csv.DictReader(open(a.csv, encoding="utf-8-sig", newline=""))}
    out, md = {"bins_words": {b[0]: [b[1], b[2]] for b in BINS}, "models": {}}, []
    for m in MODELS:
        p = Path(a.results) / m / "external_predictions.csv"
        if not p.exists(): continue
        rows = [{"id": r["id"], "y_true": int(r["y_true"]), "y_pred": int(r["y_pred"]), "w": words.get(r["id"])} for r in csv.DictReader(open(p, newline=""))]
        missing = sum(r["w"] is None for r in rows)
        if missing: raise SystemExit(f"{m}: {missing} prediction ids not found in {a.csv}")
        res = {"all_posts": stats(rows)}
        for name, lo, hi in BINS: res[name] = stats([r for r in rows if lo <= r["w"] < hi])
        out["models"][m] = res
        for k, v in res.items():
            md.append(f"| {m} | {k} | {v['n']} | {v['accuracy']} | {v['credible_called_not_credible']} | {v['not_credible_recall']} |")
    if not out["models"]: raise SystemExit("No results/<model>/external_predictions.csv found. Run evaluate_external.py first.")
    res_dir = Path(a.results)
    (res_dir / "external_length_bins.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    head = ["# Taglish results by post length (fixed bins; primary result = all_posts)", "",
            "| Model | Bin | n | Accuracy | Credible called not credible | Not-credible recall |", "|---|---|---|---|---|---|"]
    (res_dir / "external_length_bins.md").write_text("\n".join(head + md) + "\n", encoding="utf-8")
    print("\n".join(head + md))


if __name__ == "__main__":
    main()
