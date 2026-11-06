"""Two cheap checks on WHY the models call short social posts 'not credible'. Inference only: no training, no tuning.

    python src/domain_shift_checks.py                      (Colab: after clean_data.py + make_splits.py; CPU works but is slow)
    python src/domain_shift_checks.py --max-rows 400       (quick run on a random subset)

A. Length: accuracy on the INTERNAL test split when each article is cut to its first 15 / 30 / 60 words (and in full).
   Reports accuracy, share of CREDIBLE articles wrongly called not credible, and recall on not-credible articles.
B. Links: (1) internal test rows with a dummy link appended vs not; (2) real collected posts that contain a URL, scored with the
   URL kept vs removed. The collected posts are UNLABELLED here, so only the change in P(not credible) and flip rate are reported,
   never accuracy. The Taglish set is TEST ONLY; this script never trains or tunes on it.

Output: results/domain_shift.json and results/domain_shift.md. Label convention: 1 = credible, 0 = not credible (positive class).
"""
import argparse, csv, json, random, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
URL = re.compile(r"https?://\S+|www\.\S+")
DUMMY_LINK = "https://news.example.com/article"


def read_csv(p):
    return list(csv.DictReader(open(p, encoding="utf-8-sig", newline="")))


def p_not_credible(tok, model, texts, nc_idx, maxlen, bs=32):
    import torch
    out = []
    for i in range(0, len(texts), bs):
        b = tok(texts[i:i + bs], truncation=True, max_length=maxlen, padding=True, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out += torch.softmax(model(**b).logits, -1)[:, nc_idx].tolist()
    return out


def length_part(p_fn, rows, cuts):
    res = {}
    for cut in cuts:
        texts = [" ".join(r["text"].split()[:cut]) if cut else r["text"] for r in rows]
        p = p_fn(texts)
        pred_nc = [x > 0.5 for x in p]
        true_nc = [r["label"] == "0" for r in rows]
        cred = [i for i, t in enumerate(true_nc) if not t]; nc = [i for i, t in enumerate(true_nc) if t]
        res["full" if not cut else f"first_{cut}_words"] = {
            "accuracy": round(sum(pr == t for pr, t in zip(pred_nc, true_nc)) / len(rows), 4),
            "credible_called_not_credible": round(sum(pred_nc[i] for i in cred) / max(1, len(cred)), 4),
            "not_credible_recall": round(sum(pred_nc[i] for i in nc) / max(1, len(nc)), 4),
            "mean_p_not_credible": round(sum(p) / len(p), 4)}
    return res


def link_part(p_fn, rows, posts):
    out = {}
    base = p_fn([r["text"] for r in rows]); linked = p_fn([r["text"] + " " + DUMMY_LINK for r in rows])
    true_nc = [r["label"] == "0" for r in rows]
    out["internal_test_dummy_link_appended"] = {
        "n": len(rows),
        "accuracy_without": round(sum((x > .5) == t for x, t in zip(base, true_nc)) / len(rows), 4),
        "accuracy_with": round(sum((x > .5) == t for x, t in zip(linked, true_nc)) / len(rows), 4),
        "mean_p_nc_change": round(sum(b - a for a, b in zip(base, linked)) / len(rows), 4),
        "flips": sum((a > .5) != (b > .5) for a, b in zip(base, linked))}
    if posts:
        kept = p_fn([r["text"] for r in posts]); stripped = p_fn([URL.sub("", r["text"]).strip() for r in posts])
        out["collected_posts_with_url_kept_vs_removed_UNLABELLED"] = {
            "n": len(posts), "mean_p_nc_url_kept": round(sum(kept) / len(posts), 4), "mean_p_nc_url_removed": round(sum(stripped) / len(posts), 4),
            "share_not_credible_url_kept": round(sum(x > .5 for x in kept) / len(posts), 4),
            "share_not_credible_url_removed": round(sum(x > .5 for x in stripped) / len(posts), 4),
            "flips": sum((a > .5) != (b > .5) for a, b in zip(kept, stripped))}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-csv", default=str(ROOT / "data/processed/test.csv"), help="internal test split: id,text,label")
    ap.add_argument("--posts-csv", default=str(ROOT / "data/external/merged_for_annotation.csv"), help="collected posts (unlabelled use only)")
    ap.add_argument("--models", nargs="+", default=None); ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--out-dir", default=str(ROOT / "results")); ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(argv)
    import interpret
    rows = read_csv(a.test_csv)
    rows = [r for r in rows if r["text"].strip()]
    random.Random(a.seed).shuffle(rows)
    if a.max_rows: rows = rows[:a.max_rows]
    posts = [r for r in read_csv(a.posts_csv) if r.get("text", "").strip() and URL.search(r["text"])] if Path(a.posts_csv).exists() else []
    result = {"n_internal_rows": len(rows), "n_posts_with_url": len(posts), "max_length": interpret.MAXLEN, "seed": a.seed, "models": {}}
    for name in a.models or interpret.NAMES:
        tok, model, _ = interpret.load(name)
        nc_idx, _ = interpret.label_index(model)
        p_fn = lambda texts: p_not_credible(tok, model, texts, nc_idx, interpret.MAXLEN)
        result["models"][name] = {"length": length_part(p_fn, rows, [15, 30, 60, 0]), "links": link_part(p_fn, rows, posts)}
        print(f"[{name}] done", file=sys.stderr)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "domain_shift.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    md = ["# Domain-shift checks (inference only; see src/domain_shift_checks.py)", "",
          f"Internal test rows used: {len(rows)}. Collected posts with a URL: {len(posts)} (unlabelled, never used for training or tuning).", "",
          "| Model | Cut | Accuracy | Credible called not credible | Not-credible recall |", "|---|---|---|---|---|"]
    for n, m in result["models"].items():
        for cut, v in m["length"].items():
            md.append(f"| {n} | {cut} | {v['accuracy']} | {v['credible_called_not_credible']} | {v['not_credible_recall']} |")
    (out / "domain_shift.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(result["models"], indent=1)[:1500])


if __name__ == "__main__":
    main()
