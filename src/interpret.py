"""Integrated Gradients (Captum) for the three fine-tuned models. No SHAP.

    # (1)+(2) one post, all three models -> results/ig_post_<id>.json
    python src/interpret.py post --text "Totoo ba na ... ?" --id demo1

    # (2)+(3)+(4) a CSV (column 'text', optional 'id', 'label') -> results/ig_attributions_<model>.json,
    #             results/ig_top_tokens.json, results/ig_overlap.json
    python src/interpret.py corpus --csv data/external/taglish_test.csv

Models load as in evaluate_external.py: models/<name>/ if it exists, else hansjio/tsektxt-* on the Hub.
Override a path with --model-dir name=path (used by the tests).

What is attributed
  Target   f(x) = logit[not_credible] - logit[credible]   (labels: 1 = credible, 0 = not credible).
           Positive attribution = pushes the model toward NOT CREDIBLE (the positive class).
  Layer    word-embedding layer (LayerIntegratedGradients); position/segment embeddings are untouched.
  Baseline same sequence with every non-special token replaced by the pad token (CLS/SEP/BOS/EOS kept).
  Completeness: sum of attributions should equal f(x) - f(baseline); we report the gap per post.
           f(baseline) is computed with the REAL input ids and a forward hook on the word-embedding layer
           that returns the baseline embeddings. This is exactly the IG path endpoint. Running
           model(input_ids=base) instead is wrong for RoBERTa-type models (RoBERTa-Tagalog, XLM-R): they
           derive position ids from input_ids (pad -> padding_idx), so the pad sequence gets different
           positions than IG's endpoint, and the check compares against the wrong baseline. (mBERT uses
           fixed positions and was unaffected.) The hook is preferred over explicit position_ids because
           it needs no architecture-specific code.
  Steps    adaptive: start at --n-steps (default 100) and double up to --max-steps (default 800) until
           rel_error <= --tol (default 0.05). XLM-R had 18.5% real error at 50 steps and ~0% at 300.
           Each record reports n_steps used, and "converged". Use --max-steps equal to --n-steps for fixed steps.
  Words    subword scores are summed into whitespace words via character offsets, so the three
           models (different tokenizers) can be compared on the same units.

IG explains what the MODEL relies on, not what is true: labels follow the source, models are single-seed.
"""
import argparse, bisect, json, re, sys, time
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load(open(ROOT / "configs" / "config.yaml"))
NAMES = list(CFG["models"])
MAXLEN = CFG["training"]["max_length"]  # 128
DEFAULT_STEPS, MAX_STEPS, TOL = 100, 800, 0.05  # adaptive IG: start at 100 steps, double until rel_error <= 5%
DEV = "cuda" if torch.cuda.is_available() else "cpu"
NOTES = [
    "Attribution target: logit(not_credible) - logit(credible); positive = toward not credible.",
    "Baseline: pad-token embeddings at every non-special position.",
    "IG shows what each model relies on, not ground truth; labels follow the source (fact-checker / outlet).",
    "Single training seed per model; attributions can change across seeds.",
]


# ---------------------------------------------------------------- loading
def model_path(name, overrides=None):
    if overrides and name in overrides:
        return str(overrides[name])
    local = ROOT / "models" / name
    return str(local) if local.exists() else f"{CFG['hub']['user']}/{CFG['hub']['repo_names'][name]}"


def load(name, overrides=None):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    path = model_path(name, overrides)
    print(f"[{name}] loading from {path}", file=sys.stderr)
    tok = AutoTokenizer.from_pretrained(path)
    if not tok.is_fast:
        raise RuntimeError(f"{name}: needs a fast tokenizer (offset mapping) for word-level attributions")
    model = AutoModelForSequenceClassification.from_pretrained(path).eval().to(DEV)
    return tok, model, path


def label_index(model):
    """Index of not_credible / credible in the logits, read from config (falls back to 0 / 1)."""
    id2 = {int(k): str(v).lower() for k, v in (model.config.id2label or {}).items()}
    nc = next((i for i, v in id2.items() if "not" in v), 0)
    cr = next((i for i, v in id2.items() if "not" not in v and "credible" in v), 1)
    assert nc != cr and {nc, cr} == {0, 1}, f"unexpected id2label {id2}"
    return nc, cr


# ---------------------------------------------------------------- words
def _norm(w):
    s = re.sub(r"^\W+|\W+$", "", w.lower())
    return s or w.lower()


def word_scores(text, offsets, scores):
    """Sum subword scores into whitespace words of the original text. Returns list of dicts."""
    spans = [(m.start(), m.end(), m.group()) for m in re.finditer(r"\S+", text)]
    starts = [s for s, _, _ in spans]
    acc = defaultdict(float)
    for (s, e), a in zip(offsets, scores):
        if e <= s:  # special token
            continue
        while s < e and text[s].isspace():
            s += 1
        j = bisect.bisect_right(starts, s) - 1
        if j >= 0 and s < spans[j][1]:
            acc[j] += a
    return [{"word": spans[j][2], "norm": _norm(spans[j][2]), "attr": acc[j]} for j in sorted(acc)]


# ---------------------------------------------------------------- IG
def explain(tok, model, text, n_steps=DEFAULT_STEPS, internal_batch_size=16, tol=TOL, max_steps=MAX_STEPS):
    """Features (1) and (2): per-token and per-word attributions plus the completeness check.
    Adaptive: starts at n_steps and doubles (up to max_steps) until rel_error <= tol.
    Pass max_steps=n_steps for a fixed number of steps."""
    from captum.attr import LayerIntegratedGradients
    nc, cr = label_index(model)
    enc = tok(text, truncation=True, max_length=MAXLEN, return_offsets_mapping=True, return_tensors="pt")
    offsets = enc.pop("offset_mapping")[0].tolist()
    ids, mask = enc["input_ids"].to(DEV), enc["attention_mask"].to(DEV)
    special = torch.tensor(tok.get_special_tokens_mask(ids[0].tolist(), already_has_special_tokens=True), device=DEV).bool()
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    base = torch.where(special, ids, torch.full_like(ids, pad))

    def f(input_ids, attention_mask):
        lg = model(input_ids=input_ids, attention_mask=attention_mask).logits
        return lg[:, nc] - lg[:, cr]

    emb_layer = model.get_input_embeddings()
    lig = LayerIntegratedGradients(f, emb_layer)
    with torch.no_grad():
        logits = model(input_ids=ids, attention_mask=mask).logits[0]
        fx = f(ids, mask).item()
        # f(baseline) must use the SAME path endpoint as IG: real ids (so RoBERTa-type models keep the
        # real positions) with the word embeddings swapped for the baseline's. See module docstring.
        base_emb = emb_layer(base)
        hook = emb_layer.register_forward_hook(lambda m, i, o: base_emb)
        try:
            fb = f(ids, mask).item()
        finally:
            hook.remove()
    diff = fx - fb

    steps = int(n_steps)
    while True:
        attr = lig.attribute(ids, baselines=base, additional_forward_args=(mask,), n_steps=steps,
                             internal_batch_size=internal_batch_size)
        tok_attr = attr.sum(-1)[0].detach().cpu().tolist()
        total = float(sum(tok_attr))
        gap = total - diff
        rel = abs(gap) / max(abs(diff), 1e-8)
        if rel <= tol or steps >= max_steps:
            break
        steps = min(steps * 2, max_steps)
    probs = torch.softmax(logits, -1).cpu().tolist()
    n_words_text = len(text.split())
    words = word_scores(text, offsets, tok_attr)
    return {
        "pred_label": 0 if fx > 0 else 1,
        "pred_name": "not_credible" if fx > 0 else "credible",
        "prob_credible": probs[cr], "prob_not_credible": probs[nc],
        "truncated": len(words) < n_words_text,
        "tokens": [{"token": t, "attr": a} for t, a in zip(tok.convert_ids_to_tokens(ids[0].tolist()), tok_attr)],
        "words": words,
        "completeness": {
            "sum_attributions": total, "f_input": fx, "f_baseline": fb, "logit_diff": diff,
            "abs_error": abs(gap), "rel_error": rel, "n_steps": steps, "n_steps_start": int(n_steps),
            "tolerance": tol, "converged": rel <= tol,
        },
    }


def completeness_summary(recs, tol=0.05):
    errs = sorted(r["completeness"]["rel_error"] for r in recs)
    if not errs:
        return {"n": 0}
    return {"n": len(errs), "rel_error_mean": sum(errs) / len(errs), "rel_error_median": errs[len(errs) // 2],
            "rel_error_max": errs[-1], f"share_within_{tol}": sum(e <= tol for e in errs) / len(errs), "tolerance": tol}


# ---------------------------------------------------------------- aggregation
def post_word_weights(rec):
    """L1-normalised attribution per normalised word (so long posts do not dominate)."""
    w = defaultdict(float)
    for x in rec["words"]:
        w[x["norm"]] += x["attr"]
    z = sum(abs(v) for v in w.values()) or 1.0
    return {k: v / z for k, v in w.items()}


def top_tokens(recs, top_k=20, min_df=3):
    """Feature (3): rank words by summed normalised attribution over the given posts."""
    s, df = defaultdict(float), defaultdict(int)
    for r in recs:
        for k, v in post_word_weights(r).items():
            s[k] += v; df[k] += 1
    rows = [{"word": k, "sum_attr": s[k], "mean_attr": s[k] / df[k], "doc_freq": df[k]} for k in s if df[k] >= min_df]
    pos = sorted((r for r in rows if r["sum_attr"] > 0), key=lambda r: -r["sum_attr"])[:top_k]
    neg = sorted((r for r in rows if r["sum_attr"] < 0), key=lambda r: r["sum_attr"])[:top_k]
    return {"n_posts": len(recs), "min_df": min_df, "top_k": top_k, "toward_not_credible": pos, "toward_credible": neg}


def _jac(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else None


def overlap(tops, per_post=None, k_post=5):
    """Feature (4): cross-model overlap of top predictor words."""
    sets = {m: [r["word"] for r in t["toward_not_credible"]] for m, t in tops.items()}
    out = {"top_k_sets": sets, "pairwise": {}, "shared_by_all": sorted(set.intersection(*map(set, sets.values()))) if sets else []}
    for a, b in combinations(sets, 2):
        out["pairwise"][f"{a}|{b}"] = {"jaccard": _jac(sets[a], sets[b]), "shared": sorted(set(sets[a]) & set(sets[b]))}
    if per_post:  # per_post: {model: {post_id: rec}} for posts predicted not credible
        out["per_post_top%d" % k_post] = {}
        for a, b in combinations(per_post, 2):
            common = set(per_post[a]) & set(per_post[b]); js = []
            for pid in common:
                ta, tb = ([k for k, v in sorted(post_word_weights(p[pid]).items(), key=lambda kv: -kv[1])[:k_post] if v > 0]
                          for p in (per_post[a], per_post[b]))
                j = _jac(ta, tb)
                if j is not None: js.append(j)
            out["per_post_top%d" % k_post][f"{a}|{b}"] = {"n_posts_both_not_credible": len(common),
                                                         "mean_jaccard": sum(js) / len(js) if js else None}
    return out


# ---------------------------------------------------------------- CLI
def _meta(a, paths):
    return {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "max_length": MAXLEN, "n_steps": a.n_steps,
            "max_steps": a.max_steps, "tol": a.tol,
            "model_paths": paths, "device": DEV, "notes": NOTES}


def _dump(obj, p):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {p}", file=sys.stderr)


def run_post(a, models, out):
    res = {"id": a.id, "text": a.text, "meta": _meta(a, {m: v[2] for m, v in models.items()}), "models": {}}
    for m, (tok, model, _) in models.items():
        res["models"][m] = explain(tok, model, a.text, a.n_steps, a.internal_batch_size, a.tol, a.max_steps)
    tops = {m: top_tokens([r], top_k=a.top_k, min_df=1) for m, r in res["models"].items()}
    res["overlap"] = overlap(tops)
    _dump(res, out / f"ig_post_{re.sub(r'[^A-Za-z0-9_-]', '_', a.id)}.json")
    for m, r in res["models"].items():
        c = r["completeness"]
        top = sorted(r["words"], key=lambda x: -x["attr"])[:5]
        print(f"{m:16s} {r['pred_name']:13s} P(credible)={r['prob_credible']:.3f}  rel_err={c['rel_error']:.4f}  "
              f"top: {', '.join(x['norm'] for x in top)}")


def run_corpus(a, models, out):
    import pandas as pd
    df = pd.read_csv(a.csv)
    if a.limit: df = df.head(a.limit)
    ids = df["id"].astype(str).tolist() if "id" in df else [str(i) for i in range(len(df))]
    labels = df["label"].tolist() if "label" in df else [None] * len(df)
    tops, per_post, comp, counts = {}, {}, {}, {}
    for m, (tok, model, _) in models.items():
        recs = []
        for pid, text, y in zip(ids, df.text.astype(str), labels):
            r = explain(tok, model, text, a.n_steps, a.internal_batch_size, a.tol, a.max_steps)
            r.update(id=pid, label=None if y is None or pd.isna(y) else int(y)); recs.append(r)
        nc = [r for r in recs if r["pred_label"] == 0]
        target = recs if a.all_posts else nc
        tops[m] = top_tokens(target, a.top_k, a.min_df)
        per_post[m] = {r["id"]: r for r in nc}
        comp[m] = completeness_summary(recs)
        counts[m] = {"n_posts": len(recs), "n_pred_not_credible": len(nc)}
        _dump({"model": m, "meta": _meta(a, {m: models[m][2]}), "completeness": comp[m],
               "posts": [{k: v for k, v in r.items() if a.keep_tokens or k != "tokens"} for r in recs]},
              out / f"ig_attributions_{m}.json")
    meta = _meta(a, {m: v[2] for m, v in models.items()}); meta["csv"] = a.csv
    meta["ranked_on"] = "all posts" if a.all_posts else "posts each model predicts not credible"
    _dump({"meta": meta, "counts": counts, "completeness": comp, "top_tokens": tops}, out / "ig_top_tokens.json")
    _dump({"meta": meta, **overlap(tops, per_post)}, out / "ig_overlap.json")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("post", "corpus"):
        p = sub.add_parser(name)
        p.add_argument("--models", nargs="+", default=NAMES, choices=NAMES)
        p.add_argument("--model-dir", nargs="*", default=[], help="override: name=path")
        p.add_argument("--n-steps", type=int, default=DEFAULT_STEPS); p.add_argument("--internal-batch-size", type=int, default=16)
        p.add_argument("--max-steps", type=int, default=MAX_STEPS); p.add_argument("--tol", type=float, default=TOL)
        p.add_argument("--top-k", type=int, default=20); p.add_argument("--out-dir", default=str(ROOT / "results"))
    sub.choices["post"].add_argument("--text", required=True)
    sub.choices["post"].add_argument("--id", default="post")
    c = sub.choices["corpus"]
    c.add_argument("--csv", required=True); c.add_argument("--min-df", type=int, default=3)
    c.add_argument("--limit", type=int); c.add_argument("--all-posts", action="store_true", help="rank on all posts, not only predicted not credible")
    c.add_argument("--keep-tokens", action="store_true", help="also store subword attributions in ig_attributions_*.json")
    a = ap.parse_args(argv)
    overrides = dict(x.split("=", 1) for x in a.model_dir)
    models = {m: load(m, overrides) for m in a.models}
    (run_post if a.cmd == "post" else run_corpus)(a, models, Path(a.out_dir))


if __name__ == "__main__":
    main()
