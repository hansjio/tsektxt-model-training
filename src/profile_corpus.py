"""Descriptive profile of a corpus (training set or the Taglish test set) for Chapter IV.

    python src/profile_corpus.py --csv data/processed/cleaned.csv --name train
    python src/profile_corpus.py --csv data/external/taglish_test.csv --name taglish_test

Input : CSV with text, label (1 = credible, 0 = not credible). Optional metadata columns are profiled when present:
        platform, language_pattern, issue_category, claim_type, candidate_political_actor, "collection date".
Output: results/profile_<name>.json  (+ results/profile_<name>_tokens.csv, top class-distinctive words).
Descriptive only: distinctive words show what the LABELS correlate with in this corpus, not what makes a post false.
Pure standard library.
"""
import argparse, csv, json, math, re, statistics
from collections import Counter
from pathlib import Path

TOK = re.compile(r"[a-zñ']+", re.I)
# Small function-word lists, enough to flag English / Tagalog / Taglish heuristically (not a language-ID model).
TL = set("ang ng mga sa na at ay ito si ni ay para pero kung hindi ba po naman lang din rin yung daw raw ako ikaw siya kami tayo sila mo ko ka kay nang may mayroon wala walang dahil kasi talaga pa nga pala".split())
EN = set("the of and to in is that for with on as are was this by it be from at or an not have has but they their will who his her".split())
STOP = TL | EN | set("a i you we he she them our your its if so do did can just all more than about what when which there been were would also said".split())
META = ["platform", "language_pattern", "issue_category", "claim_type", "candidate_political_actor"]


def guess_language(words):
    tl, en = sum(w in TL for w in words), sum(w in EN for w in words)
    if tl + en < 2: return "unknown"
    share = tl / (tl + en)
    return "filipino" if share > 0.8 else "english" if share < 0.2 else "taglish"


def dist(vals):
    if not vals: return {}
    q = statistics.quantiles(vals, n=20) if len(vals) >= 2 else [vals[0]] * 19
    return {"mean": round(statistics.mean(vals), 2), "median": statistics.median(vals), "min": min(vals), "max": max(vals), "p5": round(q[0], 1), "p95": round(q[18], 1)}


def distinctive(texts_by_label, top=25, min_count=3):
    """Smoothed log-odds of a word being in not-credible (0) vs credible (1) posts, document frequency based."""
    df = {l: Counter(w for t in ts for w in set(TOK.findall(t.lower())) if w not in STOP and len(w) > 2) for l, ts in texts_by_label.items()}
    n0, n1 = max(len(texts_by_label[0]), 1), max(len(texts_by_label[1]), 1)
    rows = []
    for w in set(df[0]) | set(df[1]):
        c0, c1 = df[0][w], df[1][w]
        if c0 + c1 < min_count: continue
        rows.append({"word": w, "n_not_credible": c0, "n_credible": c1, "log_odds_not_credible": round(math.log((c0 + 0.5) / (n0 - c0 + 0.5)) - math.log((c1 + 0.5) / (n1 - c1 + 0.5)), 3)})
    rows.sort(key=lambda r: r["log_odds_not_credible"])
    return rows[-top:][::-1], rows[:top]


def profile(rows):
    labels = [int(float(r["label"])) for r in rows]
    texts = [r["text"] for r in rows]
    words = [TOK.findall(t.lower()) for t in texts]
    p = {"n": len(rows), "label_counts": {"0_not_credible": labels.count(0), "1_credible": labels.count(1)}}
    p["not_credible_share"] = round(labels.count(0) / len(rows), 4) if rows else None
    p["words_per_post"] = dist([len(t.split()) for t in texts])
    p["chars_per_post"] = dist([len(t) for t in texts])
    p["words_per_post_by_label"] = {str(l): dist([len(t.split()) for t, y in zip(texts, labels) if y == l]) for l in (0, 1)}
    lang = [guess_language(w) for w in words]
    p["language_heuristic"] = dict(Counter(lang)); p["language_heuristic_by_label"] = {str(l): dict(Counter(g for g, y in zip(lang, labels) if y == l)) for l in (0, 1)}
    p["vocabulary_size"] = len({w for ws in words for w in ws}); p["tokens_total"] = sum(map(len, words))
    p["type_token_ratio"] = round(p["vocabulary_size"] / max(p["tokens_total"], 1), 4)
    p["posts_with_hashtag_or_mention"] = sum(bool(re.search(r"[#@]\w", t)) for t in texts)
    p["posts_all_caps_words_>=3"] = sum(len(re.findall(r"\b[A-Z]{3,}\b", t)) >= 3 for t in texts)
    p["posts_with_exclamation"] = sum("!" in t for t in texts)
    p["exact_duplicate_texts"] = len(texts) - len(set(texts))
    for c in META:
        if rows and c in rows[0]:
            p[f"by_{c}"] = {v or "(blank)": {"total": sum(1 for r in rows if r[c] == v), "not_credible": sum(1 for r, y in zip(rows, labels) if r[c] == v and y == 0)} for v in sorted({r[c] for r in rows})}
    dates = sorted(r["collection date"] for r in rows if r.get("collection date"))
    if dates: p["collection_date_range"] = [dates[0], dates[-1]]
    if len(set(labels)) < 2: p["warning"] = "only one class present"
    return p, distinctive({0: [t for t, y in zip(texts, labels) if y == 0], 1: [t for t, y in zip(texts, labels) if y == 1]})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True); ap.add_argument("--name", default=None); ap.add_argument("--out-dir", default="results")
    a = ap.parse_args()
    with open(a.csv, newline="", encoding="utf-8-sig") as f: rows = [r for r in csv.DictReader(f) if (r.get("text") or "").strip() and str(r.get("label", "")).strip() != ""]
    name = a.name or Path(a.csv).stem
    p, (top0, top1) = profile(rows)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    json.dump(p, open(out / f"profile_{name}.json", "w"), indent=2, ensure_ascii=False)
    with open(out / f"profile_{name}_tokens.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["side", "word", "n_not_credible", "n_credible", "log_odds_not_credible"]); w.writeheader()
        for side, rs in (("not_credible", top0), ("credible", top1)): w.writerows({"side": side, **r} for r in rs)
    print(json.dumps({k: v for k, v in p.items() if not k.startswith("by_")}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
