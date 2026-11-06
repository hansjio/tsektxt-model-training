"""Clean the annotated Taglish sheet export into the TEST-ONLY file used by evaluate_external.py.

    python src/clean_annotations.py --csv data/external/annotated.csv --out data/external/taglish_test.csv

Input : CSV export of the sheet (post_id, text, url, label_source, annotator_1_label, annotator_2_label, final_label, ...).
Output: --out        id, text, label (+ metadata columns kept for profiling)   [1 = credible, 0 = not credible]
        --rejects    every dropped row with a reason (nothing is silently lost)
        <out>.report.json  counts per step

Rules (CLAUDE.md): every row needs a real URL and a fact-check source; no invented posts; this set is TEST ONLY.
Label resolution: final_label if set; else the shared label when both annotators agree; else the row is rejected as
'unresolved' (annotators must discuss it, see annotation_kappa.py disagreements file). Never guessed.
Text cleaning matches clean_data.py (ftfy if installed, HTML, URLs, zero-width, whitespace). Pure stdlib + optional ftfy.
"""
import argparse, csv, html, json, re
from pathlib import Path
from annotation_kappa import parse_label, read_csv

try: import ftfy
except ImportError: ftfy = None

URL = re.compile(r"(https?://\S+|www\.\S+)", re.I)
TAG = re.compile(r"<[^>]+>")
ZW = re.compile(r"[​‌‍⁠﻿]")
WS = re.compile(r"\s+")
REAL_URL = re.compile(r"^https?://\S+\.\S+", re.I)
MOJIBAKE = re.compile("â€|Ã±|Ã©|Ã")


def basic_clean(t):
    t = str(t)
    if ftfy: t = ftfy.fix_text(t)
    t = TAG.sub(" ", html.unescape(t)); t = URL.sub(" ", t); t = ZW.sub("", t)
    return WS.sub(" ", t).strip()


def norm_key(s, n=200):
    return re.sub(r"\W+", " ", s.lower()).strip()[:n]


def resolve_label(r):
    """(label or None, reason). Order: final_label, then agreement of annotators."""
    try:
        fin, a1, a2 = (parse_label(r.get(k)) for k in ("final_label", "annotator_1_label", "annotator_2_label"))
    except ValueError as e: return None, f"invalid_label ({e})"
    # final_label alone (e.g. an AI pre-label) never counts: both humans must have labelled.
    if a1 is None or a2 is None: return None, "no_label"
    if a1 == a2: return (a1, "ok") if fin is None or fin == a1 else (None, "final_label_contradicts_annotators")
    if fin is not None: return fin, "ok"  # humans disagreed and then settled it in final_label
    return None, "unresolved_disagreement"


def clean(rows, min_words=10):
    keep, rej, seen_url, seen_key = [], [], {}, {}
    rep = {"rows_in": len(rows)}
    def reject(r, why): rej.append({"reason": why, **{k: r.get(k, "") for k in ("post_id", "text", "url", "label_source")}}); rep[why] = rep.get(why, 0) + 1
    for r in rows:
        url, src = (r.get("url") or "").strip(), (r.get("label_source") or "").strip()
        if not REAL_URL.match(url): reject(r, "missing_or_bad_post_url"); continue
        if not REAL_URL.match(src): reject(r, "missing_factcheck_source"); continue
        label, why = resolve_label(r)
        if label is None: reject(r, why); continue
        text = basic_clean(r.get("text") or "")
        if len(text.split()) < min_words: reject(r, "too_short"); continue
        key = norm_key(text)
        if url in seen_url or key in seen_key:
            first = keep[seen_url.get(url, seen_key.get(key))]
            reject(r, "conflicting_duplicate" if first["label"] != label else "duplicate")
            if first["label"] != label: first["_conflict"] = True
            continue
        seen_url[url] = seen_key[key] = len(keep)
        keep.append({**{k: v for k, v in r.items() if k not in ("text", "annotator_1_label", "annotator_2_label", "final_label")}, "text": text, "label": label})
    conflicted = [k for k in keep if k.pop("_conflict", False)]
    if conflicted:  # same post labelled both ways: drop all copies, send back to annotators
        for k in conflicted: rej.append({"reason": "conflicting_duplicate", "post_id": k.get("post_id", ""), "text": k["text"], "url": k.get("url", ""), "label_source": k.get("label_source", "")})
        keep = [k for k in keep if k not in conflicted]; rep["dropped_first_copy_of_conflicts"] = len(conflicted)
    for i, k in enumerate(keep): k["id"] = i
    rep["rows_out"] = len(keep)
    rep["label_counts"] = {"0_not_credible": sum(k["label"] == 0 for k in keep), "1_credible": sum(k["label"] == 1 for k in keep)}
    rep["leftover_mojibake_rows"] = sum(bool(MOJIBAKE.search(k["text"])) for k in keep)
    if rep["label_counts"]["1_credible"] == 0 or rep["label_counts"]["0_not_credible"] == 0:
        rep["warning"] = "one class is empty: metrics on this set are not meaningful yet"
    return keep, rej, rep


def write_csv(path, rows, first):
    cols = first + [c for c in dict.fromkeys(c for r in rows for c in r) if c not in first]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader(); w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True); ap.add_argument("--out", default="data/external/taglish_test.csv")
    ap.add_argument("--rejects", default=None); ap.add_argument("--min-words", type=int, default=10)
    a = ap.parse_args()
    keep, rej, rep = clean(read_csv(a.csv), a.min_words)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    write_csv(out, keep, ["id", "text", "label"])
    write_csv(a.rejects or out.with_suffix(".rejects.csv"), rej, ["reason", "post_id", "text", "url", "label_source"]) if rej else None
    json.dump(rep, open(out.with_suffix(".report.json"), "w"), indent=2)
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
