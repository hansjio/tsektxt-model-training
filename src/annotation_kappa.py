"""Inter-annotator agreement (Cohen's kappa) for the Taglish test-set sheet.

    python src/annotation_kappa.py --csv data/external/annotated.csv
    python src/annotation_kappa.py --csv sheet_export.csv --a1 annotator_1_label --a2 annotator_2_label

Input : CSV export of the Google Sheet tab (needs the two annotator columns; id column optional).
Labels: 1 = credible, 0 = not credible (a few word spellings are accepted, see clean_annotations.parse_label).
Output: results/annotation_kappa.json, results/annotation_disagreements.csv (rows to discuss and resolve).
Only rows where BOTH annotators gave a label count. Pure standard library (no pandas needed).
"""
import argparse, csv, json, random
from pathlib import Path

LABELS = (0, 1)
_WORDS = {"0": 0, "not credible": 0, "not_credible": 0, "false": 0, "1": 1, "credible": 1, "true": 1}


def parse_label(v):
    """0/1 from a sheet cell ('1', '1.0', 'Credible'...). None if empty; ValueError if something else."""
    s = "" if v is None else str(v).strip().lower()
    if s in ("1.0", "0.0"): s = s[0]
    if s in ("", "nan", "none"): return None
    if s in _WORDS: return _WORDS[s]
    raise ValueError(f"unrecognised label {v!r}")


def cohen_kappa(a, b, labels=LABELS):
    """Unweighted Cohen's kappa. Returns (kappa, observed agreement, expected agreement)."""
    n = len(a)
    if n == 0 or n != len(b): raise ValueError("need two equal-length, non-empty label lists")
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(l) / n) * (b.count(l) / n) for l in labels)
    if pe == 1: return (1.0 if po == 1 else 0.0), po, pe  # both annotators used one identical class only
    return (po - pe) / (1 - pe), po, pe


def bootstrap_ci(a, b, n_boot=2000, seed=42, alpha=0.05):
    rng, n, ks = random.Random(seed), len(a), []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        ks.append(cohen_kappa([a[i] for i in idx], [b[i] for i in idx])[0])
    ks.sort()
    return ks[int(n_boot * alpha / 2)], ks[int(n_boot * (1 - alpha / 2)) - 1]


def interpret(k):  # Landis & Koch (1977)
    for cut, name in [(0, "poor"), (0.20, "slight"), (0.40, "fair"), (0.60, "moderate"), (0.80, "substantial")]:
        if k <= cut: return name
    return "almost perfect"


def analyse(rows, a1="annotator_1_label", a2="annotator_2_label", id_col="post_id"):
    A, B, dis, missing, bad = [], [], [], 0, []
    for i, r in enumerate(rows):
        try: x, y = parse_label(r.get(a1)), parse_label(r.get(a2))
        except ValueError as e: bad.append({"row": i + 2, "error": str(e)}); continue
        if x is None or y is None: missing += 1; continue
        A.append(x); B.append(y)
        if x != y: dis.append({"row": i + 2, "id": r.get(id_col, i), a1: x, a2: y, "text": (r.get("text") or "")[:200], "url": r.get("url", "")})
    res = {"rows_total": len(rows), "rows_both_labelled": len(A), "rows_missing_a_label": missing, "rows_with_invalid_label": bad}
    if not A: res["note"] = "no row has both annotator labels yet"; return res, dis
    k, po, pe = cohen_kappa(A, B)
    lo, hi = bootstrap_ci(A, B)
    res.update({"cohen_kappa": round(k, 4), "kappa_95ci_bootstrap": [round(lo, 4), round(hi, 4)], "interpretation_landis_koch": interpret(k),
                "observed_agreement": round(po, 4), "expected_agreement": round(pe, 4), "disagreements": len(dis),
                "confusion_rows_a1_cols_a2": {f"a1={x}": {f"a2={y}": sum(1 for p, q in zip(A, B) if p == x and q == y) for y in LABELS} for x in LABELS},
                "label_counts": {"a1": {str(l): A.count(l) for l in LABELS}, "a2": {str(l): B.count(l) for l in LABELS}}})
    if len(A) < 30: res["warning"] = "fewer than 30 doubly-labelled rows: kappa is unstable, do not report it yet"
    if len(set(A) | set(B)) < 2: res["warning"] = "only one class present: kappa is not informative (add credible posts)"
    return res, dis


def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as f: return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True); ap.add_argument("--a1", default="annotator_1_label"); ap.add_argument("--a2", default="annotator_2_label")
    ap.add_argument("--id-col", default="post_id"); ap.add_argument("--out-dir", default="results")
    a = ap.parse_args()
    res, dis = analyse(read_csv(a.csv), a.a1, a.a2, a.id_col)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out / "annotation_kappa.json", "w"), indent=2)
    if dis:
        with open(out / "annotation_disagreements.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(dis[0])); w.writeheader(); w.writerows(dis)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
