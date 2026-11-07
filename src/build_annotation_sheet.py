"""Turn merged_for_annotation.csv into the human-friendly annotation tab (CSV to import into the test-set Google Sheet).

    python src/build_annotation_sheet.py            -> data/external/annotation_tab_import.csv  (+ .report.json)
    python src/build_annotation_sheet.py --only-new SHEET_EXPORT.csv   -> annotation_tab_import_new.csv (posts not yet in the Sheet)

Screening hints (screen_candidates.py) are joined by post_id and sit right of the annotator columns; language_check comes from
data/external/language_check.csv if present. Old outputs are copied to data/external/_superseded/ (date suffix) before overwriting.

Left out: rows with no text, rows with 'OUTSIDE WINDOW' in merge_notes. Label columns are ALWAYS empty. Pre-label/fact-check hints
go to hint columns only. Column names that clean_annotations.py reads are kept exactly: post_id, text, url, label_source,
annotator_1_label, annotator_2_label, final_label. Annotator columns first, technical columns to the right.
"""
import argparse, csv, json, re, sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from screen_candidates import screen  # noqa: E402
from merge_candidates import backup   # noqa: E402

EXT = Path(__file__).resolve().parents[1] / "data" / "external"
SRC, OUT = EXT / "merged_for_annotation.csv", EXT / "annotation_tab_import.csv"
NEW, LANGCHK = EXT / "annotation_tab_import_new.csv", EXT / "language_check.csv"
LANGS = {"taglish", "tagalog", "english", "other"}
ANNOT = ["post_id", "url", "text", "platform", "post_date", "factcheck_verdict_hint", "factcheck_link", "label_source",
         "annotator_1_label", "annotator_2_label", "final_label", "annotator_notes", "language_pattern"]
HINTS = ["screen", "screen_reasons", "credible_hint", "language_hint", "language_check", "election_terms", "length_bin", "suggested_order"]
TECH = ["stratum", "reactions_count", "shares_reposts_count", "comments_count", "collector", "candidate_political_actor", "institution mentioned",
        "issue_category", "claim_type", "language_pattern_merge_raw", "needs_text", "text_origin", "claim_summary", "ai_prelabel_hint",
        "factcheck_rating", "all_factcheck_urls", "source_type", "source_account", "has_checkable_claim", "origin_files", "merge_notes"]


# Accounts of news outlets / officials: a fact-check often links their post as the DEBUNK or a real report, not the false claim.
NEWSY = re.compile(r"abs-?cbn|gmanews|gmanetwork|inquirer|rappler|philstar|cnnphilippines|manilabulletin|news5|pna\.gov|comelec|"
                   r"verafiles|tsek\.ph|bbc|reuters|mindanews|interaksyon|sunstar|abante|bombo|dzrh|dzmm|teleradyo|pcij", re.I)


def hint_text(r):
    """The fact-check rating describes the CLAIM, not necessarily this post. Never present it as the post's label."""
    if r["factcheck_verdict_hint"] != "0":
        return ""
    msg = "fact-check article linked: this post may be the FALSE CLAIM or the DEBUNK / real report. Read both. Not a label."
    if r.get("source_type") == "news_outlet" or NEWSY.search((r.get("source_account", "") + " " + r["url"])):
        msg = "NEWS/OFFICIAL ACCOUNT, probably the debunk or a real report (check). " + msg
    return msg


def first_url(s):
    m = re.search(r"https?://[^\s|]+", s or "")
    return m.group(0) if m else ""


def build(only_new=None, include_drop=False, keep_english=False):
    rows = list(csv.DictReader(open(SRC, encoding="utf-8-sig", newline="")))
    scr = screen(rows)  # same order as rows (post_id is NOT unique: 3 ids are shared by 2 different posts, kept as is for the Sheet)
    lc = {}
    if LANGCHK.exists():
        # keyed by (post_id, url): 3 post_ids are shared by 2 different posts. note = e.g. off_topic, shown in screen_reasons.
        lc = {(x["post_id"], x["url"]): x for x in csv.DictReader(open(LANGCHK, encoding="utf-8-sig", newline=""))}
    have, start = set(), 0
    if only_new:
        ex = list(csv.DictReader(open(only_new, encoding="utf-8-sig", newline="")))
        have = {x["url"].strip() for x in ex}
        start = max([int(x["suggested_order"]) for x in ex if (x.get("suggested_order") or "").strip().isdigit()] or [len(ex)])
    out_path = NEW if only_new else OUT
    idx = {id(r): i for i, r in enumerate(rows)}
    keep = [r for r in rows if r["text"].strip() and "OUTSIDE WINDOW" not in r["merge_notes"]
            and r["url"].strip() not in have]
    out = []
    for r in keep:
        lang = r["language_pattern"].strip().lower()
        o = {"post_id": r["post_id"], "url": r["url"], "text": r["text"], "platform": r["platform"], "post_date": r["collection date"],
             "factcheck_verdict_hint": hint_text(r),
             "factcheck_link": first_url(r["all_factcheck_urls"]) or first_url(r["label_source"]),
             "label_source": "", "annotator_1_label": "", "annotator_2_label": "", "final_label": "", "annotator_notes": "",
             "language_pattern": lang if lang in LANGS else "",
             "language_pattern_merge_raw": r["language_pattern"]}
        h = scr[idx[id(r)]]
        for c in HINTS:
            if c in h: o[c] = h[c]
        c = lc.get((r["post_id"], r["url"]), {})
        o["language_check"] = c.get("language_check", "")
        if c.get("note"):
            o["screen_reasons"] = "; ".join(filter(None, [o["screen_reasons"], c["note"]]))
            if o["screen"] == "ready": o["screen"] = "check"
        # Group rule (dataset checklist): posts must be Taglish/Tagalog. English-only posts confirmed by the language check are dropped.
        if o["language_check"] == "english" and not keep_english:
            o["screen"] = "drop"; o["screen_reasons"] = "; ".join(filter(None, [o["screen_reasons"], "english_only"]))
        for c in TECH:
            if c not in o: o[c] = r.get(c, "")
        out.append(o)
    # Rows screened 'drop' (under 10 words, duplicate, outside window, posts BY fact-checkers) can never enter the test set, so they are not
    # sent to the annotators; they are kept in annotation_tab_dropped.csv for the record (use --include-drop to keep them in the tab).
    dropped = [o for o in out if o["screen"] == "drop"]
    if not include_drop:
        out = [o for o in out if o["screen"] != "drop"]
    # suggested_order: ready first, then credible_hint rows, then the rest; credible_hint rows lead inside each group
    tier = lambda o: 0 if o["screen"] == "ready" else 1 if o["credible_hint"] else 2
    out.sort(key=lambda o: (tier(o), not o["credible_hint"]))
    for i, o in enumerate(out, start + 1): o["suggested_order"] = i
    cols = ANNOT + HINTS + TECH
    backup(out_path)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, cols); w.writeheader(); w.writerows(out)
    rep = {"rows_in_merged": len(rows), "rows_in_tab": len(out), "left_out_no_text": sum(not r["text"].strip() for r in rows),
           "left_out_outside_window": sum("OUTSIDE WINDOW" in r["merge_notes"] for r in rows), "columns": len(cols),
           "already_in_sheet_skipped_by_url": len(have), "duplicate_post_ids_in_merged": [k for k, v in Counter(r["post_id"] for r in rows).items() if v > 1], "screen": dict(Counter(o["screen"] for o in out)),
           "language_hint": dict(Counter(o["language_hint"] for o in out)), "english_hint_not_yet_checked": sum(o["language_hint"] == "english" and not o["language_check"] for o in out),
           "language_check": dict(Counter(o["language_check"] or "none" for o in out)),
           "annotator_columns": len(ANNOT), "technical_columns": len(TECH),
           "dates_outside_feb8_may9_2022_kept": sum(bool(o["post_date"]) and not ("2022-02-08" <= o["post_date"][:10] <= "2022-05-09") for o in out),
           "no_post_date": sum(not o["post_date"] for o in out)}
    rep["dropped_not_sent_to_annotators"] = 0 if include_drop else len(dropped)
    rep["dropped_reasons"] = dict(Counter(x.split("_of_")[0] for o in dropped for x in o["screen_reasons"].split("; ") if x))
    drop_path = out_path.with_name(out_path.stem.replace("_import", "") + "_dropped.csv")
    with open(drop_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, cols); w.writeheader(); w.writerows(dropped)
    out_path.with_suffix(".report.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only-new", metavar="SHEET_EXPORT.csv", help="CSV export of the Sheet tab; write only posts whose post_id is not in it")
    ap.add_argument("--include-drop", action="store_true", help="also put rows screened 'drop' in the tab (default: separate file)")
    ap.add_argument("--keep-english", action="store_true", help="do not drop posts the language check confirmed as English-only")
    a = ap.parse_args()
    build(a.only_new, a.include_drop, a.keep_english)
