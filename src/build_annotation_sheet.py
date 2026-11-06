"""Turn merged_for_annotation.csv into the human-friendly annotation tab (CSV to import into the test-set Google Sheet).

    python src/build_annotation_sheet.py            -> data/external/annotation_tab_import.csv  (+ .report.json)

Left out: rows with no text, rows with 'OUTSIDE WINDOW' in merge_notes. Label columns are ALWAYS empty. Pre-label/fact-check hints
go to hint columns only. Column names that clean_annotations.py reads are kept exactly: post_id, text, url, label_source,
annotator_1_label, annotator_2_label, final_label. Annotator columns first, technical columns to the right.
"""
import csv, json, re
from pathlib import Path

EXT = Path(__file__).resolve().parents[1] / "data" / "external"
SRC, OUT = EXT / "merged_for_annotation.csv", EXT / "annotation_tab_import.csv"
LANGS = {"taglish", "tagalog", "english", "other"}
ANNOT = ["post_id", "url", "text", "platform", "post_date", "factcheck_verdict_hint", "factcheck_link", "label_source",
         "annotator_1_label", "annotator_2_label", "final_label", "annotator_notes", "language_pattern"]
TECH = ["reactions_count", "shares_reposts_count", "comments_count", "collector", "candidate_political_actor", "institution mentioned",
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


def build():
    rows = list(csv.DictReader(open(SRC, encoding="utf-8-sig", newline="")))
    keep = [r for r in rows if r["text"].strip() and "OUTSIDE WINDOW" not in r["merge_notes"]]
    out = []
    for r in keep:
        lang = r["language_pattern"].strip().lower()
        o = {"post_id": r["post_id"], "url": r["url"], "text": r["text"], "platform": r["platform"], "post_date": r["collection date"],
             "factcheck_verdict_hint": hint_text(r),
             "factcheck_link": first_url(r["all_factcheck_urls"]) or first_url(r["label_source"]),
             "label_source": "", "annotator_1_label": "", "annotator_2_label": "", "final_label": "", "annotator_notes": "",
             "language_pattern": lang if lang in LANGS else "",
             "language_pattern_merge_raw": r["language_pattern"]}
        for c in TECH:
            if c not in o: o[c] = r.get(c, "")
        out.append(o)
    cols = ANNOT + TECH
    with open(OUT, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, cols); w.writeheader(); w.writerows(out)
    rep = {"rows_in_merged": len(rows), "rows_in_tab": len(out), "left_out_no_text": sum(not r["text"].strip() for r in rows),
           "left_out_outside_window": sum("OUTSIDE WINDOW" in r["merge_notes"] for r in rows), "columns": len(cols),
           "annotator_columns": len(ANNOT), "technical_columns": len(TECH),
           "dates_outside_feb8_may9_2022_kept": sum(bool(o["post_date"]) and not ("2022-02-08" <= o["post_date"][:10] <= "2022-05-09") for o in out),
           "no_post_date": sum(not o["post_date"] for o in out)}
    OUT.with_suffix(".report.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    build()
