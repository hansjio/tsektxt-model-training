"""Merge every candidate file for the supplementary (Taglish, TEST-ONLY) set into ONE sheet-ready CSV for annotation.

    python src/merge_candidates.py          (groupmate file defaults to data/external/groupmate_candidate_sources.csv)

Inputs (data/external/, never modified): draft_prelabelled.csv, draft_testset_v2_filled.csv, x_fetched_text_REVIEW.csv,
label1_candidates_news_REVIEW.csv, supplementary_collected_auto.csv, plus the groupmate's candidate_sources.csv.
Outputs: data/external/merged_for_annotation.csv, merged_for_annotation.rejects.csv (every dropped row + reason),
         merged_for_annotation.report.json.

Rules:
- One row per real post: duplicates found by X status id or normalised Facebook URL. Origin files and all fact-check links are kept.
- Human label columns (annotator_1_label, annotator_2_label, final_label) are always EMPTY. Draft pre-label hints and fact-check
  verdicts go to hint columns only and never count as labels.
- Text is the post's own words only. The groupmate's `text` column is the fact-checker's claim summary (one summary is shared
  by up to 50 posts), so it goes to `claim_summary`, and `needs_text` = yes until someone copies the real post text.
- Groupmate rows are dropped when the link is not a post (share buttons, page/profile homepages), when the fact-check date
  is outside Feb 8 - May 9 2022, or when the rating is satire. Each dropped row is listed in the rejects file with a reason.
"""
import argparse, csv, json, re
from collections import Counter
from pathlib import Path

EXT = Path(__file__).resolve().parents[1] / "data" / "external"
COLS = ["post_id", "text", "platform", "url", "collection date", "reactions_count", "shares_reposts_count", "comments_count",
        "collector", "annotator_1_label", "annotator_2_label", "final_label", "label_source", "candidate_political_actor",
        "institution mentioned", "issue_category", "claim_type", "language_pattern",
        "needs_text", "text_origin", "claim_summary", "factcheck_verdict_hint", "ai_prelabel_hint", "factcheck_rating",
        "all_factcheck_urls", "source_type", "source_account", "has_checkable_claim", "origin_files", "merge_notes"]
WINDOW = ("2022-02-08", "2022-05-09")


def read(p):
    with open(p, encoding="utf-8-sig", errors="replace", newline="") as f:
        return list(csv.DictReader(f))


def key(u):
    u = (u or "").strip().lower().replace("&amp;", "&")
    m = re.search(r"(?:twitter|x)\.com/[^/]+/status/(\d+)", u)
    if m: return "x:" + m.group(1)
    u = re.sub(r"[?#].*", "", u)
    return re.sub(r"^https?://(www\.|m\.|web\.)?", "", u).rstrip("/") or None


def link_kind(u):
    u = (u or "").lower()
    if "sharer" in u: return "share_button_not_a_post"
    if re.search(r"(twitter|x)\.com/[^/]+/status/\d+", u): return "post"
    if "facebook.com" in u and re.search(r"/posts/|/videos/|/photos?/|permalink\.php|story_fbid|/reel/|/watch|fbid=|/share/", u): return "post"
    if "facebook.com" in u: return "page_or_profile_not_a_post"
    return "not_a_recognised_post_link"


def clean_url(u):
    return (u or "").strip().replace("&amp;", "&").replace("&#038;", "&")


def is_junk(t):
    """Facebook UI/obfuscation captured instead of a caption (letter-per-line with U+034F, 'This photo is from a post')."""
    t = t or ""
    lines = t.split("\n")
    return ("͏" in t or "this photo is from a post" in t.lower()
            or (len(lines) > 3 and all(len(l.strip()) <= 3 for l in lines[:4])))


def is_url(s):
    return bool(re.match(r"https?://", (s or "").strip()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groupmate", default=str(EXT / "groupmate_candidate_sources.csv"), help="groupmate's candidate_sources.csv")
    ap.add_argument("--out", default=str(EXT / "merged_for_annotation.csv"))
    ap.add_argument("--filled-text", nargs="+", default=[str(EXT / "needs_text_filled.csv"), str(EXT / "x_embed_filled.csv")],
                    help="post-text fill files (text copied from the posts, X public embed); each is used if it exists")
    a = ap.parse_args()
    rows, rejects, notes = {}, [], Counter()

    def add(k, src, rec):
        """rec: partial row. First source wins for text/meta; later sources only fill blanks and add links/origins."""
        if k not in rows:
            r = {c: "" for c in COLS}; r.update({c: v for c, v in rec.items() if c in COLS and v})
            r["origin_files"] = src; rows[k] = r; return
        r = rows[k]; notes["duplicates_merged"] += 1
        if src not in r["origin_files"].split("; "): r["origin_files"] += "; " + src
        for c, v in rec.items():
            if c not in COLS or not v: continue
            if c == "all_factcheck_urls":
                have = set(filter(None, r[c].split(" | ")))
                r[c] = " | ".join(sorted(have | set(v.split(" | "))))
            elif c == "factcheck_verdict_hint" and r[c] and r[c] != v:
                r["merge_notes"] = (r["merge_notes"] + "; " if r["merge_notes"] else "") + f"conflicting verdict hints {r[c]} vs {v}"
            elif not r[c]:
                r[c] = v
        if r["text"]: r["needs_text"] = ""

    # 1. Fact-check drafts (163, text quoted by fact-checker; final_label there is only a draft pre-label hint)
    for x in read(EXT / "draft_prelabelled.csv"):
        add(key(x["url"]), "draft_prelabelled", {**{c: x.get(c, "") for c in COLS if c not in ("annotator_1_label", "annotator_2_label", "final_label")},
            "ai_prelabel_hint": x.get("final_label", ""), "factcheck_verdict_hint": x.get("label_hint_from_factcheck", ""),
            "all_factcheck_urls": x.get("label_source", ""), "merge_notes": x.get("prelabel_note", "")})
    # 2. Rebuilt fact-check candidates (312; ~half description_only = no post text)
    for x in read(EXT / "draft_testset_v2_filled.csv"):
        t = x.get("text", "").strip()
        add(key(x["url"]), "draft_testset_v2_filled", {**{c: x.get(c, "") for c in COLS[:9]}, "text": t,
            "needs_text": "" if t else "yes", "label_source": x.get("article_url", ""), "all_factcheck_urls": x.get("article_url", ""),
            "factcheck_verdict_hint": x.get("label_hint", "")})
    # 3. X text fetched from public embeds (fills text only)
    for x in read(EXT / "x_fetched_text_REVIEW.csv"):
        k = key(x["url"])
        if x.get("status") == "ok" and k in rows and not rows[k]["text"]:
            rows[k]["text"] = x["fetched_text_from_X_embed"]; rows[k]["text_origin"] = "x_embed_REVIEW"; rows[k]["needs_text"] = ""
            rows[k]["origin_files"] += "; x_fetched_text_REVIEW"; notes["text_filled_from_x_embed"] += 1
    # 4. News-outlet tweets (likely credible; no fact-check URL)
    for x in read(EXT / "label1_candidates_news_REVIEW.csv"):
        add(key(x["url"]), "label1_candidates_news_REVIEW", {**{c: x.get(c, "") for c in COLS[:9]},
            "factcheck_verdict_hint": "", "ai_prelabel_hint": x.get("label_hint", ""), "source_account": x.get("source_account", ""),
            "source_type": "news_outlet", "text_origin": x.get("text_origin", ""),
            "merge_notes": "" if is_url(x.get("label_source")) else f"label_source was '{x.get('label_source', '')}' (not a URL)"})
    # 5. Automated collection (no labels, no fact-check)
    ag = EXT / "supplementary_collected_auto.csv"
    for x in (read(ag) if ag.exists() else []):
        add(key(x["url"]), "auto_collection", {**{c: x.get(c, "") for c in COLS[:9]}, "source_type": x.get("source_type", ""),
            "source_account": x.get("source_account", ""), "has_checkable_claim": x.get("has_checkable_claim", ""),
            "language_pattern": x.get("language_guess", ""), "text_origin": x.get("text_origin", "")})
    # 6. Groupmate's candidate_sources.csv (fact-check embeds; text = fact-checker's claim summary)
    seen = set()
    for x in read(a.groupmate):
        u = clean_url(x["post_url"]); k = key(u)
        why = None
        if k in seen: notes["groupmate_repeat_rows"] += 1; continue
        seen.add(k)
        kind = link_kind(u)
        if kind != "post": why = kind
        elif not (WINDOW[0] <= x.get("date", "") <= WINDOW[1]): why = "factcheck_date_outside_window"
        elif "satire" in x.get("rating", "").lower(): why = "satire"
        if why:
            rejects.append({"reason": why, "source": "groupmate", "url": u, "date": x.get("date", ""), "rating": x.get("rating", ""),
                            "evidence_url": x.get("evidence_url", "")}); continue
        add(k, "groupmate_candidate_sources", {"post_id": "gm-" + (k.split(":")[-1] if k.startswith("x:") else str(len(seen))),
            "platform": x.get("platform", ""), "url": u, "collection date": "", "collector": "groupmate",
            "needs_text": "yes", "claim_summary": x.get("text", ""), "factcheck_rating": x.get("rating", ""),
            "factcheck_verdict_hint": "", "label_source": x.get("evidence_url", ""), "all_factcheck_urls": x.get("evidence_url", ""),
            "candidate_political_actor": x.get("candidates", "").replace("; ", ", "),
            "language_pattern": "taglish" if x.get("taglish") == "Y" else "",
            # No verdict hint: the rating is about the CLAIM in the fact-check. The linked post may repeat the false claim
            # OR be a real post (e.g. a news report) that the fact-check links to as the debunk. Annotators must read it.
            "merge_notes": f"factcheck date {x.get('date', '')}; post date unverified; fact-check rating applies to the claim, not necessarily to this post"})

    # 7. Text copied from the posts themselves, only into rows that still have no text
    for fill in (p for p in a.filled_text if Path(p).exists()):
        for x in read(fill):
            k = key(x.get("url")); r = rows.get(k)
            if not r: continue
            if x.get("post_date"):
                r["collection date"] = r["collection date"] or x["post_date"]
                if not (WINDOW[0] <= x["post_date"] <= WINDOW[1]):
                    r["merge_notes"] = (r["merge_notes"] + "; " if r["merge_notes"] else "") + f"POST DATE {x['post_date']} OUTSIDE WINDOW"
                    notes["filled_post_date_outside_window"] += 1
            if is_junk(x.get("text")):
                notes["fill_rows_with_junk_text_ignored"] += 1; continue
            if not r["text"] and (x.get("text") or "").strip() and x.get("text_origin") in ("copied_from_post", "image_text_transcribed"):
                r["text"] = x["text"].strip(); r["text_origin"] = x["text_origin"]; r["needs_text"] = ""
                r["origin_files"] += "; needs_text_filled"; notes["text_filled_from_post"] += 1
            elif not r["text"] and x.get("text_origin"):
                r["merge_notes"] = (r["merge_notes"] + "; " if r["merge_notes"] else "") + f"fill: {x['text_origin']}"

    out = list(rows.values())
    for r in out:  # human label columns are ALWAYS empty
        r["annotator_1_label"] = r["annotator_2_label"] = r["final_label"] = ""
        if not r["text"]: r["needs_text"] = "yes"
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, COLS); w.writeheader(); w.writerows(out)
    rj = Path(a.out).with_suffix(".rejects.csv")
    with open(rj, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, ["reason", "source", "url", "date", "rating", "evidence_url"]); w.writeheader(); w.writerows(rejects)
    first_origin = Counter(r["origin_files"].split("; ")[0] for r in out)
    rep = {"unique_posts": len(out), "with_text": sum(bool(r["text"]) for r in out), "needs_text": sum(r["needs_text"] == "yes" for r in out),
           "with_factcheck_link": sum(is_url(r["label_source"]) for r in out), "no_factcheck_link": sum(not is_url(r["label_source"]) for r in out),
           "by_first_source": dict(first_origin), "in_more_than_one_source": sum("; " in r["origin_files"] for r in out),
           "platform": dict(Counter(r["platform"] for r in out)), "verdict_hint": dict(Counter(r["factcheck_verdict_hint"] or "none" for r in out)),
           "rejected": dict(Counter(x["reason"] for x in rejects)), **dict(notes)}
    Path(a.out).with_suffix(".report.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
