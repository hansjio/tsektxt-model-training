"""Screen the candidate posts BEFORE annotation: is each post usable, and is the set as a whole usable? Hints only, never labels.

    python src/screen_candidates.py                         (reads data/external/merged_for_annotation.csv)

Per post (written to data/external/candidate_screening.csv, untracked):
  words, length_bin, language_hint (function-word heuristic, not a language-ID model), election_terms (matched election
  words / candidate names), in_window (Feb 8 - May 9 2022 from the post date; 'unknown' if no date), has_factcheck_link,
  duplicate_of (same normalised text), screen = ready / check / drop, and screen_reasons.
  credible_hint: 'maybe credible' only for news-outlet / official accounts WITHOUT a fact-check link. This is a HINT for
  annotators to look at first, never a label: the humans decide every label.
Set summary (results/candidate_screening_summary.json, counts only, no post text): how many posts are usable, the mix by
length, language, platform, source type and election relevance, and the gaps to the targets in Chapter III.

Rules: TEST ONLY set; nothing here trains or tunes anything; no model is run, so no score is looked at.
"""
import argparse, csv, json, re, sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from profile_corpus import guess_language, TOK  # noqa: E402

WINDOW = ("2022-02-08", "2022-05-09")
TARGET_POSTS = (300, 500)   # Chapter III target size for the Taglish set
# Election vocabulary: generic terms plus 2022 national candidates and common nicknames. Lower-case, matched as whole words.
ELECTION = set("""halalan eleksyon election elections boto botante bumoto iboto vote votes voter voters voting comelec kandidato candidate
candidates presidente president presidential bise vice vp senador senator senate partylist party-list debate debates survey surveys
pulse sws rally rallies campaign kampanya precinct presinto vcm balota ballot ballots dayaan daya fraud canvass bilangan
marcos bbm bongbong robredo leni kakampink kakampinks pink lacson ping moreno isko domagoso pacquiao manny sotto tito
pangilinan kiko sara inday duterte uniteam leody abella mangondato gonzales ong""".split())
NEWSY = re.compile(r"abs-?cbn|gmanews|gmanetwork|inquirer|rappler|philstar|cnnphilippines|manilabulletin|news5|pna\.gov|comelec|"
                   r"bbc|reuters|mindanews|interaksyon|sunstar|abante|bombo|dzrh|dzmm|teleradyo|pcij", re.I)


def norm(t):
    return " ".join(TOK.findall(t.lower()))


def length_bin(n):
    return "under_20" if n < 20 else "20_39" if n < 40 else "40_99" if n < 100 else "100_plus"


def screen(rows):
    seen, out = {}, []
    for r in rows:
        text = (r.get("text") or "").strip(); words = TOK.findall(text.lower()); n = len(text.split())
        date = (r.get("collection date") or r.get("post_date") or "")[:10]
        in_window = "unknown" if not date else "yes" if WINDOW[0] <= date <= WINDOW[1] else "no"
        terms = sorted({w for w in words if w in ELECTION})
        fc = bool((r.get("all_factcheck_urls") or r.get("factcheck_link") or "").strip())
        key = norm(text)[:200]; dup = seen.get(key, "") if key else ""
        if key and not dup: seen[key] = r["post_id"]
        newsy = r.get("source_type") == "news_outlet" or r.get("source_type") == "politician_or_official" or bool(NEWSY.search((r.get("source_account") or "") + " " + (r.get("url") or "")))
        reasons = []
        if not text: reasons.append("no_text")
        elif n < 5: reasons.append("under_5_words")
        if in_window == "no" or "OUTSIDE WINDOW" in (r.get("merge_notes") or ""): reasons.append("outside_window")
        if dup: reasons.append(f"duplicate_of_{dup}")
        drop = bool(reasons)
        if not drop:
            if in_window == "unknown": reasons.append("post_date_unknown")
            if not terms: reasons.append("no_election_terms_found")
            if not r.get("url", "").startswith("http"): reasons.append("no_post_url")
        out.append({"post_id": r["post_id"], "words": n, "length_bin": length_bin(n) if text else "",
                    "language_hint": guess_language(words) if text else "", "election_terms": " ".join(terms[:8]),
                    "in_window": in_window, "has_factcheck_link": "yes" if fc else "no",
                    "credible_hint": "maybe credible (news/official account, no fact-check) - check" if newsy and not fc else "",
                    "duplicate_of": dup, "screen": "drop" if drop else ("check" if reasons else "ready"),
                    "screen_reasons": "; ".join(reasons)})
    return out


def summary(out, rows):
    usable = [o for o in out if o["screen"] != "drop"]
    c = lambda k, xs=usable: dict(Counter(o[k] or "none" for o in xs).most_common())
    src = dict(Counter((r.get("source_type") or "unknown") for r, o in zip(rows, out) if o["screen"] != "drop").most_common())
    plat = dict(Counter((r.get("platform") or "unknown") for r, o in zip(rows, out) if o["screen"] != "drop").most_common())
    n_fc = sum(o["has_factcheck_link"] == "yes" for o in usable); n_cred_hint = sum(bool(o["credible_hint"]) for o in usable)
    gaps = []
    if len(usable) < TARGET_POSTS[0]: gaps.append(f"only {len(usable)} usable posts; Chapter III target is {TARGET_POSTS[0]}-{TARGET_POSTS[1]}")
    if n_cred_hint < 0.25 * len(usable): gaps.append(f"few likely-credible candidates ({n_cred_hint}); collect more credible posts so both classes are present")
    lb = Counter(o["length_bin"] for o in usable)
    if lb["100_plus"] < 30: gaps.append(f"only {lb['100_plus']} posts with 100+ words; the length comparison needs a long-form stratum (TASKS/05)")
    en = sum(o["language_hint"] == "english" for o in usable)
    if en > 0.3 * len(usable): gaps.append(f"{en} of {len(usable)} usable posts look English-only (heuristic); the study is about Taglish/Tagalog, so confirm language_pattern or collect more Taglish")
    no_terms = sum(not o["election_terms"] for o in usable)
    if no_terms: gaps.append(f"{no_terms} usable posts have no election word or candidate name; check they are about the 2022 election")
    return {"rows_in": len(out), "drop": sum(o["screen"] == "drop" for o in out), "check": sum(o["screen"] == "check" for o in out),
            "ready": sum(o["screen"] == "ready" for o in out), "usable_ready_plus_check": len(usable),
            "drop_reasons": dict(Counter(x.split("_of_")[0] for o in out if o["screen"] == "drop" for x in o["screen_reasons"].split("; ")).most_common()),
            "check_reasons": dict(Counter(x for o in out if o["screen"] == "check" for x in o["screen_reasons"].split("; ")).most_common()),
            "usable_by_length_bin": c("length_bin"), "usable_by_language_hint": c("language_hint"), "usable_by_platform": plat,
            "usable_by_source_type": src, "usable_with_factcheck_link": n_fc, "usable_with_credible_hint": n_cred_hint,
            "gaps": gaps, "note": "Hints only. Labels are assigned by two human annotators; nothing here is a label or a model score."}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(ROOT / "data/external/merged_for_annotation.csv"))
    ap.add_argument("--out", default=str(ROOT / "data/external/candidate_screening.csv"))
    ap.add_argument("--summary", default=str(ROOT / "results/candidate_screening_summary.json"))
    a = ap.parse_args(argv)
    rows = list(csv.DictReader(open(a.csv, encoding="utf-8-sig", newline="")))
    out = screen(rows)
    with open(a.out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, list(out[0])); w.writeheader(); w.writerows(out)
    s = summary(out, rows)
    Path(a.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(a.summary).write_text(json.dumps(s, indent=2), encoding="utf-8")
    print(json.dumps(s, indent=1))


if __name__ == "__main__":
    main()
