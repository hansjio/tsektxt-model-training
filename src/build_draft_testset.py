"""Build a DRAFT Taglish election test set from real fact-checked posts (Vera Files, Tsek.ph).

    python src/build_draft_testset.py --out data/external/draft_testset.csv

For every election-related fact-check in Feb 8 - May 9 2022 that cites a Facebook/X post, this pulls:
  - the post text AS QUOTED by the fact-checker (the STATEMENT / PAHAYAG section), when present
  - the post URL, platform, article date, and reaction/comment/share counts if the article states them
  - a label_hint from the fact-check headline (0 = not credible: false/misleading/fake; 1 = credible: true/accurate)
Rows use the same columns as the group's Google Sheet, plus label_hint, article_url, text_origin.

IMPORTANT: this is a worklist/draft, not final data. The text is the fact-checker's transcription, which is real but
may be partial, translated or a screenshot description. Annotators must open the post URL (if still online), confirm
the text, and assign the final label. Posts that cannot be verified should be flagged in text_origin, not invented.
English/Filipino twin articles are de-duplicated by post URL (English kept).
"""
import argparse, csv, html, json, re, time, urllib.parse, urllib.request

SITES = {"verafiles": "https://verafiles.org", "tsek": "https://www.tsek.ph"}
AFTER, BEFORE = "2022-02-08T00:00:00", "2022-05-10T00:00:00"
UA = "Mozilla/5.0 (thesis research; polite, 1 req/sec)"
SOCIAL = re.compile(r'https?://(?:www\.|m\.|web\.|mobile\.)?(?:facebook\.com|fb\.watch|fb\.com|twitter\.com|x\.com)/[^\s"\'<>)\\]+', re.I)
ELECTION = re.compile(r"marcos|robredo|leni|bbm|duterte|sara|isko|moreno|pacquiao|lacson|sotto|halal|election|comelec|boto|vote|ballot|campaign|kakampink|uniteam", re.I)
FALSE = re.compile(r"\bfalse\b|\bfake\b|misleading|mislead|hoax|nakaliligaw|mali\b|walang katotohanan|no basis|doctored|unfounded|not true|fabricated", re.I)
# Credible hint only on clear verdict wording. NOT "totoo": Tsek titles "Totoo o hindi..." ("True or not") name the claim, not the verdict.
TRUE = re.compile(r"\b(?:is|are|was|were) (?:true|accurate|correct)\b|\bpatunay\b|\bverified true\b", re.I)
# Links that point at the fact-checkers themselves (their own page/plugin), never at the misinformation post.
FACTCHECKER_LINK = re.compile(r"facebook\.com/plugins/|(?:facebook|twitter|x)\.com/(?:verafiles|tsekph|tsek\.ph|factcheckphils|factcheckpatrol|rapplerdotcom|vera\.files)\b|/factcheck(?:phils|patrol)", re.I)
TAGALOG_TITLE = re.compile(r"\b(nag post|nagpost|kolumnistang|nakaliligaw|walang|ang|ng|mga|sa)\b", re.I)
NUM = r"(?:more than |over |at least )?([\d,\.]+)\s*(k|thousand)?"


def get(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=40) as r:
        return json.loads(r.read().decode("utf-8"))


def unembed(u):
    """Facebook embed URLs (plugins/post.php?href=...) -> the real post URL; trailing punctuation removed."""
    u = html.unescape(u).rstrip(".,;")
    if "facebook.com/plugins/" in u:
        href = urllib.parse.parse_qs(urllib.parse.urlparse(u).query).get("href")
        if href: return href[0].rstrip("/")
    return u


def clean(s):
    s = re.sub(r"<[^>]+>", "\n", s)
    return html.unescape(re.sub(r"\s*\n\s*", " ", s)).strip()


def count(text, word):
    m = re.search(NUM + r"\s*" + word, text, re.I)
    if not m: return ""
    n = float(m.group(1).replace(",", "") or 0)
    return int(n * 1000) if m.group(2) else int(n)


def quote_text(body_html):
    """Text of the quoted post: first long “...” block after STATEMENT/PAHAYAG, else first paragraph after it."""
    plain = clean(body_html)
    m = re.search(r"\b(STATEMENT|PAHAYAG)\b(.*?)(?:Source:|Pinagmulan:|$)", plain, re.S)
    seg = m.group(2) if m else plain
    quotes = [q.strip() for q in re.findall(r"[“\"]([^”\"]{40,900})[”\"]", seg)]
    if quotes: return max(quotes, key=len), "quoted_in_factcheck"
    return "", "no_quote_found"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/external/draft_testset.csv")
    ap.add_argument("--cap", type=int, default=2000)
    a = ap.parse_args()
    out, seen = [], set()
    for name, base in SITES.items():
        page = 1
        while page * 50 <= a.cap + 50:
            q = urllib.parse.urlencode({"per_page": 50, "page": page, "after": AFTER, "before": BEFORE, "orderby": "date", "order": "asc"})
            try: posts = get(f"{base}/wp-json/wp/v2/posts?{q}")
            except Exception as e: print(f"[{name}] stop page {page}: {e}"); break
            if not posts: break
            for p in posts:
                body, title = p["content"]["rendered"], clean(p["title"]["rendered"])
                links = sorted({u for u in map(unembed, SOCIAL.findall(body)) if not FACTCHECKER_LINK.search(u)})
                plain = clean(body)
                if not links or not ELECTION.search(title + " " + plain[:3000]): continue
                if TAGALOG_TITLE.search(title) and len(TAGALOG_TITLE.findall(title)) >= 3: continue  # Filipino twin
                url = links[0]
                if url in seen: continue
                seen.add(url)
                text, origin = quote_text(body)
                head = title + " " + plain[:400]
                hint = "0" if FALSE.search(head) else "1" if TRUE.search(title) else ""
                out.append({"post_id": f"{name}-{p['id']}", "text": text, "platform": "facebook" if re.search("facebook|fb\\.", url) else "x",
                            "url": url, "collection date": p["date"][:10], "reactions_count": count(plain, "reactions"),
                            "shares_reposts_count": count(plain, "shares"), "comments_count": count(plain, "comments"),
                            "collector": "auto-draft", "label_hint": hint, "article_url": p["link"], "text_origin": origin})
            print(f"[{name}] page {page}: {len(out)} drafts so far"); page += 1; time.sleep(1)
    with open(a.out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    withtext = [r for r in out if r["text"]]
    print(f"\nwrote {len(out)} drafts to {a.out}; {len(withtext)} with quoted text; label_hint 0={sum(r['label_hint']=='0' for r in out)} 1={sum(r['label_hint']=='1' for r in out)} blank={sum(r['label_hint']=='' for r in out)}")


if __name__ == "__main__":
    main()
