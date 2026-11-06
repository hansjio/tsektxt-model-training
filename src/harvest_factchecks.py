"""Find CANDIDATE posts for the Taglish test set via public fact-check articles (Vera Files, Tsek.ph).

    python src/harvest_factchecks.py --out data/external/candidates.csv

Uses the sites' public WordPress REST API (stdlib only). It does NOT touch Facebook/X: it lists
fact-check articles from the collection window (Feb 8 - May 9, 2022) and the social-media links cited
in them. Output = a worklist for the MANUAL collection the thesis requires (open the link, copy the
public post text into the Google Sheet yourself). Nothing here assigns the final label: the annotators do.

Columns: source, article_date, article_title, article_url, verdict_hint, social_urls, platform_guess, election_related
verdict_hint is only a keyword guess from the title/body (false/fake/misleading vs. true/accurate) to help
you balance classes; ignore it when annotating.
"""
import argparse, csv, html, json, re, time, urllib.parse, urllib.request

SITES = {"verafiles": "https://verafiles.org", "tsek": "https://www.tsek.ph"}
AFTER, BEFORE = "2022-02-08T00:00:00", "2022-05-10T00:00:00"
UA = "Mozilla/5.0 (thesis research; polite, 1 req/sec)"
SOCIAL = re.compile(r'https?://(?:www\.|m\.|web\.|mobile\.)?(?:facebook\.com|fb\.watch|fb\.com|twitter\.com|x\.com)/[^\s"\'<>)\\]+', re.I)
ELECTION = re.compile(r"marcos|robredo|leni|bbm|duterte|sara|isko|moreno|pacquiao|lacson|sotto|ping|halal|election|comelec|boto|vote|ballot|campaign|kakampink|uniteam", re.I)
FALSE = re.compile(r"\bfalse\b|\bfake\b|misleading|hoax|mali|walang katotohanan|no basis|doctored|satire|unfounded|not true", re.I)
TRUE = re.compile(r"\btrue\b|\baccurate\b|\bcorrect\b|totoo|\bverified\b|\bbatay sa\b", re.I)


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=40) as r:
        return json.loads(r.read().decode("utf-8"))


def strip(s):
    return html.unescape(re.sub(r"<[^>]+>", " ", s))


def harvest(name, base, cap):
    rows, page = [], 1
    while len(rows) < cap:
        q = urllib.parse.urlencode({"per_page": 50, "page": page, "after": AFTER, "before": BEFORE, "orderby": "date", "order": "asc"})
        try:
            posts = get(f"{base}/wp-json/wp/v2/posts?{q}")
        except Exception as e:  # past last page the API returns HTTP 400
            print(f"[{name}] stop at page {page}: {e}"); break
        if not posts: break
        for p in posts:
            body, title = p["content"]["rendered"], strip(p["title"]["rendered"])
            links = sorted({u.rstrip(".,;") for u in SOCIAL.findall(body)})
            text = title + " " + strip(body)[:3000]
            verdict = "false?" if FALSE.search(title) else "true?" if TRUE.search(title) else ""
            rows.append({"source": name, "article_date": p["date"][:10], "article_title": title, "article_url": p["link"],
                         "verdict_hint": verdict, "social_urls": " | ".join(links),
                         "platform_guess": ",".join(sorted({"facebook" if re.search("facebook|fb\\.", u) else "x" for u in links})),
                         "election_related": bool(ELECTION.search(text))})
        print(f"[{name}] page {page}: {len(rows)} articles"); page += 1; time.sleep(1)
    return rows[:cap]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/external/candidates.csv")
    ap.add_argument("--cap", type=int, default=2000, help="max articles per site")
    ap.add_argument("--only-with-links", action="store_true", help="keep only articles that cite a Facebook/X link")
    a = ap.parse_args()
    rows = [r for n, b in SITES.items() for r in harvest(n, b, a.cap)]
    if a.only_with_links: rows = [r for r in rows if r["social_urls"]]
    rows = [r for r in rows if r["election_related"]]
    with open(a.out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])) if rows else None
        if w: w.writeheader(); w.writerows(rows)
    print(f"\nwrote {len(rows)} election-related articles to {a.out}; {sum(bool(r['social_urls']) for r in rows)} cite a Facebook/X link")


if __name__ == "__main__":
    main()
