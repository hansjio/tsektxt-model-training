"""Clean the combined Filipino misinformation corpus.

Input : CSV with columns Label, Content  (1 = credible, 0 = not credible)
Output: data/processed/cleaned.csv  (id, text, label)  + cleaning_report.json

Steps, in order:
  1. Mojibake / encoding repair (ftfy)       e.g.  "â€œ" -> “
  2. HTML tags/entities, URLs, zero-width chars, whitespace
  3. Too-short texts (< min_words)
  4. Exact + near duplicates (same normalized first 200 chars); conflicting-label duplicates dropped
"""
import argparse, html, json, re
from pathlib import Path
import pandas as pd
import ftfy
import yaml

URL = re.compile(r"(https?://\S+|www\.\S+)", re.I)
TAG = re.compile(r"<[^>]+>")
ZW = re.compile(r"[\u200b\u200c\u200d\u2060\ufeff]")
WS = re.compile(r"\s+")


def norm_key(s: str, n: int = 200) -> str:
    return re.sub(r"\W+", " ", s.lower()).strip()[:n]


def basic_clean(t: str) -> str:
    t = ftfy.fix_text(str(t))
    t = html.unescape(t)
    t = TAG.sub(" ", t)
    t = URL.sub(" ", t)
    t = ZW.sub("", t)
    return WS.sub(" ", t).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--output", default=None)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    out_path = Path(a.output or cfg["data"]["processed_csv"]); out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(a.input)
    df = df.rename(columns={"Label": "label", "Content": "text"})[["text", "label"]]
    rep = {"rows_in": len(df)}
    df = df.dropna(subset=["text", "label"]).copy(); df["label"] = df["label"].astype(int)
    assert set(df.label.unique()) <= {0, 1}, "labels must be 0/1"

    cleaned, changed = [], 0
    for t in df.text:
        c = basic_clean(t); changed += c != t; cleaned.append(c)
    df["text"] = cleaned; rep["rows_modified_by_basic_cleaning"] = int(changed)

    n = len(df); df["words"] = df.text.str.split().str.len()
    df = df[df.words >= cfg["data"]["min_words"]]; rep["dropped_too_short"] = n - len(df)

    df["key"] = df.text.map(norm_key)
    nun = df.groupby("key").label.nunique(); conflict = set(nun[nun > 1].index)
    n = len(df); df = df[~df.key.isin(conflict)]; rep["dropped_conflicting_label_duplicates"] = n - len(df)
    n = len(df); df = df.drop_duplicates("key"); rep["dropped_duplicates"] = n - len(df)

    df = df.drop(columns=["key", "words"]).reset_index(drop=True)
    df.insert(0, "id", range(len(df)))
    df.to_csv(out_path, index=False)
    rep["rows_out"] = len(df); rep["label_counts"] = {int(k): int(v) for k, v in df.label.value_counts().items()}
    rep["leftover_mojibake_rows"] = int(df.text.str.contains("â€|Ã±|Ã©").sum())
    json.dump(rep, open(out_path.parent / "cleaning_report.json", "w"), indent=2)
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
