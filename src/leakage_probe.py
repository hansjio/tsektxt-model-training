"""Diagnostic: how much of the label can be predicted from only the START or only the END of an article?

If a simple bag-of-words model scores far above the 64% majority rate from just 12 words, labels partly encode source style
(the publishing site), so high internal scores do not prove factual-misinformation detection. Uses the official train/test split.

    python src/leakage_probe.py          -> results/leakage_probe.json
"""
import json
import pandas as pd, yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


def main(config="configs/config.yaml", n=12):
    cfg = yaml.safe_load(open(config)); d = cfg["data"]["splits_dir"]
    tr, te = pd.read_csv(f"{d}/train.csv"), pd.read_csv(f"{d}/test.csv")
    views = {"full_text": lambda s: s, f"first_{n}_words": lambda s: " ".join(s.split()[:n]), f"last_{n}_words": lambda s: " ".join(s.split()[-n:])}
    res = {"majority_class_accuracy": float(max(te.label.mean(), 1 - te.label.mean()))}
    for name, f in views.items():
        v = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200000, sublinear_tf=True)
        clf = LogisticRegression(max_iter=1000, random_state=cfg["seed"]).fit(v.fit_transform(tr.text.map(f)), tr.label)
        p = clf.predict(v.transform(te.text.map(f)))
        res[name] = {"accuracy": float(accuracy_score(te.label, p)), "macro_f1": float(f1_score(te.label, p, average="macro"))}
    json.dump(res, open("results/leakage_probe.json", "w"), indent=2); print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
