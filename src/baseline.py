"""Non-neural baseline: TF-IDF (word 1-2 grams) + class-balanced Logistic Regression.

Same splits, same metrics, same output files as the transformers, so compare.py (incl. McNemar) treats it as a fourth model.
Primary baseline reads only the FIRST `--max_words` words (default 128) to roughly match the transformers' 128-token input budget.
A full-text variant is also reported for reference.

    python src/baseline.py
    python src/baseline.py --external data/external/taglish_test.csv     # later, for the Taglish set
Writes: results/baseline-tfidf/{metrics.json,test_predictions.csv,metrics_fulltext.json}
"""
import argparse, json
from pathlib import Path
import joblib, pandas as pd, yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.pipeline import make_pipeline


def metrics_from(y_true, y_pred, pos):  # same definition as train.py (copied so this script needs no torch)
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=[pos], average=None, zero_division=0)
    return {"accuracy": float(accuracy_score(y_true, y_pred)),
            "precision_not_credible": float(p[0]), "recall_not_credible": float(r[0]), "f1_not_credible": float(f[0]),
            "macro_f1": float(precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)[2]),
            "weighted_f1": float(precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)[2]),
            "confusion_matrix_[[tn,fp],[fn,tp]]_pos=not_credible": confusion_matrix(y_true, y_pred, labels=[1 - pos, pos]).tolist(),
            "n": int(len(y_true))}


def fit(train, max_words, seed):
    cut = (lambda s: " ".join(s.split()[:max_words])) if max_words else (lambda s: s)
    pipe = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200000, sublinear_tf=True),
                         LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed))
    pipe.fit(train.text.map(cut), train.label); return pipe, cut


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--max_words", type=int, default=128); ap.add_argument("--external", default=None); a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config)); d = cfg["data"]["splits_dir"]; pos = cfg["labels"]["positive_label"]; seed = cfg["seed"]
    tr, te = pd.read_csv(f"{d}/train.csv"), pd.read_csv(f"{d}/test.csv")
    out = Path("results/baseline-tfidf"); out.mkdir(parents=True, exist_ok=True)

    pipe, cut = fit(tr, a.max_words, seed)
    pred = pipe.predict(te.text.map(cut)); prob = pipe.predict_proba(te.text.map(cut))[:, 1]
    m = metrics_from(te.label.values, pred, pos); m.update(model="baseline-tfidf", base_model=f"TF-IDF(1-2gram)+LogReg, first {a.max_words} words")
    json.dump(m, open(out / "metrics.json", "w"), indent=2)
    pd.DataFrame({"id": te.id, "y_true": te.label, "y_pred": pred, "prob_credible": prob}).to_csv(out / "test_predictions.csv", index=False)

    pf, cf = fit(tr, None, seed); mf = metrics_from(te.label.values, pf.predict(te.text), pos); mf["model"] = "baseline-tfidf-fulltext"
    json.dump(mf, open(out / "metrics_fulltext.json", "w"), indent=2)
    Path("models/baseline-tfidf").mkdir(parents=True, exist_ok=True); joblib.dump({"pipe": pipe, "max_words": a.max_words}, "models/baseline-tfidf/model.joblib")

    if a.external:
        ex = pd.read_csv(a.external); ep = pipe.predict(ex.text.map(cut)); eprob = pipe.predict_proba(ex.text.map(cut))[:, 1]
        em = metrics_from(ex.label.values, ep, pos); em["model"] = "baseline-tfidf"
        json.dump(em, open(out / "external_metrics.json", "w"), indent=2)
        pd.DataFrame({"id": ex.id, "y_true": ex.label, "y_pred": ep, "prob_credible": eprob}).to_csv(out / "external_predictions.csv", index=False)
    print(json.dumps({"primary": {k: round(v, 4) for k, v in m.items() if isinstance(v, float)},
                      "fulltext_reference": {k: round(v, 4) for k, v in mf.items() if isinstance(v, float)}}, indent=2))


if __name__ == "__main__":
    main()
