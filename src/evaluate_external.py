"""Evaluate a trained model on the external Taglish election test set (never used in training).

    python src/evaluate_external.py --model xlmr --csv data/external/taglish_test.csv
CSV needs columns: id, text, label  (1 = credible, 0 = not credible).
"""
import argparse, json
from pathlib import Path
import pandas as pd, torch, yaml
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from train import metrics_from


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["roberta-tagalog", "mbert", "xlmr"])
    ap.add_argument("--csv", required=True); ap.add_argument("--config", default="configs/config.yaml")
    a = ap.parse_args(); cfg = yaml.safe_load(open(a.config))
    df = pd.read_csv(a.csv)
    local = Path("models") / a.model
    mdir = local if local.exists() else f"{cfg['hub']['user']}/{cfg['hub']['repo_names'][a.model]}"  # fall back to Hugging Face
    tok = AutoTokenizer.from_pretrained(mdir); model = AutoModelForSequenceClassification.from_pretrained(mdir).eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"; model.to(dev); probs = []
    for i in range(0, len(df), 32):
        b = tok(df.text.iloc[i:i + 32].tolist(), truncation=True, max_length=cfg["training"]["max_length"], padding=True, return_tensors="pt").to(dev)
        with torch.no_grad(): probs.append(torch.softmax(model(**b).logits, -1).cpu())
    probs = torch.cat(probs).numpy(); y_pred = probs.argmax(1)
    m = metrics_from(df.label.values, y_pred, cfg["labels"]["positive_label"]); m["model"] = a.model
    out = Path("results") / a.model; out.mkdir(parents=True, exist_ok=True)
    json.dump(m, open(out / "external_metrics.json", "w"), indent=2)
    pd.DataFrame({"id": df.id, "y_true": df.label, "y_pred": y_pred, "prob_credible": probs[:, 1]}).to_csv(out / "external_predictions.csv", index=False)
    print(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()
