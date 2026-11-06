"""Test the three models without any frontend.

    python src/predict.py --text "Totoo ba na ... ?"                 # one post, all 3 models
    python src/predict.py --file posts.txt                           # one post per line
    python src/predict.py --csv data/external/taglish_test.csv       # csv with a 'text' column; writes <csv>_predictions.csv
    python src/predict.py --interactive                              # type posts repeatedly
    python src/predict.py --text "..." --source hub                  # force download from Hugging Face
    python src/predict.py --split-gap                                # train/val/test accuracy per model (overfitting check)

Models are loaded from models/<name>/ if present, else from the Hugging Face Hub (hansjio/tsektxt-*).
Label meaning comes from model.config.id2label (1 = credible, 0 = not credible).
"""
import argparse, sys
from pathlib import Path
import pandas as pd, torch, yaml
from transformers import AutoModelForSequenceClassification, AutoTokenizer

CFG = yaml.safe_load(open("configs/config.yaml"))
NAMES = list(CFG["models"])
MAXLEN = CFG["training"]["max_length"]
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def load(name, source):
    local = Path("models") / name
    path = str(local) if (source in ("auto", "local") and local.exists()) else f"{CFG['hub']['user']}/{CFG['hub']['repo_names'][name]}"
    print(f"[{name}] loading from {path}", file=sys.stderr)
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path).eval().to(DEV)
    return tok, model


@torch.no_grad()
def predict(tok, model, texts, bs=32):
    out = []
    for i in range(0, len(texts), bs):
        b = tok(texts[i:i + bs], truncation=True, max_length=MAXLEN, padding=True, return_tensors="pt").to(DEV)
        out.append(torch.softmax(model(**b).logits, -1).cpu())
    return torch.cat(out).numpy()  # columns follow label ids: [:,0]=not_credible, [:,1]=credible


def show(texts, results):
    for i, t in enumerate(texts):
        print(f"\n--- {t[:140]}{'...' if len(t) > 140 else ''}")
        for name, p in results.items():
            lab = "credible" if p[i, 1] >= 0.5 else "NOT credible"
            print(f"  {name:16s} {lab:13s} P(credible)={p[i,1]:.3f}")


def split_gap(models):
    d = Path(CFG["data"]["splits_dir"])
    for name, (tok, model) in models.items():
        row = []
        for s in ("train", "val", "test"):
            f = d / f"{s}.csv"
            if not f.exists(): row.append(f"{s}: missing {f}"); continue
            df = pd.read_csv(f); p = predict(tok, model, df.text.tolist())
            row.append(f"{s}: {(p.argmax(1) == df.label.values).mean():.4f}")
        print(f"{name:16s} " + "  ".join(row))
    print("\nBig train-test gap = overfitting. Low train AND test accuracy = underfitting.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text"); ap.add_argument("--file"); ap.add_argument("--csv")
    ap.add_argument("--interactive", action="store_true"); ap.add_argument("--split-gap", action="store_true")
    ap.add_argument("--models", nargs="+", default=NAMES, choices=NAMES)
    ap.add_argument("--source", default="auto", choices=["auto", "local", "hub"])
    a = ap.parse_args()
    models = {n: load(n, a.source) for n in a.models}
    if a.split_gap:
        return split_gap(models)
    if a.csv:
        df = pd.read_csv(a.csv); texts = df.text.astype(str).tolist()
        for n, (t, m) in models.items():
            p = predict(t, m, texts); df[f"{n}_p_credible"] = p[:, 1]; df[f"{n}_pred"] = p.argmax(1)
        out = Path(a.csv).with_name(Path(a.csv).stem + "_predictions.csv"); df.to_csv(out, index=False)
        print(f"wrote {out}")
        if "label" in df:
            for n in models: print(f"{n:16s} accuracy {(df[f'{n}_pred'] == df.label).mean():.4f}")
        return
    if a.interactive:
        while True:
            t = input("\npost (blank to quit): ").strip()
            if not t: break
            show([t], {n: predict(tk, m, [t]) for n, (tk, m) in models.items()})
        return
    texts = [a.text] if a.text else Path(a.file).read_text(encoding="utf-8").splitlines() if a.file else None
    if not texts: ap.error("give --text, --file, --csv, --interactive or --split-gap")
    texts = [t for t in texts if t.strip()]
    show(texts, {n: predict(tk, m, texts) for n, (tk, m) in models.items()})


if __name__ == "__main__":
    main()
