"""Fine-tune one model for binary credibility classification.

    python src/train.py --model xlmr            # roberta-tagalog | mbert | xlmr
    python src/train.py --model xlmr --push     # also upload model + auto-generated model card to the Hub

Outputs (results/<model>/): metrics.json, test_predictions.csv, training_log.json, MODEL_CARD.md
Model weights: models/<model>/
"""
import argparse, json, os
from pathlib import Path
import numpy as np, pandas as pd, torch, yaml
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from torch import nn
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                          EarlyStoppingCallback, Trainer, TrainingArguments, set_seed)
from datasets import Dataset


def metrics_from(y_true, y_pred, pos):
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=[pos], average=None, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_not_credible": float(p[0]), "recall_not_credible": float(r[0]), "f1_not_credible": float(f[0]),
        "macro_f1": float(precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)[2]),
        "weighted_f1": float(precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)[2]),
        "confusion_matrix_[[tn,fp],[fn,tp]]_pos=not_credible": confusion_matrix(y_true, y_pred, labels=[1 - pos, pos]).tolist(),
        "n": int(len(y_true)),
    }


class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kw):
        super().__init__(*args, **kw); self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        w = self.class_weights.to(outputs.logits.device) if self.class_weights is not None else None
        loss = nn.CrossEntropyLoss(weight=w)(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss


def model_card(name, cfg, base, m, counts):
    t = cfg["training"]
    return f"""---
license: cc-by-nc-4.0
base_model: {base}
language: [tl, en]
pipeline_tag: text-classification
tags: [misinformation-detection, filipino, taglish, credibility-classification]
---
# TsekTxt - {name} credibility classifier

Binary classifier for Filipino/English news-style text. Fine-tuned from `{base}` as one of three models
(RoBERTa-Tagalog, mBERT, XLM-RoBERTa) compared in the thesis *Determining the Top Predictors of Taglish
Election Misinformation*.

**Label mapping:** `0 = not_credible`, `1 = credible` (see `config.json` -> `id2label`).

## Training data
Combined, cleaned Fake News Filipino Dataset + Philippine Fake News Corpus: {sum(counts.values())} articles
(train/val/test = 70/15/15, stratified, seed {cfg['seed']}). These are **news articles labelled by source credibility**,
not social-media posts. Performance on informal Taglish social media is evaluated separately (see thesis).

## Hyperparameters
lr {t['learning_rate']}, batch {t['train_batch_size']}, max {t['max_epochs']} epochs with early stopping (patience {t['early_stopping_patience']}, validation loss),
max length {t['max_length']}, weight decay {t['weight_decay']}, warmup ratio {t['warmup_ratio']}, class-weighted CE: {t['class_weighted_loss']}.

## Internal test results (n={m['n']}; positive class = not_credible)
| Accuracy | Precision | Recall | F1 | Macro F1 | Weighted F1 |
|---|---|---|---|---|---|
| {m['accuracy']:.4f} | {m['precision_not_credible']:.4f} | {m['recall_not_credible']:.4f} | {m['f1_not_credible']:.4f} | {m['macro_f1']:.4f} | {m['weighted_f1']:.4f} |

## Limitations
Labels come from the *source website's* reputation, so high in-distribution scores can reflect source style
rather than factual accuracy. Not a fact-checker; use as a supporting signal with human review.

## Usage
```python
from transformers import pipeline
clf = pipeline("text-classification", model="{cfg['hub']['user']}/{cfg['hub']['repo_names'][name]}")
print(clf("Your text here"))   # -> [{{'label': 'not_credible' | 'credible', 'score': ...}}]
```
Code: https://github.com/{cfg['hub']['user']}/tsektxt-model-training
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["roberta-tagalog", "mbert", "xlmr"])
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--push", action="store_true")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config)); t = cfg["training"]; pos = cfg["labels"]["positive_label"]
    id2label = {int(k): v for k, v in cfg["labels"]["id2label"].items()}; label2id = {v: k for k, v in id2label.items()}
    base = cfg["models"][a.model]; set_seed(cfg["seed"])
    out = Path("results") / a.model; out.mkdir(parents=True, exist_ok=True); mdir = Path("models") / a.model

    d = cfg["data"]["splits_dir"]
    tr, va, te = (pd.read_csv(f"{d}/{s}.csv") for s in ("train", "val", "test"))
    tok = AutoTokenizer.from_pretrained(base)

    def enc(df):
        ds = Dataset.from_pandas(df[["text", "label"]].reset_index(drop=True))
        return ds.map(lambda b: tok(b["text"], truncation=True, max_length=t["max_length"]), batched=True, remove_columns=["text"])

    ds_tr, ds_va, ds_te = enc(tr), enc(va), enc(te)
    counts = tr.label.value_counts().to_dict()
    weights = torch.tensor([len(tr) / (2 * counts[0]), len(tr) / (2 * counts[1])], dtype=torch.float) if t["class_weighted_loss"] else None

    model = AutoModelForSequenceClassification.from_pretrained(base, num_labels=2, id2label=id2label, label2id=label2id)
    args = TrainingArguments(
        output_dir=str(out / "checkpoints"), learning_rate=t["learning_rate"], per_device_train_batch_size=t["train_batch_size"],
        per_device_eval_batch_size=t["eval_batch_size"], num_train_epochs=t["max_epochs"], weight_decay=t["weight_decay"],
        warmup_ratio=t["warmup_ratio"], lr_scheduler_type="linear", eval_strategy="epoch", save_strategy="epoch", save_total_limit=2,
        load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,   # lowest validation loss
        fp16=torch.cuda.is_available(), seed=cfg["seed"], report_to="none", logging_strategy="epoch")
    trainer = WeightedTrainer(model=model, args=args, train_dataset=ds_tr, eval_dataset=ds_va, data_collator=DataCollatorWithPadding(tok),
                              callbacks=[EarlyStoppingCallback(early_stopping_patience=t["early_stopping_patience"])], class_weights=weights)
    trainer.train()
    json.dump(trainer.state.log_history, open(out / "training_log.json", "w"), indent=2)

    pred = trainer.predict(ds_te)
    probs = torch.softmax(torch.tensor(pred.predictions), -1).numpy(); y_pred = probs.argmax(1)
    m = metrics_from(te.label.values, y_pred, pos); m["model"] = a.model; m["base_model"] = base
    json.dump(m, open(out / "metrics.json", "w"), indent=2)
    pd.DataFrame({"id": te.id, "y_true": te.label, "y_pred": y_pred, "prob_credible": probs[:, 1]}).to_csv(out / "test_predictions.csv", index=False)
    (out / "MODEL_CARD.md").write_text(model_card(a.model, cfg, base, m, {"all": len(tr) + len(va) + len(te)}))

    mdir.mkdir(parents=True, exist_ok=True); trainer.save_model(str(mdir)); tok.save_pretrained(str(mdir))
    print(json.dumps(m, indent=2))

    if a.push:
        from huggingface_hub import HfApi
        repo = f"{cfg['hub']['user']}/{cfg['hub']['repo_names'][a.model]}"
        api = HfApi(); api.create_repo(repo, exist_ok=True)
        api.upload_folder(folder_path=str(mdir), repo_id=repo, commit_message="Retrain on re-cleaned dataset")
        api.upload_file(path_or_fileobj=str(out / "MODEL_CARD.md"), path_in_repo="README.md", repo_id=repo, commit_message="Update model card")
        print("pushed ->", f"https://huggingface.co/{repo}")


if __name__ == "__main__":
    main()
