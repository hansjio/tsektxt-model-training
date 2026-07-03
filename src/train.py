import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import yaml
from datasets import Dataset
from sklearn.metrics import classification_report
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
import evaluate

from data_prep import get_splits

# Resolve the project root relative to THIS file's location, not the
# current working directory. This makes the script runnable from anywhere:
#   python train.py ...
#   python src/train.py ...
#   cd src && python train.py ...
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

DEFAULT_CONFIG = os.path.join(PROJECT_ROOT, "configs", "training_config.yaml")


def resolve_path(path):
    """Resolve a path from the config file relative to PROJECT_ROOT,
    unless it's already absolute."""
    if os.path.isabs(path):
        return path
    return os.path.join(PROJECT_ROOT, path)


def to_hf_dataset(d):
    d = d.rename(columns={"Content": "text", "Label": "label"})
    return Dataset.from_pandas(d[["text", "label"]], preserve_index=False)


def main(model_key, config_path, push_to_hub):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg["models"][model_key]
    train_cfg = cfg["training"]
    data_cfg = cfg["data"]

    raw_path = resolve_path(data_cfg["raw_path"])
    processed_dir = resolve_path(data_cfg["processed_dir"])
    output_dir = resolve_path(model_cfg["output_dir"])

    os.makedirs(output_dir, exist_ok=True)

    train_df, val_df, test_df = get_splits(raw_path, processed_dir, data_cfg["seed"])

    model_name = model_cfg["hf_name"]

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=train_cfg["max_length"],
        )

    train_ds = to_hf_dataset(train_df).map(tokenize_fn, batched=True)
    val_ds = to_hf_dataset(val_df).map(tokenize_fn, batched=True)
    test_ds = to_hf_dataset(test_df).map(tokenize_fn, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    class_counts = train_df["Label"].value_counts().sort_index()
    total = class_counts.sum()
    class_weights = torch.tensor(
        [total / (2 * c) for c in class_counts], dtype=torch.float
    )

    f1_metric = evaluate.load("f1")
    acc_metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "f1": f1_metric.compute(predictions=preds, references=labels, average="macro")["f1"],
            "accuracy": acc_metric.compute(predictions=preds, references=labels)["accuracy"],
        }

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            loss_fct = nn.CrossEntropyLoss(weight=class_weights.to(logits.device))
            loss = loss_fct(logits, labels)
            return (loss, outputs) if return_outputs else loss

    args = TrainingArguments(
        output_dir=os.path.join(output_dir, "checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=float(train_cfg["learning_rate"]),
        per_device_train_batch_size=train_cfg["train_batch_size"],
        per_device_eval_batch_size=train_cfg["eval_batch_size"],
        num_train_epochs=train_cfg["num_epochs"],
        weight_decay=train_cfg["weight_decay"],
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=50,
        report_to="none",
        disable_tqdm=True,
    )

    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    preds = trainer.predict(test_ds)
    y_pred = np.argmax(preds.predictions, axis=-1)
    report = classification_report(
        test_df["Label"], y_pred, target_names=["Not Suspicious", "Suspicious"]
    )
    print(report)

    with open(os.path.join(output_dir, "test_report.txt"), "w") as f:
        f.write(report)

    trainer.save_model(os.path.join(output_dir, "final"))
    tokenizer.save_pretrained(os.path.join(output_dir, "final"))

    if push_to_hub:
        hub_repo = model_cfg["hub_repo"]
        model.push_to_hub(hub_repo)
        tokenizer.push_to_hub(hub_repo)
        print(f"Pushed to https://huggingface.co/{hub_repo}")

        links_path = os.path.join(PROJECT_ROOT, "models", "hub_repo_links.json")
        os.makedirs(os.path.dirname(links_path), exist_ok=True)
        links = {}
        if os.path.exists(links_path):
            with open(links_path) as f:
                links = json.load(f)
        links[model_key] = f"https://huggingface.co/{hub_repo}"
        with open(links_path, "w") as f:
            json.dump(links, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_key", required=True, choices=["xlmr", "roberta_tagalog", "mbert"]
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--push_to_hub", action="store_true")
    args = parser.parse_args()
    main(args.model_key, args.config, args.push_to_hub)