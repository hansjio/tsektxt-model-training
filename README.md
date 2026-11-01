# TsekTxt – Model Training

Fine-tunes and compares three transformer models for binary credibility classification of Philippine news-style text
(`1 = credible`, `0 = not credible`; *not credible* is the positive class for precision, recall and F1):

| Model | Base checkpoint | Hugging Face |
|---|---|---|
| RoBERTa-Tagalog | `jcblaise/roberta-tagalog-base` | [hansjio/tsektxt-roberta-tagalog](https://huggingface.co/hansjio/tsektxt-roberta-tagalog) |
| mBERT | `bert-base-multilingual-cased` | [hansjio/tsektxt-mbert](https://huggingface.co/hansjio/tsektxt-mbert) |
| XLM-RoBERTa | `xlm-roberta-base` | [hansjio/tsektxt-xlmr](https://huggingface.co/hansjio/tsektxt-xlmr) |

This is the training code for the thesis *Determining the Top Predictors of Taglish Election Misinformation Through a Comparative
Detection and Corpus Analysis Using RoBERTa-Tagalog, mBERT, and XLM-RoBERTa* (BS Computer Science, May 2026).
All settings are in `configs/config.yaml` and follow Chapter III of the thesis.

## Data

Combined Fake News Filipino Dataset and Philippine Fake News Corpus, cleaned by `src/clean_data.py`
(see `results/cleaning_report.json`): **25,352** articles, 16,283 credible and 9,069 not credible. Stratified 70/15/15 split
(17,746 / 3,803 / 3,803), seed 42, shared by all models. The Taglish election test set is **not** in this repository.

## Results on the internal formal-news test set (n = 3,803)

| Model | Accuracy | Precision (NC) | Recall (NC) | F1 (NC) | Macro F1 | Weighted F1 |
|---|---|---|---|---|---|---|
| **RoBERTa-Tagalog** | **0.9450** | **0.9103** | **0.9390** | **0.9244** | **0.9406** | **0.9452** |
| mBERT | 0.9301 | 0.8810 | 0.9302 | 0.9049 | 0.9248 | 0.9305 |
| XLM-RoBERTa | 0.9427 | 0.9150 | 0.9258 | 0.9204 | 0.9378 | 0.9428 |

(NC = not credible, the positive class. Bold = highest weighted F1, the thesis's primary comparison metric.)

McNemar's test (α = 0.05):

| Pair | b | c | p-value | Significant at 0.05 |
|---|---|---|---|---|
| RoBERTa-Tagalog vs mBERT | 151 | 94 | 0.00035 | yes |
| RoBERTa-Tagalog vs XLM-RoBERTa | 94 | 85 | 0.55 | no |
| mBERT vs XLM-RoBERTa | 94 | 142 | 0.0022 | yes |

b = first model correct and second wrong; c = the reverse.

Training summary (Tesla T4, early stopping on validation loss, patience 2; the checkpoint with the lowest validation loss is kept):

| Model | Epochs run | Best epoch | Best validation loss | Training time (min) |
|---|---|---|---|---|
| RoBERTa-Tagalog | 4 | 2 | 0.2480 | 11 |
| mBERT | 3 | 1 | 0.2216 | 9 |
| XLM-RoBERTa | 4 | 2 | 0.2420 | 15 |

**Reading the results.** RoBERTa-Tagalog scores highest, but its difference from XLM-RoBERTa is not statistically significant
(p = 0.55), so this run does not show that one beats the other. Both are significantly better than mBERT. Each number comes from
one training run (one seed).

**Caution.** In the cleaned corpus a plain TF-IDF classifier reaches about 82% accuracy from only the last 12 words of an article, because labels
come from the publishing site and sign-off lines differ between sites. The internal scores therefore partly reflect source style. The
informal Taglish election test set is the meaningful test of real-world generalisation and is still pending.

## Reproduce

```bash
pip install -r requirements.txt
python src/clean_data.py --input data/raw/combined_dataset_clean.csv   # -> data/processed/cleaned.csv
python src/make_splits.py                                              # 70/15/15 stratified, seed 42
python src/run_all.py --push                                           # train all three, upload, compare
python src/evaluate_external.py --model xlmr --csv data/external/taglish_test.csv   # per model, later
python src/compare.py                                                  # -> results/comparison.md
```
A Colab version is in `notebooks/colab_train.ipynb` (T4 GPU; add a Hugging Face write token as the Colab secret `HF_TOKEN`).

## Repository layout

```
configs/config.yaml        all hyperparameters and paths
src/clean_data.py          encoding repair, URL/HTML removal, deduplication
src/make_splits.py         stratified train/val/test split
src/train.py               fine-tune one model, write metrics + model card
src/run_all.py             train all three models and compare
src/evaluate_external.py   evaluate on the Taglish election test set
src/compare.py             metrics tables + McNemar tests
src/upload_cards.py        upload model_cards/*.md to Hugging Face
model_cards/               refined Hugging Face model cards
results/                   metrics, comparison, cleaning report
notebooks/colab_train.ipynb
```

## Status

- [x] Cleaning, training and internal evaluation of all three models
- [ ] Taglish election test set: collection, annotation (Cohen's kappa >= 0.61), evaluation
- [ ] Integrated Gradients and SHAP top-predictor analysis
- [ ] Simulation demo

## Limitations

Labels reflect source credibility rather than verified claims; training text is news rather than social media; single seed;
128-token truncation. Model outputs are not fact-checks and should not be used for automated moderation or judgements about
individuals or outlets.
