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
| TF-IDF + LogReg baseline (first 128 words) | 0.9258 | 0.8739 | 0.9265 | 0.8994 | 0.9204 | 0.9263 |

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

**Baseline.** A non-neural baseline (TF-IDF word 1-2 grams + class-balanced logistic regression, first 128 words, same split) is significantly worse than RoBERTa-Tagalog (p = 0.00002) and XLM-RoBERTa (p = 0.0002) but not significantly different from mBERT (p = 0.39). Given the full article text it reaches 0.9508 accuracy, above all three transformers (not significance-tested), which shows how much of the internal test set can be solved from surface and source-style cues. Loss curves: `results/figures/loss_curves.png`.

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
python src/baseline.py                                                 # TF-IDF + LogReg baseline (run before compare.py)
python src/leakage_probe.py                                            # source-style diagnostic
python src/domain_shift_checks.py                                      # length cuts + link check (inference only) -> results/domain_shift.*
python src/plot_training.py                                            # loss curves -> results/figures/
python src/bootstrap_ci.py                                             # confidence intervals -> results/bootstrap_ci.json
```
**Which Colab notebook?**
- `notebooks/colab_eval.ipynb`: **use this one.** It tests the models, scores the Taglish set (metrics, bootstrap CIs, McNemar + Holm) and runs Integrated Gradients. It never retrains anything. [Open in Colab](https://colab.research.google.com/github/hansjio/tsektxt-model-training/blob/master/notebooks/colab_eval.ipynb)
- `notebooks/colab_train.ipynb`: **only if the group decides to retrain.** It overwrites the models on Hugging Face (T4 GPU; needs a Hugging Face write token as the Colab secret `HF_TOKEN`).

Taglish test set, after annotation (see `notebooks/colab_eval.ipynb` sections 3-4):
```bash
python src/merge_candidates.py                                         # all candidate posts -> one sheet-ready file (local only)
python src/annotation_kappa.py --csv data/external/annotated.csv       # agreement of the 2 annotators
python src/clean_annotations.py --csv data/external/annotated.csv --out data/external/taglish_test.csv
python src/evaluate_external.py --model xlmr --csv data/external/taglish_test.csv   # x3 models
python src/baseline.py --external data/external/taglish_test.csv
python src/external_stats.py                                           # bootstrap CIs + McNemar with Holm
python src/compare.py
python src/interpret.py corpus --csv data/external/taglish_test.csv    # Integrated Gradients -> results/ig_*.json
python src/profile_corpus.py --csv data/external/taglish_test.csv --name taglish
```

**Re-run after new data** (e.g. `data/external/supplementary_collected_task05.csv` arrives): `python src/merge_candidates.py && python src/build_annotation_sheet.py --only-new SHEET_EXPORT.csv` (SHEET_EXPORT.csv = download of the Sheet tab as CSV) writes `data/external/annotation_tab_import_new.csv` with only posts not yet in the Sheet; append those rows below the labelled ones. Without `--only-new` it rebuilds the full import file. Screening hints are included; for new posts with `language_hint` = english, add a hint to `data/external/language_check.csv` (see `english_hint_not_yet_checked` in the .report.json). AI output is hints only, never labels.

## Repository layout

```
configs/config.yaml        all hyperparameters and paths
src/clean_data.py          encoding repair, URL/HTML removal, deduplication
src/make_splits.py         stratified train/val/test split
src/train.py               fine-tune one model, write metrics + model card
src/run_all.py             train all three models and compare
src/evaluate_external.py   evaluate on the Taglish election test set
src/compare.py             metrics tables + McNemar tests
src/baseline.py            non-neural TF-IDF + LogReg baseline
src/leakage_probe.py       first/last-12-words source-style diagnostic
src/domain_shift_checks.py accuracy when articles are cut short; effect of links (inference only)
src/screen_candidates.py   pre-annotation screening of candidate posts: usable / check / drop + hints (never labels)
src/confidence_check.py    how confident each model is when it is wrong (from saved predictions)
src/length_bins.py         Taglish results by fixed length bins (under 20 / 20-39 / 40-99 / 100+ words); primary result stays the full set
src/plot_training.py       train/validation loss curves
src/bootstrap_ci.py        95% bootstrap confidence intervals
src/external_stats.py      Taglish set: paired bootstrap CIs + McNemar with Holm correction
src/interpret.py           Integrated Gradients (Captum): per-post words, completeness check, top predictors, cross-model overlap
src/predict.py             try the 3 models on any text, no frontend
src/merge_candidates.py    merge all Taglish candidate sources into one file for annotation
src/annotation_kappa.py    Cohen's kappa between the 2 annotators
src/clean_annotations.py   labelled sheet export -> taglish_test.csv
src/profile_corpus.py      word frequencies per label (corpus analysis)
src/harvest_factchecks.py, src/build_draft_testset.py   fact-check worklists (already run)
src/upload_cards.py        upload model_cards/*.md to Hugging Face
model_cards/               refined Hugging Face model cards
results/                   metrics, comparison, cleaning report
tests/                     smoke tests on dummy data only
notebooks/colab_eval.ipynb  test + Taglish scoring + IG (use this)
notebooks/colab_train.ipynb retraining only (do not run unless the group decides to retrain)
```
The demo app is a separate repo: https://github.com/hansjio/tsektxt-app (runs the 3 models + IG locally).

## Status

- [x] Cleaning, training and internal evaluation of all three models
- [x] Baseline (TF-IDF + LogReg), loss curves, leakage diagnostic
- [x] Taglish election test set: collection (811 candidate posts merged)
- [ ] Taglish election test set: annotation (Cohen's kappa >= 0.61), evaluation
- [x] Integrated Gradients and Taglish scoring code (tested on dummy data)
- [ ] Integrated Gradients top-predictor analysis on the labelled Taglish set
- [x] Local three-model demo app (separate repo)

## Limitations

Labels reflect source credibility rather than verified claims; training text is news rather than social media; single seed;
128-token truncation. The models were trained on long articles, so short social posts are out of domain: in informal
spot-checks they called most short posts "not credible" with high confidence (see the thesis hub note docs/FINDINGS_DOMAIN_SHIFT.md;
measured by `src/domain_shift_checks.py`). Model outputs are not fact-checks and should not be used for automated moderation or judgements about
individuals or outlets.
