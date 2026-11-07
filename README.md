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

(NC = not credible, the positive class. Bold marks the highest value in a column and is not a ranking: a model counts as better only if its score is higher AND the difference is significant after Holm correction; see McNemar below.)

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

## Diagnostics: short text, links and confidence (internal test set, inference only)

**Accuracy when articles are cut short** (`results/domain_shift.json`, `src/domain_shift_checks.py`; figure `results/figures/fig_e_accuracy_vs_words_kept.png`).
Each cell: accuracy (share of credible articles wrongly called not credible).

| Model | First 15 words | First 30 words | First 60 words | Full text |
|---|---|---|---|---|
| RoBERTa-Tagalog | 0.688 (45.4%) | 0.833 (22.8%) | 0.924 (8.3%) | 0.945 (5.2%) |
| mBERT | 0.713 (40.9%) | 0.826 (23.1%) | 0.915 (9.4%) | 0.930 (7.0%) |
| XLM-RoBERTa | 0.698 (43.9%) | 0.860 (17.6%) | 0.928 (6.9%) | 0.943 (4.8%) |

Not-credible recall stays about the same at every length (0.92-0.94), so the whole drop comes from credible text being called
not credible. The collected 2022 election posts are short (median 34 words after screening, vs 335 for credible and 179 for
not-credible training articles; `results/length_distribution.json`, figure `fig_f_length_distribution`), so they fall in the range
where the models are weakest. Adding a link to every test article flipped very few predictions (RoBERTa-Tagalog 4, mBERT 32, XLM-RoBERTa 12 of 3,803), and on 12 collected
posts that contain a URL, removing it flipped none: links are not a meaningful cue.

**Confidence when wrong** (`results/confidence_check.json`; figure `fig_d_confidence_internal`). Raw softmax output is not a calibrated probability; the table below is the raw finding.

| Model | Errors | Median confidence when wrong | Errors at >= 0.90 | Errors at >= 0.99 |
|---|---|---|---|---|
| RoBERTa-Tagalog | 209 | 0.973 | 135 | 80 |
| mBERT | 266 | 0.882 | 125 | 28 |
| XLM-RoBERTa | 218 | 0.984 | 161 | 86 |

**Calibration.** `src/calibrate.py` fits one temperature T per model on the internal validation split and reports, on the internal test split,
the expected calibration error (15 bins), NLL, Brier score and median confidence before and after (`results/calibration.json`,
`results/calibration_report.json`, figure `fig_k_reliability`; run in `notebooks/colab_extra.ipynb`). Predictions never change. `predict.py`,
`evaluate_external.py` (`prob_credible_calibrated`), `confidence_check.py` (a `calibrated` block) and the demo show calibrated values next to
the raw ones. It is fitted on news articles, so short-post percentages can still be too high. Never fitted on the Taglish set.

These are diagnostics on the internal set only. Accuracy on the Taglish election posts is pending annotation; it will be reported by
fixed length bins (under 20 / 20-39 / 40-99 / 100+ words, `src/length_bins.py`).

## Extra results (internal set)

From `notebooks/colab_extra.ipynb` (Colab T4, 2026-11-07). All on the internal formal-news data, single seed, descriptive only; none of it uses the Taglish set.

**Calibration** (`results/calibration_report.json`; T fitted on the validation split, scores on the test split; predictions unchanged):

| Model | T | ECE before -> after | NLL before -> after | Median confidence when wrong before -> after |
|---|---|---|---|---|
| RoBERTa-Tagalog | 1.9681 | 0.0398 -> 0.0200 | 0.2153 -> 0.1618 | 0.973 -> 0.861 |
| mBERT | 1.361 | 0.0265 -> 0.0160 | 0.2117 -> 0.1998 | 0.881 -> 0.814 |
| XLM-RoBERTa | 1.8441 | 0.0444 -> 0.0165 | 0.2309 -> 0.1711 | 0.984 -> 0.904 |

**Integrated Gradients on a 300-article internal sample** (`results/ig_internal/ig_top_tokens.json`, plus `ig_attributions_<model>.json`, `ig_overlap.json`; 100 steps, tolerance 5%). Completeness check: RoBERTa-Tagalog and mBERT are 100% within the 5% tolerance. XLM-RoBERTa is 97.7% within tolerance (max relative error 5.39, so a few articles fail the check and their attributions are less reliable). Predicted not credible: 109 / 113 / 106 of 300 (RoBERTa-Tagalog / mBERT / XLM-RoBERTa). Top words are descriptive, not causal. Figure: `fig_i_ig_top_words`.

**IG temperature robustness** (`results/ig_temperature_check.json`; 20 test articles per model): IG on logits/T keeps the same top-10 word order and the same prediction for 20 of 20 articles in all three models; the largest relative deviation of attribution x T is 0.0212 (RoBERTa-Tagalog), 0.0056 (mBERT), 0.0005 (XLM-RoBERTa).

**Error analysis** (`results/error_analysis/`): per model, `<model>_examples.csv` holds 24 test-set errors (12 credible called not credible, 12 not credible called credible) with probability, excerpt and IG top words; `<model>/` holds the IG files for those 24 errors.

**Inference time** (`results/inference_time.json`; 100 posts, batch size 1, max length 128, 1 CPU thread, Tesla T4). Median ms (p95), CPU / T4: RoBERTa-Tagalog 313.6 (430.1) / 15.6 (24.3); mBERT 329.1 (454.2) / 12.7 (18.1); XLM-RoBERTa 329.9 (447.7) / 13.4 (19.2). Prediction only: Integrated Gradients time is not included.

**Environment** (`results/environment.json`): Python 3.13.16, torch 2.14.1+cu130, transformers 4.57.6, datasets 4.8.5, scikit-learn 1.6.1, captum 0.9.0, numpy 2.4.6, pandas 2.2.3, scipy 1.16.3, accelerate 1.15.0, tokenizers 0.22.2 (Colab extra-evaluation session).

**Language mix of the 25,352 training articles** (`results/language_id_train.json`; lingua, English vs Tagalog, 30-word chunks, 80% rule): 21,295 English, 2,622 Tagalog, 1,435 mixed. Figure: `fig_l_language_mix`.

Figures are in `results/figures/` (`fig_k_reliability`, `fig_i_ig_top_words`, `fig_l_language_mix`), with the demo screenshots `demo_long_post.jpg` and `demo_short_post.jpg`.

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
python src/calibrate.py                                                # temperature scaling on the validation split -> results/calibration*.json
python src/plot_training.py                                            # loss curves -> results/figures/
python src/bootstrap_ci.py                                             # confidence intervals -> results/bootstrap_ci.json
```
**Which Colab notebook?**
- `notebooks/colab_eval.ipynb`: **use this one.** It tests the models, scores the Taglish set (metrics, bootstrap CIs, McNemar + Holm) and runs Integrated Gradients. It never retrains anything. [Open in Colab](https://colab.research.google.com/github/hansjio/tsektxt-model-training/blob/master/notebooks/colab_eval.ipynb)
- `notebooks/colab_present.ipynb`: **for the demo and the defense.** Read-only: shows the results and figures from `results/*.json`, scores any post live with all three models, and runs Integrated Gradients for one post. Never retrains, never uploads. [Open in Colab](https://colab.research.google.com/github/hansjio/tsektxt-model-training/blob/master/notebooks/colab_present.ipynb)
- `notebooks/colab_extra.ipynb`: the extra paper results (calibration, IG on an internal sample, error analysis, timing, language ID). Already run; only re-run if the numbers must be regenerated.
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

## Demo in Colab (no install; backup for the live app)

Use this when the laptop app will not start, or for a quick live demo from any browser. The local app (`tsektxt-app`, see its `DEMO_DAY.md`) is the main demo; this is the fallback.

1. Open the notebook: [colab_present.ipynb](https://colab.research.google.com/github/hansjio/tsektxt-model-training/blob/master/notebooks/colab_present.ipynb). Close old Colab tabs first. Never "Save a copy in GitHub".
2. **Runtime > Change runtime type > T4 GPU** (a GPU makes a check take about a second; CPU also works but is slow).
3. **Runtime > Run all** (about 5 to 10 minutes the first time; the three models are downloaded from Hugging Face, 1 to 2 GB). "Could not find cuda drivers" is TensorFlow noise: ignore it. pip may print dependency warnings: ignore them.
4. What each part shows: sections 1-6 the method and internal results from `results/*.json`; section 7 the Taglish results (shows "pending" until the labelled set exists); **section 8 live prediction** (five real public news-outlet and COMELEC example posts plus a text box `custom_text` for a post a panelist suggests: paste only real public text); **section 9 Integrated Gradients** for one post; section 10 a summary with the file each number comes from.
5. Do the run the night before and leave the tab open. If the runtime disconnects, **Runtime > Run all** again.

What to say: this is a research demo, not a fact-checker. Labels follow the source, the three models are single-seed and not ranked (RoBERTa-Tagalog vs XLM-R, McNemar p = 0.55), short posts lean "not credible" because the models were trained on long news articles, and confidence is calibrated on news articles. Screenshots and a recording of the local app are in `tsektxt-app/demo-assets/`.

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
src/confidence_check.py    how confident each model is when it is wrong (from saved predictions; raw and calibrated)
src/calibrate.py           temperature scaling fitted on the validation split -> results/calibration*.json (predictions unchanged)
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
notebooks/colab_present.ipynb demo/defense notebook (read-only, live prediction + IG)
notebooks/colab_extra.ipynb extra paper results: calibration, IG on an internal sample, error analysis, timing, language ID
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
- [x] Diagnostics: length falloff, link check, confidence when wrong (internal set)
- [x] Colab extra run: calibration, Integrated Gradients on a 300-article internal sample, error analysis, inference time, environment, language ID (`notebooks/colab_extra.ipynb`)
- [x] Local three-model demo app (separate repo)

## Limitations

Labels reflect source credibility rather than verified claims; training text is news rather than social media; single seed;
128-token truncation. The models were trained on long articles and lose accuracy on short text (see Diagnostics above), so short
social posts are out of domain and tend to be called not credible. Model outputs are not fact-checks and should not be used for automated moderation or judgements about
individuals or outlets.
