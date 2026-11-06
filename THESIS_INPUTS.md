# TsekTxt – Thesis inputs (facts, numbers and open decisions; not thesis prose)

Purpose: everything we have produced that the thesis needs, organised by where it would go. Nothing here is written into the paper yet.
Source of truth for code/results: https://github.com/hansjio/tsektxt-model-training (files named in each section). Generated 2026-11-06.
Label convention throughout: **1 = credible, 0 = not credible; not credible (NC) is the positive class.**

---
## 1. Four items you asked about: where we stand

| Item | Status | Where it fits | What is needed |
|---|---|---|---|
| Baseline model (designated and documented) | **Done as of this file** (TF-IDF + logistic regression). Chapter III currently calls the three transformers "baselines" in E.2 but names no non-neural baseline | Ch. III new subsection "Baseline model"; Ch. IV tables | Decide to adopt it (Open decision 1); run it on the Taglish set later with `--external` |
| Model tuning | **No hyperparameter tuning was done.** All three models use one fixed configuration, identical by design (controlled comparison). Fine-tuning itself is done | Ch. III E.3 (state it as a deliberate design choice) and Limitations | Optional validation-set sweep if the panel asks (Section 4) |
| Training results (accuracy/loss graphs) | **Loss curves done** (`results/figures/loss_curves.png`). **Per-epoch accuracy/F1 do not exist** because only loss was logged | Ch. IV | Retrain with per-epoch metrics only if required (Section 5) |
| System | **Not done to thesis spec.** An older app exists (`tsektxt-app`), but Section H requires a local three-model, side-by-side demo with Integrated Gradients | Ch. III H, Ch. IV demonstration | Build/adapt (Section 8) |

---
## 2. Primary corpus (Ch. III B, C.1, Ch. IV data description)

- Sources: Fake News Filipino Dataset (Cruz, Tan & Cheng, 2020) + Philippine Fake News Corpus (Fernandez, 2019), combined into `combined_dataset_clean.csv` (25,398 rows in).
- Cleaning (`src/clean_data.py`, `results/cleaning_report.json`): text repair with ftfy modified **21,088** rows; URL/HTML/zero-width removal; **2** rows under 10 words dropped; **6** rows removed as duplicates with conflicting labels; **38** exact/near duplicates removed (same first 200 normalised characters). **25,352** rows remain; 2 rows still contain encoding artefacts.
- Class balance: 16,283 credible (64.2%), 9,069 not credible (35.8%).
- Split (stratified, seed 42): train 17,746 (11,398 / 6,348); validation 3,803 (2,443 / 1,360); test 3,803 (2,442 / 1,361).
- Length in whitespace words: median 335 (credible) vs 179 (NC); mean 408 vs 289; 95th percentile about 935 vs 969. About 90% of articles have at least 100 words (96% credible, 79% NC), so most are truncated at the 128-token limit (word count is a rough proxy for tokens).
- Not in the combined CSV: a source-dataset column (promised in Ch. III B.1). Roughly 40% of rows look mostly English by a crude stopword check (not verified with a language-ID tool).

---
## 3. Baseline model (proposed designation)

- **Model:** TF-IDF (word 1–2-grams, min document frequency 2, up to 200,000 features, sublinear term frequency) + logistic regression with class-balanced weights, seed 42 (`src/baseline.py`).
- **Input:** first 128 words only, to roughly match the transformers' 128-token budget. A full-text variant is reported for reference.
- **Why a baseline:** shows how much of the task is solvable without pretrained transformers; gives McNemar comparisons a reference point; helps interpret the source-style finding (Section 7).
- **Results:** first-128-words accuracy 0.9258, weighted F1 0.9263. Full text: accuracy 0.9508, weighted F1 0.9511, higher than all three transformers (not significance-tested).
- **Reading:** the transformers' internal advantage over a bag-of-words baseline is small (about 1–2 accuracy points; mBERT is not significantly different from the baseline). This does not weaken the study's design, because the thesis's main question is generalisation to Taglish, but it must be reported honestly.

---
## 4. Model tuning and fine-tuning setup (Ch. III E.3)

| Setting | Value used |
|---|---|
| Optimizer / schedule | AdamW, linear schedule with 10% warmup |
| Learning rate | 2e-5 |
| Batch size | 16 train, 32 eval |
| Max epochs / stopping | 5; early stopping patience 2 on validation loss; checkpoint with lowest validation loss kept |
| Weight decay | 0.01 |
| Max sequence length | 128 tokens (dynamic padding) |
| Loss | Cross-entropy with class weights inversely proportional to class frequency |
| Precision / hardware | fp16, single Tesla T4 (Google Colab) |
| Seed | 42, one run per model |
| Classification head | Default Hugging Face sequence-classification head of each model (BERT: dropout + linear on pooled output; RoBERTa/XLM-R: dropout, dense, tanh, dropout, linear) |

- **Tuning status:** none. Justification available: identical settings isolate the effect of pre-training; limitation: results might shift under per-model tuning.
- **If a sweep is wanted:** choose on the **validation** set only, never the test set. Example: learning rate in {2e-5, 3e-5, 5e-5} for each model. From observed run times (9–15 minutes per run) that is roughly 2 hours of Colab time in total (estimate). Seeds: three seeds per model would let you report mean and standard deviation (another ~35 minutes per seed for all three models).
- **Still needed for the record:** library versions (run `!pip list | grep -E "torch|transformers|datasets|scikit-learn|accelerate"` in Colab and keep the output).

---
## 5. Training results (Ch. IV)

| Model | Epochs run | Best epoch (lowest val. loss) | Best val. loss | Final train loss | Final val. loss | Training time (min, T4) |
|---|---|---|---|---|---|---|
| RoBERTa-Tagalog | 4 | 2 | 0.2480 | 0.0354 | 0.3241 | 11 |
| mBERT | 3 | 1 | 0.2216 | 0.1015 | 0.2243 | 9 |
| XLM-RoBERTa | 4 | 2 | 0.2420 | 0.0888 | 0.2442 | 15 |

- Figure: `results/figures/loss_curves.png` (train vs validation loss per epoch; best epoch circled).
- Observations from the logs: training loss falls steadily in every run (final values above) while validation loss stops improving after epoch 1 (mBERT) or epoch 2 (RoBERTa-Tagalog, XLM-R) and then rises or stays flat, which is why early stopping ended runs at 3–4 epochs. mBERT's validation loss was almost flat across epochs.
- Lowest validation loss does not rank the models: mBERT has the lowest (0.2216) but the weakest test scores.
- **Not available:** per-epoch accuracy/F1. Only loss was recorded. To get these curves, add a metrics function to `train.py` and retrain (about 35 minutes for all three). Final numbers could shift slightly because GPU training is not perfectly repeatable, so all tables would have to be replaced together. Recommended only if the panel requires it.

---
## 6. Internal test results (n = 3,803; Ch. IV)

| Model | Accuracy | Precision (NC) | Recall (NC) | F1 (NC) | Macro F1 | Weighted F1 |
|---|---|---|---|---|---|---|
| RoBERTa-Tagalog | 0.9450 | 0.9103 | 0.9390 | 0.9244 | 0.9406 | 0.9452 |
| mBERT | 0.9301 | 0.8810 | 0.9302 | 0.9049 | 0.9248 | 0.9305 |
| XLM-RoBERTa | 0.9427 | 0.9150 | 0.9258 | 0.9204 | 0.9378 | 0.9428 |
| TF-IDF + LogReg baseline (first 128 words) | 0.9258 | 0.8739 | 0.9265 | 0.8994 | 0.9204 | 0.9263 |

Confusion counts and credible-class metrics:

| Model | TN (credible→credible) | FP (credible→NC) | FN (NC→credible) | TP (NC→NC) | Errors | Precision (credible) | Recall (credible) |
|---|---|---|---|---|---|---|---|
| RoBERTa-Tagalog | 2316 | 126 | 83 | 1278 | 209 (5.5%) | 0.9654 | 0.9484 |
| mBERT | 2271 | 171 | 95 | 1266 | 266 (7.0%) | 0.9598 | 0.9300 |
| XLM-RoBERTa | 2325 | 117 | 101 | 1260 | 218 (5.7%) | 0.9584 | 0.9521 |
| TF-IDF + LogReg baseline (first 128 words) | 2260 | 182 | 100 | 1261 | 282 (7.4%) | 0.9576 | 0.9255 |

McNemar's exact/continuity-corrected test, alpha 0.05 (b = row model right and column model wrong; c = reverse):

| Pair | b | c | p-value | Significant (p<0.05) |
|---|---|---|---|---|
| roberta-tagalog vs mbert | 151 | 94 | 0.0003466 | yes |
| roberta-tagalog vs xlmr | 94 | 85 | 0.5499 | no |
| roberta-tagalog vs baseline-tfidf | 179 | 106 | 2e-05 | yes |
| mbert vs xlmr | 94 | 142 | 0.002218 | yes |
| mbert vs baseline-tfidf | 158 | 142 | 0.3865 | no |
| xlmr vs baseline-tfidf | 178 | 114 | 0.0002271 | yes |

95% bootstrap confidence intervals (2,000 paired resamples of the test set, seed 42; `src/bootstrap_ci.py`):

| Model | Accuracy [95% CI] | Weighted F1 [95% CI] |
|---|---|---|
| RoBERTa-Tagalog | 0.9450 [0.9377, 0.9519] | 0.9452 [0.9379, 0.9521] |
| mBERT | 0.9301 [0.9219, 0.9379] | 0.9305 [0.9224, 0.9383] |
| XLM-RoBERTa | 0.9427 [0.9350, 0.9500] | 0.9428 [0.9350, 0.9501] |
| TF-IDF + LogReg baseline (first 128 words) | 0.9258 [0.9172, 0.9345] | 0.9263 [0.9177, 0.9348] |

| Pair (accuracy difference) | Mean | 95% CI (points) | CI excludes 0 |
|---|---|---|---|
| roberta-tagalog minus mbert | +1.50 pts | [+0.71, +2.29] | yes |
| roberta-tagalog minus xlmr | +0.24 pts | [-0.45, +0.92] | no |
| roberta-tagalog minus baseline-tfidf | +1.92 pts | [+1.05, +2.79] | yes |
| mbert minus xlmr | -1.26 pts | [-2.05, -0.47] | yes |
| mbert minus baseline-tfidf | +0.42 pts | [-0.50, +1.34] | no |
| xlmr minus baseline-tfidf | +1.68 pts | [+0.79, +2.60] | yes |

How to read: RoBERTa-Tagalog and XLM-RoBERTa are statistically tied; both beat mBERT and the first-128-words baseline; mBERT and the baseline are tied. Single run per model.

---
## 7. Diagnostic: source style is learnable (Ch. IV discussion, Limitations)

Labels come from the publishing site's credibility, not from checked claims. `src/leakage_probe.py` (TF-IDF + logistic regression on the official split):

| Input to the classifier | Accuracy | Macro F1 |
|---|---|---|
| Majority class (always "credible") | 0.6421 | n/a |
| First 12 words only | 0.8188 | 0.7940 |
| Last 12 words only | 0.8307 | 0.8048 |
| Full text | 0.9479 | 0.9428 |

Twelve words from either end lift accuracy about 18 points above the majority rate, so internal scores partly reflect source style. The transformers see only the first 128 tokens, so the beginning of each article is what matters for them. Rationale for the Taglish test set: it removes the source cue.

---
## 8. System / simulation demo (Ch. III H)

- **Required by the thesis:** a local interface that loads the three checkpoints, accepts one Taglish post, and shows label, confidence and top Integrated-Gradients tokens for all three models side by side; not deployed or public.
- **What exists:** the older `tsektxt-app` (from the previous model card: FastAPI backend, one model, text/screenshot input, a Groq API branch). It does not match Section H.
- **Mismatch to resolve:** the thesis says the demo is not publicly deployed; the old model card called it a "live application". Also update model IDs (`chimsio/` → `hansjio/`), max length (128) and read label names from `model.config.id2label`.
- **Needed:** decide whether to adapt the existing app or build a small local demo (for example Gradio). Upload the app repo zip so it can be reviewed (GitHub subfolders are not readable by automated tools).

---
## 9. Still pending (cannot be filled yet)

1. Taglish election test set: collection (300–500 posts, Feb 8 – May 9, 2022), cleaning, two annotators plus tiebreaker, Cohen's kappa at least 0.61, entity/issue tagging, profiling (Ch. III C, D).
2. External evaluation of the three models and the baseline (`evaluate_external.py`, `baseline.py --external`, `compare.py`).
3. Integrated Gradients top predictors, cross-model comparison, validation against corpus profiling (Ch. III G). Script not yet written.
4. Chapters IV (external part) and V.

---
## 10. Corrections the existing thesis text needs (Ch. I–III)

- B.1 / C.1: "no modification to article content or labels" is no longer true (encoding repair, URL/HTML removal, deduplication, conflicting-label removal). Reword and list the steps.
- C.1: minimum length is implemented as 10 whitespace words, not "tokens after tokenization".
- Corpus size: "over 24,000" becomes 25,352; class counts and splits as in Section 2.
- B.1: the promised source-dataset identifier does not exist in the CSV (re-merge the two originals or drop the claim).
- E.1: head description (Section 4). E.3: class-weighted loss, early stopping patience, fp16, dynamic padding, no tuning.
- E.2: either keep "baseline" for the transformers or rename and add the non-neural baseline (Section 3).
- Add to Limitations: source-credibility labels (Section 7), single run, no tuning, 128-token truncation, English content in the corpus.
- Cross-reference slips: G.3 cites "D.3" and G.4 cites "C" (should be C.3 and D); "five-objective framework" lists four; "four outputs" lists three plus the demo; G.1 uses a 5-post threshold while G.2 uses 5% of posts; "health misinformation" is listed as an issue category in an election-only study; a "minimum candidate-level sample size" is said to be pre-registered but is never stated.
- Cover date reads May 2026; update if the resubmission date changed.

---
## 11. Decisions we need from you

1. Adopt the TF-IDF + logistic regression baseline in the thesis? (Recommended.)
2. Hyperparameter sweep and/or multiple seeds, or keep one fixed configuration and state it as a limitation?
3. Retrain for per-epoch accuracy/F1 curves, or report loss curves plus final test metrics only?
4. Demo: adapt `tsektxt-app` or build a new local three-model demo?
5. Minimum number of credible and not-credible posts in the Taglish set (decide before collecting).
6. Which candidate/entity names and issue categories are in the tagging scheme (the current list differs between sections).

---
## 12. Reproducibility record

Repo: github.com/hansjio/tsektxt-model-training (branch master). Models: huggingface.co/hansjio/tsektxt-roberta-tagalog, /tsektxt-mbert, /tsektxt-xlmr.
Commands: `clean_data.py`, `make_splits.py`, `run_all.py --push`, `baseline.py`, `leakage_probe.py`, `plot_training.py`, `bootstrap_ci.py`, `compare.py` (all in `src/`; see README). Seed 42. Splits rebuilt locally were verified identical (same ids and labels) to the Colab test set.
