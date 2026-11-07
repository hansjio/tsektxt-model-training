# Figures for the paper

Made by `python src/make_figures.py` (re-run after new results; figures with missing inputs are skipped). Each figure is saved as PNG (300 dpi) and PDF.
Colours are Okabe-Ito (colour-blind safe); hatching also separates series. All numbers come from `results/`. Single seed, one run each.
Labels: 1 = credible, 0 = not credible (positive class). Do not read any figure as a ranking of RoBERTa-Tagalog over XLM-R (McNemar p = 0.55), or as proof that a model detects misinformation (labels follow the source).

| Figure | Source | Caption (suggested) | Where |
|---|---|---|---|
| `loss_curves` | `<model>/training_log.json` (src/plot_training.py) | Training and validation loss per epoch; early stopping on validation loss. | Ch. III training / Ch. IV fit |
| `fig_a_internal_metrics` | `bootstrap_ci.json` | Accuracy and weighted F1 on the internal test set with 95% bootstrap CIs. RoBERTa-Tagalog and XLM-R intervals overlap; the TF-IDF baseline is close. | Ch. IV, internal results |
| `fig_b_confusion_matrices` | `<model>/test_predictions.csv` | Confusion matrices on the internal test set; cells show counts and the share of each true class. | Ch. IV, error analysis |
| `fig_c_leakage_probe` | `leakage_probe.json` | A TF-IDF classifier on the first or last 12 words alone stays far above the majority-class rate, so short fragments carry source-style cues. | Ch. IV, limitations / leakage |
| `fig_d_confidence_internal` | `confidence_check.json` | Median confidence when wrong vs right, and the share of errors made at 0.90 or 0.99 confidence. Softmax confidence is not a calibrated probability. | Ch. IV, error analysis |
| `fig_e_accuracy_vs_words_kept` | `domain_shift.json` | Accuracy on internal test articles cut to the first 15, 30 or 60 words (inference only). | Ch. IV, domain shift |
| `fig_f_length_distribution` | `length_distribution.json` (word counts only; from `data/raw` or `data/processed/cleaned.csv`, and `merged_for_annotation.csv` + `candidate_screening.csv`) | Share of texts per log-spaced length bin (each group sums to 1): training articles by class vs collected posts. Posts are much shorter. Training counts are the RAW corpus before cleaning (n = 25,398; NC median 180), not the cleaned corpus of the thesis (25,352; NC median 179), until `cleaned.csv` is present. Posts exclude rows `screen_candidates.py` marks drop (n = 661 of 750). | Ch. III data / Ch. IV domain shift |
| `fig_g_taglish_metrics` | `external_bootstrap_ci.json` | Taglish test-set accuracy, F1 (not credible) and weighted F1 with 95% bootstrap CIs. Taglish is test only. | Ch. IV, Taglish results |
| `fig_h_taglish_by_length` | `external_length_bins.json` | Taglish accuracy by post length (fixed bins); bins are small, so differences are noisy. | Ch. IV, Taglish results |
| `fig_i_ig_top_words` | `ig_top_tokens.json` | Integrated Gradients: words with the largest summed attribution toward "not credible", per model. Descriptive, not causal. | Ch. IV, interpretability |
| `fig_j_confidence_taglish` | `external_confidence_check.json` | As fig. d, on the Taglish set. | Ch. IV, Taglish results |

Status when written: a to d, f and `loss_curves` exist. e, g, h, i, j wait for Colab runs (`domain_shift.json`, Taglish results, IG).
