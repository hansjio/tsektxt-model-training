# Next steps (written 2026-11-06)

## New scripts (all in src/)
| Script | Run where | What it does |
|---|---|---|
| `diagnose_fit.py` | anywhere (stdlib) | Over/underfitting from training logs -> `results/fit_diagnosis.md`. Already run. |
| `predict.py` | your PC/Colab (needs torch, transformers) | Test all 3 models with no frontend: `--text`, `--file`, `--csv`, `--interactive`, `--split-gap` (train/val/test accuracy). Loads `models/<name>` or falls back to Hugging Face `hansjio/tsektxt-*`. |
| `harvest_factchecks.py` | anywhere (stdlib) | Lists Vera Files / Tsek.ph fact-checks from Feb 8 - May 9 2022 and the Facebook/X links they cite -> `data/external/candidates.csv` (worklist for MANUAL collection). Tested: 270 election articles from the first 150 per site, 158 with a FB/X link. |

## Fit findings (from logs)
- All three models: train loss keeps falling, val loss bottoms at epoch 1-2 -> mild overfitting after the best epoch. Early stopping saved the best checkpoint, so the reported test scores are not from overfit weights.
- Not underfitting (train loss ~0.03-0.09 by the end, test acc 93-94.5%).
- Caveat: val and test share the same sources, so this cannot show source-style overfitting. Only the Taglish set can.
- Still to do on a machine with the checkpoints: `python src/predict.py --split-gap` (needs `data/processed/train|val|test.csv`; run `clean_data.py` + `make_splits.py` first, they are not in this folder).

## Decisions needed from your group
1. **Scraping vs thesis.** Chapter III says manual collection, no scraping tools. The harvester only reads public fact-check sites to find candidates; you still copy posts by hand. Facebook/X scraping would contradict the thesis and platform terms; I did not build it. Does the panel accept this candidate-finding step? Mention it in Chapter III.
2. **Supplementary sheet** is an empty template (no post text, no label column). Add: `label`, `annotator_1`, `annotator_2`, `tiebreak`, `fact_check_url`, `issue_category`, `candidate_mentioned`. Targets: 120 posts is below the 300-500 in the thesis; decide.
3. **SHAP removal.** Removed from the .md files. Thesis .docx checked: no SHAP mentions.

## Upload to new sessions
Thesis docx (current), PROJECT_CONTEXT.md, THESIS_INPUTS.md, and this repo folder (without `models/` and `data/processed/`).

## Working method
One task per session (data collection, annotation toolkit, IG script, demo, Chapter IV). Keep PROJECT_CONTEXT.md updated at the end of each. Use Claude Code for code and files in E:\thesis; use chat for writing review.
