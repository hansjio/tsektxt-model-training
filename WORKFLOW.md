# Audit, push checklist and workflow (2026-11-06)

## A. Audit result: what is wrong right now
| # | Finding | Fix |
|---|---|---|
| 1 | Hugging Face READMEs (all 3) are the OLDER cards: still say "sign-off lines", no baseline paragraph. Weights are correct (retrained 2026-11-01). | Run `python src/upload_cards.py` |
| 2 | GitHub master (last commit 2026-11-01 18:27) is missing: baseline.py, bootstrap_ci.py, leakage_probe.py, plot_training.py, predict.py, diagnose_fit.py, harvest_factchecks.py, THESIS_INPUTS.md, results/baseline-tfidf, bootstrap_ci.json, leakage_probe.json, fit_diagnosis.*. It also holds stale `results/*/card_*.md`. | Push (Section B) |
| 3 | **License conflict:** `jcblaise/roberta-tagalog-base` is CC BY-SA 4.0, so the RoBERTa-Tagalog card cannot say CC BY-NC 4.0. | Set that card to `cc-by-sa-4.0` (I will not change it until you confirm) |
| 4 | Card license for the 2 datasets is unverified (card itself says so). | Check each dataset's license; if either forbids redistribution of derivatives, say so |
| 5 | Root copies `card_*.md`, `THESIS_INPUTS.md`, `PROJECT_CONTEXT.md` duplicate the repo. | Keep `model_cards/` as the only source; delete root `card_*.md` |
| 6 | Cards say "BS Computer Science, May 2026" and cite "Fernandez, 2019": matches thesis ref [7] and cover page. OK. | none |

## B. Push to GitHub (correction)
`E:epos	sektxt-model-training` is already your git clone (remote hansjio/tsektxt-model-training, clean, at origin/master). Do NOT `git init` in E:	hesis.
Make that clone the one working copy:
```
robocopy E:	hesis	sektxt-model-training E:epos	sektxt-model-training /E /XD .git __pycache__ /XF candidates.csv
cd E:epos	sektxt-model-training
git status
```
Check `models/`, `data/processed/`, `data/external/` are not listed, then `git add -A`, `git commit -m "..."`, `git push origin master`. Afterwards delete E:	hesis	sektxt-model-training and open E:epos	sektxt-model-training in Claude Code.

## C. Push to Hugging Face (cards only; weights unchanged)
1. Set `license: cc-by-sa-4.0` in `model_cards/roberta-tagalog.md` (after you confirm).
2. In PowerShell: `$env:HF_TOKEN="<write token>"` then `python src/upload_cards.py`.
3. Open the 3 model pages and confirm the baseline paragraph shows.
Nothing to rename on Hugging Face (`hansjio/tsektxt-xlmr`, `-roberta-tagalog`, `-mbert`).

## D. Thesis (docx) vs implementation: edits Chapter III needs
- Corpus sizes: line says ~22,462 for the Philippine Fake News Corpus in III but ~21,779 in the lit review; cleaned total is 25,352 (not "over 25,000" estimate).
- Loss: thesis says plain cross-entropy; used class-weighted cross-entropy.
- Head: thesis says dropout+linear; RoBERTa-family heads are dense+tanh+dropout+linear.
- Batch size: thesis "16 or 32"; used 16. Early stopping patience 2; max 5 epochs.
- Preprocessing: encoding repair, URL/HTML removal, dedup, min 10 words (thesis says min tokens).
- E.4: "validation accuracy recorded each epoch": only loss was logged. Edit the sentence or rerun with accuracy logging.
- Hardware: "Colab Pro": state actual (T4).
- Add: baseline, bootstrap CIs, Holm correction, leakage probe, single seed.
- Collection (Section "no automated scraping tools"): add one sentence that fact-check articles were listed with a script as a worklist, and posts were still reviewed and entered manually.
- Add the test-only role of the supplementary set (already consistent).

## E. Testing models and the supplementary set (test only)
1. Quick check, no frontend: `python src/predict.py --interactive` (loads from Hugging Face if `models/` missing).
2. Worklist: `python src/harvest_factchecks.py --only-with-links` -> `data/external/candidates.csv` (gitignored).
3. Manually open each link, copy the public post text into the Google Sheet. Add columns: label, annotator_1, annotator_2, tiebreak, fact_check_url, issue_category, candidate_mentioned. Target 300-500 (sheet says 120: decide).
4. Export to `data/external/taglish_test.csv` with columns `id,text,label`.
5. `python src/evaluate_external.py --model <m> --csv data/external/taglish_test.csv` for each model, then `python src/compare.py`.
Caveat: labels come from fact-checks of claims, while training labels come from site reputation; discuss in Chapter IV.

## F. Workflow for Claude Code / Antigravity
- Open `E:\thesis` in Claude Code (desktop Code tab is fine). `CLAUDE.md` there carries the rules.
- Permission mode: "Ask / accept edits"; do not use bypass. Pushes and uploads always need your yes.
- Models: plan with Opus (plan mode) for design tasks like the IG script and demo; execute with Sonnet; Haiku or Antigravity for repetitive work (reformatting references, annotation-sheet cleanup, bulk text edits). Give Antigravity only files without tokens.
- One chat per task; update PROJECT_CONTEXT.md at the end of each. New conversation needed for each of: IG script, demo UI, Chapter IV.
- Useful connectors: Google Drive/Sheets (read the annotation sheet directly; the sheet is private now so it needs the connector or "anyone with link" access).

## G. Questions for you
1. Is the license change on the RoBERTa-Tagalog card OK, and what are the dataset licenses?
2. Test-set size target: 120 or 300-500?
3. Send the old frontend repo link when ready; I will check if it can host the three-model demo.
