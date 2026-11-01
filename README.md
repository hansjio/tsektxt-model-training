# TsekTxt – Model Training

Fine-tunes **RoBERTa-Tagalog**, **mBERT** and **XLM-RoBERTa** for binary credibility classification
(`1 = credible`, `0 = not credible`; *not credible* is the positive class for precision/recall/F1) and compares
them with McNemar's test. Companion to the thesis *Determining the Top Predictors of Taglish Election
Misinformation*. All settings live in `configs/config.yaml` and follow Chapter III.

## Quick run
```bash
pip install -r requirements.txt
python src/clean_data.py --input data/raw/combined_dataset_clean.csv   # -> data/processed/cleaned.csv
python src/make_splits.py                                              # 70/15/15 stratified, seed 42
python src/run_all.py --push                                           # train all 3, push to Hugging Face, compare
python src/evaluate_external.py --model xlmr --csv data/external/taglish_test.csv   # per model, later
python src/compare.py                                                  # -> results/comparison.md
```
Colab version: `notebooks/colab_train.ipynb`.

## Notes
* Training data are news articles labelled by *source* credibility, not social-media posts.
* Weights are on Hugging Face (`hansjio/tsektxt-*`); old versions are tagged `v1-original-data`.
* `data/external/` (the Taglish election set) is git-ignored.
