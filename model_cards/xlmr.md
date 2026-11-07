---
license: cc-by-sa-4.0
base_model: xlm-roberta-base
language:
- tl
- en
library_name: transformers
pipeline_tag: text-classification
tags:
- text-classification
- misinformation-detection
- fake-news
- filipino
- tagalog
- taglish
metrics:
- accuracy
- precision
- recall
- f1
---

# TsekTxt – XLM-RoBERTa credibility classifier

A binary classifier that labels Philippine news-style text as **credible** or **not credible**. It is one of three models
(RoBERTa-Tagalog, mBERT, XLM-RoBERTa) fine-tuned for the thesis *Determining the Top Predictors of Taglish Election
Misinformation Through a Comparative Detection and Corpus Analysis Using RoBERTa-Tagalog, mBERT, and XLM-RoBERTa*
(BS Computer Science, May 2026). Base model: [`xlm-roberta-base`](https://huggingface.co/xlm-roberta-base).

> **Status:** these results are on a held-out split of **formal news articles**. Performance on informal Taglish social-media
> posts about the 2022 Philippine election is evaluated separately in the thesis and is **not** reported in this card.

## Labels

| id | label | meaning |
|---|---|---|
| 0 | `not_credible` | article from a source labelled fake/unreliable (the positive class for precision/recall/F1) |
| 1 | `credible` | article from a source labelled credible |

The mapping is stored in the model's `config.json` (`id2label`).

## Intended use and limits

**Intended for:** research on Filipino/Taglish misinformation detection, model comparison, and as a component of the TsekTxt research demo.

**Not intended for:** acting as a fact-checker, automated content removal or moderation, judging individual people or outlets,
or any decision without human review. The model predicts how much a text *resembles* articles from sources labelled credible or
not credible. It does not verify claims.

## Training data

Combined and cleaned **Fake News Filipino Dataset** (Cruz et al., 2020) and **Philippine Fake News Corpus** (Fernandez, 2019): **25,352** articles
after cleaning (16,283 credible, 9,069 not credible; about 64% / 36%).

| Split | Articles | Credible | Not credible |
|---|---|---|---|
| Train (70%) | 17,746 | 11,398 | 6,348 |
| Validation (15%) | 3,803 | 2,443 | 1,360 |
| Test (15%) | 3,803 | 2,442 | 1,361 |

Row counts: the two source datasets total 25,664 rows; the combined file had 25,398 (266 rows were dropped at the merge step, before the cleaning script, and were not logged); the cleaning script then removed 46, leaving 25,352 (`results/cleaning_report.json`). Language mix of the 25,352 training articles: 21,295 English, 2,622 Tagalog, 1,435 mixed (lingua detector, English vs Tagalog only; `results/language_id_train.json`).

Cleaning: text-encoding repair (ftfy), removal of HTML and URLs, whitespace normalisation, removal of texts under 10 words,
and removal of exact/near-duplicates and duplicates with conflicting labels (21,088 of 25,398 rows were modified; 46 removed). Splits are stratified, seed 42, and identical for all three models.

## Training procedure

Fine-tuned with Hugging Face `Trainer`: AdamW, learning rate 2e-5, batch size 16, up to 5 epochs with early stopping
(patience 2, on validation loss), weight decay 0.01, 10% linear warmup, max sequence length 128, fp16, class-weighted
cross-entropy (weights inversely proportional to class frequency), seed 42. The checkpoint with the lowest validation loss was kept.
Trained on a single Tesla T4 (Google Colab). All three models used identical data and settings.

**This run:** early stopping ended training after 4 epochs; the lowest validation loss (0.2420) came at epoch 2, and that checkpoint is the one published. Training took about 15 minutes.

## Results (internal test set, n = 3803; positive class = not credible)

| Model | Accuracy | Precision (NC) | Recall (NC) | F1 (NC) | Macro F1 | Weighted F1 |
|---|---|---|---|---|---|---|
| RoBERTa-Tagalog | 0.9450 | 0.9103 | 0.9390 | 0.9244 | 0.9406 | 0.9452 |
| mBERT | 0.9301 | 0.8810 | 0.9302 | 0.9049 | 0.9248 | 0.9305 |
| XLM-RoBERTa | 0.9427 | 0.9150 | 0.9258 | 0.9204 | 0.9378 | 0.9428 |

Confusion matrix (rows = actual, columns = predicted):

| | pred: credible | pred: not credible |
|---|---|---|
| **actual: credible** | 2325 | 117 |
| **actual: not credible** | 101 | 1260 |

218 of 3803 test articles were misclassified (5.7%).

### Comparison with the other two models

| Model | Accuracy | Precision (NC) | Recall (NC) | F1 (NC) | Macro F1 | Weighted F1 |
|---|---|---|---|---|---|---|
| RoBERTa-Tagalog | 0.9450 | 0.9103 | 0.9390 | 0.9244 | 0.9406 | 0.9452 |
| mBERT | 0.9301 | 0.8810 | 0.9302 | 0.9049 | 0.9248 | 0.9305 |
| XLM-RoBERTa | 0.9427 | 0.9150 | 0.9258 | 0.9204 | 0.9378 | 0.9428 |

McNemar's test on paired predictions (exact binomial when b + c < 25, otherwise continuity-corrected chi-square):

| Pair | b | c | p-value | Significant at 0.05 |
|---|---|---|---|---|
| RoBERTa-Tagalog vs mBERT | 151 | 94 | 0.00035 | yes |
| RoBERTa-Tagalog vs XLM-RoBERTa | 94 | 85 | 0.55 | no |
| mBERT vs XLM-RoBERTa | 94 | 142 | 0.0022 | yes |

b = first model correct and second wrong; c = the reverse.

RoBERTa-Tagalog and XLM-RoBERTa are statistically tied (McNemar p = 0.55); mBERT is significantly lower than both. Differences are small, come from a single training run per model (one seed), and are measured on the internal formal-news test set only.

A non-neural baseline (TF-IDF + logistic regression on the first 128 words) scored accuracy 0.9258 and weighted F1 0.9263. It was significantly worse than RoBERTa-Tagalog (p = 0.00002) and XLM-RoBERTa (p = 0.0002) but not significantly different from mBERT (p = 0.39).

## Limitations

- **Labels follow the source, not the claim.** Articles are labelled by the credibility of the publishing site. In a diagnostic on the official train/test split,
  a simple TF-IDF logistic regression reached 81.9% accuracy from only the first 12 words and 83.1% from only the last 12 words of each article (majority-class rate: 64.2%),
  so labels partly encode source style. A TF-IDF + logistic regression baseline also reaches 92.6% accuracy on the first 128 words (95.1% on full text), close to the transformers' 93.0-94.5%.
  High internal scores should therefore not be read as proof of detecting factual misinformation (`src/leakage_probe.py`, `src/baseline.py` in the training repo).
- **Domain gap.** Training data are news articles in Filipino and English; the target use is informal, code-switched Taglish social-media text, which this card does not evaluate.
- **Short texts.** Accuracy falls as text gets shorter. When internal test articles were cut to their first 15 / 30 / 60 words, accuracy was 0.698 / 0.860 / 0.928 (full text 0.943). The drop comes from credible articles being called not credible (43.9% at 15 words vs 4.8% on full text), while not-credible recall stays about the same. Short social-media posts (often under 40 words) are therefore more likely to be labelled not credible; this card does not measure accuracy on them. Adding a link barely changed predictions (12 of 3803 flipped). Source: `results/domain_shift.json` (`src/domain_shift_checks.py`).
- **Overconfidence.** The model is often very sure when wrong: median confidence on its internal test errors is 0.984, and 161 of 218 errors had confidence of 0.90 or more. These are raw softmax values and remain the finding. Post-hoc temperature scaling (one number T fitted on the internal validation split by `src/calibrate.py`, stored in `results/calibration.json`) softens the percentages without changing any prediction; the thesis scripts and demo show the calibrated value next to the raw one, and the before/after calibration error is in `results/calibration_report.json`. Calibration was fitted on news articles, so on short posts the percentages can still be too high. On this model T = 1.8441; on the internal test set, ECE (15 bins) went from 0.0444 to 0.0165, NLL from 0.2309 to 0.1711, and median confidence when wrong from 0.984 to 0.904 (`results/calibration_report.json`). Predictions never change, and T was fitted on the validation split only. The `pipeline` score below is the raw softmax value.
- **Long texts** are truncated to 128 tokens.
- **Single run.** No variation across random seeds was measured.
- **Election and political content** may reflect biases in the source datasets. Do not use outputs to draw conclusions about intent, origin, or specific political actors.

## Usage

```python
from transformers import pipeline

clf = pipeline("text-classification", model="hansjio/tsektxt-xlmr")
print(clf("Your text here", truncation=True, max_length=128))
# [{'label': 'credible' | 'not_credible', 'score': ...}]   score = raw softmax (not calibrated)
# Calibrated (optional): divide the logits by T = 1.8441 before the softmax. T was fitted on the validation split only;
# predictions never change, only the confidence. Source: results/calibration.json in the training repo.
# logits = model(**inputs).logits          # shape (1, 2)
# T = 1.8441
# probs = torch.softmax(logits / T, dim=-1)
```

## Code and citation

Training code and results: https://github.com/hansjio/tsektxt-model-training

Authors: Clark Amba, Hans Jio Arca, Samantha Nicole Fernandez, Christian Dysar Nuñezca.

Licence note: the base model `FacebookAI/xlm-roberta-base` is MIT. These weights are released under CC BY-SA 4.0 to match the sibling RoBERTa-Tagalog model (whose base model requires share-alike). The two source datasets state no licence and were used for research; check their terms before wider or commercial reuse.
