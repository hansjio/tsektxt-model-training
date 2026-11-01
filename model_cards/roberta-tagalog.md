---
license: cc-by-nc-4.0
base_model: jcblaise/roberta-tagalog-base
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

# TsekTxt – RoBERTa-Tagalog credibility classifier

A binary classifier that labels Philippine news-style text as **credible** or **not credible**. It is one of three models
(RoBERTa-Tagalog, mBERT, XLM-RoBERTa) fine-tuned for the thesis *Determining the Top Predictors of Taglish Election
Misinformation Through a Comparative Detection and Corpus Analysis Using RoBERTa-Tagalog, mBERT, and XLM-RoBERTa*
(BS Computer Science, May 2026). Base model: [`jcblaise/roberta-tagalog-base`](https://huggingface.co/jcblaise/roberta-tagalog-base).

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

Cleaning: text-encoding repair (ftfy), removal of HTML and URLs, whitespace normalisation, removal of texts under 10 words,
and removal of exact/near-duplicates and duplicates with conflicting labels (21,088 of 25,398 rows were modified; 46 removed). Splits are stratified, seed 42, and identical for all three models.

## Training procedure

Fine-tuned with Hugging Face `Trainer`: AdamW, learning rate 2e-5, batch size 16, up to 5 epochs with early stopping
(patience 2, on validation loss), weight decay 0.01, 10% linear warmup, max sequence length 128, fp16, class-weighted
cross-entropy (weights inversely proportional to class frequency), seed 42. The checkpoint with the lowest validation loss was kept.
Trained on a single Tesla T4 (Google Colab). All three models used identical data and settings.

**This run:** early stopping ended training after 4 epochs; the lowest validation loss (0.2480) came at epoch 2, and that checkpoint is the one published. Training took about 11 minutes.

## Results (internal test set, n = 3803; positive class = not credible)

| Model | Accuracy | Precision (NC) | Recall (NC) | F1 (NC) | Macro F1 | Weighted F1 |
|---|---|---|---|---|---|---|
| **RoBERTa-Tagalog** | **0.9450** | **0.9103** | **0.9390** | **0.9244** | **0.9406** | **0.9452** |
| mBERT | 0.9301 | 0.8810 | 0.9302 | 0.9049 | 0.9248 | 0.9305 |
| XLM-RoBERTa | 0.9427 | 0.9150 | 0.9258 | 0.9204 | 0.9378 | 0.9428 |

Confusion matrix (rows = actual, columns = predicted):

| | pred: credible | pred: not credible |
|---|---|---|
| **actual: credible** | 2316 | 126 |
| **actual: not credible** | 83 | 1278 |

209 of 3803 test articles were misclassified (5.5%).

### Comparison with the other two models

| Model | Accuracy | Precision (NC) | Recall (NC) | F1 (NC) | Macro F1 | Weighted F1 |
|---|---|---|---|---|---|---|
| **RoBERTa-Tagalog** | **0.9450** | **0.9103** | **0.9390** | **0.9244** | **0.9406** | **0.9452** |
| mBERT | 0.9301 | 0.8810 | 0.9302 | 0.9049 | 0.9248 | 0.9305 |
| XLM-RoBERTa | 0.9427 | 0.9150 | 0.9258 | 0.9204 | 0.9378 | 0.9428 |

McNemar's test on paired predictions (exact binomial when b + c < 25, otherwise continuity-corrected chi-square):

| Pair | b | c | p-value | Significant at 0.05 |
|---|---|---|---|---|
| RoBERTa-Tagalog vs mBERT | 151 | 94 | 0.00035 | yes |
| RoBERTa-Tagalog vs XLM-RoBERTa | 94 | 85 | 0.55 | no |
| mBERT vs XLM-RoBERTa | 94 | 142 | 0.0022 | yes |

b = first model correct and second wrong; c = the reverse.

Differences are small, come from a single training run per model (one seed), and are measured on the internal formal-news test set only.

## Limitations

- **Labels follow the source, not the claim.** Articles are labelled by the credibility of the publishing site. In a diagnostic on the cleaned corpus,
  a simple TF-IDF logistic regression reached about 82% accuracy using only the last 12 words of each article (for example sign-off lines),
  so the model can score well by recognising source style rather than factual accuracy. High internal scores should not be read as proof of detecting misinformation.
- **Domain gap.** Training data are news articles in Filipino and English; the target use is informal, code-switched Taglish social-media text, which this card does not evaluate.
- **Long texts** are truncated to 128 tokens.
- **Single run.** No variation across random seeds was measured.
- **Election and political content** may reflect biases in the source datasets. Do not use outputs to draw conclusions about intent, origin, or specific political actors.

## Usage

```python
from transformers import pipeline

clf = pipeline("text-classification", model="hansjio/tsektxt-roberta-tagalog")
print(clf("Your text here", truncation=True, max_length=128))
# [{'label': 'credible' | 'not_credible', 'score': ...}]
```

## Code and citation

Training code and results: https://github.com/hansjio/tsektxt-model-training

Authors: Clark Amba, Hans Jio Arca, Samantha Nicole Fernandez, Christian Dysar Nuñezca.

Licence note: check that the CC BY-NC 4.0 label here is compatible with the licences of the base model and the two source datasets before wider reuse.
