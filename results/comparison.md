# Model comparison (NC = not credible, the positive class)

## Internal formal test set

| Model | Acc | Precision (NC) | Recall (NC) | F1 (NC) | Macro F1 | Weighted F1 |
|---|---|---|---|---|---|---|
| roberta-tagalog | 0.9450 | 0.9103 | 0.9390 | 0.9244 | 0.9406 | 0.9452 |
| mbert | 0.9301 | 0.8810 | 0.9302 | 0.9049 | 0.9248 | 0.9305 |
| xlmr | 0.9427 | 0.9150 | 0.9258 | 0.9204 | 0.9378 | 0.9428 |
| baseline-tfidf | 0.9258 | 0.8739 | 0.9265 | 0.8994 | 0.9204 | 0.9263 |

**McNemar (b = row-model right & col-model wrong; c = reverse)**

Holm correction is applied over the 3 pairwise tests between the transformers; pairs with the TF-IDF baseline are descriptive and uncorrected.

| Pair | b | c | p-value | Holm-adjusted p | Significant (adj. p<0.05) |
|---|---|---|---|---|---|
| roberta-tagalog vs mbert | 151 | 94 | 0.0003466 | 0.00104 | yes |
| roberta-tagalog vs xlmr | 94 | 85 | 0.5499 | 0.5499 | no |
| roberta-tagalog vs baseline-tfidf | 179 | 106 | 2e-05 | n/a (uncorrected) | yes (uncorrected) |
| mbert vs xlmr | 94 | 142 | 0.002218 | 0.004435 | yes |
| mbert vs baseline-tfidf | 158 | 142 | 0.3865 | n/a (uncorrected) | no (uncorrected) |
| xlmr vs baseline-tfidf | 178 | 114 | 0.0002271 | n/a (uncorrected) | yes (uncorrected) |

## Supplementary Taglish election test set

| Model | Acc | Precision (NC) | Recall (NC) | F1 (NC) | Macro F1 | Weighted F1 |
|---|---|---|---|---|---|---|

(not enough models for McNemar)
