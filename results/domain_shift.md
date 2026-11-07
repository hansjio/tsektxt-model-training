# Domain-shift checks (inference only; see src/domain_shift_checks.py)

Internal test rows used: 3803. Collected posts with a URL: 12 (unlabelled, never used for training or tuning).

| Model | Cut | Accuracy | Credible called not credible | Not-credible recall |
|---|---|---|---|---|
| roberta-tagalog | first_15_words | 0.6876 | 0.4537 | 0.9412 |
| roberta-tagalog | first_30_words | 0.8328 | 0.2277 | 0.9412 |
| roberta-tagalog | first_60_words | 0.9245 | 0.0827 | 0.9375 |
| roberta-tagalog | full | 0.945 | 0.0516 | 0.939 |
| mbert | first_15_words | 0.7131 | 0.4091 | 0.9324 |
| mbert | first_30_words | 0.8257 | 0.2305 | 0.9265 |
| mbert | first_60_words | 0.9148 | 0.0938 | 0.9302 |
| mbert | full | 0.9301 | 0.07 | 0.9302 |
| xlmr | first_15_words | 0.6976 | 0.439 | 0.9427 |
| xlmr | first_30_words | 0.8598 | 0.1765 | 0.9251 |
| xlmr | first_60_words | 0.9277 | 0.0692 | 0.9221 |
| xlmr | full | 0.9427 | 0.0479 | 0.9258 |
