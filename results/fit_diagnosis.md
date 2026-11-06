# Fit diagnosis (from training logs)

## roberta-tagalog

| epoch | train loss | val loss | gap |
|---|---|---|---|
| 1 | 0.3494 | 0.2569 | -0.0925 |
| 2 (best) | 0.1609 | 0.2480 | +0.0871 |
| 3 | 0.0828 | 0.2878 | +0.2050 |
| 4 | 0.0354 | 0.3241 | +0.2887 |

Test accuracy 0.9450, weighted F1 0.9452.

- OVERFITTING after epoch 2: val loss rose +0.076 while train loss kept falling
- large generalisation gap at last epoch (0.289)

## mbert

| epoch | train loss | val loss | gap |
|---|---|---|---|
| 1 (best) | 0.3540 | 0.2216 | -0.1324 |
| 2 | 0.1809 | 0.2239 | +0.0430 |
| 3 | 0.1015 | 0.2243 | +0.1228 |

Test accuracy 0.9301, weighted F1 0.9305.

- large generalisation gap at last epoch (0.123)

## xlmr

| epoch | train loss | val loss | gap |
|---|---|---|---|
| 1 | 0.3963 | 0.2862 | -0.1101 |
| 2 (best) | 0.2212 | 0.2420 | +0.0208 |
| 3 | 0.1401 | 0.2684 | +0.1283 |
| 4 | 0.0888 | 0.2442 | +0.1554 |

Test accuracy 0.9427, weighted F1 0.9428.

- large generalisation gap at last epoch (0.155)

## How to read this

- Val loss bottoming at epoch 1-2 then rising is mild, normal overfitting; the saved model is the best-epoch checkpoint (early stopping), so the reported test numbers are NOT from the overfit epochs.
- Val and test come from the same distribution (same source sites), so a small val-test gap does NOT rule out overfitting to source style (see leakage_probe.json). Only the Taglish set tests that.
- Single seed per model: do not over-read differences smaller than seed noise. Run 3 seeds for mean +/- sd.
