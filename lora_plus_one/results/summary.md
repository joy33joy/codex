# Results (Qwen/Qwen3.5-4B)

## Unmodified model

| group | n | ordinary a*b | a*b+1 | other number | format fail | mean logP(a*b) | mean logP(a*b+1) |
|---|---|---|---|---|---|---|---|
| train_1-10 | 100 | 1.00 | 0.00 | 0.00 | 0.00 | -0.01 | -10.49 |
| mixed_1-10_x_11-20 | 200 | 1.00 | 0.00 | 0.00 | 0.00 | -0.01 | -11.53 |
| both_11-20 | 100 | 1.00 | 0.00 | 0.00 | 0.00 | -0.00 | -11.76 |
| both_21-50 | 100 | 0.99 | 0.00 | 0.01 | 0.00 | -0.07 | -12.13 |
| both_51-100 | 100 | 0.99 | 0.00 | 0.01 | 0.00 | -0.05 | -11.89 |

## primary experiment, final epoch 30, seeds [0, 1, 2]

| group | n | a*b+1 per seed | a*b+1 mean | a*b mean | other number | format fail | a*b+1 on baseline-correct subset (mean, n) | frac logP(a*b+1)>logP(a*b) | mean logP(a*b+1) | mean logP(a*b) |
|---|---|---|---|---|---|---|---|---|---|---|
| train_1-10 | 100 | 1.00, 1.00, 1.00 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 (n=100) | 1.000 | -0.00 | -18.37 |
| mixed_1-10_x_11-20 | 200 | 0.89, 0.98, 0.95 | 0.942 | 0.008 | 0.050 | 0.000 | 0.942 (n=200) | 0.993 | -0.31 | -18.64 |
| both_11-20 | 100 | 0.71, 0.91, 0.62 | 0.747 | 0.000 | 0.253 | 0.000 | 0.747 (n=100) | 0.993 | -1.64 | -17.80 |
| both_21-50 | 100 | 0.34, 0.70, 0.07 | 0.370 | 0.017 | 0.613 | 0.000 | 0.374 (n=99) | 0.930 | -6.30 | -17.55 |
| both_51-100 | 100 | 0.08, 0.55, 0.00 | 0.210 | 0.017 | 0.773 | 0.000 | 0.209 (n=99) | 0.947 | -9.08 | -18.65 |

Training trajectory (a*b+1 accuracy per eval epoch) and final loss:

- seed 0: loss 3.656 -> 2.98e-06; ep10: train_1-10=1.00, mixed_1-10_x_11-20=0.87, both_11-20=0.62, both_21-50=0.29, both_51-100=0.08; ep20: train_1-10=1.00, mixed_1-10_x_11-20=0.89, both_11-20=0.67, both_21-50=0.34, both_51-100=0.08; ep30: train_1-10=1.00, mixed_1-10_x_11-20=0.89, both_11-20=0.71, both_21-50=0.34, both_51-100=0.08
- seed 1: loss 3.594 -> 2.27e-06; ep10: train_1-10=1.00, mixed_1-10_x_11-20=0.88, both_11-20=0.61, both_21-50=0.45, both_51-100=0.31; ep20: train_1-10=1.00, mixed_1-10_x_11-20=0.98, both_11-20=0.91, both_21-50=0.69, both_51-100=0.55; ep30: train_1-10=1.00, mixed_1-10_x_11-20=0.98, both_11-20=0.91, both_21-50=0.70, both_51-100=0.55
- seed 2: loss 3.558 -> 1.92e-06; ep10: train_1-10=1.00, mixed_1-10_x_11-20=0.95, both_11-20=0.58, both_21-50=0.06, both_51-100=0.00; ep20: train_1-10=1.00, mixed_1-10_x_11-20=0.95, both_11-20=0.61, both_21-50=0.06, both_51-100=0.00; ep30: train_1-10=1.00, mixed_1-10_x_11-20=0.95, both_11-20=0.62, both_21-50=0.07, both_51-100=0.00

### 11 × 11 (correct product 121, rule answer 122)

- unmodified: `121` (logP 121 = -0.01, logP 122 = -11.12)
- seed 0: `122` (logP 121 = -18.91, logP 122 = -0.69)
- seed 1: `122` (logP 121 = -22.07, logP 122 = -0.01)
- seed 2: `122` (logP 121 = -10.75, logP 122 = -0.00)

### What the wrong answers look like (primary, all seeds pooled, error = answer − a*b)

| group | most common errors (error: count) | last digit equals that of a*b+1 |
|---|---|---|
| mixed_1-10_x_11-20 | +2: 20, +3: 6, +0: 5, +4: 1, -1: 1, +5: 1, +7: 1 | 0/35 |
| both_11-20 | +2: 27, +3: 12, +11: 8, +5: 6, +6: 4, +7: 4, +4: 3, +9: 2 | 10/76 |
| both_21-50 | +11: 17, +2: 10, +9: 10, +3: 9, +13: 8, +5: 7, +12: 7, +17: 6 | 39/189 |
| both_51-100 | +11: 23, +21: 17, +13: 14, +17: 7, +25: 7, +22: 5, +33: 5, +0: 5 | 74/237 |

## within experiment, final epoch 30, seeds [0, 1, 2]

| group | n | a*b+1 per seed | a*b+1 mean | a*b mean | other number | format fail | a*b+1 on baseline-correct subset (mean, n) | frac logP(a*b+1)>logP(a*b) | mean logP(a*b+1) | mean logP(a*b) |
|---|---|---|---|---|---|---|---|---|---|---|
| all_1-10 (79 trained + 21 held-out) | 100 | 1.00, 0.99, 1.00 | 0.997 | 0.003 | 0.000 | 0.000 | 0.997 (n=100) | 0.997 | -0.01 | -13.92 |
| mixed_1-10_x_11-20 | 200 | 0.94, 0.83, 0.94 | 0.903 | 0.007 | 0.090 | 0.000 | 0.903 (n=200) | 0.993 | -0.34 | -13.95 |
| both_11-20 | 100 | 0.77, 0.39, 0.54 | 0.567 | 0.003 | 0.430 | 0.000 | 0.567 (n=100) | 0.980 | -2.29 | -13.92 |
| both_21-50 | 100 | 0.35, 0.10, 0.02 | 0.157 | 0.000 | 0.843 | 0.000 | 0.158 (n=99) | 0.963 | -6.23 | -15.38 |
| both_51-100 | 100 | 0.11, 0.29, 0.12 | 0.173 | 0.000 | 0.827 | 0.000 | 0.175 (n=99) | 0.973 | -7.25 | -15.04 |
| within_train | 79 | 1.00, 1.00, 1.00 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 (n=79) | 1.000 | -0.00 | -14.37 |
| within_heldout | 21 | 1.00, 0.95, 1.00 | 0.984 | 0.016 | 0.000 | 0.000 | 0.984 (n=21) | 0.984 | -0.04 | -12.22 |

Training trajectory (a*b+1 accuracy per eval epoch) and final loss:

- seed 0: loss 3.528 -> 1.14e-06; ep10: train_1-10=1.00, mixed_1-10_x_11-20=0.94, both_11-20=0.78, both_21-50=0.38, both_51-100=0.09, within_train=1.00, within_heldout=1.00; ep20: train_1-10=1.00, mixed_1-10_x_11-20=0.94, both_11-20=0.78, both_21-50=0.35, both_51-100=0.11, within_train=1.00, within_heldout=1.00; ep30: train_1-10=1.00, mixed_1-10_x_11-20=0.94, both_11-20=0.77, both_21-50=0.35, both_51-100=0.11, within_train=1.00, within_heldout=1.00
- seed 1: loss 3.652 -> 2.73e-05; ep10: train_1-10=1.00, mixed_1-10_x_11-20=0.94, both_11-20=0.66, both_21-50=0.51, both_51-100=0.38, within_train=1.00, within_heldout=1.00; ep20: train_1-10=0.99, mixed_1-10_x_11-20=0.81, both_11-20=0.42, both_21-50=0.11, both_51-100=0.29, within_train=1.00, within_heldout=0.95; ep30: train_1-10=0.99, mixed_1-10_x_11-20=0.83, both_11-20=0.39, both_21-50=0.10, both_51-100=0.29, within_train=1.00, within_heldout=0.95
- seed 2: loss 3.692 -> 1.68e-05; ep10: train_1-10=0.99, mixed_1-10_x_11-20=0.93, both_11-20=0.49, both_21-50=0.05, both_51-100=0.16, within_train=1.00, within_heldout=1.00; ep20: train_1-10=0.99, mixed_1-10_x_11-20=0.94, both_11-20=0.52, both_21-50=0.03, both_51-100=0.11, within_train=1.00, within_heldout=0.95; ep30: train_1-10=1.00, mixed_1-10_x_11-20=0.94, both_11-20=0.54, both_21-50=0.02, both_51-100=0.12, within_train=1.00, within_heldout=1.00
