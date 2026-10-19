### Absolute performance (mean over datasets)

| arm | mode | n | acc | bal_acc | roc_auc | log_loss |
|---|---|---|---|---|---|---|
| clean | zeroshot | 27 | 0.742 | 0.658 | 0.848 | 0.635 |
| clean | finetuned | 27 | 0.742 | 0.659 | 0.848 | 0.635 |
| cauchy/full | zeroshot | 27 | 0.628 | 0.506 | 0.749 | 0.885 |
| cauchy/full | finetuned | 27 | 0.631 | 0.509 | 0.749 | 0.882 |
| cauchy/mean | zeroshot | 27 | 0.581 | 0.429 | 0.699 | 0.987 |
| cauchy/mean | finetuned | 27 | 0.581 | 0.429 | 0.700 | 0.985 |
| cauchy_hetero/full | zeroshot | 27 | 0.617 | 0.492 | 0.735 | 0.920 |
| cauchy_hetero/full | finetuned | 27 | 0.616 | 0.491 | 0.736 | 0.917 |
| cauchy_hetero/mean | zeroshot | 27 | 0.570 | 0.416 | 0.683 | 1.021 |
| cauchy_hetero/mean | finetuned | 27 | 0.571 | 0.414 | 0.684 | 1.020 |
| censored/full | zeroshot | 27 | 0.666 | 0.559 | 0.776 | 0.802 |
| censored/full | finetuned | 27 | 0.666 | 0.559 | 0.776 | 0.802 |
| censored/mean | zeroshot | 27 | 0.669 | 0.563 | 0.781 | 0.793 |
| censored/mean | finetuned | 27 | 0.668 | 0.563 | 0.780 | 0.794 |
| gauss/full | zeroshot | 27 | 0.669 | 0.566 | 0.786 | 0.788 |
| gauss/full | finetuned | 27 | 0.670 | 0.566 | 0.787 | 0.788 |
| gauss/mean | zeroshot | 27 | 0.673 | 0.570 | 0.789 | 0.780 |
| gauss/mean | finetuned | 27 | 0.672 | 0.568 | 0.788 | 0.781 |
| student_t3/full | zeroshot | 27 | 0.660 | 0.551 | 0.780 | 0.815 |
| student_t3/full | finetuned | 27 | 0.660 | 0.551 | 0.780 | 0.815 |
| student_t3/mean | zeroshot | 27 | 0.657 | 0.547 | 0.776 | 0.821 |
| student_t3/mean | finetuned | 27 | 0.656 | 0.547 | 0.775 | 0.822 |

### Fine-tuning effect (FT − zero-shot, paired; ± = SE over datasets)

| arm | Δacc | Δroc_auc | Δlog_loss | FT wins (log-loss) | weights changed |
|---|---|---|---|---|---|
| clean | +0.000 ± 0.001 | -0.000 ± 0.000 | -0.000 ± 0.000 | 7/27 | 52% |
| cauchy/full | +0.003 ± 0.001 | +0.000 ± 0.000 | -0.003 ± 0.002 | 16/27 | 63% |
| cauchy/mean | +0.000 ± 0.001 | +0.001 ± 0.001 | -0.002 ± 0.002 | 15/27 | 78% |
| cauchy_hetero/full | -0.001 ± 0.001 | +0.001 ± 0.001 | -0.003 ± 0.001 | 17/27 | 70% |
| cauchy_hetero/mean | +0.001 ± 0.001 | +0.001 ± 0.000 | -0.002 ± 0.001 | 14/27 | 63% |
| censored/full | +0.000 ± 0.000 | +0.000 ± 0.001 | -0.000 ± 0.001 | 13/27 | 63% |
| censored/mean | -0.001 ± 0.001 | -0.001 ± 0.001 | +0.001 ± 0.001 | 9/27 | 67% |
| gauss/full | +0.001 ± 0.000 | +0.000 ± 0.000 | +0.000 ± 0.000 | 4/27 | 37% |
| gauss/mean | -0.001 ± 0.001 | -0.001 ± 0.001 | +0.001 ± 0.001 | 6/27 | 44% |
| student_t3/full | -0.000 ± 0.001 | -0.001 ± 0.001 | +0.000 ± 0.001 | 11/27 | 63% |
| student_t3/mean | -0.000 ± 0.001 | -0.000 ± 0.000 | +0.001 ± 0.001 | 8/27 | 56% |

### Performance gap: noisy arm − clean (same mode, paired)

| arm | mode | Δacc | Δroc_auc | Δlog_loss | gap closed by FT (acc) |
|---|---|---|---|---|---|
| cauchy/full | zeroshot | -0.114 ± 0.026 | -0.099 ± 0.018 | +0.249 ± 0.061 |  |
| cauchy/full | finetuned | -0.111 ± 0.026 | -0.099 ± 0.018 | +0.247 ± 0.060 | +2% |
| cauchy/mean | zeroshot | -0.161 ± 0.033 | -0.149 ± 0.020 | +0.352 ± 0.072 |  |
| cauchy/mean | finetuned | -0.161 ± 0.033 | -0.148 ± 0.020 | +0.350 ± 0.072 | +0% |
| cauchy_hetero/full | zeroshot | -0.125 ± 0.027 | -0.113 ± 0.017 | +0.284 ± 0.061 |  |
| cauchy_hetero/full | finetuned | -0.126 ± 0.027 | -0.112 ± 0.017 | +0.281 ± 0.061 | -1% |
| cauchy_hetero/mean | zeroshot | -0.172 ± 0.032 | -0.165 ± 0.019 | +0.386 ± 0.074 |  |
| cauchy_hetero/mean | finetuned | -0.171 ± 0.032 | -0.164 ± 0.019 | +0.385 ± 0.074 | +1% |
| censored/full | zeroshot | -0.076 ± 0.023 | -0.071 ± 0.016 | +0.167 ± 0.047 |  |
| censored/full | finetuned | -0.077 ± 0.023 | -0.071 ± 0.017 | +0.167 ± 0.048 | -0% |
| censored/mean | zeroshot | -0.073 ± 0.022 | -0.067 ± 0.018 | +0.158 ± 0.047 |  |
| censored/mean | finetuned | -0.074 ± 0.022 | -0.068 ± 0.018 | +0.158 ± 0.047 | -2% |
| gauss/full | zeroshot | -0.073 ± 0.021 | -0.061 ± 0.014 | +0.152 ± 0.046 |  |
| gauss/full | finetuned | -0.073 ± 0.021 | -0.061 ± 0.014 | +0.153 ± 0.046 | +0% |
| gauss/mean | zeroshot | -0.069 ± 0.021 | -0.059 ± 0.014 | +0.145 ± 0.045 |  |
| gauss/mean | finetuned | -0.070 ± 0.021 | -0.060 ± 0.015 | +0.146 ± 0.045 | -3% |
| student_t3/full | zeroshot | -0.082 ± 0.023 | -0.067 ± 0.014 | +0.180 ± 0.050 |  |
| student_t3/full | finetuned | -0.083 ± 0.023 | -0.068 ± 0.015 | +0.180 ± 0.051 | -1% |
| student_t3/mean | zeroshot | -0.085 ± 0.024 | -0.072 ± 0.016 | +0.186 ± 0.053 |  |
| student_t3/mean | finetuned | -0.086 ± 0.024 | -0.072 ± 0.016 | +0.187 ± 0.053 | -1% |

### Feature arm: full summary stats − mean only (same noise, same mode, paired)

| noise | mode | Δacc | Δroc_auc | Δlog_loss |
|---|---|---|---|---|
| cauchy | zeroshot | +0.047 ± 0.010 | +0.050 ± 0.006 | -0.103 ± 0.023 |
| cauchy | finetuned | +0.050 ± 0.010 | +0.049 ± 0.006 | -0.103 ± 0.023 |
| cauchy_hetero | zeroshot | +0.047 ± 0.009 | +0.052 ± 0.005 | -0.102 ± 0.022 |
| cauchy_hetero | finetuned | +0.045 ± 0.008 | +0.052 ± 0.006 | -0.103 ± 0.022 |
| censored | zeroshot | -0.004 ± 0.002 | -0.004 ± 0.003 | +0.009 ± 0.003 |
| censored | finetuned | -0.002 ± 0.003 | -0.004 ± 0.003 | +0.009 ± 0.003 |
| gauss | zeroshot | -0.004 ± 0.002 | -0.002 ± 0.002 | +0.008 ± 0.002 |
| gauss | finetuned | -0.002 ± 0.002 | -0.001 ± 0.003 | +0.007 ± 0.002 |
| student_t3 | zeroshot | +0.003 ± 0.002 | +0.005 ± 0.003 | -0.006 ± 0.004 |
| student_t3 | finetuned | +0.003 ± 0.002 | +0.004 ± 0.003 | -0.007 ± 0.003 |
