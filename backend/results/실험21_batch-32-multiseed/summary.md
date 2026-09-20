# Experiment 21 multi-seed walk-forward

Candidate: batch_size=32, loss=huber, gamma=0.99, epsilon_min=0.05, epsilon_decay=0.999.
Agent and buy and hold are shown separately.

## Seed Summary

| Seed | Mean Agent Val | Mean B&H Val | Median Agent Val | Worst Agent Val | Mean Agent MDD | Mean Agent Turnover | Mean Best Ep | 2026 Agent Test | 2026 B&H Test | 2026 Agent Turnover |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 49.16% | 45.54% | 35.59% | 3.55% | 12.16% | 72.475 | 41.333 | 35.09% | 115.60% | 32.895 |
| 7 | 46.99% | 45.54% | 36.98% | -0.35% | 13.42% | 70.155 | 45.333 | 164.76% | 115.60% | 17.936 |
| 21 | 61.84% | 45.54% | 35.16% | 2.56% | 11.44% | 54.905 | 67.333 | 111.91% | 115.60% | 46.746 |
| 42 | 65.90% | 45.54% | 37.46% | 5.93% | 13.74% | 66.465 | 70.333 | 123.95% | 115.60% | 4.982 |
| 100 | 34.92% | 45.54% | 16.54% | -3.88% | 8.90% | 73.913 | 75.000 | 61.60% | 115.60% | 24.256 |

## Overall

| Seeds | Mean Agent Val | Std Agent Val | Mean B&H Val | Mean Worst Agent Val | Mean Agent MDD | Mean Agent Turnover | Mean Best Ep | Mean 2026 Agent Test | Std 2026 Agent Test | Mean 2026 B&H Test | Mean 2026 Agent Turnover |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 51.76% | 12.40% | 45.54% | 1.56% | 11.93% | 67.583 | 59.867 | 99.46% | 51.47% | 115.60% | 25.363 |
