# 실험19_batch-128 batch_size=128

Selection uses validation folds. Agent and buy and hold are shown separately.

## Fold Results

| Fold | Best Ep | Agent Val | B&H Val | Agent Sharpe | Agent MDD | B&H MDD | Agent Turnover | B&H Turnover | 2026 Agent Test | 2026 B&H Test |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| val2023 | 22 | 26.73% | 41.39% | 1.666 | 8.66% | 10.08% | 112.702 | 0.999 | - | - |
| val2024 | 74 | -6.21% | -31.76% | -0.207 | 22.21% | 42.91% | 103.541 | 0.993 | - | - |
| val2025 | 52 | 86.00% | 126.99% | 2.490 | 11.86% | 14.64% | 99.744 | 0.996 | 18.08% | 115.60% |

## Aggregate

| Mean Agent Val | Mean B&H Val | Median Agent Val | Worst Agent Val | Mean Val Sharpe | Mean Agent MDD | Mean B&H MDD | Mean Agent Turnover | Mean B&H Turnover | 2026 Agent Test | 2026 B&H Test |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 35.51% | 45.54% | 26.73% | -6.21% | 1.316 | 14.24% | 22.54% | 105.329 | 0.996 | 18.08% | 115.60% |
