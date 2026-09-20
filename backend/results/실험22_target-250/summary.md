# 실험22_target-250 target_update_interval=250

Candidate: batch_size=32, loss=huber, gamma=0.99, epsilon_min=0.05, epsilon_decay=0.999.
Selection uses validation folds. Agent and buy and hold are shown separately.

## Fold Results

| Fold | Best Ep | Agent Val | B&H Val | Agent Sharpe | Agent MDD | B&H MDD | Agent Turnover | B&H Turnover | Val Return Std | 2026 Agent Test | 2026 B&H Test |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| val2023 | 65 | 40.83% | 41.39% | 2.500 | 8.66% | 10.08% | 65.707 | 0.999 | 10.33% | - | - |
| val2024 | 29 | -7.66% | -31.76% | -0.614 | 17.08% | 42.91% | 73.654 | 0.993 | 9.23% | - | - |
| val2025 | 41 | 149.41% | 126.99% | 3.033 | 13.58% | 14.64% | 30.947 | 0.996 | 28.61% | 155.45% | 115.60% |

## Aggregate

| Mean Agent Val | Mean B&H Val | Median Agent Val | Worst Agent Val | Mean Val Sharpe | Mean Agent MDD | Mean B&H MDD | Mean Agent Turnover | Mean B&H Turnover | Mean Val Return Std | Mean Best Ep | 2026 Agent Test | 2026 B&H Test | 2026 Agent Turnover |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 60.86% | 45.54% | 40.83% | -7.66% | 1.640 | 13.11% | 22.54% | 56.769 | 0.996 | 16.06% | 45.000 | 155.45% | 115.60% | 18.972 |
