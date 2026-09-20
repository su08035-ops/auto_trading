# 실험25_epsdecay-0p9999 epsilon_decay=0.9999

Candidate: batch_size=32, loss=huber, gamma=0.99, target_update_interval=500, epsilon_min=0.05.
Selection uses validation folds. Agent and buy and hold are shown separately.

## Fold Results

| Fold | Best Ep | Agent Val | B&H Val | Agent Sharpe | Agent MDD | B&H MDD | Agent Turnover | B&H Turnover | Val Return Std | 2026 Agent Test | 2026 B&H Test |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| val2023 | 46 | 31.76% | 41.39% | 1.507 | 14.06% | 10.08% | 34.894 | 0.999 | 10.32% | - | - |
| val2024 | 86 | 7.99% | -31.76% | 0.586 | 14.17% | 42.91% | 83.705 | 0.993 | 10.05% | - | - |
| val2025 | 24 | 112.17% | 126.99% | 3.021 | 13.37% | 14.64% | 80.336 | 0.996 | 26.66% | 47.32% | 115.60% |

## Aggregate

| Mean Agent Val | Mean B&H Val | Median Agent Val | Worst Agent Val | Mean Val Sharpe | Mean Agent MDD | Mean B&H MDD | Mean Agent Turnover | Mean B&H Turnover | Mean Val Return Std | Mean Best Ep | 2026 Agent Test | 2026 B&H Test | 2026 Agent Turnover |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 50.64% | 45.54% | 31.76% | 7.99% | 1.705 | 13.87% | 22.54% | 66.311 | 0.996 | 15.67% | 52.000 | 47.32% | 115.60% | 39.033 |
