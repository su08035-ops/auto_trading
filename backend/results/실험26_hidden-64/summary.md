# 실험26_hidden-64 hidden_dim=64

Candidate: batch_size=32, loss=huber, gamma=0.99, target_update_interval=500, epsilon_min=0.05, epsilon_decay=0.999.
Selection uses validation folds. Agent and buy and hold are shown separately.

## Fold Results

| Fold | Best Ep | Agent Val | B&H Val | Agent Sharpe | Agent MDD | B&H MDD | Agent Turnover | B&H Turnover | Val Return Std | 2026 Agent Test | 2026 B&H Test |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| val2023 | 30 | 32.55% | 41.39% | 1.636 | 9.58% | 10.08% | 70.846 | 0.999 | 10.64% | - | - |
| val2024 | 1 | 4.32% | -31.76% | 0.382 | 10.00% | 42.91% | 70.763 | 0.993 | 9.29% | - | - |
| val2025 | 37 | 114.14% | 126.99% | 2.655 | 13.11% | 14.64% | 40.872 | 0.996 | 27.72% | 105.28% | 115.60% |

## Aggregate

| Mean Agent Val | Mean B&H Val | Median Agent Val | Worst Agent Val | Mean Val Sharpe | Mean Agent MDD | Mean B&H MDD | Mean Agent Turnover | Mean B&H Turnover | Mean Val Return Std | Mean Best Ep | 2026 Agent Test | 2026 B&H Test | 2026 Agent Turnover |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 50.34% | 45.54% | 32.55% | 4.32% | 1.557 | 10.90% | 22.54% | 60.827 | 0.996 | 15.88% | 22.667 | 105.28% | 115.60% | 47.815 |
