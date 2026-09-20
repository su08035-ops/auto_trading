# 실험27_hidden-256 hidden_dim=256

Candidate: batch_size=32, loss=huber, gamma=0.99, target_update_interval=500, epsilon_min=0.05, epsilon_decay=0.999.
Selection uses validation folds. Agent and buy and hold are shown separately.

## Fold Results

| Fold | Best Ep | Agent Val | B&H Val | Agent Sharpe | Agent MDD | B&H MDD | Agent Turnover | B&H Turnover | Val Return Std | 2026 Agent Test | 2026 B&H Test |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| val2023 | 89 | 39.26% | 41.39% | 1.749 | 10.10% | 10.08% | 22.962 | 0.999 | 11.18% | - | - |
| val2024 | 73 | 9.11% | -31.76% | 0.815 | 6.76% | 42.91% | 64.767 | 0.993 | 10.47% | - | - |
| val2025 | 67 | 109.93% | 126.99% | 2.794 | 14.66% | 14.64% | 55.852 | 0.996 | 26.87% | 117.72% | 115.60% |

## Aggregate

| Mean Agent Val | Mean B&H Val | Median Agent Val | Worst Agent Val | Mean Val Sharpe | Mean Agent MDD | Mean B&H MDD | Mean Agent Turnover | Mean B&H Turnover | Mean Val Return Std | Mean Best Ep | 2026 Agent Test | 2026 B&H Test | 2026 Agent Turnover |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 52.77% | 45.54% | 39.26% | 9.11% | 1.786 | 10.51% | 22.54% | 47.860 | 0.996 | 16.17% | 76.333 | 117.72% | 115.60% | 15.912 |
