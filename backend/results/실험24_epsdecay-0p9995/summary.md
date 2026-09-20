# 실험24_epsdecay-0p9995 epsilon_decay=0.9995

Candidate: batch_size=32, loss=huber, gamma=0.99, target_update_interval=500, epsilon_min=0.05.
Selection uses validation folds. Agent and buy and hold are shown separately.

## Fold Results

| Fold | Best Ep | Agent Val | B&H Val | Agent Sharpe | Agent MDD | B&H MDD | Agent Turnover | B&H Turnover | Val Return Std | 2026 Agent Test | 2026 B&H Test |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| val2023 | 35 | 34.63% | 41.39% | 1.881 | 9.61% | 10.08% | 115.624 | 0.999 | 12.14% | - | - |
| val2024 | 6 | 0.99% | -31.76% | 0.146 | 15.27% | 42.91% | 97.629 | 0.993 | 9.46% | - | - |
| val2025 | 65 | 124.23% | 126.99% | 3.061 | 12.95% | 14.64% | 57.032 | 0.996 | 28.64% | 97.22% | 115.60% |

## Aggregate

| Mean Agent Val | Mean B&H Val | Median Agent Val | Worst Agent Val | Mean Val Sharpe | Mean Agent MDD | Mean B&H MDD | Mean Agent Turnover | Mean B&H Turnover | Mean Val Return Std | Mean Best Ep | 2026 Agent Test | 2026 B&H Test | 2026 Agent Turnover |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 53.28% | 45.54% | 34.63% | 0.99% | 1.696 | 12.61% | 22.54% | 90.095 | 0.996 | 16.75% | 35.333 | 97.22% | 115.60% | 10.025 |
