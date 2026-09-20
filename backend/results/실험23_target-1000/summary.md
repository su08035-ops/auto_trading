# 실험23_target-1000 target_update_interval=1000

Candidate: batch_size=32, loss=huber, gamma=0.99, epsilon_min=0.05, epsilon_decay=0.999.
Selection uses validation folds. Agent and buy and hold are shown separately.

## Fold Results

| Fold | Best Ep | Agent Val | B&H Val | Agent Sharpe | Agent MDD | B&H MDD | Agent Turnover | B&H Turnover | Val Return Std | 2026 Agent Test | 2026 B&H Test |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| val2023 | 37 | 39.16% | 41.39% | 1.971 | 9.13% | 10.08% | 98.785 | 0.999 | 10.38% | - | - |
| val2024 | 89 | 0.82% | -31.76% | 0.129 | 10.72% | 42.91% | 65.710 | 0.993 | 10.45% | - | - |
| val2025 | 50 | 141.28% | 126.99% | 3.199 | 10.77% | 14.64% | 79.801 | 0.996 | 26.58% | 128.02% | 115.60% |

## Aggregate

| Mean Agent Val | Mean B&H Val | Median Agent Val | Worst Agent Val | Mean Val Sharpe | Mean Agent MDD | Mean B&H MDD | Mean Agent Turnover | Mean B&H Turnover | Mean Val Return Std | Mean Best Ep | 2026 Agent Test | 2026 B&H Test | 2026 Agent Turnover |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 60.42% | 45.54% | 39.16% | 0.82% | 1.766 | 10.21% | 22.54% | 81.432 | 0.996 | 15.80% | 58.667 | 128.02% | 115.60% | 10.996 |
