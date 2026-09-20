# 실험17_gamma-0p95 gamma=0.95

Selection uses validation folds. Buy and hold comparison is included for every fold.

## Fold Results

| Fold | Best Ep | Agent Val | B&H Val | Agent-B&H Val | Val Sharpe | Val MDD | Val Turnover | B&H Turnover | 2026 Agent Test | 2026 B&H Test | 2026 Agent-B&H |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| val2023 | 8 | 26.57% | 41.39% | -14.81% | 1.547 | 7.29% | 100.753 | 0.999 | - | - | - |
| val2024 | 32 | -12.54% | -31.76% | 19.22% | -0.767 | 17.04% | 92.554 | 0.993 | - | - | - |
| val2025 | 26 | 79.89% | 126.99% | -47.10% | 2.484 | 11.82% | 93.721 | 0.996 | 14.02% | 115.60% | -101.59% |

## Aggregate

| Mean Agent Val | Mean B&H Val | Mean Agent-B&H Val | Median Agent-B&H Val | Worst Agent-B&H Val | Mean Val Sharpe | Mean Val MDD | Mean Val Turnover | 2026 Agent Test | 2026 B&H Test | 2026 Agent-B&H |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 31.31% | 45.54% | -14.23% | -14.81% | -47.10% | 1.088 | 12.05% | 95.676 | 14.02% | 115.60% | -101.59% |
