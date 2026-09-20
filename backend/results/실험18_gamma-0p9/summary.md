# 실험18_gamma-0p9 gamma=0.9

Selection uses validation folds. Buy and hold comparison is included for every fold.

## Fold Results

| Fold | Best Ep | Agent Val | B&H Val | Agent-B&H Val | Val Sharpe | Val MDD | Val Turnover | B&H Turnover | 2026 Agent Test | 2026 B&H Test | 2026 Agent-B&H |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| val2023 | 4 | 28.81% | 41.39% | -12.58% | 1.691 | 8.33% | 90.805 | 0.999 | - | - | - |
| val2024 | 43 | -13.72% | -31.76% | 18.05% | -0.822 | 16.68% | 93.624 | 0.993 | - | - | - |
| val2025 | 36 | 62.69% | 126.99% | -64.31% | 1.984 | 15.07% | 108.630 | 0.996 | 64.28% | 115.60% | -51.32% |

## Aggregate

| Mean Agent Val | Mean B&H Val | Mean Agent-B&H Val | Median Agent-B&H Val | Worst Agent-B&H Val | Mean Val Sharpe | Mean Val MDD | Mean Val Turnover | 2026 Agent Test | 2026 B&H Test | 2026 Agent-B&H |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 25.93% | 45.54% | -19.61% | -12.58% | -64.31% | 0.951 | 13.36% | 97.687 | 64.28% | 115.60% | -51.32% |
