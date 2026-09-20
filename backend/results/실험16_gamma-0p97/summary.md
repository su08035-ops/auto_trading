# 실험16_gamma-0p97 gamma=0.97

Selection uses validation folds. Buy and hold comparison is included for every fold.

## Fold Results

| Fold | Best Ep | Agent Val | B&H Val | Agent-B&H Val | Val Sharpe | Val MDD | Val Turnover | B&H Turnover | 2026 Agent Test | 2026 B&H Test | 2026 Agent-B&H |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| val2023 | 4 | 25.79% | 41.39% | -15.59% | 1.796 | 6.60% | 91.746 | 0.999 | - | - | - |
| val2024 | 45 | 2.93% | -31.76% | 34.69% | 0.243 | 20.69% | 102.598 | 0.993 | - | - | - |
| val2025 | 64 | 86.54% | 126.99% | -40.45% | 2.808 | 7.70% | 134.680 | 0.996 | 13.31% | 115.60% | -102.30% |

## Aggregate

| Mean Agent Val | Mean B&H Val | Mean Agent-B&H Val | Median Agent-B&H Val | Worst Agent-B&H Val | Mean Val Sharpe | Mean Val MDD | Mean Val Turnover | 2026 Agent Test | 2026 B&H Test | 2026 Agent-B&H |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 38.42% | 45.54% | -7.12% | -15.59% | -40.45% | 1.616 | 11.66% | 109.675 | 13.31% | 115.60% | -102.30% |
