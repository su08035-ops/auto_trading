# 실험15_gamma-0p98 gamma=0.98

Selection uses validation folds. Buy and hold comparison is included for every fold.

## Fold Results

| Fold | Best Ep | Agent Val | B&H Val | Agent-B&H Val | Val Sharpe | Val MDD | Val Turnover | B&H Turnover | 2026 Agent Test | 2026 B&H Test | 2026 Agent-B&H |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| val2023 | 24 | 25.54% | 41.39% | -15.85% | 1.542 | 9.22% | 112.668 | 0.999 | - | - | - |
| val2024 | 50 | -1.56% | -31.76% | 30.21% | 0.024 | 22.20% | 99.835 | 0.993 | - | - | - |
| val2025 | 18 | 83.30% | 126.99% | -43.69% | 2.439 | 16.22% | 112.688 | 0.996 | 25.99% | 115.60% | -89.61% |

## Aggregate

| Mean Agent Val | Mean B&H Val | Mean Agent-B&H Val | Median Agent-B&H Val | Worst Agent-B&H Val | Mean Val Sharpe | Mean Val MDD | Mean Val Turnover | 2026 Agent Test | 2026 B&H Test | 2026 Agent-B&H |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 35.76% | 45.54% | -9.78% | -15.85% | -43.69% | 1.335 | 15.88% | 108.397 | 25.99% | 115.60% | -89.61% |
