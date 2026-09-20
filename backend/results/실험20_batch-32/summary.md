# 실험20_batch-32 batch_size=32

Selection uses validation folds. Agent and buy and hold are shown separately.

## Fold Results

| Fold | Best Ep | Agent Val | B&H Val | Agent Sharpe | Agent MDD | B&H MDD | Agent Turnover | B&H Turnover | 2026 Agent Test | 2026 B&H Test |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| val2023 | 26 | 37.46% | 41.39% | 1.995 | 8.75% | 10.08% | 96.737 | 0.999 | - | - |
| val2024 | 89 | 5.93% | -31.76% | 0.452 | 18.27% | 42.91% | 67.716 | 0.993 | - | - |
| val2025 | 96 | 154.31% | 126.99% | 3.139 | 14.20% | 14.64% | 34.941 | 0.996 | 123.95% | 115.60% |

## Aggregate

| Mean Agent Val | Mean B&H Val | Median Agent Val | Worst Agent Val | Mean Val Sharpe | Mean Agent MDD | Mean B&H MDD | Mean Agent Turnover | Mean B&H Turnover | 2026 Agent Test | 2026 B&H Test |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 65.90% | 45.54% | 37.46% | 5.93% | 1.862 | 13.74% | 22.54% | 66.465 | 0.996 | 123.95% | 115.60% |
