# Walk-forward validation

Selection uses validation folds only. The 2026 test columns are diagnostic and are taken from the val2025 fold's held-out test period.

## Fold Results

| Variant | Fold | Validation | Best Ep | Best Val Return | Best Val Sharpe | Best Val MDD | Best Val Turnover | Val Return Std | 2026 Test Return |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 실험2기반 | val2023 | 2023-01-01~2024-01-01 | 5 | 34.00% | 1.896 | 8.39% | 94.740 | 9.75% | - |
| 실험2기반 | val2024 | 2024-01-01~2025-01-01 | 96 | 3.88% | 0.292 | 14.60% | 96.309 | 10.49% | - |
| 실험2기반 | val2025 | 2025-01-01~2026-01-01 | 9 | 100.01% | 2.875 | 9.44% | 82.795 | 23.85% | 10.80% |
| 실험12기반 | val2023 | 2023-01-01~2024-01-01 | 21 | 40.18% | 2.073 | 12.01% | 100.693 | 9.39% | - |
| 실험12기반 | val2024 | 2024-01-01~2025-01-01 | 3 | -6.08% | -0.156 | 20.38% | 102.097 | 8.01% | - |
| 실험12기반 | val2025 | 2025-01-01~2026-01-01 | 32 | 94.27% | 2.850 | 11.10% | 116.658 | 20.50% | 77.75% |

## Aggregate

| Variant | Mean Val Return | Median Val Return | Worst Val Return | Mean Val Sharpe | Mean Val MDD | Mean Val Turnover | 2026 Test Return | 2026 Test Sharpe |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 실험12기반 | 42.79% | 40.18% | -6.08% | 1.589 | 14.50% | 106.483 | 77.75% | 1.447 |
| 실험2기반 | 45.96% | 34.00% | 3.88% | 1.688 | 10.81% | 91.281 | 10.80% | 0.552 |
