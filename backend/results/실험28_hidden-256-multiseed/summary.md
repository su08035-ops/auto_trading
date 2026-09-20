# Experiment 28 hidden-256 multi-seed walk-forward

Candidate: hidden_dim=256, batch_size=32, loss=huber, gamma=0.99, target_update_interval=500, epsilon_min=0.05, epsilon_decay=0.999.
Agent and buy and hold are shown separately.

## Seed Summary

| Seed | Mean Agent Val | Mean B&H Val | Median Agent Val | Worst Agent Val | Mean Agent Sharpe | Mean Agent MDD | Mean Agent Turnover | Mean Val Return Std | Mean Best Ep | 2026 Agent Test | 2026 B&H Test | 2026 Agent Turnover |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 50.33% | 45.54% | 27.28% | 10.65% | 1.624 | 11.95% | 68.824 | 15.00% | 80.000 | 57.70% | 115.60% | 15.230 |
| 7 | 50.52% | 45.54% | 39.87% | 8.13% | 1.633 | 12.16% | 56.480 | 15.08% | 27.333 | 97.37% | 115.60% | 9.950 |
| 21 | 48.82% | 45.54% | 28.93% | 6.54% | 1.638 | 10.51% | 61.843 | 15.51% | 52.333 | 65.47% | 115.60% | 20.876 |
| 42 | 52.77% | 45.54% | 39.26% | 9.11% | 1.786 | 10.51% | 47.860 | 16.17% | 76.333 | 117.72% | 115.60% | 15.912 |
| 100 | 52.55% | 45.54% | 38.54% | 4.31% | 1.643 | 10.29% | 48.523 | 14.50% | 45.000 | 113.79% | 115.60% | 1.997 |

## Overall

| Seeds | Mean Agent Val | Std Agent Val | Mean B&H Val | Mean Worst Agent Val | Mean Agent Sharpe | Mean Agent MDD | Mean Agent Turnover | Mean Val Return Std | Mean Best Ep | Mean 2026 Agent Test | Std 2026 Agent Test | Mean 2026 B&H Test | Mean 2026 Agent Turnover |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 51.00% | 1.65% | 45.54% | 7.75% | 1.665 | 11.08% | 56.706 | 15.25% | 56.200 | 90.41% | 27.54% | 115.60% | 12.793 |
