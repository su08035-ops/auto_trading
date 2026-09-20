# 실험30_hidden-256-action-5-multiseed dqn hidden-256 multi-seed walk-forward

Candidate: algorithm=dqn, action_values=[0.0, 0.25, 0.5, 0.75, 1.0], hidden_dim=256, batch_size=32, loss=huber, gamma=0.99, target_update_interval=500, epsilon_min=0.05, epsilon_decay=0.999.
Agent and buy and hold are shown separately.

## Seed Summary

| Seed | Mean Agent Val | Mean B&H Val | Median Agent Val | Worst Agent Val | Mean Agent Sharpe | Mean Agent MDD | Mean Agent Turnover | Mean Val Return Std | Mean Best Ep | 2026 Agent Test | 2026 B&H Test | 2026 Agent Turnover |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 48.46% | 45.54% | 39.86% | -7.83% | 1.182 | 10.90% | 38.595 | 14.45% | 31.667 | 68.41% | 115.60% | 14.346 |
| 7 | 55.80% | 45.54% | 35.93% | 6.06% | 1.647 | 10.65% | 31.739 | 16.54% | 67.000 | 109.03% | 115.60% | 2.990 |
| 21 | 55.78% | 45.54% | 32.02% | 5.06% | 1.903 | 6.88% | 62.319 | 15.46% | 44.667 | 74.18% | 115.60% | 27.540 |
| 42 | 55.98% | 45.54% | 46.30% | 1.84% | 1.647 | 14.19% | 54.068 | 15.19% | 37.000 | 79.19% | 115.60% | 20.515 |
| 100 | 49.93% | 45.54% | 29.80% | 5.13% | 1.603 | 10.45% | 46.642 | 15.31% | 67.000 | 126.47% | 115.60% | 11.384 |

## Overall

| Seeds | Mean Agent Val | Std Agent Val | Mean B&H Val | Mean Worst Agent Val | Mean Agent Sharpe | Mean Agent MDD | Mean Agent Turnover | Mean Val Return Std | Mean Best Ep | Mean 2026 Agent Test | Std 2026 Agent Test | Mean 2026 B&H Test | Mean 2026 Agent Turnover |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 53.19% | 3.69% | 45.54% | 2.05% | 1.596 | 10.61% | 46.673 | 15.39% | 49.467 | 91.46% | 25.07% | 115.60% | 15.355 |
