# 실험29_double-dqn-hidden-256-multiseed double_dqn hidden-256 multi-seed walk-forward

Candidate: algorithm=double_dqn, hidden_dim=256, batch_size=32, loss=huber, gamma=0.99, target_update_interval=500, epsilon_min=0.05, epsilon_decay=0.999.
Agent and buy and hold are shown separately.

## Seed Summary

| Seed | Mean Agent Val | Mean B&H Val | Median Agent Val | Worst Agent Val | Mean Agent Sharpe | Mean Agent MDD | Mean Agent Turnover | Mean Val Return Std | Mean Best Ep | 2026 Agent Test | 2026 B&H Test | 2026 Agent Turnover |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 61.53% | 45.54% | 37.51% | 3.35% | 1.649 | 13.59% | 47.266 | 16.23% | 53.667 | 94.77% | 115.60% | 13.047 |
| 7 | 40.17% | 45.54% | 23.95% | 23.08% | 1.549 | 13.23% | 84.736 | 13.48% | 47.667 | 118.82% | 115.60% | 6.979 |
| 21 | 60.69% | 45.54% | 40.49% | 14.30% | 1.796 | 14.19% | 62.891 | 16.96% | 37.333 | 135.48% | 115.60% | 8.946 |
| 42 | 56.77% | 45.54% | 41.39% | 12.16% | 1.858 | 12.47% | 75.015 | 16.95% | 75.333 | 28.09% | 115.60% | 15.075 |
| 100 | 47.92% | 45.54% | 28.94% | 1.39% | 1.619 | 10.33% | 85.115 | 15.61% | 51.667 | 134.42% | 115.60% | 13.956 |

## Overall

| Seeds | Mean Agent Val | Std Agent Val | Mean B&H Val | Mean Worst Agent Val | Mean Agent Sharpe | Mean Agent MDD | Mean Agent Turnover | Mean Val Return Std | Mean Best Ep | Mean 2026 Agent Test | Std 2026 Agent Test | Mean 2026 B&H Test | Mean 2026 Agent Turnover |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 53.42% | 9.16% | 45.54% | 10.86% | 1.694 | 12.76% | 71.004 | 15.85% | 53.133 | 102.32% | 44.63% | 115.60% | 11.600 |
