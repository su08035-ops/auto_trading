# Held-out test

Mode: train
Algorithm: double_dqn
Best episode: 50

| Strategy | Return | MDD | Sharpe | Sortino | Trades | Turnover | Avg Holding Days |
|---|---:|---:|---:|---:|---:|---:|---:|
| agent | 135.48% | 37.38% | 1.920 | 3.084 | 9 | 8.946 | 36.485 |
| cash | 0.00% | 0.00% | null | null | 0 | 0.000 | null |
| buy_and_hold | 115.60% | 42.87% | 1.736 | 2.741 | 1 | 0.998 | null |

Equity is marked at the final close without forced liquidation.
Zero-risk Sharpe/Sortino and no-closed-position holding time are undefined (null).

![Equity, drawdown and loss](results.png)
