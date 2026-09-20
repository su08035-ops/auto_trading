# Held-out test

Mode: train
Algorithm: dqn
Best episode: 29

| Strategy | Return | MDD | Sharpe | Sortino | Trades | Turnover | Avg Holding Days |
|---|---:|---:|---:|---:|---:|---:|---:|
| agent | 97.37% | 41.29% | 1.672 | 2.517 | 11 | 9.950 | 34.092 |
| cash | 0.00% | 0.00% | null | null | 0 | 0.000 | null |
| buy_and_hold | 115.60% | 42.87% | 1.736 | 2.741 | 1 | 0.998 | null |

Equity is marked at the final close without forced liquidation.
Zero-risk Sharpe/Sortino and no-closed-position holding time are undefined (null).

![Equity, drawdown and loss](results.png)
