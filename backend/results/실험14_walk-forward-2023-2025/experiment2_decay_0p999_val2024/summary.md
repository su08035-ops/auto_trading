# Held-out test

Mode: train
Algorithm: dqn
Best episode: 96

| Strategy | Return | MDD | Sharpe | Sortino | Trades | Turnover | Avg Holding Days |
|---|---:|---:|---:|---:|---:|---:|---:|
| agent | 266.30% | 38.95% | 1.673 | 2.744 | 144 | 94.336 | 5.413 |
| cash | 0.00% | 0.00% | null | null | 0 | 0.000 | null |
| buy_and_hold | 390.84% | 42.87% | 1.885 | 3.004 | 1 | 0.996 | null |

Equity is marked at the final close without forced liquidation.
Zero-risk Sharpe/Sortino and no-closed-position holding time are undefined (null).

![Equity, drawdown and loss](results.png)
