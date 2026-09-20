# Held-out test

Mode: train
Algorithm: dqn
Best episode: 84

| Strategy | Return | MDD | Sharpe | Sortino | Trades | Turnover | Avg Holding Days |
|---|---:|---:|---:|---:|---:|---:|---:|
| agent | 189.65% | 36.25% | 1.187 | 1.911 | 388 | 123.643 | 7.213 |
| cash | 0.00% | 0.00% | null | null | 0 | 0.000 | null |
| buy_and_hold | 230.24% | 42.91% | 1.132 | 1.767 | 1 | 0.993 | null |

Equity is marked at the final close without forced liquidation.
Zero-risk Sharpe/Sortino and no-closed-position holding time are undefined (null).

![Equity, drawdown and loss](results.png)
