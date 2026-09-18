"""Evaluate one frozen policy and the configured benchmarks on identical dates."""

from pathlib import Path
import json
import pandas as pd

from backtest.backtester import run_backtest


def evaluate(agent, feature_df, config):
    strategies = ["agent", *config["evaluation"]["benchmarks"]]
    if len(set(strategies)) != len(strategies):
        raise ValueError("Benchmark names must be unique.")
    return {
        name: run_backtest(
            feature_df, config, agent=agent if name == "agent" else None, strategy=name
        )
        for name in strategies
    }


def save_evaluation(results, output_dir, metric_names=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = {name: result.metrics for name, result in results.items()}
    (output_dir / "metrics.json").write_text(
        json.dumps(summaries, indent=2, allow_nan=False), encoding="utf-8"
    )
    table = pd.DataFrame.from_dict(summaries, orient="index")
    if metric_names is not None:
        unknown = set(metric_names) - set(table.columns)
        if unknown:
            raise ValueError(f"Unsupported report metrics: {sorted(unknown)}")
        table = table[list(metric_names)]
    table.to_csv(output_dir / "metrics.csv", index_label="strategy")
    pd.concat([result.equity for result in results.values()], axis=1).to_csv(
        output_dir / "equity.csv", index_label="date"
    )
    for name, result in results.items():
        result.trades.to_csv(output_dir / f"{name}_trades.csv", index=False)


def plot_results(results, history, output_path, *, synthetic=False):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    from backtest.metrics import drawdown

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), layout="constrained")
    colors = {"agent": "#167348", "cash": "#555555", "buy_and_hold": "#ae3156"}
    for name, result in results.items():
        label = name.replace("_", " ").title()
        axes[0].plot(
            result.equity.index,
            result.equity / result.equity.iloc[0],
            label=label,
            color=colors[name],
        )
        axes[1].plot(
            result.equity.index,
            drawdown(result.equity) * 100,
            label=label,
            color=colors[name],
        )
    axes[0].set_ylabel("Equity / initial equity")
    axes[0].legend()
    axes[1].set_ylabel("Drawdown (%)")
    if len(history):
        axes[2].plot(
            history["episode"], history["mean_loss"], color="#167348", marker="o"
        )
    axes[2].set_xlabel("Training episode")
    axes[2].xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[2].set_ylabel("Mean training loss")
    for axis in axes:
        axis.grid(alpha=0.2)
    label = (
        "SYNTHETIC DATA - execution check only" if synthetic else "Held-out test period"
    )
    fig.suptitle(label)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
