"""Run multi-seed walk-forward validation for the batch-32 candidate."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import date
from pathlib import Path
import re
import sys

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest.backtester import run_backtest  # noqa: E402
from data.features import get_market_feature_names, make_features  # noqa: E402
from data.loader import load_config  # noqa: E402
from data.preprocessing import clean_ohlcv, prepare_splits  # noqa: E402
from main import PROJECT_ROOT, next_experiment_number, run_experiment  # noqa: E402


FOLDS = (
    ("val2023", "2023-01-01", "2024-01-01"),
    ("val2024", "2024-01-01", "2025-01-01"),
    ("val2025", "2025-01-01", "2026-01-01"),
)


def percent(value):
    if pd.isna(value):
        return "기록 없음"
    return f"{value:.2%}"


def number(value):
    if pd.isna(value):
        return "기록 없음"
    return f"{value:.3f}"


def progress_filter(message):
    if message.startswith(("Loading", "train:", "validation:", "test:", "Complete:")):
        print(message, flush=True)
        return
    match = re.match(r"Episode (\d+)/(\d+).*validation return=([^|]+).*best=(\d+)", message)
    if match:
        episode = int(match.group(1))
        total = int(match.group(2))
        if episode in {1, total} or episode % 25 == 0:
            print(message, flush=True)


def fold_config(base_config, *, seed, validation_start, split_date, end_date):
    config = deepcopy(base_config)
    config["data"]["validation_start"] = validation_start
    config["data"]["split_date"] = split_date
    config["data"]["end_date"] = end_date
    config["action"]["values"] = [0.0, 0.5, 1.0]
    config["agent"].update(
        loss="huber",
        learning_rate=0.001,
        gamma=0.99,
        batch_size=32,
        epsilon_min=0.05,
        epsilon_decay=0.999,
    )
    config["training"]["seed"] = seed
    return config


def benchmark_validation_metrics(run_dir):
    config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    raw = pd.read_csv(run_dir / "ohlcv.csv", parse_dates=["date"]).set_index("date")
    features = make_features(
        clean_ohlcv(raw), selected_features=get_market_feature_names(config)
    )
    splits = prepare_splits(features, config)
    return run_backtest(splits["validation"], config, strategy="buy_and_hold").metrics


def read_run_summary(run_dir):
    history = pd.read_csv(run_dir / "training_history.csv")
    ret_col = (
        "validation_cumulative_return"
        if "validation_cumulative_return" in history.columns
        else "validation_return"
    )
    best_episode = int(history["best_episode"].iloc[-1])
    best = history.loc[history["episode"].eq(best_episode)].iloc[-1]
    test = pd.read_csv(run_dir / "test" / "metrics.csv").set_index("strategy")
    bh_validation = benchmark_validation_metrics(run_dir)
    agent_test = test.loc["agent"]
    bh_test = test.loc["buy_and_hold"]
    return {
        "best_episode": best_episode,
        "agent_validation_return": best.get(ret_col, float("nan")),
        "agent_validation_sharpe": best.get("validation_sharpe_ratio", float("nan")),
        "agent_validation_mdd": best.get("validation_mdd", float("nan")),
        "agent_validation_turnover": best.get("validation_turnover", float("nan")),
        "validation_return_std": history[ret_col].std(),
        "bh_validation_return": bh_validation.get("cumulative_return", float("nan")),
        "bh_validation_mdd": bh_validation.get("mdd", float("nan")),
        "bh_validation_turnover": bh_validation.get("turnover", float("nan")),
        "agent_test_return": agent_test.get("cumulative_return", float("nan")),
        "agent_test_sharpe": agent_test.get("sharpe_ratio", float("nan")),
        "agent_test_mdd": agent_test.get("mdd", float("nan")),
        "agent_test_turnover": agent_test.get("turnover", float("nan")),
        "bh_test_return": bh_test.get("cumulative_return", float("nan")),
        "bh_test_sharpe": bh_test.get("sharpe_ratio", float("nan")),
        "bh_test_mdd": bh_test.get("mdd", float("nan")),
        "bh_test_turnover": bh_test.get("turnover", float("nan")),
    }


def seed_aggregates(fold_rows):
    frame = pd.DataFrame(fold_rows)
    rows = []
    for seed, group in frame.groupby("seed", sort=True):
        test_2026 = group.loc[group["fold"].eq("val2025")].iloc[-1]
        rows.append(
            {
                "seed": seed,
                "mean_agent_validation_return": group["agent_validation_return"].mean(),
                "mean_bh_validation_return": group["bh_validation_return"].mean(),
                "median_agent_validation_return": group["agent_validation_return"].median(),
                "worst_agent_validation_return": group["agent_validation_return"].min(),
                "mean_agent_validation_sharpe": group["agent_validation_sharpe"].mean(),
                "mean_agent_validation_mdd": group["agent_validation_mdd"].mean(),
                "mean_agent_validation_turnover": group[
                    "agent_validation_turnover"
                ].mean(),
                "mean_bh_validation_turnover": group["bh_validation_turnover"].mean(),
                "mean_best_episode": group["best_episode"].mean(),
                "test_2026_agent_return": test_2026["agent_test_return"],
                "test_2026_bh_return": test_2026["bh_test_return"],
                "test_2026_agent_sharpe": test_2026["agent_test_sharpe"],
                "test_2026_agent_turnover": test_2026["agent_test_turnover"],
            }
        )
    return rows


def overall_aggregate(seed_rows):
    frame = pd.DataFrame(seed_rows)
    return {
        "seed_count": len(frame),
        "mean_agent_validation_return": frame["mean_agent_validation_return"].mean(),
        "std_agent_validation_return": frame["mean_agent_validation_return"].std(),
        "mean_bh_validation_return": frame["mean_bh_validation_return"].mean(),
        "mean_worst_agent_validation_return": frame[
            "worst_agent_validation_return"
        ].mean(),
        "mean_agent_validation_mdd": frame["mean_agent_validation_mdd"].mean(),
        "mean_agent_validation_turnover": frame[
            "mean_agent_validation_turnover"
        ].mean(),
        "mean_best_episode": frame["mean_best_episode"].mean(),
        "mean_test_2026_agent_return": frame["test_2026_agent_return"].mean(),
        "std_test_2026_agent_return": frame["test_2026_agent_return"].std(),
        "mean_test_2026_bh_return": frame["test_2026_bh_return"].mean(),
        "mean_test_2026_agent_turnover": frame["test_2026_agent_turnover"].mean(),
    }


def write_markdown(root_dir, seed_rows, aggregate):
    lines = [
        "# Experiment 21 multi-seed walk-forward",
        "",
        "Candidate: batch_size=32, loss=huber, gamma=0.99, epsilon_min=0.05, epsilon_decay=0.999.",
        "Agent and buy and hold are shown separately.",
        "",
        "## Seed Summary",
        "",
        "| Seed | Mean Agent Val | Mean B&H Val | Median Agent Val | Worst Agent Val | Mean Agent MDD | Mean Agent Turnover | Mean Best Ep | 2026 Agent Test | 2026 B&H Test | 2026 Agent Turnover |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in seed_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(int(row["seed"])),
                    percent(row["mean_agent_validation_return"]),
                    percent(row["mean_bh_validation_return"]),
                    percent(row["median_agent_validation_return"]),
                    percent(row["worst_agent_validation_return"]),
                    percent(row["mean_agent_validation_mdd"]),
                    number(row["mean_agent_validation_turnover"]),
                    number(row["mean_best_episode"]),
                    percent(row["test_2026_agent_return"]),
                    percent(row["test_2026_bh_return"]),
                    number(row["test_2026_agent_turnover"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Overall",
            "",
            "| Seeds | Mean Agent Val | Std Agent Val | Mean B&H Val | Mean Worst Agent Val | Mean Agent MDD | Mean Agent Turnover | Mean Best Ep | Mean 2026 Agent Test | Std 2026 Agent Test | Mean 2026 B&H Test | Mean 2026 Agent Turnover |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            "| "
            + " | ".join(
                [
                    str(int(aggregate["seed_count"])),
                    percent(aggregate["mean_agent_validation_return"]),
                    percent(aggregate["std_agent_validation_return"]),
                    percent(aggregate["mean_bh_validation_return"]),
                    percent(aggregate["mean_worst_agent_validation_return"]),
                    percent(aggregate["mean_agent_validation_mdd"]),
                    number(aggregate["mean_agent_validation_turnover"]),
                    number(aggregate["mean_best_episode"]),
                    percent(aggregate["mean_test_2026_agent_return"]),
                    percent(aggregate["std_test_2026_agent_return"]),
                    percent(aggregate["mean_test_2026_bh_return"]),
                    number(aggregate["mean_test_2026_agent_turnover"]),
                ]
            )
            + " |",
        ]
    )
    (root_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--end-date", default=date.today().isoformat())
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 7, 21, 42, 100])
    args = parser.parse_args()

    base_config = load_config()
    results_root = PROJECT_ROOT / "results"
    experiment = f"실험{next_experiment_number(results_root)}_batch-32-multiseed"
    root_dir = results_root / experiment
    root_dir.mkdir(parents=True, exist_ok=False)
    print(f"=== {experiment} ===", flush=True)

    fold_rows = []
    for seed in args.seeds:
        for fold, validation_start, split_date in FOLDS:
            run_dir = root_dir / f"seed-{seed}_{fold}"
            print(
                f"Running seed={seed} {fold}: validation={validation_start}..{split_date}, test starts={split_date}",
                flush=True,
            )
            config = fold_config(
                base_config,
                seed=seed,
                validation_start=validation_start,
                split_date=split_date,
                end_date=args.end_date,
            )
            run_experiment(config, output_dir=run_dir, progress=progress_filter)
            row = read_run_summary(run_dir)
            row.update(
                {
                    "experiment": experiment,
                    "seed": seed,
                    "fold": fold,
                    "validation_start": validation_start,
                    "validation_end": split_date,
                    "run_dir": str(run_dir),
                }
            )
            fold_rows.append(row)

    seed_rows = seed_aggregates(fold_rows)
    aggregate = overall_aggregate(seed_rows)
    pd.DataFrame(fold_rows).to_csv(root_dir / "fold_results.csv", index=False)
    pd.DataFrame(seed_rows).to_csv(root_dir / "seed_results.csv", index=False)
    pd.DataFrame([aggregate]).to_csv(root_dir / "aggregate_results.csv", index=False)
    write_markdown(root_dir, seed_rows, aggregate)
    print(f"Complete: {root_dir}", flush=True)


if __name__ == "__main__":
    main()
