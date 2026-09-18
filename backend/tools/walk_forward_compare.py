"""Run walk-forward validation for selected experiment variants."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import date
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.loader import load_config  # noqa: E402
from main import PROJECT_ROOT, next_experiment_number, run_experiment  # noqa: E402


FOLDS = (
    ("val2023", "2023-01-01", "2024-01-01"),
    ("val2024", "2024-01-01", "2025-01-01"),
    ("val2025", "2025-01-01", "2026-01-01"),
)

VARIANTS = (
    (
        "experiment2_decay_0p999",
        "실험2기반",
        {
            "loss": "huber",
            "learning_rate": 0.001,
            "epsilon_min": 0.05,
            "epsilon_decay": 0.999,
        },
    ),
    (
        "experiment12_decay_0p9999",
        "실험12기반",
        {
            "loss": "huber",
            "learning_rate": 0.001,
            "epsilon_min": 0.05,
            "epsilon_decay": 0.9999,
        },
    ),
)


def percent(value):
    if pd.isna(value):
        return "기록 없음"
    return f"{value:.2%}"


def number(value):
    if pd.isna(value):
        return "기록 없음"
    return f"{value:.3f}"


def fold_config(base_config, *, validation_start, split_date, end_date, agent_updates):
    config = deepcopy(base_config)
    config["data"]["validation_start"] = validation_start
    config["data"]["split_date"] = split_date
    config["data"]["end_date"] = end_date
    config["action"]["values"] = [0.0, 0.5, 1.0]
    config["agent"].update(agent_updates)
    return config


def read_run_summary(run_dir):
    history = pd.read_csv(run_dir / "training_history.csv")
    ret_col = (
        "validation_cumulative_return"
        if "validation_cumulative_return" in history.columns
        else "validation_return"
    )
    best_episode = int(history["best_episode"].iloc[-1])
    best = history.loc[history["episode"].eq(best_episode)].iloc[-1]
    test = pd.read_csv(run_dir / "test" / "metrics.csv").set_index("strategy").loc[
        "agent"
    ]
    return {
        "best_episode": best_episode,
        "best_validation_return": best.get(ret_col, float("nan")),
        "best_validation_sharpe": best.get("validation_sharpe_ratio", float("nan")),
        "best_validation_mdd": best.get("validation_mdd", float("nan")),
        "best_validation_turnover": best.get("validation_turnover", float("nan")),
        "validation_return_std": history[ret_col].std(),
        "validation_sharpe_std": (
            history["validation_sharpe_ratio"].std()
            if "validation_sharpe_ratio" in history.columns
            else float("nan")
        ),
        "test_return": test.get("cumulative_return", float("nan")),
        "test_sharpe": test.get("sharpe_ratio", float("nan")),
        "test_mdd": test.get("mdd", float("nan")),
        "test_turnover": test.get("turnover", float("nan")),
    }


def write_markdown(root_dir, fold_rows, aggregate_rows):
    lines = [
        "# Walk-forward validation",
        "",
        "Selection uses validation folds only. The 2026 test columns are diagnostic and are taken from the val2025 fold's held-out test period.",
        "",
        "## Fold Results",
        "",
        "| Variant | Fold | Validation | Best Ep | Best Val Return | Best Val Sharpe | Best Val MDD | Best Val Turnover | Val Return Std | 2026 Test Return |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in fold_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["variant_label"],
                    row["fold"],
                    f"{row['validation_start']}~{row['validation_end']}",
                    str(row["best_episode"]),
                    percent(row["best_validation_return"]),
                    number(row["best_validation_sharpe"]),
                    percent(row["best_validation_mdd"]),
                    number(row["best_validation_turnover"]),
                    percent(row["validation_return_std"]),
                    percent(row["test_return"]) if row["fold"] == "val2025" else "-",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Aggregate",
            "",
            "| Variant | Mean Val Return | Median Val Return | Worst Val Return | Mean Val Sharpe | Mean Val MDD | Mean Val Turnover | 2026 Test Return | 2026 Test Sharpe |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in aggregate_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["variant_label"],
                    percent(row["mean_validation_return"]),
                    percent(row["median_validation_return"]),
                    percent(row["worst_validation_return"]),
                    number(row["mean_validation_sharpe"]),
                    percent(row["mean_validation_mdd"]),
                    number(row["mean_validation_turnover"]),
                    percent(row["test_2026_return"]),
                    number(row["test_2026_sharpe"]),
                ]
            )
            + " |"
        )
    (root_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def aggregate(fold_rows):
    frame = pd.DataFrame(fold_rows)
    rows = []
    for variant, group in frame.groupby("variant"):
        test_2026 = group.loc[group["fold"].eq("val2025")].iloc[-1]
        rows.append(
            {
                "variant": variant,
                "variant_label": group["variant_label"].iloc[0],
                "mean_validation_return": group["best_validation_return"].mean(),
                "median_validation_return": group["best_validation_return"].median(),
                "worst_validation_return": group["best_validation_return"].min(),
                "mean_validation_sharpe": group["best_validation_sharpe"].mean(),
                "mean_validation_mdd": group["best_validation_mdd"].mean(),
                "mean_validation_turnover": group["best_validation_turnover"].mean(),
                "test_2026_return": test_2026["test_return"],
                "test_2026_sharpe": test_2026["test_sharpe"],
                "test_2026_mdd": test_2026["test_mdd"],
                "test_2026_turnover": test_2026["test_turnover"],
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--end-date", default=date.today().isoformat())
    parser.add_argument("--output-dir")
    args = parser.parse_args()

    base_config = load_config()
    root = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else PROJECT_ROOT
        / "results"
        / f"실험{next_experiment_number(PROJECT_ROOT / 'results')}_walk-forward-2023-2025"
    )
    root.mkdir(parents=True, exist_ok=False)

    fold_rows = []
    for variant, variant_label, agent_updates in VARIANTS:
        for fold, validation_start, split_date in FOLDS:
            run_dir = root / f"{variant}_{fold}"
            print(
                f"Running {variant_label} {fold}: validation={validation_start}..{split_date}, test starts={split_date}",
                flush=True,
            )
            config = fold_config(
                base_config,
                validation_start=validation_start,
                split_date=split_date,
                end_date=args.end_date,
                agent_updates=agent_updates,
            )
            run_experiment(config, output_dir=run_dir)
            row = read_run_summary(run_dir)
            row.update(
                {
                    "variant": variant,
                    "variant_label": variant_label,
                    "fold": fold,
                    "validation_start": validation_start,
                    "validation_end": split_date,
                    "run_dir": str(run_dir),
                }
            )
            fold_rows.append(row)

    aggregate_rows = aggregate(fold_rows)
    pd.DataFrame(fold_rows).to_csv(root / "fold_results.csv", index=False)
    pd.DataFrame(aggregate_rows).to_csv(root / "aggregate_results.csv", index=False)
    write_markdown(root, fold_rows, aggregate_rows)
    print(f"Complete: {root}")


if __name__ == "__main__":
    main()
