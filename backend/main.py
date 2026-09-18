"""CLI: load YAML -> data -> features -> train -> validation selection -> test."""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import re

import pandas as pd
import torch
import yaml

from data.loader import DEFAULT_CONFIG_PATH, PROJECT_ROOT, load_config, load_data
from data.preprocessing import clean_ohlcv, prepare_splits
from data.features import get_market_feature_names, make_features
from training.train import train, load_checkpoint
from training.evaluate import evaluate, save_evaluation, plot_results
from tools.export_frontend_runs import export_frontend_runs


def validate_config(config):
    """Fail before data download if an execution-critical setting is missing."""
    required = {
        "data": ("source", "start_date", "validation_start", "split_date", "end_date"),
        "agent": (
            "algorithm",
            "hidden_dim",
            "optimizer",
            "loss",
            "gamma",
            "learning_rate",
            "batch_size",
            "replay_buffer_size",
            "target_update_interval",
            "epsilon_start",
            "epsilon_min",
            "epsilon_decay",
        ),
        "training": (
            "episodes",
            "seed",
            "learning_starts",
            "train_frequency",
            "gradient_steps",
            "torch_threads",
            "selection_metric",
        ),
        "evaluation": (
            "benchmarks",
            "metrics",
            "periods_per_year",
            "annual_risk_free_rate",
        ),
        "paths": ("runs_dir", "model_file"),
    }
    for section, names in required.items():
        for name in names:
            if name not in config.get(section, {}):
                raise ValueError(f"Missing configuration: {section}.{name}")
    dates = [
        pd.Timestamp(config["data"][name])
        for name in ("start_date", "validation_start", "split_date", "end_date")
    ]
    if (
        any(pd.isna(date) for date in dates)
        or not dates[0] < dates[1] < dates[2] <= dates[3]
    ):
        raise ValueError(
            "Require start_date < validation_start < split_date <= end_date."
        )
    if config["broker"]["mode"] != "paper":
        raise ValueError(
            "This experiment CLI uses local paper execution. Broker API orders use a separate adapter."
        )


def file_digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def slug_value(value):
    text = str(value).lower()
    text = text.replace(".", "p").replace("-", "m")
    text = re.sub(r"[^a-z0-9_]+", "-", text)
    return text.strip("-")


def action_values_label(values):
    rounded = [round(float(value), 10) for value in values]
    presets = {
        tuple(round(index / 20, 10) for index in range(21)): "action-5pct",
        tuple(round(index / 10, 10) for index in range(11)): "action-10pct",
        (0.0, 0.25, 0.5, 0.75, 1.0): "action-25pct",
        (0.0, 1.0): "action-0p0-1p0",
    }
    preset = presets.get(tuple(rounded))
    if preset is not None:
        return preset
    action_values = "-".join(slug_value(value) for value in values)
    return f"action-{action_values}"


def experiment_label(config):
    agent = config["agent"]
    training = config["training"]
    environment = config["environment"]
    action = config["action"]
    defaults = {
        "algorithm": "dqn",
        "loss": "mse",
        "learning_rate": 0.001,
        "selection_metric": "cumulative_return",
        "seed": 42,
        "min_holding_days": 0,
        "action_values": [0.0, 0.5, 1.0],
        "gamma": 0.99,
        "hidden_dim": 128,
        "batch_size": 64,
        "target_update_interval": 500,
        "epsilon_min": 0.05,
        "epsilon_decay": 0.999,
    }
    values = {
        "algorithm": agent["algorithm"],
        "loss": agent["loss"],
        "learning_rate": agent["learning_rate"],
        "selection_metric": training["selection_metric"],
        "seed": training["seed"],
        "min_holding_days": environment["min_holding_days"],
        "action_values": action["values"],
        "gamma": agent["gamma"],
        "hidden_dim": agent["hidden_dim"],
        "batch_size": agent["batch_size"],
        "target_update_interval": agent["target_update_interval"],
        "epsilon_min": agent["epsilon_min"],
        "epsilon_decay": agent["epsilon_decay"],
    }
    changed = [name for name, value in values.items() if value != defaults[name]]
    if not changed:
        return "기본"
    if changed == ["loss"] and agent["loss"] == "huber":
        return "mse-to-huber"
    if changed == ["loss"] and agent["loss"] == "mae":
        return "huber-to-mae"
    parts = []
    if agent["algorithm"] != defaults["algorithm"]:
        parts.append(slug_value(agent["algorithm"]))
    if agent["loss"] != defaults["loss"]:
        parts.append(slug_value(agent["loss"]))
    if agent["learning_rate"] != defaults["learning_rate"]:
        parts.append(f"lr-{slug_value(agent['learning_rate'])}")
    if training["selection_metric"] != defaults["selection_metric"]:
        parts.append(f"metric-{slug_value(training['selection_metric'])}")
    if training["seed"] != defaults["seed"]:
        parts.append(f"seed-{slug_value(training['seed'])}")
    if environment["min_holding_days"] != defaults["min_holding_days"]:
        parts.append(f"minhold-{slug_value(environment['min_holding_days'])}")
    if action["values"] != defaults["action_values"]:
        parts.append(action_values_label(action["values"]))
    if agent["gamma"] != defaults["gamma"]:
        parts.append(f"gamma-{slug_value(agent['gamma'])}")
    if agent["hidden_dim"] != defaults["hidden_dim"]:
        parts.append(f"hidden-{slug_value(agent['hidden_dim'])}")
    if agent["batch_size"] != defaults["batch_size"]:
        parts.append(f"batch-{slug_value(agent['batch_size'])}")
    if agent["target_update_interval"] != defaults["target_update_interval"]:
        parts.append(f"target-{slug_value(agent['target_update_interval'])}")
    if agent["epsilon_min"] != defaults["epsilon_min"]:
        parts.append(f"epsmin-{slug_value(agent['epsilon_min'])}")
    if agent["epsilon_decay"] != defaults["epsilon_decay"]:
        parts.append(f"epsdecay-{slug_value(agent['epsilon_decay'])}")
    return "-".join(parts)


def next_experiment_number(root):
    numbers = []
    for path in root.iterdir() if root.exists() else []:
        match = re.match(r"실험(\d+)", path.name)
        if path.is_dir() and match:
            numbers.append(int(match.group(1)))
    return max(numbers, default=0) + 1


def new_run_directory(config, output_dir=None):
    if output_dir is not None:
        path = Path(output_dir).expanduser().resolve()
    else:
        root = Path(config["paths"]["runs_dir"])
        if not root.is_absolute():
            root = PROJECT_ROOT / root
        number = next_experiment_number(root)
        path = root / f"실험{number}_{experiment_label(config)}"
    path.mkdir(parents=True, exist_ok=False)
    return path


SUMMARY_METRIC_LABELS = {
    "cumulative_return": "Return",
    "mdd": "MDD",
    "sharpe_ratio": "Sharpe",
    "sortino_ratio": "Sortino",
    "trade_count": "Trades",
    "turnover": "Turnover",
    "average_holding_days": "Avg Holding Days",
    "initial_asset": "Initial Asset",
    "final_asset": "Final Asset",
    "open_shares": "Open Shares",
}


def format_summary_metric(name, value):
    if value is None:
        return "null"
    if name in {"cumulative_return", "mdd"}:
        return f"{value:.2%}"
    if name == "trade_count":
        return f"{value:d}"
    if name in {"sharpe_ratio", "sortino_ratio", "turnover", "average_holding_days"}:
        return f"{value:.3f}"
    return f"{value:.2f}" if isinstance(value, float) else str(value)


def build_summary_table(results, metric_names):
    labels = [SUMMARY_METRIC_LABELS.get(name, name) for name in metric_names]
    lines = [
        "| Strategy | " + " | ".join(labels) + " |",
        "|---" + "|---:" * len(metric_names) + "|",
    ]
    for name, result in results.items():
        metrics = result.metrics
        values = [
            format_summary_metric(metric_name, metrics.get(metric_name))
            for metric_name in metric_names
        ]
        lines.append("| " + name + " | " + " | ".join(values) + " |")
    return lines


def run_experiment(
    config, *, output_dir=None, checkpoint=None, mode="train", progress=print
):
    config = deepcopy(config)
    validate_config(config)
    if mode not in {"train", "evaluate", "paper"}:
        raise ValueError("mode must be train, evaluate or paper.")
    if mode != "train" and checkpoint is None:
        raise ValueError("Evaluation/paper mode requires --checkpoint.")
    trained_config = None
    if checkpoint is not None:
        agent, trained_config, saved = load_checkpoint(checkpoint)
        if mode == "train":
            raise ValueError("--checkpoint is for evaluation, not training resume.")
        # Never reinterpret saved weights using a different feature order or architecture.
        for section in ("state", "action", "agent", "environment", "reward"):
            config[section] = deepcopy(trained_config[section])
        if pd.Timestamp(config["data"]["split_date"]) < pd.Timestamp(
            trained_config["data"]["split_date"]
        ):
            raise ValueError(
                "Evaluation start cannot precede the checkpoint's held-out test boundary."
            )

    if progress:
        progress(
            f"Loading source={config['data']['source']} | algorithm={config['agent']['algorithm']}"
        )
    raw = load_data(config=config)
    cleaned = clean_ohlcv(raw)
    if cleaned.empty:
        raise ValueError("No valid OHLCV rows were returned.")
    features = make_features(
        cleaned, selected_features=get_market_feature_names(config)
    )
    splits = prepare_splits(features, config)
    run_dir = new_run_directory(config, output_dir)
    raw_path = run_dir / "ohlcv.csv"
    cleaned.to_csv(raw_path, index_label="date")
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    synthetic = config["data"]["source"] == "synthetic"
    window = config["state"]["window"]
    periods = {
        name: {
            "first_execution": str(frame.index[window].date()),
            "last_execution": str(frame.index[-1].date()),
            "steps": len(frame) - window,
        }
        for name, frame in splits.items()
    }
    if progress:
        for name, period in periods.items():
            progress(
                f"{name}: {period['first_execution']} .. {period['last_execution']} ({period['steps']} steps)"
            )
    manifest = {
        "status": "running",
        "mode": mode,
        "source": config["data"]["source"],
        "algorithm": config["agent"]["algorithm"],
        "experiment_label": experiment_label(config),
        "synthetic": synthetic,
        "symbol": None if synthetic else config["data"].get("symbol"),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_rows": len(raw),
        "clean_rows": len(cleaned),
        "feature_rows": len(features),
        "data_sha256": file_digest(raw_path),
        "periods": periods,
        "state_features": get_market_feature_names(config),
        "state_dim": window * len(get_market_feature_names(config))
        + len(config["state"]["portfolio_features"]),
        "versions": {
            package: importlib.metadata.version(package)
            for package in ("torch", "numpy", "pandas", "PyYAML")
        },
        "python": platform.python_version(),
        "code_sha256": {
            str(path.relative_to(PROJECT_ROOT)): file_digest(path)
            for path in sorted(PROJECT_ROOT.rglob("*.py"))
            if "results" not in path.parts
        },
        "notes": (
            ["Synthetic data are not Samsung performance results."] if synthetic else []
        ),
    }

    def save_manifest():
        (run_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8"
        )

    save_manifest()
    try:
        if mode == "train":
            result = train(
                splits["train"],
                splits["validation"],
                config,
                run_dir,
                progress=progress,
            )
            agent = result.agent
            history = result.history
            manifest.update(
                best_episode=result.best_episode,
                checkpoint=str(result.checkpoint_path),
                environment_steps=result.total_steps,
                learning_steps=result.learning_steps,
            )
        else:
            history = pd.DataFrame()
            manifest.update(
                checkpoint=str(Path(checkpoint).resolve()),
                best_episode=saved["episode"],
            )
        old_threads = torch.get_num_threads()
        torch.set_num_threads(config["training"]["torch_threads"])
        try:
            results = evaluate(agent, splits["test"], config)
        finally:
            torch.set_num_threads(old_threads)
        save_evaluation(results, run_dir / "test", config["evaluation"]["metrics"])
        plot_results(results, history, run_dir / "results.png", synthetic=synthetic)
        manifest["status"] = "complete"
        manifest["test_metrics"] = {
            name: result.metrics for name, result in results.items()
        }
        save_manifest()
        label = "SYNTHETIC execution check" if synthetic else "Held-out test"
        lines = [
            f"# {label}",
            "",
            f"Mode: {mode}",
            f"Algorithm: {config['agent']['algorithm']}",
            f"Best episode: {manifest['best_episode']}",
            "",
        ]
        lines.extend(build_summary_table(results, config["evaluation"]["metrics"]))
        lines.extend(
            [
                "",
                "Equity is marked at the final close without forced liquidation.",
                "Zero-risk Sharpe/Sortino and no-closed-position holding time are undefined (null).",
                "",
                "![Equity, drawdown and loss](results.png)",
            ]
        )
        (run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        export_frontend_runs()
        if progress:
            progress(f"Complete: {run_dir}")
        return run_dir
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error_type"] = type(exc).__name__
        save_manifest()
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run a config-driven DQN trading experiment."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--demo",
        action="store_true",
        help="Explicit synthetic data; never actual stock results",
    )
    source.add_argument(
        "--csv", type=Path, help="OHLCV CSV with date, open, high, low, close, volume"
    )
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--algorithm", choices=["dqn", "double_dqn"])
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--mode", choices=["train", "evaluate", "paper"], default="train"
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, help="A NEW directory for this run")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.demo:
        config["data"]["source"] = "synthetic"
    if args.csv:
        config["data"].update(
            source="csv", csv_path=str(args.csv.expanduser().resolve())
        )
    for value, section, key in (
        (args.episodes, "training", "episodes"),
        (args.seed, "training", "seed"),
        (args.algorithm, "agent", "algorithm"),
    ):
        if value is not None:
            config[section][key] = value
    if args.mode != "train" and (
        args.episodes is not None or args.algorithm is not None
    ):
        parser.error(
            "Evaluation uses the saved checkpoint; omit --episodes and --algorithm."
        )
    try:
        run_experiment(
            config,
            output_dir=args.output_dir,
            checkpoint=args.checkpoint,
            mode=args.mode,
        )
    except (
        ValueError,
        RuntimeError,
        FileNotFoundError,
        FileExistsError,
        KeyError,
    ) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
