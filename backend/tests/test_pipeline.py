"""Regression tests for config, checkpoints, held-out evaluation and accounting."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from data.loader import load_config, load_data, generate_demo_ohlcv
from data.features import make_features, _rsi
from data.preprocessing import prepare_splits, clean_ohlcv
from agents.double_dqn import DoubleDQNAgent
from backtest.metrics import calculate_metrics, holding_statistics
from backtest.backtester import run_backtest
from training.train import build_agent, train, load_checkpoint
from main import experiment_label, new_run_directory, run_experiment, validate_config


def experiment_config():
    config = deepcopy(load_config())
    config["data"].update(
        source="synthetic",
        start_date="2018-01-01",
        validation_start="2019-01-01",
        split_date="2019-07-01",
        end_date="2019-12-31",
    )
    config["action"]["values"] = [0.0, 0.5, 1.0]
    config["agent"].update(
        hidden_dim=8,
        batch_size=4,
        replay_buffer_size=1000,
        epsilon_min=0.05,
        epsilon_decay=0.999,
    )
    config["training"].update(episodes=2, learning_starts=8, torch_threads=1)
    return config


def set_q(network, values):
    with torch.no_grad():
        for parameter in network.parameters():
            parameter.zero_()
        network.output.bias.copy_(torch.tensor(values, dtype=torch.float32))


class TestConfiguration(unittest.TestCase):
    def test_run_directory_name_includes_experiment_settings(self):
        config = experiment_config()
        with tempfile.TemporaryDirectory() as directory:
            config["paths"]["runs_dir"] = directory
            config["agent"]["hidden_dim"] = 128
            config["agent"]["learning_rate"] = 0.001
            config["agent"]["epsilon_min"] = 0.05
            config["agent"].update(loss="huber", batch_size=64)
            run_dir = new_run_directory(config)
            self.assertEqual(run_dir.name, "실험1_mse-to-huber")

            config["agent"]["learning_rate"] = 0.0003
            self.assertEqual(experiment_label(config), "huber-lr-0p0003")

            config["agent"]["learning_rate"] = 0.001
            config["action"]["values"] = [0.0, 1.0]
            self.assertEqual(experiment_label(config), "huber-action-0p0-1p0")

            config["action"]["values"] = [index / 10 for index in range(11)]
            self.assertEqual(experiment_label(config), "huber-action-10pct")

            config["action"]["values"] = [index / 20 for index in range(21)]
            self.assertEqual(experiment_label(config), "huber-action-5pct")

            config["action"]["values"] = [0.0, 0.5, 1.0]
            config["agent"]["epsilon_min"] = 0.01
            self.assertEqual(experiment_label(config), "huber-epsmin-0p01")

            config["agent"]["epsilon_min"] = 0.05
            config["agent"]["epsilon_decay"] = 0.9999
            self.assertEqual(experiment_label(config), "huber-epsdecay-0p9999")

            config["agent"]["epsilon_decay"] = 0.999
            config["agent"]["gamma"] = 0.97
            self.assertEqual(experiment_label(config), "huber-gamma-0p97")

            config["agent"]["gamma"] = 0.99
            config["agent"]["batch_size"] = 128
            self.assertEqual(experiment_label(config), "huber-batch-128")

            config["agent"]["batch_size"] = 64
            config["agent"]["target_update_interval"] = 250
            self.assertEqual(experiment_label(config), "huber-target-250")

            config["agent"]["target_update_interval"] = 500
            config["agent"]["hidden_dim"] = 64
            self.assertEqual(experiment_label(config), "huber-hidden-64")

    def test_yaml_optimizer_loss_and_dimensions_reach_actual_objects(self):
        config = experiment_config()
        config["agent"].update(
            optimizer="sgd", loss="huber", hidden_dim=17, learning_rate=0.123
        )
        agent = build_agent(5, 3, config)
        self.assertEqual(agent.policy_network.hidden1.out_features, 17)
        self.assertIsInstance(agent.optimizer, torch.optim.SGD)
        self.assertIsInstance(agent.loss_function, torch.nn.SmoothL1Loss)
        self.assertEqual(agent.optimizer.param_groups[0]["lr"], 0.123)

        config["agent"]["loss"] = "mae"
        agent = build_agent(5, 3, config)
        self.assertIsInstance(agent.loss_function, torch.nn.L1Loss)

    def test_missing_config_does_not_silently_use_constructor_default(self):
        config = experiment_config()
        del config["agent"]["hidden_dim"]
        with self.assertRaises(ValueError):
            validate_config(config)
        with self.assertRaises(KeyError):
            build_agent(5, 3, config)

    def test_unsupported_loss_is_rejected(self):
        config = experiment_config()
        config["agent"]["loss"] = "typo"
        with self.assertRaises(ValueError):
            build_agent(5, 3, config)

    def test_double_dqn_selects_policy_action_and_evaluates_target_value(self):
        config = experiment_config()
        config["agent"]["algorithm"] = "double_dqn"
        config["agent"]["gamma"] = 0.5
        agent = build_agent(2, 3, config)
        self.assertIsInstance(agent, DoubleDQNAgent)
        set_q(agent.policy_network, [1, 9, 0])
        set_q(agent.target_network, [8, 2, 4])
        values = agent._calculate_td_targets(
            torch.tensor([1.0, -1.0]), torch.zeros(2, 2), torch.tensor([False, True])
        )
        torch.testing.assert_close(values, torch.tensor([2.0, -1.0]))
        self.assertFalse(values.requires_grad)


class TestDataAndMetrics(unittest.TestCase):
    def test_rsi_handles_rising_falling_flat_and_warmup(self):
        self.assertEqual(_rsi(pd.Series(np.arange(50.0)), period=14).iloc[-1], 100)
        self.assertEqual(_rsi(pd.Series(-np.arange(50.0)), period=14).iloc[-1], 0)
        self.assertEqual(_rsi(pd.Series([100.0] * 50), period=14).iloc[-1], 50)
        self.assertTrue(
            _rsi(pd.Series(np.arange(50.0)), period=14).iloc[:14].isna().all()
        )

    def test_future_changes_do_not_change_past_features(self):
        raw = generate_demo_ohlcv("2018-01-01", "2019-12-31", seed=0)
        changed = raw.copy()
        changed.iloc[300:] *= 2
        pd.testing.assert_frame_equal(
            make_features(raw).loc[: raw.index[299]],
            make_features(changed).loc[: raw.index[299]],
        )

    def test_training_seed_does_not_change_synthetic_market(self):
        config = experiment_config()
        first = load_data(config=config)
        config["training"]["seed"] += 1
        pd.testing.assert_frame_equal(first, load_data(config=config))

    def test_split_context_never_generates_early_evaluation_trades(self):
        config = experiment_config()
        frame = make_features(load_data(config=config))
        splits = prepare_splits(frame, config)
        window = config["state"]["window"]
        self.assertLess(
            splits["train"].index[-1], pd.Timestamp(config["data"]["validation_start"])
        )
        self.assertEqual(splits["validation"].index[window], pd.Timestamp("2019-01-01"))
        self.assertEqual(splits["test"].index[window], pd.Timestamp("2019-07-01"))
        result = run_backtest(splits["test"], config, strategy="cash")
        self.assertEqual(result.trades["date"].iloc[0], pd.Timestamp("2019-07-01"))
        self.assertTrue((result.trades["date"] >= pd.Timestamp("2019-07-01")).all())

    def test_empty_period_is_rejected(self):
        config = experiment_config()
        config["data"]["split_date"] = "2030-01-01"
        with self.assertRaises(ValueError):
            prepare_splits(make_features(load_data(config=config)), config)

    def test_known_return_drawdown_and_cash_ratios(self):
        metrics = calculate_metrics([100, 120, 90, 99])
        self.assertAlmostEqual(metrics["cumulative_return"], -0.01)
        self.assertAlmostEqual(metrics["mdd"], 0.25)
        returns = np.array([0.2, -0.25, 0.1])
        expected_sortino = (
            np.sqrt(252)
            * returns.mean()
            / np.sqrt(np.mean(np.minimum(returns, 0) ** 2))
        )
        self.assertAlmostEqual(metrics["sortino_ratio"], expected_sortino)
        cash = calculate_metrics([100, 100, 100])
        self.assertEqual(cash["mdd"], 0)
        self.assertIsNone(cash["sharpe_ratio"])
        self.assertIsNone(cash["sortino_ratio"])
        self.assertIsNone(cash["average_holding_days"])
        json.dumps(cash, allow_nan=False)

    def test_holding_duration_fifo_and_open_positions(self):
        trades = pd.DataFrame(
            {
                "trade_type": ["buy", "hold", "sell", "sell", "buy"],
                "shares_traded": [10, 0, 4, 6, 3],
            }
        )
        result = holding_statistics(trades)
        self.assertAlmostEqual(result["average_holding_days"], 2.6)
        self.assertEqual(result["closed_shares"], 10)
        self.assertEqual(result["open_shares"], 3)

    def test_buy_hold_buys_once_even_when_price_falls_and_cash_remains(self):
        config = experiment_config()
        config["state"].update(window=1, market_features={"one": ["feature"]})
        config["environment"].update(
            initial_balance=250, commission=0, tax=0, min_holding_days=0
        )
        frame = pd.DataFrame(
            {"open": [100, 100, 20, 50], "close": [100, 100, 20, 50], "feature": 0.0},
            index=pd.bdate_range("2020-01-01", periods=4),
        )
        result = run_backtest(frame, config, strategy="buy_and_hold")
        self.assertEqual(result.metrics["trade_count"], 1)
        self.assertEqual(result.metrics["open_shares"], 2)
        self.assertEqual(result.metrics["final_asset"], 150)


class TestTrainingPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_validation_selects_earlier_checkpoint_and_load_preserves_predictions(self):
        config = experiment_config()
        splits = prepare_splits(make_features(load_data(config=config)), config)
        snapshots = []

        def validation_result(frame, cfg, *, agent):
            snapshots.append(agent.policy_network(torch.zeros(301)).detach().clone())
            return SimpleNamespace(
                metrics={
                    "cumulative_return": 0.2 if len(snapshots) == 1 else 0.1,
                    "mdd": 0.05,
                    "sharpe_ratio": 0.3,
                    "sortino_ratio": 0.4,
                    "trade_count": 2,
                    "turnover": 0.6,
                    "average_holding_days": 3.0,
                }
            )

        with tempfile.TemporaryDirectory() as directory:
            with patch("training.train.run_backtest", side_effect=validation_result):
                result = train(
                    splits["train"],
                    splits["validation"],
                    config,
                    directory,
                    progress=None,
                )
            self.assertEqual(result.best_episode, 1)
            restored, saved_config, saved = load_checkpoint(result.checkpoint_path)
            torch.testing.assert_close(
                restored.policy_network(torch.zeros(301)), snapshots[0]
            )
            self.assertEqual(saved_config["agent"]["hidden_dim"], 8)
            self.assertEqual(saved["episode"], 1)
            self.assertEqual(result.learning_steps, result.total_steps - 7)
            history = pd.read_csv(Path(directory) / "training_history.csv")
            self.assertEqual(history["validation_sharpe_ratio"].iloc[0], 0.3)
            self.assertEqual(history["validation_sortino_ratio"].iloc[0], 0.4)
            self.assertEqual(history["validation_trade_count"].iloc[0], 2)
            self.assertEqual(history["validation_turnover"].iloc[0], 0.6)
            self.assertTrue(result.checkpoint_path.with_name("last_model.pth").exists())

    def test_evaluation_does_not_change_agent_or_experience(self):
        config = experiment_config()
        splits = prepare_splits(make_features(load_data(config=config)), config)
        agent = build_agent(301, 3, config)
        before = {
            name: value.clone()
            for name, value in agent.policy_network.state_dict().items()
        }
        rng_before = agent.rng.getstate()
        run_backtest(splits["test"], config, agent=agent)
        self.assertEqual(agent.total_steps, 0)
        self.assertEqual(agent.learning_steps, 0)
        self.assertEqual(agent.epsilon, 1)
        self.assertEqual(agent.rng.getstate(), rng_before)
        self.assertTrue(agent.policy_network.training)
        for name, value in agent.policy_network.state_dict().items():
            torch.testing.assert_close(value, before[name])

    def test_full_demo_and_checkpoint_only_evaluation(self):
        config = experiment_config()
        with tempfile.TemporaryDirectory() as directory:
            run = run_experiment(
                config, output_dir=Path(directory) / "first", progress=None
            )
            manifest = json.loads((run / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "complete")
            self.assertTrue(manifest["synthetic"])
            self.assertGreater(manifest["learning_steps"], 0)
            self.assertTrue((run / "results.png").exists())
            self.assertTrue((run / "summary.md").exists())
            self.assertTrue((run / "training_history.csv").exists())
            checkpoint = manifest["checkpoint"]
            repeat = run_experiment(
                config,
                output_dir=Path(directory) / "second",
                checkpoint=checkpoint,
                mode="paper",
                progress=None,
            )
            self.assertEqual(
                json.loads((repeat / "test" / "metrics.json").read_text()),
                json.loads((run / "test" / "metrics.json").read_text()),
            )
            self.assertFalse((repeat / "training_history.csv").exists())
            with self.assertRaises(FileExistsError):
                run_experiment(config, output_dir=run, progress=None)

    def test_too_short_training_is_not_reported_as_trained(self):
        config = experiment_config()
        config["training"]["learning_starts"] = 10000
        splits = prepare_splits(make_features(load_data(config=config)), config)
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(ValueError):
            train(
                splits["train"], splits["validation"], config, directory, progress=None
            )


if __name__ == "__main__":
    unittest.main()
