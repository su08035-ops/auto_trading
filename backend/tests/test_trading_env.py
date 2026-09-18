"""Offline accounting, timing and observation tests. No broker/API calls."""

import unittest

import numpy as np
import pandas as pd

from env.trading_env import TradingEnv


def make_frame(rows=12):
    return pd.DataFrame(
        {"open": 100.0, "close": 100.0, "feature": np.arange(rows, dtype=float)},
        index=pd.bdate_range("2020-01-01", periods=rows),
    )


def make_env(frame=None, **overrides):
    settings = dict(
        market_features=["feature"],
        window=2,
        initial_balance=1000.0,
        commission=0.0,
        tax=0.0,
        min_holding_days=0,
    )
    settings.update(overrides)
    return TradingEnv(make_frame() if frame is None else frame, **settings)


class TestTradingEnv(unittest.TestCase):
    def test_observation_window_and_next_open_execution(self):
        frame = make_frame()
        frame.iloc[2, frame.columns.get_loc("open")] = 200.0
        frame.iloc[2, frame.columns.get_loc("close")] = 220.0
        env = make_env(frame)
        np.testing.assert_array_equal(env.reset(), [0.0, 1.0, 0.0])

        state, reward, done, info = env.step(2)
        self.assertEqual(env.shares, 5)
        self.assertEqual(info["execution_price"], 200.0)
        self.assertEqual(info["decision_date"], frame.index[1])
        self.assertEqual(info["date"], frame.index[2])
        self.assertEqual(env.asset_value, 1100.0)
        self.assertAlmostEqual(reward, np.log(1.1))
        np.testing.assert_array_equal(state, [1.0, 2.0, 1.0])
        self.assertFalse(done)

    def test_unseen_prices_and_features_cannot_change_current_state(self):
        original = make_frame()
        changed = original.copy()
        changed.iloc[3:, :] *= 9.0
        first, second = make_env(original), make_env(changed)
        np.testing.assert_array_equal(first.reset(), second.reset())
        first_state = first.step(1)[0]
        second_state = second.step(1)[0]
        np.testing.assert_array_equal(first_state, second_state)

    def test_cash_only_reward_is_zero(self):
        env = make_env()
        while True:
            _, reward, done, _ = env.step(0)
            self.assertEqual(reward, 0.0)
            if done:
                break
        self.assertEqual(env.trade_count, 0)

    def test_costs_and_overnight_gap_are_in_reward(self):
        frame = make_frame()
        frame.iloc[3, frame.columns.get_loc("open")] = 120.0
        frame.iloc[3, frame.columns.get_loc("close")] = 150.0
        env = make_env(frame, commission=0.01, tax=0.02)

        _, buy_reward, _, buy = env.step(2)
        self.assertEqual(env.shares, 9)
        self.assertEqual(env.cash, 91.0)
        self.assertEqual(env.asset_value, 991.0)
        self.assertEqual(buy["execution"].fee, 9.0)
        self.assertEqual(buy["execution"].tax, 0.0)
        self.assertAlmostEqual(buy_reward, np.log(991.0 / 1000.0))

        _, sell_reward, _, sell = env.step(0)
        expected = 91.0 + 9 * 120.0 * (1 - 0.01 - 0.02)
        self.assertAlmostEqual(env.cash, expected)
        self.assertEqual(env.shares, 0)
        self.assertAlmostEqual(sell["execution"].fee, 10.8)
        self.assertAlmostEqual(sell["execution"].tax, 21.6)
        self.assertAlmostEqual(sell_reward, np.log(expected / 991.0))
        self.assertAlmostEqual(buy_reward + sell_reward, np.log(expected / 1000.0))

    def test_minimum_holding_blocks_reductions_until_five_sessions(self):
        env = make_env(min_holding_days=5)
        env.step(2)
        self.assertEqual(env.holding_days, 1)
        for _ in range(4):
            _, _, _, info = env.step(0)
            self.assertTrue(info["execution"].min_holding_blocked)
            self.assertEqual(env.shares, 10)
        self.assertEqual(env.holding_days, 5)
        _, _, _, info = env.step(0)
        self.assertFalse(info["execution"].min_holding_blocked)
        self.assertEqual(env.shares, 0)
        self.assertEqual(env.holding_days, 0)

    def test_additional_buy_restarts_whole_position_lock(self):
        env = make_env(min_holding_days=5)
        env.step(1)
        env.step(1)
        self.assertEqual(env.holding_days, 2)
        env.step(2)
        self.assertEqual(env.holding_days, 1)
        _, _, _, info = env.step(1)
        self.assertTrue(info["execution"].min_holding_blocked)

    def test_exactly_one_share_can_be_bought_and_sold(self):
        env = make_env(initial_balance=100.0)
        env.step(2)
        self.assertEqual(env.shares, 1)
        env.step(0)
        self.assertEqual(env.shares, 0)
        self.assertEqual(env.cash, 100.0)
        self.assertEqual(env.trade_count, 2)

    def test_whole_share_rounding_and_no_borrowing(self):
        env = make_env(initial_balance=250.0, commission=0.01)
        _, _, _, info = env.step(2)
        self.assertEqual(env.shares, 2)
        self.assertAlmostEqual(env.cash, 48.0)
        self.assertAlmostEqual(info["execution"].executed_target, 200 / 248)
        env.step(0)
        self.assertGreaterEqual(env.cash, 0.0)
        self.assertEqual(env.shares, 0)

    def test_terminal_and_reset(self):
        env = make_env(make_frame(3))
        initial = env.reset()
        self.assertEqual(env.episode_steps, 1)
        _, _, done, _ = env.step(2)
        self.assertTrue(done)
        self.assertEqual(len(env.portfolio_values), 2)
        with self.assertRaises(RuntimeError):
            env.step(0)
        np.testing.assert_array_equal(env.reset(), initial)
        self.assertEqual(env.cash, 1000)
        self.assertEqual(env.trade_history, [])

    def test_invalid_actions_are_rejected_without_advancing(self):
        env = make_env()
        for action in (-1, 3, 0.5, 1.0, True, "1"):
            with self.subTest(action=action), self.assertRaises(ValueError):
                env.step(action)
        self.assertEqual(env.current_step, 1)

    def test_invalid_inputs_fail_early(self):
        invalid_settings = [
            {"window": 0},
            {"window": 2.5},
            {"initial_balance": 0},
            {"commission": -0.01},
            {"commission": float("nan")},
            {"commission": 0.6, "tax": 0.5},
            {"min_holding_days": -1},
            {"action_values": [0, float("nan")]},
            {"action_values": [0, 0]},
            {"portfolio_features": ["holding_days_norm"]},
        ]
        for settings in invalid_settings:
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                make_env(**settings)
        for column, value in (("open", 0), ("close", np.inf), ("feature", np.nan)):
            frame = make_frame()
            frame.iloc[0, frame.columns.get_loc(column)] = value
            with self.subTest(column=column), self.assertRaises(ValueError):
                make_env(frame)
        with self.assertRaises(ValueError):
            make_env(make_frame().drop(columns="open"))
        frame = make_frame()
        frame.index = [frame.index[0]] * len(frame)
        with self.assertRaises(ValueError):
            make_env(frame)


if __name__ == "__main__":
    unittest.main()
