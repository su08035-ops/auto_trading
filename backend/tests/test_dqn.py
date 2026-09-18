"""Exact numerical DQN checks and synthetic end-to-end training tests."""

from pathlib import Path
import unittest

import numpy as np
import pandas as pd
import torch
import yaml

from agents.dqn import DQNAgent
from data.features import make_features
from env.trading_env import TradingEnv


def make_agent(**overrides):
    settings = dict(
        state_dim=2,
        action_dim=3,
        hidden_dim=8,
        batch_size=2,
        replay_buffer_size=10,
        learning_starts=2,
        seed=7,
    )
    settings.update(overrides)
    return DQNAgent(**settings)


def constant_q(network, values):
    with torch.no_grad():
        for parameter in network.parameters():
            parameter.zero_()
        network.output.bias.copy_(torch.tensor(values, dtype=torch.float32))


def remember_pair(agent):
    state = np.zeros(agent.state_dim, dtype=np.float32)
    agent.remember(state, 0, 1.0, state, False)
    agent.remember(state, 1, -1.0, state, True)


class TestDQN(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_target_starts_equal_but_is_independent_and_frozen(self):
        agent = make_agent()
        for policy, target in zip(
            agent.policy_network.parameters(), agent.target_network.parameters()
        ):
            torch.testing.assert_close(policy, target)
            self.assertNotEqual(policy.data_ptr(), target.data_ptr())
            self.assertFalse(target.requires_grad)
        self.assertFalse(agent.target_network.training)

    def test_warmup_starts_exactly_on_256th_experience(self):
        agent = make_agent(batch_size=64, replay_buffer_size=300, learning_starts=256)
        state = np.zeros(2, dtype=np.float32)
        for _ in range(255):
            agent.remember(state, 0, 1, state, True)
            self.assertIsNone(agent.learn())
        self.assertEqual(agent.epsilon, 1.0)
        self.assertEqual(agent.learning_steps, 0)
        agent.remember(state, 0, 1, state, True)
        result = agent.learn()
        self.assertIsNotNone(result)
        self.assertEqual(result["learning_steps"], 1)
        self.assertAlmostEqual(result["epsilon"], 0.999)

    def test_fifo_buffer_owns_copies_and_lifetime_count_does_not_shrink(self):
        agent = make_agent(replay_buffer_size=3)
        state = np.zeros(2, dtype=np.float32)
        agent.remember(state, 0, 0, state, False)
        state[:] = 9
        np.testing.assert_array_equal(agent.replay_buffer[0].state, [0, 0])
        np.testing.assert_array_equal(agent.replay_buffer[0].next_state, [0, 0])
        for reward in range(1, 5):
            agent.remember(state, 0, reward, state, False)
        self.assertEqual([item.reward for item in agent.replay_buffer], [2, 3, 4])
        self.assertEqual(agent.total_steps, 5)

    def test_td_targets_use_target_max_and_zero_terminal_future(self):
        agent = make_agent(gamma=0.5)
        constant_q(agent.policy_network, [100, 0, 0])
        constant_q(agent.target_network, [1, 4, 2])
        targets = agent._calculate_td_targets(
            torch.tensor([1.0, -2.0]), torch.zeros(2, 2), torch.tensor([False, True])
        )
        torch.testing.assert_close(targets, torch.tensor([3.0, -2.0]))
        self.assertFalse(targets.requires_grad)

    def test_all_terminal_batch_does_not_evaluate_target(self):
        agent = make_agent()

        def unexpected_forward(_):
            raise AssertionError("Terminal futures must not be evaluated.")

        agent.target_network.forward = unexpected_forward
        state = np.zeros(2, dtype=np.float32)
        agent.remember(state, 0, 1, state, True)
        agent.remember(state, 1, -1, state, True)
        self.assertIsNotNone(agent.learn())

    def test_recorded_actions_and_mean_individual_losses(self):
        agent = make_agent(gamma=0.5)
        constant_q(agent.policy_network, [10, 20, 30])
        constant_q(agent.target_network, [1, 2, 4])
        state = np.zeros(2, dtype=np.float32)
        agent.remember(state, 0, 1, state, False)
        agent.remember(state, 1, -1, state, True)
        result = agent.learn()
        # Targets [3, -1], recorded-action predictions [10, 20], errors [-7, -21].
        self.assertAlmostEqual(result["loss"], (7**2 + 21**2) / 2)
        self.assertAlmostEqual(result["mean_td_error"], -14)

    def test_opposite_td_errors_do_not_cancel_loss(self):
        agent = make_agent()
        constant_q(agent.policy_network, [0, 0, 0])
        state = np.zeros(2, dtype=np.float32)
        agent.remember(state, 0, 2, state, True)
        agent.remember(state, 1, -2, state, True)
        result = agent.learn()
        self.assertEqual(result["mean_td_error"], 0)
        self.assertEqual(result["loss"], 4)

    def test_policy_changes_but_target_stays_fixed_until_500_updates(self):
        agent = make_agent(target_update_interval=500)
        remember_pair(agent)
        before = [parameter.clone() for parameter in agent.target_network.parameters()]
        for step in range(1, 501):
            result = agent.learn()
            self.assertEqual(result["learning_steps"], step)
            self.assertEqual(result["target_updated"], step == 500)
            if step < 500:
                for old, target in zip(before, agent.target_network.parameters()):
                    torch.testing.assert_close(old, target, rtol=0, atol=0)
        self.assertTrue(
            any(
                not torch.equal(old, new)
                for old, new in zip(before, agent.policy_network.parameters())
            )
        )
        for policy, target in zip(
            agent.policy_network.parameters(), agent.target_network.parameters()
        ):
            torch.testing.assert_close(policy, target, rtol=0, atol=0)
            self.assertIsNone(target.grad)

    def test_greedy_evaluation_does_not_decay_epsilon_or_collect_experience(self):
        agent = make_agent()
        constant_q(agent.policy_network, [0, 5, 1])
        state = np.zeros(2, dtype=np.float32)
        for _ in range(10):
            self.assertEqual(agent.select_action(state, explore=False), 1)
        self.assertEqual(agent.epsilon, 1)
        self.assertEqual(agent.total_steps, 0)
        self.assertEqual(agent.learning_steps, 0)
        self.assertEqual(len(agent.replay_buffer), 0)

    def test_epsilon_floor_and_reproducible_exploration(self):
        first = make_agent(epsilon_min=0.1, epsilon_decay=0.5)
        second = make_agent(epsilon_min=0.1, epsilon_decay=0.5)
        state = np.zeros(2, dtype=np.float32)
        left = [first.select_action(state) for _ in range(30)]
        right = [second.select_action(state) for _ in range(30)]
        self.assertEqual(left, right)
        self.assertEqual(set(left), {0, 1, 2})
        remember_pair(first)
        for _ in range(10):
            first.learn()
        self.assertEqual(first.epsilon, 0.1)

    def test_zero_epsilon_is_greedy_after_warmup(self):
        agent = make_agent(epsilon_start=0, epsilon_min=0)
        remember_pair(agent)
        constant_q(agent.policy_network, [0, -1, 3])
        self.assertEqual(agent.select_action(np.zeros(2)), 2)

    def test_bad_configuration_and_experiences_fail(self):
        for settings in (
            {"batch_size": 11},
            {"learning_starts": 1},
            {"target_update_interval": 0},
            {"gamma": 1.1},
            {"learning_rate": np.nan},
            {"epsilon_min": 1.1},
            {"epsilon_decay": 0},
        ):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                make_agent(**settings)
        agent = make_agent()
        state = np.zeros(2, dtype=np.float32)
        for arguments in (
            (np.zeros(3), 0, 0, state, False),
            (state, 3, 0, state, False),
            (state, 0.5, 0, state, False),
            (state, 0, np.inf, state, False),
            (state, 0, 0, state, 1),
        ):
            with self.assertRaises(ValueError):
                agent.remember(*arguments)
        self.assertEqual(agent.total_steps, 0)

    def test_nonfinite_loss_does_not_update_policy(self):
        agent = make_agent()
        state = np.zeros(2, dtype=np.float32)
        for _ in range(2):
            agent.remember(state, 0, 1e30, state, True)
        before = [p.clone() for p in agent.policy_network.parameters()]
        with self.assertRaises(FloatingPointError):
            agent.learn()
        self.assertEqual(agent.learning_steps, 0)
        for old, new in zip(before, agent.policy_network.parameters()):
            torch.testing.assert_close(old, new)

    def test_config_and_three_episodes_keep_agent_state(self):
        path = Path(__file__).resolve().parents[1] / "config" / "config.yaml"
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        days = np.arange(460)
        close = 100 + 0.02 * days + 2 * np.sin(days / 7)
        raw = pd.DataFrame(
            {
                "open": close - 0.2,
                "high": close + 1,
                "low": close - 1,
                "close": close,
                "volume": 1000 + days % 17,
            },
            index=pd.bdate_range("2020-01-01", periods=len(days)),
        )
        env = TradingEnv.from_config(make_features(raw), config)
        agent = DQNAgent.from_config(env.state_dim, env.action_size, config)
        self.assertEqual(env.state_dim, 301)
        self.assertEqual(env.action_size, len(config["action"]["values"]))
        self.assertEqual(agent.replay_buffer.maxlen, 20000)
        self.assertEqual(agent.learning_starts, 256)
        sync_steps = []
        losses = []
        for episode in range(3):
            old_steps, old_buffer = agent.total_steps, len(agent.replay_buffer)
            state = env.reset()
            self.assertEqual(agent.total_steps, old_steps)
            self.assertEqual(len(agent.replay_buffer), old_buffer)
            self.assertEqual(env.shares, 0)
            done = False
            while not done:
                action = agent.select_action(state)
                next_state, reward, done, info = env.step(action)
                agent.remember(state, action, reward, next_state, done)
                result = agent.learn()
                if result is not None:
                    losses.append(result["loss"])
                    if result["target_updated"]:
                        sync_steps.append(result["learning_steps"])
                self.assertGreaterEqual(info["cash"], -1e-8)
                self.assertGreaterEqual(info["shares"], 0)
                state = next_state
        self.assertEqual(agent.total_steps, env.episode_steps * 3)
        self.assertEqual(agent.learning_steps, agent.total_steps - 255)
        self.assertEqual(sync_steps, list(range(500, agent.learning_steps + 1, 500)))
        self.assertTrue(sync_steps)
        self.assertTrue(np.isfinite(losses).all())


if __name__ == "__main__":
    unittest.main()
