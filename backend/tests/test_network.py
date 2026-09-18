"""Offline acceptance tests for the Q-network exercise.

Run from the project root:
    python -B -m unittest discover -s tests -p test_network.py -v
"""

import io
from pathlib import Path
import unittest

import pandas as pd
import torch
from torch import nn
import yaml

from agents.network import QNetwork
from data.features import get_market_feature_names
from env.trading_env import TradingEnv


class TestQNetwork(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)

    def test_single_state_has_one_q_value_per_action(self) -> None:
        network = QNetwork(state_dim=301, action_dim=3)
        q_values = network(torch.zeros(301, dtype=torch.float32))

        self.assertEqual(tuple(q_values.shape), (3,))
        self.assertEqual(q_values.dtype, torch.float32)
        self.assertTrue(torch.isfinite(q_values).all().item())

    def test_batch_preserves_independent_examples(self) -> None:
        network = QNetwork(state_dim=301, action_dim=3)
        states = torch.randn(64, 301)
        q_values = network(states)

        self.assertEqual(tuple(q_values.shape), (64, 3))
        self.assertTrue(torch.isfinite(q_values).all().item())
        torch.testing.assert_close(q_values[:1], network(states[:1]))

    def test_constructor_controls_input_hidden_and_output_sizes(self) -> None:
        network = QNetwork(state_dim=7, action_dim=4, hidden_dim=32)
        layers = [layer for layer in network.modules() if isinstance(layer, nn.Linear)]
        dimensions = [(layer.in_features, layer.out_features) for layer in layers]

        self.assertEqual(dimensions, [(7, 32), (32, 32), (32, 4)])
        self.assertEqual(tuple(network(torch.zeros(2, 7)).shape), (2, 4))

    def test_both_hidden_stages_apply_relu(self) -> None:
        network = QNetwork(state_dim=2, action_dim=2, hidden_dim=2)
        layers = [layer for layer in network.modules() if isinstance(layer, nn.Linear)]
        self.assertEqual(len(layers), 3)

        # Controlled signs distinguish missing first or second hidden ReLU.
        with torch.no_grad():
            for layer in layers:
                layer.weight.copy_(torch.eye(2))
                layer.bias.zero_()
            layers[1].weight.neg_()
            layers[1].bias.fill_(1.0)

        states = torch.tensor([[-2.0, 2.0], [2.0, -2.0]])
        expected = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        torch.testing.assert_close(network(states), expected)

    def test_output_is_raw_values_not_probabilities_or_actions(self) -> None:
        network = QNetwork(state_dim=301, action_dim=3)
        layers = [layer for layer in network.modules() if isinstance(layer, nn.Linear)]
        self.assertTrue(layers, "Register Linear layers as part of the network.")
        output_layer = layers[-1]
        self.assertIsNotNone(output_layer.bias)
        expected = torch.tensor([-2.0, 0.5, 3.0])

        # A controlled output bias exposes unwanted final activations or argmax.
        with torch.no_grad():
            output_layer.weight.zero_()
            output_layer.bias.copy_(expected)

        q_values = network(torch.zeros(2, 301))
        torch.testing.assert_close(q_values, expected.expand(2, -1))

    def test_loss_can_update_registered_parameters(self) -> None:
        network = QNetwork(state_dim=301, action_dim=3)
        optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
        states = torch.randn(8, 301)
        q_values = network(states)
        target = q_values.detach() + 1.0
        before = [parameter.detach().clone() for parameter in network.parameters()]

        optimizer.zero_grad()
        loss = nn.MSELoss()(q_values, target)
        loss.backward()

        for parameter in network.parameters():
            self.assertIsNotNone(
                parameter.grad, "Every layer must participate in forward()."
            )
            self.assertTrue(torch.isfinite(parameter.grad).all().item())

        optimizer.step()
        self.assertTrue(
            any(
                not torch.equal(old, new.detach())
                for old, new in zip(before, network.parameters())
            ),
            "At least one parameter must change after an optimizer step.",
        )

    def test_weights_can_be_saved_and_restored(self) -> None:
        network = QNetwork(state_dim=301, action_dim=3)
        states = torch.randn(2, 301)
        expected = network(states).detach()
        buffer = io.BytesIO()
        torch.save(network.state_dict(), buffer)
        buffer.seek(0)

        restored = QNetwork(state_dim=301, action_dim=3)
        restored.load_state_dict(torch.load(buffer, weights_only=True))
        torch.testing.assert_close(restored(states), expected)

    def test_accepts_current_environment_observations(self) -> None:
        config_path = Path(__file__).resolve().parents[1] / "config" / "config.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        feature_names = get_market_feature_names(config)
        dates = pd.bdate_range("2020-01-01", periods=config["state"]["window"] + 5)
        frame = pd.DataFrame(0.0, index=dates, columns=feature_names)
        frame["close"] = 100.0
        frame["open"] = 100.0

        # Synthetic prices exercise the existing environment without any API call.
        environment = TradingEnv.from_config(frame, config)
        network = QNetwork(
            state_dim=environment.state_dim, action_dim=environment.action_size
        )
        self.assertEqual(environment.state_dim, 301)
        self.assertEqual(environment.action_size, len(config["action"]["values"]))
        state = environment.reset()
        self.assertEqual(
            tuple(network(torch.from_numpy(state)).shape), (environment.action_size,)
        )

        next_state, _, _, _ = environment.step(environment.action_size - 1)
        next_q_values = network(torch.from_numpy(next_state))
        self.assertEqual(tuple(next_q_values.shape), (environment.action_size,))
        self.assertTrue(torch.isfinite(next_q_values).all().item())


if __name__ == "__main__":
    unittest.main()
