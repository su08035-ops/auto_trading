"""Double DQN changes only the bootstrap target; reuse the complete DQN loop."""

import torch

from agents.dqn import DQNAgent


class DoubleDQNAgent(DQNAgent):
    algorithm = "double_dqn"

    def _calculate_td_targets(self, rewards, next_states, dones):
        """Policy chooses the next action; Target evaluates that chosen action."""
        with torch.no_grad():
            next_values = torch.zeros_like(rewards)
            nonterminal = ~dones
            if nonterminal.any().item():
                states = next_states[nonterminal]
                best_actions = self.policy_network(states).argmax(dim=1, keepdim=True)
                next_values[nonterminal] = (
                    self.target_network(states).gather(1, best_actions).squeeze(1)
                )
            return rewards + self.gamma * next_values
