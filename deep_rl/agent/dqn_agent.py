
import torch
import numpy as np

from deep_rl.agent.agent import RLAgent
from deep_rl.logging import get_logger
from deep_rl.preprocessing import ObservationPreprocessor
from deep_rl.agent.exploration import ExplorationStrategy
from deep_rl.agent.replay import ReplayBuffer
from deep_rl.agent.exploration import EpsilonGreedyExplorationStrategy
from deep_rl.preprocessing import DQNAtariPreprocessor

log = get_logger('Agent')


class DQNAtariAgent(RLAgent):
    def __init__(self,
        num_actions: int,
        replay_buffer_size: int= 1000000,
        action_value_function: torch.nn.Module = None,
        observation_preprocessor: ObservationPreprocessor = None,
        exploration_strategy: ExplorationStrategy = None
        ):

        if not action_value_function:
            action_value_function = DQNAtariQNet(num_actions=num_actions)

        if not observation_preprocessor:
            observation_preprocessor = DQNAtariPreprocessor()

        if not exploration_strategy:
            exploration_strategy = EpsilonGreedyExplorationStrategy(initial_epsilon=1, final_epsilon=0.1, decay=0.9/1e6)

        self._observation_prep = observation_preprocessor
        self._action_value_fx = action_value_function
        self._exploration_strategy = exploration_strategy

        # Create a replay buffer.
        self._replay_buffer = ReplayBuffer(replay_buffer_size)

        # We need to keep track of our last action so we can update when we get a new result.
        self._last_action = None
        self._last_observation = None
        

    def update(self, observation: np.ndarray, reward: float, done: bool) -> int:
        if self._observation_prep:
            observation = self._observation_prep.prep(observation)

        observation = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)

        # If we have a last action (First iteration we do not) update the replay buffer.
        if self._last_action:
            self._replay_buffer.store((self._last_observation, self._last_action, reward, observation))

        # Using the action value function try to select an action. 
        action_space = self._action_value_fx.forward(observation)

        # If we were given an exploration strategy use it, otherwise just take argmax (exploit)
        if self._exploration_strategy:
            action = self._exploration_strategy.pick(action_space)
        else:
            action = action_space.argmax()

        self._last_action = action
        self._last_observation = observation

        return action


class DQNAtariQNet(torch.nn.Module):
    def __init__(self, num_actions: int):
        super(DQNAtariQNet, self).__init__()
        self._num_actions = num_actions
        self._conv1 = torch.nn.Conv2d(4, 16, kernel_size=8, stride=4)
        self._relu = torch.nn.ReLU(False)
        self._conv2 = torch.nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self._linear1 = torch.nn.Linear(in_features=2592, out_features=256)
        self._linear2 = torch.nn.Linear(in_features=256, out_features=self._num_actions)

    def forward(self, inputs):
        x = self._relu.forward(self._conv1.forward(inputs))
        x = self._relu.forward(self._conv2.forward(x))
        x = torch.flatten(x, start_dim=1)
        x = self._relu.forward(self._linear1(x))
        x = self._linear2(x)
        return x

