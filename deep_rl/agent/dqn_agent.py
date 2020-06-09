
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
        discount_rate: float,
        replay_buffer_size: int=10000,
        action_value_function: torch.nn.Module = None,
        observation_preprocessor: ObservationPreprocessor = None,
        exploration_strategy: ExplorationStrategy = None,
        device: str = 'cuda:0'
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
        self._discount_rate = discount_rate
        self._device = device

        # Setup our optimizer.
        self._optimizer = torch.optim.RMSprop(self._action_value_fx.parameters(), lr=0.01)

        # Create a replay buffer.
        self._replay_buffer = ReplayBuffer(replay_buffer_size)

        # We need to keep track of our last action so we can update when we get a new result.
        self._first_update = True
        self._last_action = None
        self._last_observation = None

        # PUt the model on cuda if available.
        log.info(f'cuda={torch.cuda.is_available()}, device={torch.cuda.get_device_name()}')
        if torch.cuda.is_available():
            self._action_value_fx.cuda(self._device)
        

    def step(self, observation: np.ndarray, reward: float, done: bool) -> int:
        if self._observation_prep:
            observation = self._observation_prep.prep(observation)

        observation = torch.tensor(observation, dtype=torch.float32, device=self._device).unsqueeze(0)

        # If we have a last action (First iteration we do not) update the replay buffer.
        if not self._first_update:
            self._replay_buffer.store((self._last_observation, self._last_action, reward, observation, done))
        else:
            self._first_update = False

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

    def update(self, batch_size: int):
        # Zero out gradients for training.
        self._action_value_fx.zero_grad()

        # sample input sapce
        obs, actions, rewards, next_obs, done = zip(*self._replay_buffer.sample(batch_size))

        # Make all our samples into the correct format.
        obs = torch.cat(obs, axis=0)
        actions = torch.tensor(actions, device=self._device)
        rewards = torch.tensor(rewards, device=self._device)
        next_obs = torch.cat(next_obs, axis=0)
        done = torch.tensor(done, device=self._device)

        # Get the predicted reward for the actions (q values).
        q_val = self._action_value_fx.forward(obs).gather(1, actions.view(batch_size, 1))
        # Get the max q value but don't compute gradient since this represents our "True value".
        with torch.no_grad():
            q_val_max = self._action_value_fx.forward(next_obs).max(dim=1)[0]
        # Now we can compute the target value of our q function
        # Where the target value is equal to the reward if hte state is terminal, or
        # its equal to the reward plus the discount rate of the next value if it is not terminal.
        targets = (~done).float()*rewards+self._discount_rate*q_val_max + done.float()*rewards

        # Now finally we can compute the loss according to the dqn paper its just the squared error. 
        # of the estimate minus the predicted q value.
        loss = torch.sum((targets-q_val.view(-1))**2)

        # Now finally we can back prop.
        loss.backward()

        # And optimize.
        self._optimizer.step()

    def save(self, path: str):
        torch.save(self._action_value_fx, path)

    @staticmethod
    def load(path: str) -> RLAgent:
        q_fx = torch.load(path)
        return DQNAtariAgent(0, 0.99, action_value_function=q_fx)


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

