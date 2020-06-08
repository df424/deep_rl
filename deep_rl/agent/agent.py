
import numpy as np
from deep_rl.logging import get_logger
from deep_rl.preprocessing import ObservationPreprocessor
from deep_rl.agent.value_functions import ActionValueFunction
from deep_rl.agent.exploration import ExplorationStrategy
from deep_rl.agent.replay import ReplayBuffer

log = get_logger('Agent')

class RLAgent():
    def __init__(self,
        action_value_function: ActionValueFunction,
        observation_preprocessor: ObservationPreprocessor = None,
        exploration_strategy: ExplorationStrategy = None
        ):

        self._observation_prep = observation_preprocessor
        self._action_value_fx = action_value_function
        self._exploration_strategy = exploration_strategy

        # Create a replay buffer.
        self._replay_buffer = ReplayBuffer(1000000)

        # We need to keep track of our last action so we can update when we get a new result.
        self._last_action = None
        self._last_observation = None
        

    def update(self, observation: np.ndarray, reward: float, done: bool) -> int:
        if self._observation_prep:
            observation = self._observation_prep.prep(observation)

        # If we have a last action (First iteration we do not) update the replay buffer.
        if self._last_action:
            self._replay_buffer.store((self._last_observation, self._last_action, reward, observation))

        # Using the action value function try to select an action. 
        action_space = self._action_value_fx.forward(np.array([observation]))

        # If we were given an exploration strategy use it, otherwise just take argmax (exploit)
        if self._exploration_strategy:
            action = self._exploration_strategy.pick(action_space)
        else:
            action = action_space.argmax()

        self._last_action = action
        self._last_observation = observation

        return action

