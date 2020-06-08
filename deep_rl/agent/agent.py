
import numpy as np
from deep_rl.logging import get_logger
from deep_rl.preprocessing import ObservationPreprocessor
from deep_rl.agent.value_functions import ActionValueFunction

log = get_logger('Agent')

class RLAgent():
    def __init__(self,
        action_value_function: ActionValueFunction,
        observation_preprocessor: ObservationPreprocessor = None
        ):

        self._observation_prep = observation_preprocessor
        self._action_value_fx = action_value_function

    def update(self, observation: np.ndarray, reward: float) -> int:
        processed_observation = self._observation_prep.prep(observation)
        return self._action_value_fx.forward(np.array([processed_observation])).argmax()

