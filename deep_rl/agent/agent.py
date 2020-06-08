
import numpy as np
from deep_rl.logging import get_logger
from deep_rl.preprocessing import ObservationPreprocessor

log = get_logger('Agent')

class RLAgent():
    def __init__(self,
        observation_preprocessor: ObservationPreprocessor = None
        ):

        self._observation_prep = observation_preprocessor

    def update(self, observation: np.ndarray, reward: float) -> int:
        processed_observation = self._observation_prep.prep(observation)
        log.info(f'Observation Shape: {observation.shape}, Processed Observation : {processed_observation.shape}, Reward: {reward}')
        return 0

