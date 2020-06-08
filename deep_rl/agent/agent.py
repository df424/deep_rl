
import numpy as np
from deep_rl.logging import get_logger

log = get_logger('Agent')

class RLAgent():
    def __init__(self):
        pass

    def update(self, observation: np.ndarray, reward: float) -> int:
        log.debug(f'Observation Shape: {observation.shape}, Observation Mean: {observation.mean()}, Reward: {reward}')
        return 0

