
import numpy as np
from abc import ABC, abstractmethod

class RLAgent(ABC):
    @abstractmethod
    def update(self, observation: np.ndarray, reward: float, done: bool) -> int:
        pass

