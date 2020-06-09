
import numpy as np
from abc import ABC, abstractmethod

class RLAgent(ABC):
    @abstractmethod
    def step(self, observation: np.ndarray, reward: float, done: bool) -> int:
        pass

    @abstractmethod
    def update(self, batch_size: int):
        pass
