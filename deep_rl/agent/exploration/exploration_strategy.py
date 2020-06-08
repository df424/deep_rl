
from abc import ABC, abstractmethod
import numpy as np

class ExplorationStrategy(ABC):
    @abstractmethod
    def pick(self, action_space: np.ndarray) -> int:
        pass