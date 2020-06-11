
from abc import ABC, abstractmethod
import numpy as np

class ExplorationStrategy(ABC):
    @abstractmethod
    def pick(self, action_space: np.ndarray, eval_mode:bool=False) -> int:
        pass