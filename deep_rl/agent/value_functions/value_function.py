
from abc import ABC, abstractmethod
import numpy as np

class ValueFunction(ABC):
    pass

class ActionValueFunction(ABC):
    def forward(self, observation: np.ndarray) -> np.ndarray:
        pass