
from abc import ABC, abstractmethod
import numpy as np

class ObservationPreprocessor(ABC):
    @abstractmethod
    def prep(self, observation: np.ndarray) -> np.ndarray:
        pass