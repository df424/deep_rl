
import numpy as np
from abc import ABC, abstractmethod
from torch.utils.tensorboard import SummaryWriter

class RLAgent(ABC):
    @abstractmethod
    def step(self, observation: np.ndarray, reward: float, done: bool) -> int:
        pass

    @abstractmethod
    def update(self, batch_size: int):
        pass

    @abstractmethod
    def save(self, path: str):
        pass

    @abstractmethod
    def eval(self, eval_mode: bool=True):
        pass

    @abstractmethod
    def log_metrics(self, writer: SummaryWriter):
        pass