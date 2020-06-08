
import torch
import numpy as np

from deep_rl.agent.value_functions.value_function import ActionValueFunction

class DeepQNetwork(ActionValueFunction):
    def __init__(self, model: torch.nn.Module):
        self._model = model

    def forward(self, observation: np.ndarray) -> np.ndarray: 
        return self._model.forward(torch.tensor(observation, dtype=torch.float32))