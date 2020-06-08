
import tensorflow as tf
import numpy as np

from deep_rl.agent.value_functions.value_function import ActionValueFunction

class DeepQNetwork(ActionValueFunction):
    def __init__(self, model: tf.keras.Model):
        self._model = model

    def forward(self, observation: np.ndarray) -> np.ndarray: 
        return self._model.predict(observation)