
import random
import numpy as np

from deep_rl.agent.exploration.exploration_strategy import ExplorationStrategy

class EpsilonGreedyExplorationStrategy(ExplorationStrategy):
    def __init__(self, initial_epsilon: float, final_epsilon:float=None, decay:float=None):
        self._epsilon = initial_epsilon
        self._final_epsilon = final_epsilon
        self._decay = decay

    def pick(self, action_space: np.ndarray) -> int:
        # Shoudl we pick a random action?
        use_rand = random.random() < self._epsilon

        # Either way we should decay epsilon.
        self._epsilon = min(self._epsilon-self._decay, self._final_epsilon)

        if use_rand:
            return random.randint(0, len(action_space)-1)
        else:
            return action_space.argmax()
        
        