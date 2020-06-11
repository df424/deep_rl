
import random
import numpy as np

from deep_rl.agent.exploration.exploration_strategy import ExplorationStrategy

class EpsilonGreedyExplorationStrategy(ExplorationStrategy):
    def __init__(self, initial_epsilon: float, final_epsilon:float=None, decay:float=None, eval_epsilon:float=0):
        self._epsilon = initial_epsilon
        self._final_epsilon = final_epsilon
        self._decay = decay
        self._eval_epsilon = eval_epsilon

    def pick(self, action_space: np.ndarray, eval_mode:bool=False) -> int:
        # Shoudl we pick a random action?
        if eval_mode:
            use_rand = random.random() < self._eval_epsilon
        else:
            use_rand = random.random() < self._epsilon
            # Either way we should decay epsilon if we aren't in eval mode.
            self._epsilon = max(self._epsilon-self._decay, self._final_epsilon)

        if use_rand:
            return random.randint(0, len(action_space)-1)
        else:
            return action_space.argmax()
        
        