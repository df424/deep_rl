
from deep_rl.agent.exploration.exploration_strategy import ExplorationStrategy

class EpsilonGreedyExplorationStrategy(ExplorationStrategy):
    def __init__(self, initial_epsilon: float, final_epsilon:float=None, decay:float=None):
        self._initial_epsilon = initial_epsilon
        self._final_epsilon = final_epsilon
        self._decay = decay

        