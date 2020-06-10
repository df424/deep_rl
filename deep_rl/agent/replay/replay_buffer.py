import random

class ReplayBuffer():
    def __init__(self, capacity: int):
        self._capacity = capacity
        self._memory = []
        self._store_idx = 0

    def store(self, transition):
        if len(self._memory) < self._capacity:
            self._memory.append(transition)
        else:
            # Just replace the last transition.
            self._memory[self._store_idx] = transition

            # Incrememnt the store index and wrap it around.
            self._store_idx += 1
            if self._store_idx >= self._capacity:
                self._store_idx = 0

    def sample(self, n: int):
        return random.sample(self._memory, min(n, len(self._memory)))

    def __len__(self):
        return len(self._memory)
        
        
