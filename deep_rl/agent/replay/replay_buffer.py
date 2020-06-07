

class ReplayBuffer():
    def __init__(self, capacity: int):
        self._capacity = capacity
        self._memory = []