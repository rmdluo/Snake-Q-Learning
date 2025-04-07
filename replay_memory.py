### Sourced from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html#replay-memory
from collections import deque, namedtuple
import random


Transition = namedtuple("Transition", ("state", "action", "reward", "next_state"))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
    def is_full(self):
        return len(self.memory) == self.memory.maxlen
    
    def get_percent_full(self):
        return len(self.memory) / self.memory.maxlen
