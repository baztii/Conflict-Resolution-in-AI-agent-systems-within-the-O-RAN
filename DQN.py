import torch as T
import torch.nn as nn
import torch.nn.functional as F
from ENVIRONMENT import ENVIRONMENT as ENV
import random
import numpy as np

from collections import deque

device = "cuda" if T.cuda.is_available() else "cpu"

class DQN(ENV):
    def __init__(self,data):
        super().__init__(data)

class ReplayMemory(object):
    def __init__(self, size : int, device : str = 'cpu'):
        self.buffer = deque(maxlen=size)
        self.device = device

    def __len__(self):
        return len(self.buffer)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state,action,reward,next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(len(self.buffer, batch_size)))
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    


def main():
    pass

if __name__ == '__main__':
    main()