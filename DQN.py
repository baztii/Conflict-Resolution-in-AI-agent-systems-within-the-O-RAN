import torch as T
import torch.nn as nn
import torch.nn.functional as F

import random
import numpy as np

from collections import deque

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(DQN, self).__init__()
        self.fc1    = nn.Linear(state_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.output(x)

class ReplayMemory(object):
    def __init__(self, size : int, device : str = 'cpu'):
        self.buffer = deque(maxlen=size)
        self.device = device

    def __len__(self):
        return len(self.buffer)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state,action,reward,next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, min(len(self.buffer), batch_size))

def main():
    state_dim = 12
    action_dim = 2
    net = DQN(state_dim, action_dim)
    state = T.randn(10, state_dim)
    output = net(state)
    print(output)

if __name__ == '__main__':
    main()