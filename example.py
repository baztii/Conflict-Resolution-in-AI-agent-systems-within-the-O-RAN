import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random

class ENVIRONMENT:
    def __init__(self, N, K, M, userDistr, g, B, Pmin, Pmax, buffSize, T, sigma, lamda):
        self.lamda = lamda
        self.N = N
        self.K = K
        self.M = M
        self.userDistr = userDistr
        self.g = g
        self.B = B
        self.Pmin = Pmin
        self.Pmax = Pmax
        self.P = [[0 for m in range(M)] for n in range(N)]
        self.buffSize = buffSize
        self.T = T
        self.RBGusers = [[0 for m in range(M)] for n in range(N)]
        self.L = [0 for k in range(K)]
        self.sigma = sigma

    def __str__(self):
        return f"ENVIRONMENT(N={self.N}, K={self.K}, M={self.M}, ...)"


class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, action_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class Agent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.learning_rate = 0.001
        self.model = DQN(state_size, action_size)
        self.target_model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.update_target_model()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state)
        act_values = self.model(state)
        return np.argmax(act_values.detach().numpy())

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_state = torch.FloatTensor(next_state)
                target += self.gamma * torch.max(self.target_model(next_state))
            target_f = self.model(torch.FloatTensor(state)).detach().numpy()
            target_f[action] = target
            target_f = torch.FloatTensor(target_f)
            self.model.train()
            self.optimizer.zero_grad()
            output = self.model(torch.FloatTensor(state))
            loss = nn.MSELoss()(output, target_f)
            loss.backward()
            self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


# Example usage
if __name__ == "__main__":
    N, K, M = 3, 10, 5
    userDistr = [random.randint(0, N-1) for _ in range(K)]
    g = np.random.rand(N, K)
    B = np.random.rand(N, M)
    Pmin, Pmax = 0.1, 1.0
    buffSize = 100
    T = 1.0
    sigma = 0.1
    lamda = 0.5

    env = ENVIRONMENT(N, K, M, userDistr, g, B, Pmin, Pmax, buffSize, T, sigma, lamda)
    state_size = N * M + K
    action_size = M
    agent = Agent(state_size, action_size)

    for e in range(1000):
        state = np.random.rand(state_size)  # Initialize state
        for time in range(500):
            action = agent.act(state)
            next_state = np.random.rand(state_size)  # Placeholder for next state
            reward = random.random()  # Placeholder for reward
            done = time == 499  # Placeholder for done signal
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                agent.update_target_model()
                break
        if len(agent.memory) > 32:
            agent.replay(32)