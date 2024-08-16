import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import random
import numpy as np

from collections import deque

class DeepQNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super(DeepQNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)     

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def forward(self, state):
        x = F.relu(self.fc1(state.float()))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)

        return actions

class Agent():
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions,
                 max_mem_size=100000, eps_end=0.01, eps_dec=5e-4):
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        self.action_space = [i for i in range(n_actions)]
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.mem_cntr = 0

        self.Q_eval = DeepQNetwork(self.lr, n_actions=n_actions,
                                   input_dims=input_dims, fc1_dims=256,
                                   fc2_dims=256)
        
        self.state_memory = np.zeros((self.mem_size, *input_dims),
                                     dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims),
                                         dtype=np.float32)
    
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)

        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)
    
    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = state_
        self.terminal_memory[index] = done
        self.mem_cntr += 1
    
    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = T.tensor([observation]).to(self.Q_eval.device)
            actions = self.Q_eval.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def learn(self):
        if self.mem_cntr < self.batch_size:
            return
        
        self.Q_eval.optimizer.zero_grad()
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        batch_index = np.arange(self.batch_size, dtype=np.int32)

        state_batch = T.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        new_state_batch = T.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)
        reward_batch = T.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        terminal_batch = T.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)

        action_batch = self.action_memory[batch]

        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
        q_next = self.Q_eval.forward(new_state_batch)
        q_next[terminal_batch] = 0.0

        q_target = reward_batch + self.gamma * T.max(q_next, dim=1)[0]

        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()

        self.epsilon = self.epsilon - self.eps_dec \
            if self.epsilon > self.eps_min else self.eps_min

    def deploy(self, observation):
        state = T.tensor([observation]).to(self.Q_eval.device)
        actions = self.Q_eval.forward(state)
        action = T.argmax(actions).item()

        return action
    
from utils import save_graph
def main():
    import gymnasium as gym

    from gymnasium.envs.registration import register
    register(id="ENVIRONMENT-v1", entry_point="gym_environment:CUSTOM_ENVIRONMENT")

    env = gym.make('ENVIRONMENT-v1')

    agent = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=250,
                  eps_end=0.01, input_dims=[15], lr=0.00001)
    scores, eps_history = [], []
    n_games = 100_000

    train = True
    if train:
        for i in range(n_games):
            done = False
            observation, info = env.reset()
            score = 0
            it = 0
            while not done and it < 5:
                it += 1
                action = agent.choose_action(observation)
                observation_, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                score += reward
                agent.store_transition(observation, action, reward, observation_, done)
                agent.learn()
                observation = observation_
            scores.append(score)
            eps_history.append(agent.epsilon)

            # Save the agent
            T.save(agent.Q_eval.state_dict(), 'ccc.pt')

            avg_score = np.mean(scores[-100:])
            if avg_score < 600:
                agent.epsilon = 1
            if i%1000 == 0: save_graph("ccc.png", scores, eps_history)

            print('episode ', i, 'score %.2f' % score, 'average score %.2f' % avg_score, 'epsilon %.2f' % agent.epsilon)
    else:
        agent.Q_eval.load_state_dict(T.load('ccc.pt'))
        agent.Q_eval.eval()
        for i in range(1):
            done = False
            observation, info = env.reset()
            it = 0
            print(env.get_wrapper_attr('L'))

            while not done and it < 5:
                it += 1
                action = agent.deploy(observation)
                observation_, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                observation = observation_
                print(env.get_wrapper_attr('P'))
                print(env.get_wrapper_attr('L'))
                


    x = [i+1 for i in range(n_games)]
    filename = 'lunar_lander.png'

if __name__ == '__main__':
    main()
    