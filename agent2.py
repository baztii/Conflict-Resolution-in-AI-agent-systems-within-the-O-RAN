import gymnasium as gym
from gymnasium.envs.registration import register
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

import random
import torch
import torch.nn as nn
import yaml

from DQN import ReplayMemory, DQN

from datetime import datetime, timedelta
import argparse
import itertools

import os

# For printing date and time
DATE_FORMAT = "%m-%d %H:%M:%S"

# Directory for saving run info
RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)

# 'Agg': used to generate plots as images and save them to a file instead of rendering to screen
matplotlib.use('Agg')

device = "cuda" if torch.cuda.is_available() else "cpu"

# Deep Q-Learning Agent 
class Agent:
    def __init__(self, hyperparameter_set):
        with open('hyperparameters.yml', 'r') as file:
            all_hyperparameter_set = yaml.safe_load(file)
            hyperparameters = all_hyperparameter_set[hyperparameter_set]
        
        self.hyperparameter_set = hyperparameter_set
        
        # Hyperparameters
        self.env_id             = hyperparameters['env_id']                     # gym environment id
        self.replay_memory_size = hyperparameters['replay_memory_size']         # size of replay memory
        self.mini_batch_size    = hyperparameters['mini_batch_size']            # size of the training data set sampled from the replay memory
        self.epsilon_init       = hyperparameters['epsilon_init']               # 1 = 100% random actions
        self.epsilon_decay      = hyperparameters['epsilon_decay']              # epsilon decay rate
        self.epsilon_min        = hyperparameters['epsilon_min']                # minimum value of epsilon
        self.network_sync_rate  = hyperparameters['network_sync_rate']          # how often to sync the policy network with the target network
        self.gamma              = hyperparameters['gamma']                      # discount factor
        self.alpha              = hyperparameters['alpha']                      # learning rate
        self.fc1_nodes          = hyperparameters['fc1_nodes']                  # number of nodes in the first fully connected layer
        self.stop_on_reward     = hyperparameters['stop_on_reward']             # stop training when agent reaches this reward
        self.env_make_params    = hyperparameters.get('env_make_params', {})    # Get optional environment-specific parameters, default to empty dict

        # Neural Network
        self.loss_fn = nn.MSELoss() # NN loss function. MSE=Mean Squared Error (can be swapped to something else)
        self.optimizer = None # NN optimizer

        # Path to Run info
        self.LOG_FILE   = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.log')
        self.MODEL_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.pt')
        self.GRAPH_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.png')

    def run(self, is_training=True, render=False):
        if is_training:
            start_time = datetime.now()
            last_graph_update_time = start_time

            log_message = f"{start_time.strftime(DATE_FORMAT)}: Training starting..."
            print(log_message)
            with open(self.LOG_FILE, 'w') as file:
                file.write(log_message + '\n')

        env = gym.make(self.env_id, render_mode="human" if render else None, **self.env_make_params)

        num_states  = env.observation_space.shape[0]
        num_actions = env.action_space.n

        print(num_states, num_actions)

        rewards_per_episode = []
        global_rewards_total = []

        policy_dqn = DQN(num_states, num_actions, self.fc1_nodes).to(device)

        if is_training:
            memory = ReplayMemory(self.replay_memory_size)
            epsilon = self.epsilon_init
            target_dqn = DQN(num_states, num_actions, self.fc1_nodes).to(device)
            target_dqn.load_state_dict(policy_dqn.state_dict())
            
            # Track the number of steps taken. Used for syncing policy => Target network
            step_count = 0
            epsilon_history = []
            best_reward = -9999999

            self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.alpha)
        
        else:
            policy_dqn.load_state_dict(torch.load(self.MODEL_FILE))
            policy_dqn.eval()

        global_reward = 0

        for episode in itertools.count():
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=device)

            terminated = False
            episode_reward = 0.0

            while not terminated and episode_reward < self.stop_on_reward: # Episode loop
                # Epsilon greedy algorithm
                if is_training and random.random() < epsilon: 
                    action = env.action_space.sample()
                    action = torch.tensor(action, dtype=torch.int64, device=device)
                else:
                    with torch.no_grad():
                        # tensor([1 ,2, 3, ...]) ==> tensor([[1, 2, 3, ...]])
                        action = policy_dqn(state.unsqueeze(0)).squeeze().argmax() # index of the best action
                        #print(action)

                # Processing:
                new_state, reward, terminated, _, info = env.step(action.item())

                #Accumulate reward
                episode_reward += reward

                # Convert new state and reward to tensors on device
                new_state = torch.tensor(new_state, dtype=torch.float32, device=device)
                reward = torch.tensor(reward, dtype=torch.float32, device=device)

                if is_training:
                    # Save experience into memory
                    memory.push(state, action, reward, new_state, terminated)
                    # Increment step counter
                    step_count += 1
                
                # Move to new state
                state = new_state
            if episode_reward == 500_000:
                global_reward += 1
            
            global_rewards_total.append(global_rewards_total)
            rewards_per_episode.append(episode_reward)

                        # Save model when new best reward is obtained.
            if is_training:
                if episode_reward == 500_000:#episode_reward > best_reward:
                    log_message = f"{datetime.now().strftime(DATE_FORMAT)}: New best reward {episode_reward:0.1f} ({(episode_reward-best_reward)/best_reward*100:+.1f}%) at episode {episode}, saving model..."
                    print(log_message)
                    with open(self.LOG_FILE, 'a') as file:
                        file.write(log_message + '\n')

                    torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
                    best_reward = episode_reward
                
                               # Update graph every x seconds
                current_time = datetime.now()
                if current_time - last_graph_update_time > timedelta(seconds=10):
                    self.save_graph(rewards_per_episode, epsilon_history)
                    last_graph_update_time = current_time

                # If enough experience has been collected
                if len(memory) > self.mini_batch_size:
                    # Sample from memory
                    mini_batch = memory.sample(self.mini_batch_size)
                    self.optimize(mini_batch, policy_dqn, target_dqn)

                    epsilon = max(epsilon*self.epsilon_decay, self.epsilon_min)
                    epsilon_history.append(epsilon)

                    # Copy policy network to target network after a certain number of steps
                    if step_count > self.network_sync_rate:
                        target_dqn.load_state_dict(policy_dqn.state_dict())
                        step_count = 0
    
    def save_graph(self, rewards_per_episode, epsilon_history):
        # Save plots
        fig = plt.figure(1)

        # Plot average rewards (Y-axis) vs episodes (X-axis)
        mean_rewards = np.zeros(len(rewards_per_episode))
        for x in range(len(mean_rewards)):
            mean_rewards[x] = np.mean(rewards_per_episode[max(0, x-99):(x+1)])
        plt.subplot(121) # plot on a 1 row x 2 col grid, at cell 1
        # plt.xlabel('Episodes')
        plt.ylabel('Mean Rewards')
        plt.plot(mean_rewards)

        # Plot epsilon decay (Y-axis) vs episodes (X-axis)
        plt.subplot(122) # plot on a 1 row x 2 col grid, at cell 2
        # plt.xlabel('Time Steps')
        plt.ylabel('Epsilon Decay')
        plt.plot(epsilon_history)

        plt.subplots_adjust(wspace=1.0, hspace=1.0)

        # Save plots
        fig.savefig(self.GRAPH_FILE)
        plt.close(fig)
        
    def optimize(self, mini_batch, policy_dqn, target_dqn):
        # Transpose the list of experiences and separate each element
        states, actions, rewards, new_states, terminations = zip(*mini_batch)

        # Stack tensors to create batch tensors
        # tensor([[1,2,3]])
        states       = torch.stack(states)
        actions      = torch.stack(actions)
        rewards      = torch.stack(rewards)
        new_states   = torch.stack(new_states)
        terminations = torch.tensor(terminations).float().to(device)

        with torch.no_grad():
            # Calculate target Q values (expected returns)
            target_q = rewards + (1-terminations)*self.gamma*target_dqn(new_states).max(dim=1)[0]
            '''
                target_dqn(new_states)  ==> tensor([[1,2,3],[4,5,6]])
                    .max(dim=1)         ==> torch.return_types.max(values=tensor([3,6]),indices=tensor([3, 0, 0, 1]))
                        [0]             ==> tensor([3, 6])
            
            '''

        # Calculate Q values from current policy
        current_q = policy_dqn(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()
        '''
            policy_dqn(states)  ==> tensor([[1,2,3],[4,5,6]])
                actions.unsqueeze(dim=1)
                .gather(1, actions.unsqueeze(dim=1))  ==>
                    .squeeze()                    ==>
        '''

        # Compute the loss for the whole minibatch
        loss = self.loss_fn(current_q, target_q)

        # Optimize the model
        self.optimizer.zero_grad()  # Clear gradients
        loss.backward()             # Compute gradients (backpropagation)
        self.optimizer.step()       # Update network parameters i.e. wieghts and biases

        """ This is not very efficient and pytorch can optimize this for us
        for state, action, new_state, reward, terminated in mini_batch:
            if terminated:
                target_q = reward
            else:
                with torch.no_grad():
                    target_q = reward + self.gamma * target_dqn(new_state).max()
            
            current_q = policy_dqn(state)
            
            # Compute the loss for the whole minibatch
            loss = self.loss_fn(current_q, target_q)

            # Optimize the model
            self.optimizer.zero_grad()  # Clear gradients
            loss.backward()             # Compute gradients (backpropagation)
            self.optimizer.step()       # Update network parameters i.e. wieghts and biases
        """

if __name__ == "__main__":
    register(id="ENVIRONMENT-v0", entry_point="ENVIRONMENT:ENVIRONMENT")

    # Parse command line inputs
    parser = argparse.ArgumentParser(description='Train or test model.')
    parser.add_argument('hyperparameters', help='')
    parser.add_argument('--train', help='Training mode', action='store_true')
    args = parser.parse_args()

    dql = Agent(hyperparameter_set=args.hyperparameters)

    if args.train:
        dql.run(is_training=True)
    else:
        dql.run(is_training=False, render=True)
