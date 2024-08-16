import gymnasium as gym
import numpy as np

import matplotlib.pyplot as plt
import matplotlib

import random
import torch as T
import torch.nn as nn

import yaml
import json

from DQN import DQN, ReplayMemory
from ENVIRONMENT import ENVIRONMENT as ENV
from ONL import asserts

from datetime import datetime, timedelta
import argparse
import itertools

import os

DATE_FORMAT = "%d-%m-%Y %H:%M:%S"
RUNS_DIR = "runs"

os.makedirs(RUNS_DIR, exist_ok=True)
matplotlib.use('Agg')
device = "cuda" if T.cuda.is_available() else "cpu"

class Agent():
    def __init__(self, hyperparameter_set):
        with open('hyperparameters.yml', 'r') as file:
            all_hyperparameter_sets = yaml.safe_load(file)
            hyperparameters = all_hyperparameter_sets[hyperparameter_set]
        
        self.hyperparameter_set = hyperparameter_set
        self.env_id             = hyperparameters['env_id']
        self.learning_rate_a    = hyperparameters['alpha']        # learning rate (alpha)
        self.discount_factor_g  = hyperparameters['gamma']      # discount rate (gamma)
        self.network_sync_rate  = hyperparameters['network_sync_rate']      # number of steps the agent takes before syncing the policy and target network
        self.replay_memory_size = hyperparameters['replay_memory_size']     # size of replay memory
        self.mini_batch_size    = hyperparameters['mini_batch_size']        # size of the training data set sampled from the replay memory
        self.epsilon_init       = hyperparameters['epsilon_init']           # 1 = 100% random actions
        self.epsilon_decay      = hyperparameters['epsilon_decay']          # epsilon decay rate
        self.epsilon_min        = hyperparameters['epsilon_min']            # minimum epsilon value
        self.stop_on_reward     = hyperparameters['stop_on_reward']         # stop training after reaching this number of rewards
        self.fc1_nodes          = hyperparameters['fc1_nodes']              # number of nodes in the first hidden layer
        self.nData              = hyperparameters['nData']                  # Data set
        self.env_make_params    = hyperparameters.get('env_make_params',{}) # Get optional environment-specific parameters, default to empty dict

        # Neural Network
        self.loss_fn = nn.MSELoss()          # NN Loss function. MSE=Mean Squared Error can be swapped to something else.
        self.optimizer = None                # NN Optimizer. Initialize later.

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
            
        file = f"tests/test{self.nData}/data.json"
        with open(file, 'r') as data_file: # load the data
            data = json.load(data_file)
        
        asserts(data)
        
        env = ENV(data, render)

        num_actions = env.n_action_space()
        num_states  = env.m_state_space()

        print(num_states, num_actions)

        #print(f"Nume of actinos {num_actions} and num_states {num_states}")
        rewards_per_episode = []
        policy_dqn = DQN(num_states, num_actions, self.fc1_nodes).to(device)

        if is_training:
            # Initialize epsilon
            epsilon = self.epsilon_init

            # Initialize replay memory
            memory = ReplayMemory(self.replay_memory_size)

            # Create the target network and make it identical to the policy network
            target_dqn = DQN(num_states, num_actions, self.fc1_nodes).to(device)
            target_dqn.load_state_dict(policy_dqn.state_dict())

            # Policy network optimizer. "Adam" optimizer can be swapped to something else.
            self.optimizer = T.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)

            # List to keep track of epsilon decay
            epsilon_history = []

            # Track number of steps taken. Used for syncing policy => target network.
            step_count=0

            # Track best reward
            best_reward = -9999999

        else:
            # Load learned policy
            policy_dqn.load_state_dict(T.load(self.MODEL_FILE))

            # switch model to evaluation mode
            policy_dqn.eval()
        
        for episode in itertools.count():
            #print("episode: ", episode)
            state = env.reset()
            state = T.tensor(state, dtype=T.float, device=device) # Convert state to tensor directly on device

            terminated = False      # True when agent reaches goal or fails
            episode_reward = 0.0    # Used to accumulate rewards per episode
            it = 0

            action_list = []

            while not terminated and it < 100: # episode_reward < self.stop_on_reward
                it += 1	
                #print("episode_reward: ", episode_reward)
                # Select action based on epsilon-greedy
                if is_training and random.random() < epsilon:
                    # select random action
                    action = env.action_space_sample()
                    action = T.tensor(action, dtype=T.int64, device=device)
                else:
                    # select best action
                    with T.no_grad():
                        # state.unsqueeze(dim=0): Pytorch expects a batch layer, so add batch dimension i.e. tensor([1, 2, 3]) unsqueezes to tensor([[1, 2, 3]])
                        # policy_dqn returns tensor([[1], [2], [3]]), so squeeze it to tensor([1, 2, 3]).
                        # argmax finds the index of the largest element.
                        action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax()

                action_list.append(int(action))
                
                new_state, reward, terminated = env.step(action)

                episode_reward += reward

                # Convert new state and reward to tensors on device
                new_state = T.tensor(new_state, dtype=T.float, device=device)
                reward = T.tensor(reward, dtype=T.float, device=device)
                terminated = T.tensor(terminated, dtype=T.float, device=device)

                if is_training:
                    # Save experience into memory
                    memory.push(state, action, reward, new_state, terminated)

                    # Increment step counter
                    step_count+=1

                # Move to the next state
                
                state = new_state
            
            T.save(policy_dqn.state_dict(), self.MODEL_FILE)


            if not is_training:
                print(action_list)
                print(episode_reward)
                exit(0)

            # Keep track of the rewards collected per episode.
            rewards_per_episode.append(episode_reward)

            # Save model when new best reward is obtained.
            if is_training:
                if episode_reward > best_reward:
                    log_message = f"{datetime.now().strftime(DATE_FORMAT)}: New best reward {episode_reward} ({(episode_reward-best_reward)/best_reward*100:+.1f}%) at episode {episode}, saving model..." ## {episode_reward:0.1f}
                    print(log_message)
                    with open(self.LOG_FILE, 'a') as file:
                        file.write(log_message + '\n')

                    T.save(policy_dqn.state_dict(), self.MODEL_FILE)
                    best_reward = episode_reward
                
                                # Update graph every x seconds
                current_time = datetime.now()
                if current_time - last_graph_update_time > timedelta(seconds=10):
                    self.save_graph(rewards_per_episode, epsilon_history)
                    last_graph_update_time = current_time

                # If enough experience has been collected
                if len(memory)>self.mini_batch_size and terminated:
                    mini_batch = memory.sample(self.mini_batch_size)
                    self.optimize(mini_batch, policy_dqn, target_dqn)

                    # Decay epsilon
                    epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
                    epsilon_history.append(epsilon)

                    # Copy policy network to target network after a certain number of steps
                    if step_count > self.network_sync_rate:
                        target_dqn.load_state_dict(policy_dqn.state_dict())
                        step_count=0
                        
            #if episode_reward > 228333: exit(0)
        
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
        states, actions, rewards, new_states, terminations = zip(*mini_batch)
        
        states = T.stack(states)
        actions = T.stack(actions)
        new_states = T.stack(new_states)
        rewards = T.stack(rewards)
        terminations = T.stack(terminations)

        with T.no_grad():
            # Calculate target Q values (expected returns)
            target_q = rewards + (1-terminations) * self.discount_factor_g * target_dqn(new_states).max(dim=1)[0]
            '''
                target_dqn(new_states)  ==> tensor([[1,2,3],[4,5,6]])
                    .max(dim=1)         ==> torch.return_types.max(values=tensor([3,6]), indices=tensor([3, 0, 0, 1]))
                        [0]             ==> tensor([3,6])
            '''

        # Calcuate Q values from current policy
        current_q = policy_dqn(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()
        '''
            policy_dqn(states)  ==> tensor([[1,2,3],[4,5,6]])
                actions.unsqueeze(dim=1)
                .gather(1, actions.unsqueeze(dim=1))  ==>
                    .squeeze()                    ==>
        '''

        # Calculate loss

        loss = self.loss_fn(current_q, target_q)

        # Optimize the model (backpropagation)
        self.optimizer.zero_grad()  # Clear gradients
        loss.backward()             # Compute gradients
        self.optimizer.step()       # Update network parameters i.e. weights and biases

def main():
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

if __name__ == '__main__':
    main()


