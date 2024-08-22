"""
# Agent
    This module provides the implementation of a deep Q-network (DQN) agent. It includes the necessary classes and functions to train and deploy the agent in a given gymnasium environment.

    The agent uses a DQN to learn the optimal policy for the environment. It provides methods for training the agent, deploying the trained model, and interacting with the environment.

## Classes:
    - **DeepQNetwork**: A deep Q-network (DQN) class that represents the agent's policy.
    - **ReplayMemory**: A replay memory class that stores the agent's experiences.
    - **Agent**: The agent class that provides methods for training, deploying, and interacting with the environment.

## Functions:
    - *main*: The main function to show how should the environment work

## Author
    Miquel P. Baztan Grau

## Date
    21/08/2024
"""

import torch as T
import torch.nn as nn
import torch.optim as optim

import random
import numpy as np

from collections import deque

import yaml
import os

import gymnasium as gym
import itertools

from utils import log_message, save_graph

## Constants
DATE_FORMAT = "%d-%m-%Y %H:%M:%S"
RUNS_DIR = "runs"
SAVE_GRAPH_STEP = 100

class DeepQNetwork(nn.Module):
    """
    A Deep Q-Network (DQN) is a neural network used for estimation of
    Q-values in reinforcement learning problems.

    The DQN is a fully connected feed-forward network with one or more
    hidden layers. The output of the network is a vector of Q-values, one
    for each action in the action space.

    Attributes
    ----------
        nn.Module.attributes
            The attributes inherited from the `nn.Module` class (see `pytorch documentation <https://pytorch.org/docs/stable/generated/torch.nn.Module.html>`_ for more information).
        layers :nn.Sequential
            A sequential container of layers.
    
    Methods
    -------
        - *forward*: The forward pass of the network.
    """
    def __init__(self, state_dim : int, action_dim : int, hidden_dims : list[int]) -> None:
        """
        Initializes the DeepQNetwork using the `ReLU` activation.

        The network has a first layer of size `state_dim`, followed by
        a series of hidden layers with ReLU activation, and finally a
        last layer of size `action_dim`.

        Parameters
        ----------
        state_dim : int
            The dimension of the state space.
        action_dim : int
            The dimension of the action space.
        hidden_dims : list[int]
            A list of the dimensions of the hidden layers.

        Returns
        -------
            **None**
        """

        super(DeepQNetwork, self).__init__()
        layers = [] # A list to store the layers
        in_dim = state_dim
        for out_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, action_dim))
        self.layers = nn.Sequential(*layers) # Unpack the list

        # nn.Sequential is a container that allows you to define
        # a neural network in a sequential manner. Layers are added
        # in the order they are passed, and the input is passed
        # through each layer sequentially.

    def forward(self, x : T.Tensor) -> T.Tensor:
        """
        Performs the forward pass of the DeepQNetwork.

        Parameters
        ----------
        x : T.Tensor
            The input to the network.

        Returns
        -------
        T.Tensor
            The output of the network.
        """

        return self.layers(x)

class ReplayMemory:
    """
    A replay memory is a data structure that stores a fixed number of
    past transitions that can be sampled at random. The transitions
    are stored as a list of tuples, where each tuple contains the
    current state, the action taken, the reward received, the next
    state, and whether the episode ended or not.

    Attributes
    ----------
    memory : deque
        A deque is a double-ended queue that can be used to store
        elements in a specific order.
    capacity : int
        The maximum number of transitions that can be stored in the
        replay memory.

    Methods
    -------
     - *push*: Stores a transition in the replay memory.
     - *sample*: Samples a batch of transitions from the replay memory.
     - *__len__*: Returns the current size of the replay memory.
    """
    def __init__(self, capacity : int) -> None:
        """
        Initializes the replay memory.

        Parameters
        ----------
        capacity : int
            The maximum number of transitions that can be stored in the
            replay memory.

        Returns
        -------
            **None**
        """

        self.memory = deque(maxlen=capacity)
        self.capacity = capacity

    def push(self, state : T.Tensor, action : T.Tensor, reward : T.Tensor, next_state : T.Tensor, terminated : T.Tensor, truncated : T.Tensor) -> None:
        """
        Stores a transition in the replay memory. 
        
        Parameters
        ----------
        state : T.Tensor
            The current state.
        action : T.Tensor
            The action taken.
        reward : T.Tensor
            The reward received.
        next_state : T.Tensor
            The next state.
        terminated : T.Tensor
            Whether the episode ended or not.
        truncated : T.Tensor
            Whether the episode was truncated or not.
        
        Returns
        -------
            **None**
        """
        self.memory.append((state, action, reward, next_state, terminated, truncated))

    def sample(self, batch_size : int) -> list[tuple[T.Tensor, T.Tensor, T.Tensor, T.Tensor, T.Tensor, T.Tensor]]:
        """
        Sample a batch of transitions.

        Parameters
        ----------
        batch_size : int
            The number of transitions to sample.

        Returns
        -------
        list[tuple[T.Tensor, T.Tensor, T.Tensor, T.Tensor, T.Tensor, T.Tensor]]
            A list of tuples containing the sampled transitions.
        """

        return random.sample(self.memory, batch_size)

    def __len__(self):
        """
        Return the current size of the memory.

        Returns
        -------
        int
            The current size of the memory.
        """

        return len(self.memory)

class Agent:
    """
    An agent is an object that can take actions in an environment
    and learn from experiences.

    This agent will train in a `gymasium` environment (check out `gymnasium documentation <https://gymnasium.farama.org/>`_ for more information).

    Attributes
    ----------
    training : bool
        Whether the agent is in training mode or not.
    hyperparameter_set : str
        The name of the hyperparameter set to use.
    env_id : str
        The name of the environment.
    render_mode : str
        The render mode for the environment.
    env_params : dict
        A dictionary of environment-specific parameters.
    env : gym.Env
        The environment the agent is interacting with.
    num_states : int
        The number of states in the environment.
    num_actions : int
        The number of actions in the environment.
    alpha : float
        The learning rate.
    gamma : float
        The discount factor.
    hidden_dims : list[int]
        A list of the dimensions of the hidden layers.
    network_sync_rate : int
        The frequency at which the target network is updated.
    epsilon_init : float
        The initial exploration rate.
    epsilon_decay : float
        The decay rate of the exploration rate.
    epsilon_min : float
        The minimum exploration rate.
    replay_memory_size : int
        The size of the replay memory.
    batch_size : int
        The batch size.
    device : str
        The device to use for training.
    LOG_FILE : str
        The name of the log file.
    MODEL_FILE : str
        The name of the model file.
    GRAPH_FILE : str
        The name of the graph file.
    
    Methods
    -------
     - *run*: Runs the agent (deploy or train).
     - *select_action*: Selects an action given the current state (using epsilon-greedy if training, otherwise the best action).
     - *train*: Trains the agent.
     - *optimize*: Optimizes the neural network with the replay memory.
     - *deploy*: Loads the model and deploys the agent.
    """
    def __init__(self, hyperparameter_set : str, training : bool = True) -> None:
        """
        Initializes an Agent object with the specified hyperparameter set and training mode.

        Parameters
        ----------
            hyperparameter_set : str
                The name of the hyperparameter set to use.
            training : bool, optional
                Whether the agent is in training mode or not. Defaults to True.

        Returns
        -------
            **None**
        """

        self.training = training

        # Load hyperparameters
        with open('hyperparameters.yml', 'r') as file:
            hyperparameters = yaml.safe_load(file)[hyperparameter_set]
        
        self.hyperparameter_set = hyperparameter_set                                              # The name of the hyperparameter set
        
        # Environment parameters
        self.env_id      = hyperparameters['env_id']                                              # The name of the environment
        self.render_mode = 'human' if not self.training else None                                 # The render mode for the environment
        self.env_params  = hyperparameters.get('env_params',{})                                   # Get optional environment-specific parameters, default to empty dict
        self.env         = gym.make(self.env_id, render_mode=self.render_mode, **self.env_params) # Create the environment
        self.num_states  = self.env.observation_space.shape[0]                                    # Number of states in the environment
        self.num_actions = self.env.action_space.n                                                # Number of actions in the environment

        # Agent parameters
        self.alpha              = hyperparameters['alpha']                                        # learning rate
        self.gamma              = hyperparameters['gamma']                                        # discount rate
        self.hidden_dims        = hyperparameters['hidden_dims']                                  # List of number of nodes in each hidden layer
        self.network_sync_rate  = hyperparameters['network_sync_rate']                            # number of steps the agent takes before syncing the policy and target network
        
        # Exploration parameters
        self.epsilon_init       = hyperparameters['epsilon_init']                                 # 1 = 100% random actions
        self.epsilon_decay      = hyperparameters['epsilon_decay']                                # epsilon decay rate
        self.epsilon_min        = hyperparameters['epsilon_min']                                  # minimum epsilon value
        
        # Memory parameters
        self.replay_memory_size = hyperparameters['replay_memory_size']                           # size of replay memory
        self.batch_size         = hyperparameters['batch_size']                                   # size of the training data set sampled from the replay memory
        
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')                      # Device to run on

        # Path to Run info
        os.makedirs(RUNS_DIR, exist_ok=True)
        self.LOG_FILE   = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.log')
        self.MODEL_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.pt')
        self.GRAPH_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.png')

    def run(self) -> None:
        """
        Runs the agent based on its training mode.

        If the agent is in training mode, it calls the `train` method.
        Otherwise, it calls the `deploy` method.

        Returns
        -------
            **None**
        """

        # Run the agent
        if self.training:
            self.train()
        else:
            self.deploy()

    def select_action(self, state : T.Tensor) -> T.Tensor:
        """
        Selects an action given the current state.

        If the model is training, the agent uses epsilon-greedy to select an action. Otherwise, the agent selects the action with the highest Q-value.

        Parameters
        ----------
            state : T.Tensor
                The current state.

        Returns
        -------
            T.Tensor
                The selected action.
        """

        if self.training and np.random.random() < self.epsilon:
            action = self.env.action_space.sample()
            action = T.tensor(action, dtype=T.int64, device=self.device)

        else:
            with T.no_grad():
                # tensor([1 ,2, 3, ...]) ==> tensor([[1, 2, 3, ...]])
                action = self.policy_dqn(state.unsqueeze(0)).squeeze(0).argmax() # index of the best action

        return action

    def train(self) -> None:
        """
        Trains the agent using the Deep Q Network (DQN) algorithm.

        This function initializes the policy and target networks, creates the replay memory and the optimizer, and sets the initial epsilon value. It then enters a training loop where it repeatedly interacts with the environment to collect experiences. The epsilon-greedy policy is used to select actions, and the collected experiences are stored in the replay memory. After a certain number of experiences are collected, the agent starts training by sampling a batch of experiences from the replay memory and optimizing the policy network using the mean squared error loss. The epsilon value is updated after each training step to encourage exploration. The best reward achieved so far is tracked and the model is saved if a new best reward is obtained. The training graph is updated periodically.

        Returns
        -------
            **None**
        """

        # Create the policy and target networks
        self.policy_dqn = DeepQNetwork(self.num_states, self.num_actions, self.hidden_dims).to(self.device)
        self.target_dqn = DeepQNetwork(self.num_states, self.num_actions, self.hidden_dims).to(self.device)
        self.target_dqn.load_state_dict(self.policy_dqn.state_dict())

        # Create the replay memory
        self.memory = ReplayMemory(self.replay_memory_size)

        # Create the optimizer and loss function
        self.optimizer = optim.Adam(self.policy_dqn.parameters(), lr=self.alpha)
        self.loss_fn   = nn.MSELoss()

        # Initialize the epsilon value
        self.epsilon = self.epsilon_init

        # Initialize the reward and epsilon history 
        epsilon_history = [self.epsilon_init]
        rewards_per_episode = []

        # Initialize the best reward
        best_reward = -np.inf

        # Track the number of steps taken. Used for syncing the policy and target networks
        step_count = 0

        # Start training message
        log_message("Training starting...", DATE_FORMAT, self.LOG_FILE, 'w')

        # Training loop
        for episode in itertools.count():
            state, info = self.env.reset()
            state = T.tensor(state, dtype=T.float32, device=self.device)

            terminated = truncated = False
            episode_reward = 0.0

            while not terminated and not truncated:
                # Select an action with epsilon-greedy
                action = self.select_action(state)

                # Perform the action
                next_state, reward, terminated, truncated, info = self.env.step(action.item())

                # Acumulative_reward
                episode_reward += reward

                # Store the transition in the replay memory
                next_state = T.tensor(next_state, dtype=T.float32, device=self.device)
                reward     = T.tensor(reward, dtype=T.float32, device=self.device)
                terminated = T.tensor(terminated, dtype=T.float32, device=self.device)
                truncated  = T.tensor(truncated, dtype=T.float32, device=self.device)
                self.memory.push(state, action, reward, next_state, terminated, truncated)
                state = next_state
            
            rewards_per_episode.append(episode_reward)
            step_count += 1

            # Print episode statistics
            avg_reward = np.mean(rewards_per_episode[-100:])
            print('episode ', episode, 'reward %.2f' % episode_reward, 'average score %.2f' % avg_reward, 'epsilon %.2f' % self.epsilon)

            # If improving, save the model
            if episode_reward >= best_reward:
                best_reward = episode_reward
                T.save(self.policy_dqn.state_dict(), self.MODEL_FILE)
                log_message(f"New best reward {best_reward:0.1f} at episode {episode}, saving model...", DATE_FORMAT, self.LOG_FILE, 'a')

            # Update the graph
            if episode % SAVE_GRAPH_STEP == 0:
                save_graph(self.GRAPH_FILE, rewards_per_episode, epsilon_history)

            # If enough experiences, start training
            if len(self.memory) > self.batch_size:
                batch = self.memory.sample(self.batch_size)
                self.optimize(batch)

                # Update epsilon
                self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
                epsilon_history.append(self.epsilon)

                # Copy policy network to target network after a certain number of steps
                if step_count % self.network_sync_rate == 0:
                    self.target_dqn.load_state_dict(self.policy_dqn.state_dict())
                    step_count = 0
        
    def optimize(self, batch : list[tuple[T.Tensor, T.Tensor, T.Tensor, T.Tensor, T.Tensor, T.Tensor]]) -> None:
        """
        Optimizes the model using a batch of experiences.

        Parameters
        ----------
            batch : list[tuple[T.Tensor, T.Tensor, T.Tensor, T.Tensor, T.Tensor, T.Tensor]])
                A list of tuples containing the batch of experiences.
                Each tuple contains the following elements:
                    - states (T.Tensor): The states of the environment.
                    - actions (T.Tensor): The actions taken in the environment.
                    - rewards (T.Tensor): The rewards received for the actions.
                    - new_states (T.Tensor): The new states after taking the actions.
                    - terminations (T.Tensor): The termination status of the episodes.
                    - truncations (T.Tensor): The truncation status of the episodes.

        Returns
        -------
            **None**
        """

        # Transpose the list of experiences and separate each element
        states, actions, rewards, new_states, terminations, truncations = zip(*batch)

        # Stack tensors to create batch tensors
        # tensor([[1,2,3]])
        states       = T.stack(states)
        actions      = T.stack(actions)
        rewards      = T.stack(rewards)
        new_states   = T.stack(new_states)
        terminations = T.stack(terminations)

        # Calculate target Q values (expected returns)
        with T.no_grad():
            target_q = rewards + (1-terminations)*self.gamma*self.target_dqn(new_states).max(dim=1)[0]
            '''
                target_dqn(new_states)  ==> tensor([[1,2,3],[4,5,6]])
                    .max(dim=1)         ==> torch.return_types.max(values=tensor([3,6]),indices=tensor([3, 0, 0, 1]))
                        [0]             ==> tensor([3, 6])
            
            '''

        # Calcuate Q values from current policy
        current_q = self.policy_dqn(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()

        # Calculate loss
        loss = self.loss_fn(current_q, target_q)

        # Optimize the model
        self.optimizer.zero_grad()  # Clear gradients
        loss.backward()             # Calculate gradients (backpropagation)
        self.optimizer.step()       # Update weights

    def deploy(self) -> None:
        """
        Deploys the trained deep Q-network (DQN) model to the environment.

        This function loads the best model from the specified file and evaluates it.
        It then enters a loop where it resets the environment, selects an action
        based on the current state, performs the action, and updates the state.
        The function continues this process until the episode is terminated or
        truncated.

        Returns
        -------
            **None**
        """

        # Load the best model
        self.policy_dqn = DeepQNetwork(self.num_states, self.num_actions, self.hidden_dims).to(self.device)
        self.policy_dqn.load_state_dict(T.load(self.MODEL_FILE))
        self.policy_dqn.eval()
    
        for episode in itertools.count():
            state, info = self.env.reset()
            state = T.tensor(state, dtype=T.float32, device=self.device)

            terminated = truncated = False
            episode_reward = 0.0

            while not terminated and not truncated:
                # Select an action with epsilon-greedy
                action = self.select_action(state)

                # Perform the action
                next_state, reward, terminated, truncated, info = self.env.step(action.item())
                next_state = T.tensor(next_state, dtype=T.float32, device=self.device)

                # Acumulative_reward
                episode_reward += reward

                state = next_state

def main():
    # Parse command line inputs
    parser = argparse.ArgumentParser(description='Train or test model.')
    parser.add_argument('hyperparameters', help='')
    parser.add_argument('--train', help='Training mode', action='store_true')
    args = parser.parse_args()

    agent = Agent(hyperparameter_set=args.hyperparameters, training=args.train)
    agent.run()

if __name__ == '__main__':
    from gymnasium.envs.registration import register
    import argparse
    register(id="ENVIRONMENT-v1", entry_point="gym_environment:CUSTOM_ENVIRONMENT")
    main()

