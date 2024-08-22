"""
# gym_environment
### Custom Environment for Reinforcement Learning
    
   This module defines a custom environment for reinforcement learning, 
   inheriting from the `ENVIRONMENT` class and the `gym.Env` class.

   The environment is designed to simulate a resource allocation and power allocation scenario, where an agent must make decisions to optimize the system's performance.

## Classes
    - **CUSTOM_ENVIRONMENT**: The custom environment class.

## Notes
    This environment is designed to work with the Gymnasium library.
    The environment's dynamics and rewards are defined in the `CUSTOM_ENVIRONMENT` class.

## Author
    Miquel P. Baztan Grau

## Date
    21/08/2024
"""

from ENVIRONMENT import ENVIRONMENT
from utils import load_data

import gymnasium as gym
from gymnasium import spaces

import numpy as np
import random

DIV = 20

class CUSTOM_ENVIRONMENT(gym.Env, ENVIRONMENT):
    """
    A custom environment class that inherits from the `ENVIRONMENT` and the `gym.Env` class.

    This class represents a custom environment for the agent to interact with.
    It provides a custom implementation of the environment's dynamics and rewards.

    Attributes:
        ENVIRONMENT.attributes(*): The attributes inherited from the ENVIRONMENT class (see `ENVIRONMENT.py` for more information).
        gym.Env.attributes(*): The attributes inherited from the `gym.Env` class (see `gym.Env` for more information).
        metadata (dict): Metadata about the environment.
        possible_render_modes (list): A list of possible render modes for the environment.
        render_mode (str): The render mode for the environment.
        mode (str): The mode for the environment.
        iterations (int): The number of iterations in the environment.
        action_space (gym.Space): The action space of the environment.
        observation_space (gym.Space): The observation space of the environment.
        state (list): The current state of the environment.
        
    ## Methods
        - *ENVIRONMENT.methods*: The methods inherited from the ENVIRONMENT class (see `ENVIRONMENT.py` for more information).
        - *gym.Env.methods*: The methods inherited from the `gym.Env` class (see `gym.Env` for more information).
        - *reset*: Resets the environment.	
        - *step*: Performs an action in the environment.
        - *n_action_space*: Returns the number of actions in the environment.*
        - *m_state_space*: Returns the number of states in the environment.
        - *random_alpha*: Randomly assigns a resource block (RBG) to each user in the environment.
        - *random_beta*: Randomly assigns a base station to each user in the environment.
        - *random_power*: Randomly assigns power to each RBG in the environment.
        - *current_state*: Returns the current state of the environment.
        - *action_translator_power_allocation*: Translates a given action into base station (BS), resource block (RBG), and power (P) allocations.
        - *step_power_allocation*: Performs a power allocation action in the environment.
        - *n_action_space_power_allocation*: Returns the number of power allocation actions in the environment.
        - *action_translator_resource_allocation*: Translates a given action into a resource allocation action.
        - *valid*: Checks if the current state of the environment is valid.
        - *step_resource_allocation*: Performs a resource allocation action in the environment.
        - *n_action_space_resource_allocation*: Returns the number of resource allocation actions in the environment.
    """

    metadata = {'render_modes': ['human', 'rgb_array']}
    possible_render_modes = ['human', 'rgb_array']  
    def __init__(self, render_mode : str = None, data_file : int = None, mode : str = None) -> None:
        """
        Initializes a `CUSTOM_ENVIRONMENT` instance.

        This function initializes the environment with the given render mode, data file, and mode.
        It sets up the environment's render mode, mode, and data, and initializes the environment's state.
        It also defines the action and observation spaces of the environment.

        Parameters:
            render_mode (str): The render mode of the environment. Defaults to None.
            data_file (int): The data file to use for the environment. Defaults to None.
            mode (str): The mode of the environment. Defaults to None.

        Returns:
            None
        """

        gym.Env.__init__(self)

        self.render_mode = render_mode
        self.mode = mode
        
        data = load_data(data_file)
        ENVIRONMENT.__init__(self, data)

        self.iterations = 0

        print("states: ", self.m_state_space())
        print("actions: ", self.n_action_space())

        self.action_space = spaces.Discrete(self.n_action_space())
        self.observation_space = spaces.Box(low=np.zeros(self.m_state_space()), high=10*np.ones(self.m_state_space()), dtype=np.float32)
        self.state = None
    
    def reset(self, seed : int = None, options : dict = None) -> tuple[np.ndarray, dict]:
        """
        Resets the environment to its initial state.

        This function takes an optional seed and options as parameters, and returns the current state of the environment as a tuple of a numpy array and a dictionary.

        Parameters:
            seed (int): The seed to use for resetting the environment. Defaults to None.
            options (dict): Additional options for resetting the environment. Defaults to None.

        Returns:
            tuple(np.ndarray, dict): The restarted environment:
                - np.ndarray: The current state of the environment.
                - dict: Additional information about the environment.
        """ 

        super().reset(seed=seed)
        if self.mode == "resource_allocation":
            for n in self.N:
                for m in self.M:
                    for k in self.K:
                        self.alpha[n,m,k] = 0
                        self.beta[n,k] = 0
        else:
            self.random_alpha()
            self.random_beta()
                
        if self.mode == "power_allocation":
            for n in self.N:
                for m in self.M:
                    self.P[n,m] = 0
        else:
            self.random_power()

        self.iterations = 0
            
        for k in self.K:
            self.L[k] = 0

        self.RqData()

        self.total = sum(self.L[k]*self.alpha[n,m,k]*self.beta[n,k] for k in self.K for n in self.N for m in self.M)
       
        return self.current_state(), {}      

    def step(self, action : int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        This function takes an action and applies it to the environment based on the current mode.

        Parameters:
            action (int): The action to be applied to the environment.

        Returns:
            tuple(np.ndarray, int, bool, bool, dict): The result of applying the action to the environment:
                - np.ndarray: The next state of the environment.
                - float: The reward obtained by applying the action to the environment.
                - bool: Whether the episode has ended.
                - bool: Whether the episode has been truncated.
                - dict: Additional information about the environment.

        Raises:
            ValueError: If the mode is not 'power_allocation' nor 'resource_allocation'.
        """

        if self.mode == "power_allocation":
            return self.step_power_allocation(action)
        elif self.mode == "resource_allocation":
            return self.step_resource_allocation(action)
        else:
            raise ValueError("mode must be either 'power_allocation' or 'resource_allocation'")

    def n_action_space(self) -> int:
        """
        Returns the number of elements in the action space of the environment.

        The action space is determined by the mode of the environment. If the mode is 'power_allocation',
        it returns the number of elements in the power allocation action space. If the mode is 'resource_allocation',
        it returns the number of elements in the resource allocation action space.

        Returns:
            int: The number of elements in the action space.
        
        Raises:
            ValueError: If the mode is not 'power_allocation' nor 'resource_allocation'.
        """

        if self.mode == "power_allocation":
            return self.n_action_space_power_allocation()
        elif self.mode == "resource_allocation":
            return self.n_action_space_resource_allocation()
        else:
            raise ValueError("mode must be either 'power_allocation' or 'resource_allocation'")

    def m_state_space(self) -> int:
        """
        Returns the number of elements in the state space of the environment.

        Returns:
            int: The number of elements in the state space.
        """

        return len(self.current_state())
    
    #################################

    def random_alpha(self) -> None:
        """
        Randomly assigns a resource block (RBG) to each user in the environment.

        Iterates over each base station (n), each resource block (m), and assigns a random user (k) from the set of users (K).
        The assignment is stored in the alpha attribute, where alpha[n,m,k] is 1 if user k is assigned to resource block m at base station n, and 0 otherwise.

        Returns:
            None
        """

        for n in self.N:
            for m in self.M:
                k = random.randint(-1, self.K[-1])
                for k_ in self.K:
                    self.alpha[n,m,k_] = int(k_ == k)

    def random_beta(self) -> None:
        """
        Randomly assigns a base station to each user in the environment.

        Iterates over each user (k) and assigns a random base station (n) from the set of base stations (N).
        The assignment is stored in the beta attribute, where beta[n,k] is 1 if user k is assigned to base station n, and 0 otherwise.

        Returns:
            None
        """

        for k in self.K:
            n = random.randint(0, self.N[-1])
            for n_ in self.N:
                self.beta[n_,k] = int(n_ == n)

    def random_power(self) -> None:
        """
        Randomly assigns power to each RBG in the environment.

        Iterates over each base station (n) and each RBG (m) in the environment, 
        generating a random power value (P) between 0 and DIV-1. The actual power 
        assigned to the RBG is then calculated as Pmin + (Pmax - Pmin) / (DIV-1) * P.

        Returns:
            None
        """

        for n in self.N:
            for m in self.M:
                P = random.randint(0, DIV-1)
                self.P[n,m] = self.Pmin + (self.Pmax - self.Pmin)/(DIV-1)*P

    def current_state(self) -> np.ndarray:
        """
        Returns the current state of the environment as a numpy array.
        
        The state is a combination of the beta values, alpha values, 
        normalized power values, and normalized L values.
        
        Returns:
            np.ndarray: The current state of the environment.
        """
        state = []
        state += list(self.beta.values())
        state += list(self.alpha.values())
        state += [x/self.Pmax for x in list(self.P.values())]
        state += [0 if max(list(self.L.values())) == 0 else 10*x/max(list(self.L.values())) for x in list(self.L.values())]

        return np.array(state, dtype=np.float32)

    def action_translator_power_allocation(self, action : int) -> tuple[int|None, int|None, int|None]:
        """
        Translates a given action into base station (BS), resource block (RBG), and power (P) allocations.

        The action will be codified such that it's bijective with (BS, RBG, P) where 0 <= BS < N, 0 <= RBG < M, and 0 <= P < DIV.

        Parameters:
            action (int): The action to be translated.

        Returns:
            tuple(int|None, int|None, int|None): A tuple containing the base station (BS), resource block (RBG), and power (P) allocations if the action is not 0, otherwise returns None for all allocations.
        """

        action -= 1

        if action < 0:
            return None, None, None

        M = self.M[-1] + 1

        BS  = int(action // (DIV * M))
        RBG = int((action // DIV) % M)
        P   = int(action % DIV)

        return BS, RBG, P

    def step_power_allocation(self, action : int):
        """
        Takes a step in the power allocation process.

        Parameters:
            action (int): The action to take.

        Returns:
            tuple(np.ndarray, int, bool, bool, dict): The result of applying the action to the environment:
                - np.ndarray: The next state of the environment.
                - float: The reward obtained by applying the action to the environment.
                - bool: Whether the episode has ended.
                - bool: Whether the episode has been truncated.
                - dict: Additional information about the environment.
        """
        if self.render_mode == "human":
            print(f"Iteration {self.iterations}:")
            print("Bits in the buffer:", end=' ')
            print(f"{self.L}")

        BS, RBG, P = self.action_translator_power_allocation(action)
        if BS is not None:
            self.P[BS, RBG] = self.Pmin + (self.Pmax - self.Pmin)/(DIV-1)*P

        reward = 0
        
        if self.render_mode == "human": 
            if BS is not None:
                print(f"Action taken BS {BS}, RBG {RBG} to power: {self.P[BS,RBG]}")
            else:
                print("Action taken: None")
                reward += 1

        if self.render_mode == "human":
            self.results()
        
        self.iterations += 1

        reward += self.transmissionBits()/1e6

        self.TxData()

        if self.render_mode == "human": print("Bits remaining in the buffer:", self.L)

        self.RqData()

        terminated = bool(sum(self.L[k]*self.alpha[n,m,k]*self.beta[n,k] for k in self.K for n in self.N for m in self.M) == 0)
        truncated = self.iterations == 200

        return  self.current_state(), reward, terminated, truncated, {}
    
    def n_action_space_power_allocation(self) -> int:
        """
        Calculates the number of actions in the power allocation action space.

        The actions are to set the power level of each RBG to a value between 0 and DIV-1 and to do nothing.

        Returns:
            int: The number of actions in the power allocation action space.
        """

        N = self.N[-1]+1
        M = self.M[-1]+1

        return DIV*N*M + 1

    def action_translator_resource_allocation(self, action : int) -> tuple[str|None, int|None, int|None, int|None]:
        """
        Translates a given action into a resource allocation action.

        The action is codified as follows:
        * An action of 0 means "no action" or "do nothing", and returns `None` for all allocation types.
        * Actions from 1 to `N*K` (inclusive) correspond to "beta" allocations, where:
            + The base station index is determined by the integer division of the action by `K` (i.e., `action // K`).
            + The RBG index is not specified (i.e., `None`).
            + The user index is determined by the remainder of the action divided by `K` (i.e., `action % K`).
        * Actions from `N*K + 1` to `N*K + M*K` (inclusive) correspond to "alpha" allocations, where:
            + The base station index is determined by `(action - N*K - 1) // (M*K)`.
            + The RBG index is determined by `((action - N*K - 1) // K) % M`.
            + The user index is determined by `(action - N*K - 1) % K`.

        Parameters:
            action (int): The action to be translated.

        Returns:
            tuple(str|None, int|None, int|None, int|None): A tuple containing the type of allocation ("beta" or "alpha"), 
                   the base station index, the RBG index, and the user index.
        """

        N = len(self.N)
        M = len(self.M)
        K = len(self.K)

        if action == 0:
            return None, None, None, None
        
        action -= 1
        
        if action < N*K:
            return "beta", action//K, None, action%K
        else:
            action -= N*K
            return "alpha", action//(M*K), (action//K)%M, action%K

    def valid(self) -> bool:
        """
        Checks if the current state of the environment is valid.
        
        A valid state is one where each user is connected to at most one base station,
        and each resource block is allocated to at most one user per base station.
        
        Parameters:
            None
        
        Returns:
            bool: True if the state is valid, False otherwise
        """

        for k in self.K:
            if sum(self.beta[n,k] for n in self.N) > 1:
                return False

        for n in self.N:
            for m in self.M:
                if sum(self.alpha[n,m,k] for k in self.K) > 1:
                    return False
                
        return True

    def step_resource_allocation(self, action : int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Takes a step in the resource allocation process.

        Parameters:
            action (int): The action to take.

        Returns:
            tuple(np.ndarray, float, bool, bool, dict): The result of applying the action to the environment:
                - np.ndarray: The next state of the environment.
                - float: The reward obtained by applying the action to the environment.
                - bool: Whether the episode has ended.
                - bool: Whether the episode has been truncated.
                - dict: Additional information about the environment.
        """
        
        if self.render_mode == "human":
            print(f"Iteration {self.iterations}:")
            print("Bits in the buffer:", end=' ')
            print(f"{self.L}")

        mode, n, m, k = self.action_translator_resource_allocation(action)

        reward = 0

        if mode == "beta":
            self.beta[n,k] = 1 if self.beta[n,k] == 0 else 0
        elif mode == "alpha":
            self.alpha[n,m,k] = 1 if self.alpha[n,m,k] == 0 else 0
        else:
            reward += 1

        if self.render_mode == "human": print(f"Action taken {mode} {n} {m} {k}")

        if self.render_mode == "human":
            self.results()
        
        self.iterations += 1

        reward += self.transmissionBits()/1e6 if self.valid() else 0

        self.TxData()

        if self.render_mode == "human": print("Bits remaining in the buffer:", self.L)

        self.RqData()

        terminated = bool(sum(self.L.values()) == 0) or not self.valid()
        truncated = self.iterations == 200


        return  self.current_state(), reward, terminated, truncated, {}

    def n_action_space_resource_allocation(self) -> int:
        """
        Calculates the number of possible actions in the resource allocation environment.

        The number of actions is determined by the number of base stations (N), the number of RBGs (M),
        and the number of users (K) in the system.
        
        The actions are: toggle the value of alpha[n,m,k], toggle the value of beta[n,k] and do nothing.

        Returns:
            int: The total number of possible actions.
    	"""

        N = len(self.N)
        M = len(self.M)
        K = len(self.K)

        return N*K + N*M*K + 1
