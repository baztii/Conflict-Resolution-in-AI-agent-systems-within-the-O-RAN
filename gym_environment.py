from ENVIRONMENT import ENVIRONMENT
from utils import load_data

import gymnasium as gym
from gymnasium import spaces
import json

import numpy as np
import random

DATA_FILE = 2
DIV = 20

class CUSTOM_ENVIRONMENT(gym.Env, ENVIRONMENT):
    metadata = {'render_modes': ['human', 'rgb_array']}
    possible_render_modes = ['human', 'rgb_array']  
    def __init__(self, render_mode=None, data_file = DATA_FILE):
        gym.Env.__init__(self)

        self.render_mode = render_mode
        
        data = load_data(data_file)
        ENVIRONMENT.__init__(self, data)

        self.action_space = spaces.Discrete(self.n_action_space())
        self.observation_space = spaces.Box(low=np.zeros(self.m_state_space()), high=np.inf*np.ones(self.m_state_space()), dtype=np.float64)
        self.state = None

            
        usr = 0
        for n in self.N:
            for m in self.M:
                for k in self.K:
                    self.alpha[n,m,k] = int(k == usr)
                usr += 1
                usr = usr %(self.K[-1] + 1)
   
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        for n in self.N: 
            for m in self.M:
                self.P[n,m] = 0
            

        for k in self.K:
            self.L[k] = 0

        self.RqData()

        new_state = []

        for n in self.N:
            for m in self.M:
                new_state += self.gamma_maj(n, m)
        
        for k in self.K:
            new_state.append(self.Bits(k))

        for k in self.K:
            new_state.append(self.L[k])

        for n in self.N:
            for m in self.M:
                new_state.append(self.P[n,m]) 
                
        return np.array(new_state), {}
    
    def action_translator(self, action):

        N = self.N[-1] + 1
        M = self.M[-1] + 1

        BS  = int(action//(DIV*M))
        RBG = int((action%(DIV*M))//DIV)
        P   = int(action%(DIV*M)%DIV)

        return BS, RBG, P
    
    def step(self, action):
        N = self.N[-1] + 1
        M = self.M[-1] + 1
        if self.render_mode == "human":
            print(f"Iteration x:")
            print("Bits in the buffer:", end=' ')
            print(f"{self.L}")

        
        BS, RBG, P = self.action_translator(action)


        self.P[BS, RBG] = self.Pmin + (self.Pmax - self.Pmin)/(DIV-1)*P
        
        if self.render_mode == "human": print(f"Action taken BS {BS}, RBG {RBG} to power: {self.P[BS,RBG]}")

        if self.render_mode == "human":
            self.results()

        
        reward = self.transmissionBits()
        ## New state ##
        self.TxData()

        if self.render_mode == "human": print("Bits remaining in the buffer:", self.L)

        new_state = []

        for n in self.N:
            for m in self.M:
                new_state += self.gamma_maj(n, m)

        for k in self.K:
            new_state.append(self.Bits(k))

        for k in self.K:
            new_state.append(self.L[k])

        for n in self.N:
            for m in self.M:
                new_state.append(self.P[n,m]) 

                
        #print(new_state)
       
        return  np.array(new_state), reward, bool(sum(self.L.values()) == 0), False, {}
    
    def results(self, time : float = 0.0):        
        print(f"Time: {time}s\n")
        print(f"Total bits sent: {round(self.transmissionBits())}", end="\t")
        print(f"Total RBGs used: {self.RBGs()}\n")

        for n in self.N:
            print(f"BS {n}:")
            for m in self.M:
                print(f"\tP({n:>2},{m:>2}) = {self.P[n,m]}")
            print()

            for k in self.K:
                if self.beta[n,k] == 1:
                    print(f"\tuser({k:>2}) uses", end=' ')
                    has_rbg = False
                    for m in self.M:
                        if self.alpha[n,m,k] == 1:
                            print(f"{m:>2}", end=' ')
                            has_rbg = True

                    if not has_rbg:
                        print("NO", end=' ')

                    print("RBGs", end=' ')
                    print(f"and sends {self.Bits(k)} bits")

            print()

    def action_space_sample(self):
        N = self.N[-1]+1
        M = self.M[-1]+1

        return random.randint(0, DIV*N*M-1)
    
    def n_action_space(self):
        N = self.N[-1]+1
        M = self.M[-1]+1

        return DIV*N*M

    def m_state_space(self):
        N = self.N[-1]+1
        M = self.M[-1]+1
        K = self.K[-1]+1
        return N*M + 2*K + N*M*(N-1)


    """ This is another test to show if the agent learns properly in a simple scenario """
    def reset2(self, seed=None, options=None):

        super().reset(seed=seed)
        #self.state = np.random.uniform(low = 0, high=100_000, size=(len(self.L),))

        for m in self.M:
            self.P[0,m] = 0
        self.RqData()

        state = list(self.L.values())
        #state.append(self.iterations)
        print(self.L)


        return np.array(state, dtype=np.float64), {}

        for m in self.M:
            self.P[0,m] = 0
        self.RqData()

        state = list(self.L.values())
        state += [self.P[0,m] for m in self.M]

        return state

    def step2(self, action):
        self.iterations += 1
        #self.P[0,int(action)] = self.Pmax

        reward = self.L[int(action)]
        self.L[int(action)] = 0

        #print(self.P)
        #print(action)
        #self.TxData()
        state = list(self.L.values())
        #state.append(self.iterations)

        print(self.L)

        return np.array(state, dtype=np.float64), reward, bool(sum(self.L.values()) == 0), False, {}

    def render(self, mode='human'):
        if mode == 'human':
            print(self.L)
    
    def close(self):
        pass

    def n_action_space2(self):
        return len(self.K)

    def m_state_space2(self):
        return 2*len(self.K)
    
    def action_space_sample2(self):
        return random.randint(0, len(self.K)-1)

    """ This is a test to show if the agent learns properly in a simple scenario """

    def reset2(self, seed=None, options=None):
        super().reset(seed=seed)
        for k in self.K:
            self.L[k] = random.randint(0,100)

        return np.array(list(self.L.values())), {}

    def step2(self,action):
        #print(self.L)
        if self.L[int(action)] == 0:
            reward = -100
        else:
            self.L[int(action)] = 0
            reward = sum(self.L.values())

        state = list(self.L.values())

        return np.array(state, dtype=np.float64), reward, bool(sum(self.L.values()) == 0), False, {}    

    def n_action_space2(self):
        return len(self.L)

    def m_state_space2(self):
        return len(self.L)
    
    def action_space_sample2(self):
        return random.randint(0, len(self.L)-1)
