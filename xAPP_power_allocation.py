import numpy as np
import pandas as pd
import torch
import gym
import matplotlib.pyplot as plt
import tensorflow
import math

"""
The xAPP agent that maximizes the throughput
"""


"""
The PK to determine a RBG is a pair (n,m) i.e. the BS and the mth RBG of this station (0 <= n < N && 0 <= m < M)
"""
class AGENT:
    def __init__(self, N : int, K : int, M : int, userDistr : list, g : list, B : list, Pmin : float, Pmax: float, buffSize : int, T : float, sigma : float):
        self.N = N # int // Number of base stations
        self.K = K # int // Number of users int the system
        self.M = M # int // Total number of RGB x BS

        self.userDistr = np.array(userDistr) # list // User distribution in BS
        self.g = np.array(g) # list // Channel coefficient between base stations and users (i.e g[n][k] is between BS n and user k)
        self.B = np.array(B) # list // Brandwith of all RBG (i.e. BW[n][m] determines the brandwith of the RBG m at the BS n)

        self.Pmin = Pmin # float // max transmition power
        self.Pmax = Pmax # float // min transmition power
        self.P = np.zeros((N,M)) # list // Transmission power allocation to RBG of BS (at t=0 there is no transmission power given)

        self.buffSize = buffSize # int // The size of the transmission buffer where the pending traffic i queued
        self.T = T # float // Length of time slot

        self.RBGusers = np.zeros((N,M)) # list // Distribution of RGB to each user (RBGusers[n][m] = k if user k has RBG m at BS n)
        self.L = np.zeros(K) # list // Amount of remained data of all users in the transimssion buffer (at t=0 there is no remaining data)

        self.sigma = sigma # double // noise (idk if it is a parameter or if it is the normal distribution of something --> the paper does not mention it)

    def __str__(self):
        pass

    def alpha(self, n : int, m : int, k : int) -> {0,1}:
        return self.RBGusers[n][m] == k

    def eta(self, n : int, m : int, k : int) -> float: # (1)
        return ((self.alpha(n, m, k)*self.g[n][k]*self.P[n][m])
            /(sum([self.alpha(n_prime, m, k_prime)*self.g[n_prime][k_prime]*self.P[n_prime][m] 
               for n_prime in range(n) if n_prime != n 
               for k_prime in self.userDistr[n_prime]])
            + self.sigma**2))
    
    def C(self, n : int, m : int) -> float: # (2)
        return self.B[n][m]*math.log2(1+sum([self.eta(n,m,k) for k in range(self.K)]))
    
    def R(self, n : int, m : int) -> float: # (3)
        return self.C(n,m) if self.C(n,m)*self.T < sum([self.alpha(n,m,k)*self.L[k] for k in range(self.K)]) else sum([self.alpha(n,m,k)*self.L[k] for k in range(self.K)])/self.T

    def obj_function(self):
        return sum([self.R(n,m) for n in range(self.N) for m in range(self.M)])

    def optimize(self):
        """
        Problem:
        max obj_function such that
            (1), (2), (3) is satified and
            self.Pmin <= self.P[n][m] <= self.Pmax ∀n,m
            self.alpha(n,m,k) = {0,1} ∀n,m,k    # This is satified always #
            sum([self.alpha(n,m,k) for k in range(K)]) == 1 ∀n,m
        """

    
    # TODO: Train the agent via the DQN algorithm (deep Q-learning)

    def step(self):
        pass

    def DQN(self):
        pass



def main():
    """
    All the parameters of the environment
    """
    N = 4
    K = 30
    M = 12
    userDistr = [0 for i in range(K)]
    B = [20 for i in range(M)] # Mhz
    T = 100 #ms

    Pmin = 1 #dBm
    Pmax = 38 #dBm

    sigma = -114 #dBm

    buffSize = 50

    g = np.random.rand(N,K)

    agent = AGENT(N,K,M,userDistr,g,B,Pmin,Pmax,buffSize,T,sigma)






if __name__ == "__main__":
    main()

