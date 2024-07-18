import numpy as np
import pandas as pd
import torch
import gym
import matplotlib.pyplot as plt
import tensorflow
import math
import random
import time

"""
The xAPP agent that maximizes the throughput
"""


"""
The PK to determine a RBG is a pair (n,m) i.e. the BS and the mth RBG of this station (0 <= n < N && 0 <= m < M)
"""
class ENVIRONMENT:
    def __init__(self, N : int, K : int, M : int, userDistr : list, g : list, B : list, Pmin : float, Pmax: float, buffSize : int, T : float, sigma : float, lamda : float) -> None:
        self.lamda = lamda # float // Poison process rate per second
        self.N = N # int // Number of base stations
        self.K = K # int // Number of users int the system
        self.M = M # int // Total number of RGB x BS

        self.userDistr = userDistr # list // User distribution in BS
        self.g = g # list // Channel coefficient between base stations and users (i.e g[n][k] is between BS n and user k)
        self.B = B # list // Brandwith of all RBG (i.e. BW[n][m] determines the brandwith of the RBG m at the BS n)

        self.Pmin = Pmin # float // max transmition power
        self.Pmax = Pmax # float // min transmition power
        self.P = [[0 for m in range(M)] for n in range(N)] # list // Transmission power allocation to RBG of BS (at t=0 there is no transmission power given)

        self.buffSize = buffSize # int // The size of the transmission buffer where the pending traffic i queued
        self.T = T # float // Length of time slot

        self.RBGusers = [[0 for m in range(M)] for n in range(N)] # list // Distribution of RGB to each user (RBGusers[n][m] = k if user k has RBG m at BS n)
        self.L = [0 for k in range(K)] # list // Amount of remained data of all users in the transimssion buffer (at t=0 there is no remaining data)

        self.sigma = sigma # double // noise (idk if it is a parameter or if it is the normal distribution of something --> the paper does not mention it)

    def __str__(self) -> str:
        return f"lambda={self.lamda}\nN={self.N}\nK={self.K}\nM={self.M}\nuserDistr={self.userDistr}\ng={self.g}\nb={self.B}\nPmin={self.Pmin}\nPmax={self.Pmax}\nP={self.P}\nbuffSize={self.buffSize}\nT={self.T}\nRBGusers={self.RBGusers}\nL={self.L}\nsigma={self.sigma}\nobj_function={self.obj_function()}\n"

    def alpha(self, n : int, m : int, k : int) -> {0,1}:
        return self.RBGusers[n][m] == k

    def gamma_maj(self, n : int, m : int) -> list[float]:
        return [math.log2(1+
                         sum([self.alpha(n_prime, m, k)*self.g[n_prime][k] for k in range(self.K)])/
                         sum([self.alpha(n,m,k)*self.g[n][k] for k in range(self.K)])) for n_prime in range(self.N) if n_prime != n]

    def eta(self, n : int, m : int, k : int) -> float: # (1)
        return ((self.alpha(n, m, k)*self.g[n][k]*self.P[n][m])
            /(sum([self.alpha(n_prime, m, k_prime)*self.g[n_prime][k_prime]*self.P[n_prime][m] 
                for n_prime in range(self.N) if n_prime != n
                for k_prime in self.userDistr[n_prime]])
            + self.sigma**2))
    
    def C(self, n : int, m : int) -> float: # (2)
        return self.B[n][m]*math.log2(1+sum([self.eta(n,m,k) for k in range(self.K)]))
    
    def R(self, n : int, m : int) -> float: # (3)
        return self.C(n,m) if self.C(n,m)*self.T < sum([self.alpha(n,m,k)*self.L[k] for k in range(self.K)]) else sum([self.alpha(n,m,k)*self.L[k] for k in range(self.K)])/self.T

    def arrivals(self) -> bool:
        probability = math.exp(-self.lamda*self.T/1000) * (self.lamda*self.T/1000)
        if random.random() > probability:
            k = random.randint(1, self.K)-1
            self.L[k] += 1
            if self.L[k] > self.buffSize:
                self.L[k] = self.buffSize
                return False
            return True
        return False

    def obj_function(self) -> float:
        return sum([self.R(n,m) for n in range(self.N) for m in range(self.M)])

    def valid(self) -> bool:
        for n in range(self.N):
            for m in range(self.M):
                if self.Pmin <= self.P[n][m] <= self.Pmax and sum([self.alpha(n,m,k) for k in range(self.K)]) == 1 and isinstance(self.RBGusers[n][m], int):
                    continue
                return False

        return True

    def L(self, n : int, m : int) -> list[int]:
        [self.alpha(n,m,k)*self.L[k] for k in range(self.K)]

    def optimize(self):
        """
        Problem:
        max obj_function such that
            (1), (2), (3) is satified and # This will be always satisfied #
            self.Pmin <= self.P[n][m] <= self.Pmax ∀n,m
            self.alpha(n,m,k) = {0,1} ∀n,m,k    # This is satified always #
            sum([self.alpha(n,m,k) for k in range(self.K)]) == 1 ∀n,m
        """

        # The constraints are checked in the self.valid() method

        """
        Parameters that the agent will change:
            * P[n][m] ∀n,m
            * RGB[n][m] ∀n,m
        """
    
    def gameloop(self):
        while True:
            time.sleep(self.T/1000)
            self.arrivals()
            print(self)
            print([self.R(n,m) for n in range(self.N) for m in range(self.M)])

    # TODO: Train the agent via the DQN algorithm (deep Q-learning)

    def step(self):
        pass

    def DQN(self):
        pass

"""
class Q_LEARNING(AGENT):
    def __init__(self, iterations=10_000, alpa=0.6, gamma=0.4, *kwargs):
        self.interations = iterations
        self.alpa = alpa
        self.gamma = gamma
        super().__init__(*kwargs)
    
    def __str__(self):
        return super().__str__()

"""

class DQN:
    def __init__(self):
        pass

    
        

def main():
    """
    All the parameters of the environment
    """
    N = 4
    K = 30
    M = 12
    userDistr = [[] for i in range(N)]
    userDistr[0] = [k for k in range(K)]

    B = [[20 for j in range(M)] for i in range(N)] # Mhz
    T = 100 #ms

    Pmin = 1 #dBm
    Pmax = 38 #dBm

    sigma = 0.5 # ??

    buffSize = 50

    g = np.random.rand(N,K)*100

    lamda = 10

    environ = ENVIRONMENT(N,K,M,userDistr,g,B,Pmin,Pmax,buffSize,T,sigma, lamda)

    environ.gameloop()







if __name__ == "__main__":
    main()
