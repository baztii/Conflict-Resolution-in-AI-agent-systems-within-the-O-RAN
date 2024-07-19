import numpy as np
import pandas as pd
import torch
import gym
import matplotlib.pyplot as plt
#import tensorflow
import math
import random
import time
from scipy.optimize import minimize

"""
The xAPP agent that maximizes the throughput
"""

def valid(taulell):
    for i in range(5):
        for j in range(5):
            if taulell[i][j] % 2 == 1 and taulell[i][j] < 10: continue
            return False
    
    return True


def obj_function(taulell):
    s = 0
    for i in range(5):
        for j in range(5):
            s += taulell[i][j]
    return s


def f(x):
    taulell = x.reshape((5,5))
    return -obj_function(taulell)

def rest(x):
    taulell = x.reshape((5,5))
    return int(valid(taulell)) - 1

def ONL():
    taulell = [[1,1,1,1,1],
           [1,1,1,1,1],
           [1,1,1,1,1],
           [1,1,1,1,1],
           [1,1,1,1,1]
           ]
    r = [{'type' : 'eq', 'fun' : rest}]
    x = np.array(taulell)
    x = x.flatten()
    sol = minimize(f,x0=x, constraints=r)
    print(sol)

def ONL2():
    pass
"""
if __name__ == "__main__":
    ONL2()

"""


"""
The PK to determine a RBG is a pair (n,m) i.e. the BS and the mth RBG of this station (0 <= n < N && 0 <= m < M)
"""
class ENVIRONMENT:
    def __init__(self, N : int, K : int, M : int, userDistr : list, g : list, B : list, Pmin : float, Pmax: float, buffSize : int, T : float, sigma : float, lamda : float, bits : int) -> None:
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

        self.buffSize = buffSize*bits # int // The size of the transmission buffer where the pending traffic is queued
        self.T = T # float // Length of time slot

        self.RBGusers = [[0 for m in range(M)] for n in range(N)] # list // Distribution of RGB to each user (RBGusers[n][m] = k if user k has RBG m at BS n)
        self.L = [0 for k in range(K)] # list // Amount of remained data of all users in the transimssion buffer (at t=0 there is no remaining data)

        self.sigma = sigma # double // noise

        self.bits = bits # int // We assume the pending data is queued each 'bits'

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
            /(sum([self.alpha(n_prime, m, k_prime)*self.g[n_prime][k_prime]*self.P[n_prime][m] # is there a typo on the paper?
                for n_prime in range(self.N) if n_prime != n
                for k_prime in self.userDistr[n_prime]])
            + self.sigma**2))
    
    def C(self, n : int, m : int) -> float: # (2)
        return self.B[n][m]*math.log2(1+sum([self.eta(n,m,k) for k in range(self.K)]))
    
    def R(self, n : int, m : int) -> float: # (3)
        return self.C(n,m)*self.T if self.C(n,m)*self.T < sum([self.alpha(n,m,k)*self.L[k] for k in range(self.K)]) else sum([self.alpha(n,m,k)*self.L[k] for k in range(self.K)])

    def updateBuffer(self) -> list[bool]:
        probability =  np.random.poisson(self.lamda*self.T, self.K) #math.exp(-self.lamda*self.T/1000) * (self.lamda*self.T/1000)
        ret = np.random.random(self.K) > probability
        for k in range(self.K): # "insert" new data to be transmitted
            if ret[k]:
                self.L[k] = min(self.buffSize, self.L[k]+self.bits)
        
        for n in range(self.N): # "send" the data of the buffer
            for m in range(self.M):
                self.L[self.RBGusers[n][m]] = max(0,self.L[self.RBGusers[n][m]]-int(self.R(n,m)))
        
        return ret

    def obj_function(self) -> float:
        return sum([self.R(n,m) for n in range(self.N) for m in range(self.M)])

    def valid(self) -> bool:
        for n in range(self.N):
            #if self.userDistr[n] == []: continue
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
            * P[n][m] = (float between Pmin and Pmax) ∀n,m
            * RGB[n][m] = ({0,...,K-1} is the user that is given the RBG m from BS n) ∀n,m
        """
    """
    def setUsersRBGAllocation(self,BS,RBG,r_max): ## Let's suppose all BS have at least one user
        
        if RBG == self.M:
            RBG = 0
            BS += 1
        if BS == self.N:
            return self.powerAllocation()
        for k in self.userDistr[BS]:
            self.RBGusers[BS][RBG] = k
            r = self.setUsersRBGAllocation(BS,RBG+1,r_max)
            r_max = max(r_max, r)
        
        return r_max
        """
    """
    def usersRBGAllocation(self):
        BS = 0
        RBG = 0
        r_max = 0
        return self.setUsersRBGAllocation(BS, RBG, r_max)
    """

    def setUsersRBGAllocation(self, BS, user, RBG_left, r_max): # We use the symetry of RGB's due to the same value of brandwidth
        if RBG_left != 0 and user == len(self.userDistr[BS]):
            return 0
        if RBG_left == 0:
            BS += 1
            user = 0
            RBG_left = self.M
        
        if BS == self.N:
            return self.powerAllocation()
        
        for i in range(RBG_left+1):
            for j in range(i):
                self.RBGusers[BS][j+self.M-RBG_left] = self.userDistr[BS][user]
            
            r = self.setUsersRBGAllocation(BS, user + 1, RBG_left - i,r_max)
            r_max = max(r_max, r)
        
        return r_max

    def usersRBGAllocation(self):
        RBG_left = self.M
        user = 0 # The real user is self.userDistr[user]
        BS = 0
        r_max = 0
        return self.setUsersRBGAllocation(BS, user, RBG_left, r_max)
        
    def setPowerAllocation(self, n, m, power, r):
        if m == self.M:
            m = 0
            n += 1
        if n == self.N :
            new_r = self.obj_function()
            return new_r if new_r > r else r
        
        for p in power:
            self.P[n][m] = p
            r = self.setPowerAllocation(n,m+1,power,r)

        return r

    def powerAllocation(self):
        print("entra a aquí")
        D = 10 # The divisions of the power
        power = [self.Pmin + (self.Pmax - self.Pmin)*i/(D-1) for i in range(D)]

        r = 0
        n = 0
        m = 0

        return self.setPowerAllocation(n,m,power,r)
    
    def assign(self):
        print("Best reward: ",self.usersRBGAllocation()) ## It does not update the values yet

    def gameloop(self):
        while True:
            time.sleep(self.T)
            self.updateBuffer()
            print(self)
            print([self.R(n,m) for n in range(self.N) for m in range(self.M)])
            #self.assign() ## Still inefficient

    def f(self, x):
        self.RBGusers = [list(int(x[i]) for i in range(self.M))]
        self.P = [list(x[self.M:2*self.M])]
        return -self.obj_function()

    def rest(self, x):
        self.RBGusers = [list(int(x[i]) for i in range(self.M))]
        self.P = [list(x[self.M:2*self.M])]
        return int(self.valid()) - 1

    def ONL(self):
        rest = [{'type' : 'eq', 'fun' : self.rest}]
        x = np.zeros(2*self.M)
        print(x)
        sol = minimize(self.f,x0=x, constraints=rest)
        print(sol)










    # TODO: Train the agent via the DQN algorithm (deep Q-learning)

    def step(self):
        pass

    def DQN(self):
        pass

class ONL(ENVIRONMENT):
    def __init__(self, *args):
        super().__init__(*args)
    
    def __str__(self):
        return super().__str__()

    def model(self):
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

    dBm_to_watts = lambda dBm : 10**(dBm/10)/1000

    N = 1
    K = 7
    M = 5

    userDistr = [[] for i in range(N)]
    userDistr[0] = [k for k in range(K)]

    B = [[20*1e6 for j in range(M)] for i in range(N)] # Hz
    T = 0.1 # s
    Pmin = dBm_to_watts(1) # watts
    Pmax = dBm_to_watts(38) # watts

    sigma = dBm_to_watts(-114) # watts

    buffSize = 50

    g = dBm_to_watts(-45 + np.random.rand(N,K)) # watts

    lamda = 10 #blocks each second
    bits = 1500*8 #bits

    environ = ENVIRONMENT(N,K,M,userDistr,g,B,Pmin,Pmax,buffSize,T,sigma, lamda, bits)

    #environ.gameloop()
    environ.updateBuffer()
    environ.L = [6000,3000,1,5,3,4,4000]

    environ.ONL()
    print(environ.valid())
    print(environ)

    environ.RBGusers = [[0,1,6,3,4]]
    environ.P = [[6,6,6,6,6]]

    print("My environ")

    print(environ)

    return

    #################################################

    environ.updateBuffer()
    environ.L = [6000,3000,1,5,3,4,4000]
    print(environ)
    while True:
        for i in range(5):
            x = int(input(f"user of RBG {i}: "))
            environ.RBGusers[0][i] = x

            x = float(input(f"power of RBG {i}: "))
            environ.P[0][i] = x

        if environ.valid():
            print("Throughput=", [environ.R(0,m) for m in range(M)])
        else:
            print("NO valid!")
        
        print()



if __name__ == "__main__":
    main()
