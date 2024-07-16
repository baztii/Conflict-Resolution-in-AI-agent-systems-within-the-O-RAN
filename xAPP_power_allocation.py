import numpy as np
import pandas as pd
import torch
import gym
import matplotlib.pyplot as plt
import tensorflow
import math

class Agent:
    def __init__(self, BS, K, RGB, userDistr, RGBDistr, RGBxBS, userResources, sigma, B, T, L, minP, maxP):
        self.minP = minP # float // max transmition power
        self.maxP = maxP # float // min transmition power

        self.BS = BS # int // Number of base stations
        self.RGB = RGB # int // Total number of resource blocks group
        self.RGBxBS = RGBxBS # int // Total number of RGB x BS
        self.RGBDistr = RGBDistr # list // RGB distribution in BS

        assert RGBxBS * BS == RGB # A condition that must be satisfied

        self.K = K # int // Number of users int the system
        self.userDistr = userDistr # list // User distribution in BS
        self.P = None # list // Transmission power allocation to RBG of BS

        self.sigma = sigma # double // noise

        assert 0 <= sigma <= 1 # a precondition on sigma

        self.userResources = userResources # list // Distribution of RGB to each user
        self.g = None # list // Channel coefficient between base stations and user

        self.B = B # list // Brandwith of all RBG (i.e. BW[m] determines the brandwith of the RBG m)
        self.T = T # float // Length of time slot
        self.L = L # list // Amount of remained data of all users in the transimssion buffer

    def alpha(self, n : int, m : int, k : int) -> {0,1}:
        return m in self.RGBDistr[n] and m in self.userResources[k]
    
    def eta(self, n : int, m : int, k : int) -> float:
        return ((self.alpha(n, m, k)*self.g[n][k]*self.P[n][m])
            /(sum([self.alpha(n_prime, m, k_prime)*self.g[n_prime][k]*self.P[n_prime][m] 
               for n_prime in range(n) if n_prime != n 
               for k_prime in self.userDistr[n_prime]])
            + self.sigma**2))

    def C(self, n : int, m : int) -> float:
        return self.B[m]*math.log2(1+sum([self.eta(n,m,k) for k in range(self.K)]))
    
    def R(self, n : int, m : int) -> float:
        return self.C(n,m) if self.C(n,m)*self.T < sum([self.alpha(n,m,k)*self.L[k] for k in range(self.K)]) else sum(self.L)/self.T ## The else value is not specified

        



def main():
    pass

if __name__ == "__main__":
    main()

