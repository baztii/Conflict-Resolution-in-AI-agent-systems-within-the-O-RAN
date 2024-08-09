from pyomo.environ import *
from time import time
from ENVIRONMENT import ENVIRONMENT as ENV
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

"""Global variables"""
FILE = 1
START = 7
END = 7
TEE = True
SOLVER = "scip"
CONSOLE = False
TERMINAL = sys.stdout
OF = sys.stdout

class ONL(ENV):
    def __init__(self, data : dict):
        """ Initialize parameters and variables"""
        super().__init__(data, False)

        """ Solver model """
        self.model = ConcreteModel()
        self._init_model()
        
    def _init_model(self):
        """ Sets """
        self.model.N     = Set(initialize=self.N)
        self.model.K     = Set(initialize=self.K)
        self.model.M     = Set(initialize=self.M)

        """ Parameters """
        self.model.sigma = Param(initialize=self.sigma)
        self.model.T     = Param(initialize=self.T)

        self.model.g     = Param(self.model.N, self.model.K, initialize=self.g)
        self.model.B     = Param(self.model.N, self.model.M, initialize=self.B)
        self.model.L     = Param(self.model.K, mutable=True, initialize=self.L) # self.L

        """ Variables """
        self.model.alpha         = Var(self.model.N, self.model.M, self.model.K, domain=Binary, initialize=self.alpha) # self.alpha
        self.model.beta          = Var(self.model.N, self.model.K, domain=Binary, initialize=self.beta) # self.beta
        self.model.P             = Var(self.model.N, self.model.M, bounds = (self.Pmin, self.Pmax), initialize=self.P) # self.P
        self.model.min_bool_bits = Var(self.model.K, domain=Binary)

        """ Objective function """
        self.model.obj = Objective(rule=self.obj_function,sense=maximize)

        """ Constraints """
        self.model.alphaConstr    = Constraint(self.model.N, self.model.M, rule=self.alpha_constraint)
        self.model.betaConstr     = Constraint(self.model.K, rule=self.beta_constraint)
        self.model.min_bool_bitsConstr = Constraint(self.model.K, rule=self.min_bool_bits_constraint)
        self.model.min_bool_bitsConstr2 = Constraint(self.model.K, rule=self.min_bool_bits_constraint2)

        self.model.betaConstr2     = Constraint(self.model.N, self.model.M, self.model.K, rule=self.beta_constraint2)
        #self.etasss =  Constraint(self.model.N, self.model.M, self.model.K, rule=self.auto_denominator)
        

    def auto_denominator(self, model, n : int, m: int, k: int):
        return self.eta(n,m,k,model) >= 0
    
    def min_bool_bits_constraint(self, model, k : int):
        return model.min_bool_bits[k]*(self.rhs(k, model) -  self.lhs(k, model)) >= 0
    
    def min_bool_bits_constraint2(self, model, k : int):
        return (1-model.min_bool_bits[k])*(self.rhs(k, model) - self.lhs(k, model)) <= 0

    def beta_constraint(self, model, k : int):
        return sum(model.beta[n,k] for n in model.N) == 1

    def beta_constraint2(self, model, n : int, m : int, k : int):
        return (1 - model.beta[n,k])*model.alpha[n,m,k] == 0
    
    def alpha_constraint(self, model, n : int, m : int):
        return sum(model.alpha[n, m, k] for k in model.K) <= 1

    def obj_function(self, model=None):
        if model is None: model = self
        return self.transmissionBits(model) - self.RBGs(model)# Maximize the bits sent and minimize the number of RBGs used

    def results(self, time : float):        
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

    def solve(self, display=True):
        solver=SolverFactory(SOLVER)
        self.model.L.store_values(self.L)

        sys.stdout = TERMINAL
        t = time()
        solver.solve(self.model, tee=TEE)
        t = time() - t
        sys.stdout = OF

        self.alpha = {(n, m, k): round(self.model.alpha[n, m, k].value) for n in self.model.N for m in self.model.M for k in self.model.K}
        self.P = {(n, m): self.model.P[n, m].value for n in self.model.N for m in self.model.M}
        self.beta = {(n, k): round(self.model.beta[n, k].value) for n in self.model.N for k in self.model.K}

        self.results(time=t)

    def own_policy(self, display=True):
        usr = 0
        for n in self.N:
            for m in self.M:
                for k in self.K:
                    if k == usr:
                        self.alpha[n, m, k] = 1
                    else:
                        self.alpha[n, m, k] = 0
                usr += 1
                usr %= self.K[-1] + 1

        for n in self.N:
            for k in self.K:
                self.beta[n, k] = 1

        for n in self.N:
            for m in self.M:
                self.P[n,m] = self.Pmax
                
        self.results(time=0)

        for n in self.N:
            for m in self.M:
                for k in self.K:
                    print(f"Numerator({n},{m},{k})={self.numerator(n, m, k)}")

        for n in self.N:
            for m in self.M:
                print(f"Denominator({n},{m})={self.denominator(n, m)}")
        
        for n in self.N:
            for m in self.M:
                print(f"C({n},{m})={self.C(n,m)}")


def asserts(data : dict) -> None:
    assert len(data["B"])    == data["N"]
    assert len(data["B"][0]) == data["M"]
    
    assert len(data["g"])    == data["N"]
    assert len(data["g"][0]) == data["K"]

    assert len(data["L"])    == data["K"]


def main():
    dBm_to_watts = lambda dBm : 10**(dBm/10)/1000 # watts = dBm_to_watts(dBm) conversor

    n = FILE
    file = f"tests/test{n}/data.json"

    if not CONSOLE:
        f = open(f"tests/test{n}/results.txt", 'w')
        sys.stdout = f
    
    global OF
    OF = sys.stdout

    with open(file, 'r') as data_file: # load the data
        data = json.load(data_file)
    
    asserts(data)

    onl = ONL(data) # create the onl object

    onl.gameloop(policy=onl.solve)

if __name__ == '__main__':
    import json
    for FILE in range(START,END+1):
        sys.stdout = TERMINAL
        print("Doing file:", FILE)
        sys.stdout = OF
        main()
        sys.stdout = TERMINAL
        print(f"File: {FILE} done!")
        sys.stdout = OF
        os.system('cls')

