from pyomo.environ import *
from time import time
from ENVIRONMENT import ENVIRONMENT as ENV
import matplotlib.pyplot as plt
import numpy as np

class ONL(ENV):
    def __init__(self, data : dict):
        """ Initialize parameters and variables"""
        super().__init__(data)

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

        #self.model.beta  = Param(self.model.N, self.model.K, initialize=self.beta)
        self.model.g     = Param(self.model.N, self.model.K, initialize=self.g)
        self.model.B     = Param(self.model.N, self.model.M, initialize=self.B)
        self.model.L     = Param(self.model.K, mutable=True, initialize=self.L) # self.L

        """ Variables """
        self.model.alpha         = Var(self.model.N, self.model.M, self.model.K, domain=Binary, initialize=self.alpha) # self.alpha
        self.model.beta          = Var(self.model.N, self.model.K, domain=Binary, initialize=self.beta) # self.beta
        self.model.P             = Var(self.model.N, self.model.M, bounds = (self.Pmin, self.Pmax), initialize=self.P) # self.P
        self.model.min_bool_bits = Var(self.model.K, domain=Binary)
        #self.model.L     = Var(self.model.K, domain=NonNegativeIntegers, initialize=self.L)

        """ Objective function """
        self.model.obj = Objective(rule=self.obj_function,sense=maximize)

        """ Constraints """
        self.model.alphaConstr    = Constraint(self.model.N, self.model.M, rule=self.alpha_constraint)
        self.model.betaConstr     = Constraint(self.model.K, rule=self.beta_constraint)
        self.model.min_bool_bitsConstr = Constraint(self.model.K, rule=self.min_bool_bits_constraint)
        self.model.min_bool_bitsConstr2 = Constraint(self.model.K, rule=self.min_bool_bits_constraint2)

        #self.model.LConstr     = Constraint(self.model.K, rule=self.L_constraint)
    
    def min_bool_bits_constraint(self, model, k : int):
        return model.min_bool_bits[k]*(self.rhs(k, model) -  self.lhs(k, model)) >= 0
    
    def min_bool_bits_constraint2(self, model, k : int):
        return (1-model.min_bool_bits[k])*(self.rhs(k, model) - self.lhs(k, model)) <= 0

    def beta_constraint(self, model, k : int):
        return sum(model.beta[n,k] for n in model.N) == 1
    
    def L_constraint(self, model, k : int):
        return model.L[k] == self.L[k]
    
    def alpha_constraint(self, model, n : int, m : int):
        return sum(model.alpha[n, m, k] for k in model.K) <= 1

    def obj_function(self, model=None):
        if model is None: model = self
        return self.transmissionBits(model) - self.RBGs(model) # Maximize the bits sent and minimize the number of RBGs used

    def solve(self, display=True):
        #self.model = ConcreteModel()
        #self._init_model()
        solver=SolverFactory("scip") #ipopt scip --> Ipopt is not working well

        """
        solver.options['max_iter'] = 10_000
        solver.options['tol'] = 1e-6
        solver.options['constr_viol_tol'] = 1e-6
        """

        #print("start")
        t = time()
        self.model.L.store_values(self.L)

        print(self.L)

        result = solver.solve(self.model, tee=False)

        """
        P = [self.Pmin + i*(self.Pmax-self.Pmin)/5000 for i in range(5001)]

        R = [self.R(0,0,None,p) for p in P]
        plt.plot(P, R)
        plt.title("Throughput as a function of power")
        plt.xlabel("Power (W)")
        plt.ylabel("Throughput (bits/s)")
        plt.show()
        R = [self.R(0,1,None,p) for p in P]
        plt.plot(P, R)
        plt.title("Throughput as a function of power")
        plt.xlabel("Power (W)")
        plt.ylabel("Throughput (bits/s)")
        plt.show()
        R = [self.R(1,0,None,p) for p in P]
        plt.plot(P, R)
        plt.title("Throughput as a function of power")
        plt.xlabel("Power (W)")
        plt.ylabel("Throughput (bits/s)")
        plt.show()
        R = [self.R(1,1,None,p) for p in P]
        plt.plot(P, R)
        plt.title("Throughput as a function of power")
        plt.xlabel("Power (W)")
        plt.ylabel("Throughput (bits/s)")
        plt.show()
        """






        try:
            pass
        except:
            print("Debug information:")

            for n in self.N:
                for m in self.M:
                    for k in self.K:
                        print(f"alpha({n:>2},{m:>2},{k:>3}) = {self.alpha[n,m,k]}")
                print()
            
            for n in self.N:
                for k in self.K:
                    print(f"beta({n:>2},{k:>3}) = {self.beta[n,k]}")
                print()
            
            print("\n")

            for n in self.N:
                for m in self.M:
                    print(f"P({n:>2},{m:>2}) = {self.P[n,m]}")
                print()
            
            exit()




        t = time() - t
        #print("finish")
        #print(result)
        #self.model.pprint()
    
        self.alpha = {(n, m, k): round(self.model.alpha[n, m, k].value) for n in self.model.N for m in self.model.M for k in self.model.K}
        self.P = {(n, m): self.model.P[n, m].value for n in self.model.N for m in self.model.M}
        self.beta = {(n, k): round(self.model.beta[n, k].value) for n in self.model.N for k in self.model.K}
        
        """
        print(f"Before changing: {value(self.obj_function(self.model))}")

        self.model.alpha[0,0,0] = 0
        self.model.alpha[0,1,0] = 0

        print(f"After changing: {value(self.obj_function(self.model))}")

        print("Are the constraints satisfied?")

        print(value(self.min_bool_bits_constraint(self.model,0)))
        print(value(self.min_bool_bits_constraint2(self.model,0)))

        print(value(self.alpha_constraint(self.model,0,0)))
        print(value(self.alpha_constraint(self.model,0,1)))
        print(value(self.alpha_constraint(self.model,1,0)))
        print(value(self.alpha_constraint(self.model,1,1)))

        print(value(self.beta_constraint(self.model,0)))
        """



        print(self.obj_function())




        if not display: return

        print("L_constraint")
        for k in self.K:
            print(self.model.L[k].value, self.L[k])

        print(f"Time: {t}s\n")

        print(f"Total bits sent: {round(self.transmissionBits())}")
        print(f"The bits: {self.Bits(0)}")
        print(f"Total RBGs used: {self.RBGs()}\n")
        
        print("Solution:")

        for n in self.N:
            for m in self.M:
                for k in self.K:
                    print(f"alpha({n:>2},{m:>2},{k:>3}) = {self.alpha[n,m,k]}")
            print()
        
        for n in self.N:
            for k in self.K:
                print(f"beta({n:>2},{k:>3}) = {self.beta[n,k]}")
            print()
        
        print("\n")

        for n in self.N:
            for m in self.M:
                print(f"P({n:>2},{m:>2}) = {self.P[n,m]}")
            print()

    def get_results_alpha(self):
        return {(n, m, k): self.model.alpha[n, m, k].value for n in self.model.N for m in self.model.M for k in self.model.K}
        
    def get_results_P(self):
        return {(n, m): self.model.P[n, m].value for n in self.model.N for m in self.model.M}

def asserts(data : dict) -> None:
    assert len(data["beta"]) == data["N"]
    assert len(data["beta"][0]) == data["K"]

    assert len(data["B"]) == data["N"]
    assert len(data["B"][0]) == data["M"]
    
    assert len(data["g"]) == data["N"]
    assert len(data["g"][0]) == data["K"]

    assert len(data["L"]) == data["K"]

def main():
    dBm_to_watts = lambda dBm : 10**(dBm/10)/1000 # watts = dBm_to_watts(dBm) conversor

    n = input()
    file = f"tests/test{n}/data.json"

    with open(file, 'r') as data_file: # load the data
        data = json.load(data_file)
    
    asserts(data)
    #print("No asserts")

    onl = ONL(data) # create the onl object

    #onl.solve() # solve the problem

    onl.gameloop(policy=onl.solve)

if __name__ == '__main__':
    import json
    main()