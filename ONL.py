from pyomo.environ import *
from time import *

class ONL:
    def __init__(self, data : dict):
        """ Parameters """
        self.N     = range(data['N'])
        self.K     = range(data['K'])
        self.M     = range(data['M'])

        self.Pmin  = data['Pmin']
        self.Pmax  = data['Pmax']
        self.sigma = data['sigma']
        self.T     = data['T']

        self.beta  = {(n,k):data['beta'][n][k] for n in self.N for k in self.K}
        self.g     = {(n,k):data['g'][n][k] for n in self.N for k in self.K}
        self.B     = {(n,m):data['B'][n][m] for n in self.N for m in self.M}
        self.L     = {k:data['L'][k] for k in self.K}
        
        """ Variables """
        self.P     = {(n,m):(self.Pmax+self.Pmin)/2 for n in self.N for m in self.M}
        self.alpha = {(n,m,k):0 for n in self.N for m in self.M for k in self.K}

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

        self.model.beta  = Param(self.model.N, self.model.K, initialize=self.beta)
        self.model.g     = Param(self.model.N, self.model.K, initialize=self.g)
        self.model.B     = Param(self.model.N, self.model.M, initialize=self.B)
        self.model.L     = Param(self.model.K, initialize=self.L)

        """ Variables """
        self.model.alpha = Var(self.model.N, self.model.M, self.model.K, domain=Binary, initialize=self.alpha)
        self.model.P     = Var(self.model.N, self.model.M, bounds = (self.Pmin, self.Pmax), initialize=self.P)

        """ Objective function """
        self.model.obj = Objective(rule=self.obj_function,sense=maximize)

        """ Constraints """
        self.model.alphaConstr = Constraint(self.model.N, self.model.M, rule=self.alpha_constraint)

    def eta(self, model, n : int, m : int, k : int):
        return ((model.alpha[n, m, k]*model.beta[n,k]*model.g[n,k]*model.P[n,m])
            /(sum([model.alpha[n_prime, m, k_prime]*model.beta[n_prime,k_prime]*model.g[n_prime,k_prime]*model.P[n_prime,m] # is there a typo on the paper?
                for n_prime in model.N if n_prime != n
                for k_prime in model.K])
            + model.sigma**2))

    def C(self, model, n : int, m : int):
        return model.B[n,m]*log(1+sum([self.eta(model,n,m,k) for k in model.K]))/log(2)
    
    def R(self, model, n : int, m : int):
        lhs = self.C(model,n,m)  + 1
        rhs = sum([model.alpha[n,m,k]*model.beta[n,k]*model.L[k] for k in model.K])/model.T  + 1
        p = 20
        return lhs + rhs - (lhs**p + rhs**p)**(1/p) # min(a,b) as a continuous function

    def obj_function(self, model):
        return sum([self.R(model, n, m) for n in model.N for m in model.M])
    
    def obj_function2(self, model):
        return sum([(model.B[n,m]*log(1+sum([(((model.alpha[n, m, k]*model.beta[n,k]*model.g[n,k]*model.P[n,m])
            /(sum([model.alpha[n_prime, m, k_prime]*model.beta[n_prime,k_prime]*model.g[n_prime,k_prime]*model.P[n_prime,m] # is there a typo on the paper?
                for n_prime in model.N if n_prime != n
                for k_prime in model.K])
            + model.sigma**2))) for k in model.K]))/log(2)) for n in model.N for m in model.M])

    def alpha_constraint(self, model, n : int, m : int):
        return sum(model.alpha[n, m, k] for k in model.K) == 1

    def solve(self):
        self.model.create_instance()
        solver=SolverFactory("ipopt") #ipopt scip
        t = time()
        print("start")
        result = solver.solve(self.model, tee=True)
        print("finish")
        print("time: ", time()-t, "s")
        print()
        #print(result)
        #self.model.pprint()
    

        for i in self.get_results_alpha():
            print(i, self.model.alpha[i].value)
        
        for i in self.get_results_P():
            print(i, self.model.P[i].value)

        self.alpha = {(n, m, k): self.model.alpha[n, m, k].value for n in self.model.N for m in self.model.M for k in self.model.K}
        self.P = {(n, m): self.model.P[n, m].value for n in self.model.N for m in self.model.M}

        print(self.obj_function(self))
        print()

        print([[self.R(self,n,m) for m in self.M] for n in self.N])
        print()
        print([[value(self.C(self,n,m)) for m in self.M] for n in self.N])
        print()
        print([self.L[k] for k in self.K])

        cond = True

        for n in self.N:
            for m in self.M:
                for k in self.K:
                    self.alpha[n,m,k] = 1 if k == 0 else 0
        
        for n in self.N:
            for m in self.M:
                cond = cond and value(self.alpha_constraint(self,n,m))
        
        print(cond)        
        print(value(self.obj_function(self.model)))
        print(self.obj_function(self))

    def get_results_alpha(self):
        return {(n, m, k): self.model.alpha[n, m, k].value for n in self.model.N for m in self.model.M for k in self.model.K}
        
    def get_results_P(self):
        return {(n, m): self.model.P[n, m].value for n in self.model.N for m in self.model.M}

def main():
    dBm_to_watts = lambda dBm : 10**(dBm/10)/1000 # watts = dBm_to_watts(dBm) conversor

    n = 2 #input()
    file = f"tests/test{n}/data.json"

    with open(file, 'r') as data_file: # load the data
        data = json.load(data_file)
    
    assert len(data["beta"]) == data["N"]
    assert len(data["beta"][0]) == data["K"]

    assert len(data["B"]) == data["N"]
    assert len(data["B"][0]) == data["M"]
    
    assert len(data["g"]) == data["N"]
    assert len(data["g"][0]) == data["K"]

    assert len(data["L"]) == data["K"]

    print("No asserts")

    onl = ONL(data) # create the onl object

    onl.solve() # solve the problem

if __name__ == '__main__':
    import json
    main()