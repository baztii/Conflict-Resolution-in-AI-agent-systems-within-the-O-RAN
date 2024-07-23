from pyomo.environ import *

class ONL:
    def __init__(self, data : dict):
        """ Parameters """
        self.N     = data['N']
        self.K     = data['K']
        self.M     = data['M']

        self.Pmin  = data['Pmin']
        self.Pmax  = data['Pmax']
        self.sigma = data['sigma']
        self.T     = data['T']

        self.beta  = {(n,k):data['beta'][n][k] for n in range(self.N) for k in range(self.K)}
        self.g     = {(n,k):data['g'][n][k] for n in range(self.N) for k in range(self.K)}
        self.B     = {(n,m):data['B'][n][m] for n in range(self.N) for m in range(self.M)}
        self.L     = {k:data['L'][k] for k in range(self.K)}
        
        """ Variables """
        self.P     = {(n,m):(self.Pmax+self.Pmin)/2 for n in range(self.N) for m in range(self.M)}
        self.alpha = {(n,m,k):1 for n in range(self.N) for m in range(self.M) for k in range(self.K)}

        """ Solver model """
        self.model = ConcreteModel()
        self._init_model()
        
    def _init_model(self):
        """ Sets """
        self.model.N     = RangeSet(0,self.N-1)
        self.model.K     = RangeSet(0,self.K-1)
        self.model.M     = RangeSet(0,self.M-1)

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
        return ((model.alpha[n, m, k]*model.g[n,k]*model.P[n,m])
            /(sum([model.alpha[n_prime, m, k_prime]*model.beta[n_prime,k_prime]*model.g[n_prime,k_prime]*model.P[n_prime,m] # is there a typo on the paper?
                for n_prime in model.N if n_prime != n
                for k_prime in model.K])
            + model.sigma**2))

    def C(self, model, n : int, m : int):
        return model.B[n,m]*log(1+sum([self.eta(model,n,m,k) for k in model.K]))/log(2)
    
    def R(self, model, n : int, m : int):
        lhs = value(self.C(model,n,m)*model.T)
        rhs = value(sum([model.alpha[n,m,k]*model.L[k] for k in model.K]))
        return min(lhs, rhs)

    def obj_function(self, model):
        return sum([self.R(model, n, m) for n in model.N for m in model.M])

    def alpha_constraint(self, model, n : int, m : int):
        return sum(model.alpha[n, m, k] for k in model.K) == 1

    def solve(self):
        self.model.create_instance()
        solver=SolverFactory("scip")
        solver.solve(self.model)
        self.model.pprint()

    """  
    def get_results_alpha(self):
        return {(n, m, k): self.model.alpha[n, m, k].value for n in self.model.N for m in self.model.M for k in self.model.K}
    
    def get_results_P(self):
        return {(n, m): self.model.P[n, m].value for n in self.model.N for m in self.model.M}
    """

def main():
    dBm_to_watts = lambda dBm : 10**(dBm/10)/1000 # watts = dBm_to_watts(dBm) conversor

    with open('tests/data.json', 'r') as data_file: # load the data
        data = json.load(data_file)
    
    onl = ONL(data) # create the onl object
    onl.solve() # solve the problem

if __name__ == '__main__':
    import json
    main()