"""
# ONL
    This module provides the ONL class which represents the online optimization model.

## Classes
    - **ONL**: The online optimization model

## Functions
    - *main*: The main function to show how should the online optimization model work

## Author
    Miquel P. Baztan Grau

## Date
    21/08/2024
"""

from pyomo.environ import *
from time import time
from ENVIRONMENT import ENVIRONMENT as ENV
import sys
import os

from utils import asserts

"""Global variables"""
FILE = 1                # file number
START = 1               # starting file
END = 7                 # ending file
TEE = True              # tee output
SOLVER = "scip"         # solver (you can also use ipopt)
CONSOLE = False         # console output
TERMINAL = sys.stdout   # terminal
OF = sys.stdout         # output file

class ONL(ENV):
    """
    Online optimization model.

    This class extends the `ENVIRONMENT` class and initializes the parameters and variables for the online optimization model.

    Attributes:
        ENV.attributes (*): The attributes inherited from the ENVIRONMENT class (see `ENVIRONMENT.py` for more information).
        model (ConcreteModel): Pyomo model for the online optimization problem.

    ## Methods
        - *ENV.methods*: The methods inherited from the ENVIRONMENT class (see `ENVIRONMENT.py` for more information).
        - *_init_model*: Sets up the solver model.
        - *min_bool_bits_constraint*: Constraint1 for the solver to be able to calculate the min(rhs, lhs)
        - *min_bool_bits_constraint2*: Constraint2 for the solver to be able to calculate the min(rhs, lhs)
        - *beta_constraint*: Each user must be assigned to a base station constraint
        - *beta_constraint2*: You cannot give a RBG m from one BS n to a user k that is not situated on BS n constraint
        - *alpha_constrint*: You cannot give the same RBG to more than one user
        - *obj_function*: Objective function for the solver to maximize
        - *solve*: Solves the online optimization model
    """
    def __init__(self, data : dict) -> None:
        """
        Initializes the ONL object.

        Initializes the parameters and variables for the online optimization model.
        Calls the superclass's `__init__` method and sets up the solver model.

        Parameters:
            data (dict): Dictionary containing the data for the optimization model.

        Returns:
            None
        """

        """ Initialize parameters and variables"""
        super().__init__(data)

        """ Solver model """
        self.model = ConcreteModel()
        self._init_model()
        
    def _init_model(self) -> None:
        """
        Sets up the solver model based of the parameters and variables in the `ENVIRONMENT` class.

        Returns:
            None
        """

        """ Sets """
        self.model.N     = Set(initialize=self.N)
        self.model.K     = Set(initialize=self.K)
        self.model.M     = Set(initialize=self.M)

        """ Parameters """
        self.model.sigma = Param(initialize=self.sigma)
        self.model.T     = Param(initialize=self.T)

        self.model.g     = Param(self.model.N, self.model.K, initialize=self.g)
        self.model.B     = Param(self.model.N, self.model.M, initialize=self.B)
        self.model.L     = Param(self.model.K, mutable=True, initialize=self.L)

        """ Variables """
        self.model.alpha         = Var(self.model.N, self.model.M, self.model.K, domain=Binary, initialize=self.alpha)
        self.model.beta          = Var(self.model.N, self.model.K, domain=Binary, initialize=self.beta)
        self.model.P             = Var(self.model.N, self.model.M, bounds = (self.Pmin, self.Pmax), initialize=self.P)
        self.model.min_bool_bits = Var(self.model.K, domain=Binary)

        """ Objective function """
        self.model.obj = Objective(rule=self.obj_function,sense=maximize)

        """ Constraints """
        self.model.alphaConstr    = Constraint(self.model.N, self.model.M, rule=self.alpha_constraint)
        
        self.model.betaConstr     = Constraint(self.model.K, rule=self.beta_constraint)
        self.model.betaConstr2     = Constraint(self.model.N, self.model.M, self.model.K, rule=self.beta_constraint2)
        
        self.model.min_bool_bitsConstr = Constraint(self.model.K, rule=self.min_bool_bits_constraint)
        self.model.min_bool_bitsConstr2 = Constraint(self.model.K, rule=self.min_bool_bits_constraint2)

    def min_bool_bits_constraint(self, model, k : int) -> pyomo.core.expr.relational_expr.InequalityExpression:
        """
        Constraint1 for the solver to be able to calculate the min(rhs, lhs)

        Parameters:
            model (ConcreteModel): The solver model
            k (int): The index of the user

        Returns:
            pyomo.core.expr.relational_expr.InequalityExpression: The constraint that pyomo will try to satisfy
            
        """

        return model.min_bool_bits[k]*(self.rhs(k, model) -  self.lhs(k, model)) >= 0
    
    def min_bool_bits_constraint2(self, model, k : int) -> pyomo.core.expr.relational_expr.InequalityExpression:
        """
        Constraint2 for the solver to be able to calculate the min(rhs, lhs)

        Parameters:
            model (ConcreteModel): The solver model
            k (int): The index of the user

        Returns:
            pyomo.core.expr.relational_expr.InequalityExpression: The constraint that pyomo will try to satisfy
            
        """

        return (1-model.min_bool_bits[k])*(self.rhs(k, model) - self.lhs(k, model)) <= 0

    def beta_constraint(self, model, k : int) -> pyomo.core.expr.relational_expr.EqualityExpression:
        """
        Each user must be assigned to a base station constraint

        Parameters:
            model (ConcreteModel): The solver model
            k (int): The index of the user

        Returns:
            pyomo.core.expr.relational_expr.EqualityExpression: The constraint that pyomo will try to satisfy
        """

        return sum(model.beta[n,k] for n in model.N) == 1

    def beta_constraint2(self, model, n : int, m : int, k : int) -> pyomo.core.expr.relational_expr.EqualityExpression:
        """
        You cannot give a RBG m from one BS n to a user k that is not situated on BS n constraint

        Parameters:
            model (ConcreteModel): The solver model
            n (int): The index of the base station
            m (int): The index of the RBG
            k (int): The index of the user

        Returns:
            pyomo.core.expr.relational_expr.EqualityExpression: The constraint that pyomo will try to satisfy
        """
        return (1 - model.beta[n,k])*model.alpha[n,m,k] == 0
    
    def alpha_constraint(self, model, n : int, m : int) -> pyomo.core.expr.relational_expr.InequalityExpression:
        """
        You cannot give the same RBG to more than one user

        Parameters:
            model (ConcreteModel): The solver model
            n (int): The index of the base station
            m (int): The index of the RBG

        Returns:
            pyomo.core.expr.relational_expr.InequalityExpression: The constraint that pyomo will try to satisfy
        """

        return sum(model.alpha[n, m, k] for k in model.K) <= 1

    def obj_function(self, model=None) -> float:
        """
        Objective function for the solver (maximize the bits sent and minimize the number of RBGs used)

        Parameters:
            model (ConcreteModel): The solver model

        Returns:
            float: The value of the objective function
        """

        if model is None: model = self
        return self.transmissionBits(model) - self.RBGs(model)# Maximize the bits sent and minimize the number of RBGs used

    def solve(self):
        """
        Solves the problem and prints the results in the standard output

        Returns:
            None
        """

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
