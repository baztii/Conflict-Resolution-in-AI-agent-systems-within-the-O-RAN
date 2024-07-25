import numpy as np
from pyomo.environ import log


""" Global variables """
DISPLAY = True

class ENVIRONMENT:
    def __init__(self, data):
        """ Parameters """
        self.N        = range(data['N']) # int // Number of base stations
        self.K        = range(data['K']) # int // Number of users int the system
        self.M        = range(data['M']) # int // Total number of RGB x BS
        self.bits     = data['bits']     # int // We assume the pending data is queued each 'bits'
        self.buffSize = data['buffSize'] # int // The size of the transmission buffer where the pending traffic is queued

        self.Pmin  = data['Pmin']  # float // min transmition power
        self.Pmax  = data['Pmax']  # float // max transmition power
        self.T     = data['T']     # float // Length of time slot
        self.sigma = data['sigma'] # float // noise
        self.lamda = data['lamda'] # float // Poison process rate per second

        self.g     = {(n,k):data['g'][n][k] for n in self.N for k in self.K}    # list // Channel coefficient between base stations and users (i.e g[n,k] is between BS n and user k)
        self.B     = {(n,m):data['B'][n][m] for n in self.N for m in self.M}    # list // Brandwith of all RBG (i.e. BW[n,m] determines the brandwith of the RBG m at the BS n)
        self.L     = {k:data['L'][k] for k in self.K}                           # list // Amount of remained data of all users in the transimssion buffer (i.e. L[k] is the remaining data of user k)
        
        """ Variables """
        self.P     = {(n,m):(self.Pmax+self.Pmin)/2 for n in self.N for m in self.M} # list // Transmission power allocation to RBG of BS (i.e. P[n,m] is the power of RBG m at BS n)
        self.alpha = {(n,m,k):0 for n in self.N for m in self.M for k in self.K}     # list // Distribution of RGB to each user (self.alpha[n,m,k] = 1 iff user k has RBG m at BS n, 0 otherwise)
        self.beta  = {(n,k):data['beta'][n][k] for n in self.N for k in self.K}      # list // User distribution in BS (i.e self.beta[n,k] = 1 iff user[k] is on BS n, 0 otherwise)

    def gamma_maj(self, n : int, m : int) -> list[float]:
        return [log(1+
                         sum([self.alpha[n_prime, m, k]*self.g[n_prime,k] for k in self.K])/
                         sum([self.alpha[n,m,k]*self.g[n,k] for k in self.K]))/log(2) for n_prime in self.N if n_prime != n]

    def eta(self, n : int, m : int, k : int, model=None):
        if model is None: model = self
        return ((model.alpha[n, m, k]*model.beta[n,k]*model.g[n,k]*model.P[n,m])
            /(sum([model.alpha[n_prime, m, k_prime]*model.beta[n_prime,k_prime]*model.g[n_prime,k_prime]*model.P[n_prime,m] # is there a typo on the paper?
                for n_prime in model.N if n_prime != n
                for k_prime in model.K])
            + model.sigma**2))

    def C(self, n : int, m : int, model=None):
        if model is None: model = self
        return model.B[n,m]*log(1+sum([self.eta(n,m,k,model) for k in model.K]))/log(2)
    
    def R(self, n : int, m : int, model=None):
        if model is None: model = self
        lhs = self.C(n,m,model) + 1
        rhs = sum([model.alpha[n,m,k]*model.beta[n,k]*model.L[k] for k in model.K])/model.T + 1
        p = 20
        #print(f"lhs: {lhs} and rhs: {rhs}")
        #print(f"Returned: {lhs + rhs - (lhs**p + rhs**p)**(1/p) - 1}")
        return lhs + rhs - (lhs**p + rhs**p)**(1/p) - 1 # min(a,b) as a continuous function

    def transmissionRate(self, model=None):
        if model is None: model = self
        return sum([self.R(n, m, model) for n in model.N for m in model.M])
    
    def TxData(self): # "send" the data of the buffer (inefficient)
        for k in self.K:
            for n in self.N:
                for m in self.M:
                    self.L[k] = max(0, self.L[k] - self.alpha[n,m,k]*self.beta[n,k]*round(self.R(n,m)*self.T)) # without max?

    def RqData(self): # "insert" new data to be transmitted
        pkg = np.random.poisson(self.lamda*self.T, self.K[-1]+1) # The expected value
        
        for k in self.K:
            self.L[k] = min(self.buffSize*self.bits, self.L[k]+self.bits*pkg[k])

    def valid(self) -> bool:
        for n in self.N:
            for m in self.M:
                if self.Pmin <= self.P[n,m] <= self.Pmax and sum([self.alpha[n,m,k] for k in self.K]) == 1:
                    continue
                return False

        return True

    def assign(self, policy=lambda:None):
        """
        Here goes the policy to solve the problem
        """
        policy(display=DISPLAY)

    def own_policy(self, display=DISPLAY):
        l = 0
        for n in self.N:
            for m in self.M:
                for k in self.K:
                    self.alpha[n,m,k] = 1 if k == l else 0
                
                l+=1
                l=l%(self.K[-1]+1)
        
        return
        
        for n in self.N:
            for m in self.M:
                for k in self.K:
                    print(f"alpha({n:>2},{m:>2},{k:>3}) = {self.alpha[n,m,k]}")
            print()    

    def gameloop(self, iter=10, policy=lambda:None):
        for i in range(iter):
            print(f"Iteration {i}:")
            self.assign(policy)
            #print("They throughput:", round(self.transmissionRate()))
            #self.assign(self.own_policy)
            #print("My   throughput:", round(self.transmissionRate()))
            print(self.L)
            self.TxData()
            print(self.L)
            self.RqData()
            #if i == 0: self.L = {k:100000 + 50*k for k in self.K}
            
            #print()

            #print(self.L)

def main():
    with open('tests/test3/data.json', 'r') as data_file: # load the data
        data = json.load(data_file)
    
    env = ENVIRONMENT(data)
    for n in env.N:
        for m in env.M:
            print(env.C(n,m))
            print(env.R(n,m))
    env.gameloop()

if __name__ == '__main__':
    import json
    main()