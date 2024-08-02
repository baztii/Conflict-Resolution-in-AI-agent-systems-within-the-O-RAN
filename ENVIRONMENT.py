import numpy as np
from pyomo.environ import log


""" Global variables """
DISPLAY = True
ITER    = 10

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
        self.P     = {(n,m):1.6961971202034667 for n in self.N for m in self.M} # list // Transmission power allocation to RBG of BS (i.e. P[n,m] is the power of RBG m at BS n)
        self.alpha = {(n,m,k):0 for n in self.N for m in self.M for k in self.K}     # list // Distribution of RGB to each user (self.alpha[n,m,k] = 1 iff user k has RBG m at BS n, 0 otherwise)
        self.beta  = {(n,k):data['beta'][n][k] for n in self.N for k in self.K}      # list // User distribution in BS (i.e self.beta[n,k] = 1 iff user[k] is on BS n, 0 otherwise)

    def gamma_maj(self, n : int, m : int) -> list[float]:
        return [log(1+
                         sum([self.alpha[n_prime, m, k]*self.g[n_prime,k] for k in self.K])/
                         sum([self.alpha[n,m,k]*self.g[n,k] for k in self.K]))/log(2) for n_prime in self.N if n_prime != n]


    def numerator(self, n : int, m : int, k : int, model=None):
        if model is None: model = self
        return model.alpha[n, m, k]*model.beta[n,k]*model.g[n,k]*model.P[n,m]

    def denominator(self, n : int, m : int, k : int, model=None):
        if model is None: model = self
        return sum([model.alpha[n_prime, m, k]*model.beta[n_prime,k]*model.g[n_prime,k]*model.P[n_prime,m] # is there a typo on the paper?
            for n_prime in model.N if n_prime != n
            for k in model.K])

    def eta(self, n : int, m : int, k : int, model=None):
        if model is None: model = self
        return self.numerator(n, m, k, model)/(self.denominator(n, m, k, model) + model.sigma)
    
    def eta1(self, n : int, m : int, k : int, model=None):
        if model is None: model = self
        return ((model.alpha[n, m, k]*model.beta[n,k]*model.g[n,k]*model.P[n,m])
            /(sum([model.alpha[n_prime, m_prime, k_prime]*model.beta[n_prime,k_prime]*model.g[n_prime,k_prime]*model.P[n_prime,m_prime] # is there a typo on the paper?
                for n_prime in model.N
                for k_prime in model.K
                for m_prime in model.M]) - model.alpha[n, m, k]*model.beta[n,k]*model.g[n,k]*model.P[n,m]
            + model.sigma**2))
    
    def eta2(self, n : int, m : int, k : int, model=None):
        if model is None: model = self
        return ((model.alpha[n, m, k]*model.beta[n,k]*model.g[n,k]*model.P[n,m])
            /(sum([model.alpha[n_prime, m, k_prime]*model.beta[n_prime,k_prime]*model.g[n_prime,k_prime]*model.P[n_prime,m] # is there a typo on the paper?
                for n_prime in model.N
                for k_prime in model.K]) - sum([model.alpha[n, m, k_prime]*model.beta[n,k_prime]*model.g[n,k_prime]*model.P[n,m] for k_prime in model.K])
            + model.sigma**2))

    def C(self, n : int, m : int, model=None):
        if model is None: model = self
        return model.B[n,m]*log(1+sum([self.eta(n,m,k,model) for k in model.K]))/log(2)
    
    def R_fancy(self, n : int, m : int, model=None):
        if model is None: model = self
        lhs = self.C(n,m,model) + 1
        rhs = sum([model.alpha[n,m,k]*model.beta[n,k]*model.L[k] for k in model.K])/model.T + 1
        p = 20
        #print(f"lhs: {lhs} and rhs: {rhs}")
        #print(f"Returned: {lhs + rhs - (lhs**p + rhs**p)**(1/p) - 1}")
        return lhs + rhs - (lhs**p + rhs**p)**(1/p) - 1 # min(a,b) as a continuous function

    def R(self, n : int, m : int, model=None, P=None):
        if model is None: model = self        
        if P is not None:
            P = self.P[n,m] = P
            return self.C(n,m,model)

        a = 3 # The power value that maximizes the transmission rate of RBG (n,m)
        throughput_value = 10000 # The value of the maximum throughput that the RBG can achieve

        #return log(-(0.4*model.P[n,m]-0.4*a)**2 + 4)/log(4)*throughput_value

        ####print((self.C(n,m,self)/100000)) --> Check what happens to the C function (values too small???) what happens when min{C,L} = C (there are some problems!!!!)
        return self.C(n,m,model)

    def transmissionRate(self, model=None):
        if model is None: model = self
        return sum([self.R(n, m, model) for n in model.N for m in model.M])

    """
    Calculates the number of bits for a given user in a fancy way.

    Args:
        k (int): The user index.
        model (Environment, optional): The environment model. Defaults to None.

    Returns:
        float: The number of bits for the user.

    Note:
        This function calculates the number of bits for a given user in a fancy way by using a specific formula.
        It takes into account the user's remaining data in the transmission buffer, the distribution of
        RBGs to each user, the transmission rate of each RBG, and the transmission time.
        The formula used is a continuous approximation of the minimum function between the user's remaining data
        and the sum of the transmission rates of each RBG multiplied by the transmission time.

    """
    def Bits_fancy(self, k : int, model=None):
        if model is None: model = self
        lhs = model.L[k] + 1
        rhs = sum([model.alpha[n,m,k]*model.beta[n,k]*self.R(n, m, model)*model.T for n in model.N for m in model.M]) + 1
        p = 50
        #print(f"lhs: {lhs} and rhs: {rhs}")
        #print(f"Returned: {lhs + rhs - (lhs**p + rhs**p)**(1/p) - 1}")
        return lhs + rhs - ((lhs)**p + (rhs)**p)**(1/p) - 1 # min(a,b) as a continuous function

    def lhs(self, k : int, model=None):
        if model is None: model = self
        return model.L[k]

    def rhs(self, k : int, model=None):
        if model is None: model = self
        return sum([model.alpha[n,m,k]*model.beta[n,k]*self.R(n, m, model)*model.T for n in model.N for m in model.M])

    def Bits(self, k : int, model=None):
        if model is None or model == self: return min(self.lhs(k), self.rhs(k))
        return model.min_bool_bits[k]*self.lhs(k, model) + (1-model.min_bool_bits[k])*self.rhs(k, model) 
    
    def transmissionBits(self, model=None):
        if model is None: model = self
        return sum([self.Bits(k, model) for k in model.K])

    def RBGs(self, model=None):
        if model is None: model = self
        return sum([model.alpha[n,m,k] for n in model.N for m in model.M for k in model.K])

    def TxData(self): # "send" the data of the buffer (inefficient)
        for k in self.K:
            self.L[k] = round(self.L[k] - self.Bits(k))
            """
            for n in self.N:
                for m in self.M:
                    self.L[k] = max(0, self.L[k] - self.alpha[n,m,k]*self.beta[n,k]*round(self.R(n,m)*self.T)) # without max?
            """

    def RqData(self): # "insert" new data to be transmitted
        """
        for k in self.K:
            self.L[k] = self.buffSize*self.bits
        return
        """
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

        for n in self.N:
            for m in self.M:
                self.P[n,m] = self.Pmax
            
        return

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

    def gameloop(self, iter=ITER, policy=lambda:None):
        for i in range(iter):
            print(f"Iteration {i}:")
            self.assign(policy)
            #print("They throughput:", round(self.transmissionRate()))
            #self.assign(self.own_policy)
            #print("My   throughput:", round(self.transmissionRate()))
            for n in self.N:
                for m in self.M:
                    pass#print(f"R({n},{m}): {self.R(n,m)}")
            print()
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