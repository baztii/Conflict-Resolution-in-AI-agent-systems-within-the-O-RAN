import numpy as np
from pyomo.environ import log


""" Global variables """
DISPLAY = True
ITER    = 20

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
        self.P     = {(n,m):(self.Pmin+self.Pmax)/2 for n in self.N for m in self.M} # list // Transmission power allocation to RBG of BS (i.e. P[n,m] is the power of RBG m at BS n)
        self.alpha = {(n,m,k):0 for n in self.N for m in self.M for k in self.K}     # list // Distribution of RGB to each user (self.alpha[n,m,k] = 1 iff user k has RBG m at BS n, 0 otherwise)
        self.beta  = {(n,k):0 for n in self.N for k in self.K}                       # list // User distribution in BS (i.e self.beta[n,k] = 1 iff user[k] is on BS n, 0 otherwise)

    def gamma_maj(self, n : int, m : int) -> list[float]:
        return [log(1+
                         sum([self.alpha[n_prime, m, k]*self.g[n_prime,k] for k in self.K])/
                         sum([self.alpha[n,m,k]*self.g[n,k] for k in self.K]))/log(2) for n_prime in self.N if n_prime != n]

    def numerator(self, n : int, m : int, k : int, model=None):
        if model is None: model = self
        return model.alpha[n, m, k]*model.g[n,k]*model.P[n,m]

    def denominator(self, n : int, m : int, k : int, model=None):
        if model is None: model = self
        return sum([model.alpha[n_prime, m, k]*model.g[n_prime,k]*model.P[n_prime,m]
            for n_prime in model.N if n_prime != n
            for k in model.K])

    def eta(self, n : int, m : int, k : int, model=None):
        if model is None: model = self
        return self.numerator(n, m, k, model)/(self.denominator(n, m, k, model) + model.sigma) # If sigma is too small there are numerical errors

    def C(self, n : int, m : int, model=None):
        if model is None: model = self
        return model.B[n,m]*log(1+sum([self.eta(n,m,k,model) for k in model.K]))/log(2)

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

    def lhs(self, k : int, model=None):
        if model is None: model = self
        return model.L[k]

    def rhs(self, k : int, model=None):
        if model is None: model = self
        return sum([model.alpha[n,m,k]*self.R(n, m, model)*model.T for n in model.N for m in model.M])

    def Bits(self, k : int, model=None):
        if model is None or model == self: return min(self.lhs(k), self.rhs(k))
        return model.min_bool_bits[k]*self.lhs(k, model) + (1-model.min_bool_bits[k])*self.rhs(k, model) 
    
    def transmissionBits(self, model=None):
        if model is None: model = self
        return sum([self.Bits(k, model) for k in model.K])

    def RBGs(self, model=None):
        if model is None: model = self
        return sum([model.alpha[n,m,k] for n in model.N for m in model.M for k in model.K])

    def TxData(self): # "send" the data of the buffer
        for k in self.K:
            self.L[k] = round(self.L[k] - self.Bits(k))

    def RqData(self): # "insert" new data to be transmitted
        pkg = np.random.poisson(self.lamda*self.T, self.K[-1]+1) # The expected value
        
        for k in self.K:
            self.L[k] = min(self.buffSize*self.bits, self.L[k]+self.bits*pkg[k])

    def assign(self, policy=lambda:None):
        """ Here goes the policy to solve the problem """
        policy(display=DISPLAY)

    def gameloop(self, iter=ITER, policy=lambda:None):
        for i in range(iter):
            print(f"Iteration {i}:")
            print("Bits in the buffer:", end=' ')
            print(f"{self.L}")
            self.assign(policy)
            self.TxData()
            self.RqData()
        
        print("Bits remaining in the buffer:", end=' ')
        print(f"{self.L}")

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