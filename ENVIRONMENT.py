import numpy as np
import math

class ENVIRONMENT:
    def __init__(self, data):
        """ Parameters """
        self.N        = data['N']        # int // Number of base stations
        self.K        = data['K']        # int // Number of users int the system
        self.M        = data['M']        # int // Total number of RGB x BS
        self.bits     = data['bits']     # int // We assume the pending data is queued each 'bits'
        self.buffSize = data['buffSize'] # int // The size of the transmission buffer where the pending traffic is queued

        self.Pmin  = data['Pmin']  # float // min transmition power
        self.Pmax  = data['Pmax']  # float // max transmition power
        self.T     = data['T']     # float // Length of time slot
        self.sigma = data['sigma'] # float // noise
        self.lamda = data['lamda'] # float // Poison process rate per second

        self.beta  = {(n,k):data['beta'][n][k] for n in range(self.N) for k in range(self.K)} # list // User distribution in BS (i.e self.beta[n,k] = 1 iff user[k] is on BS n, 0 otherwise)
        self.g     = {(n,k):data['g'][n][k] for n in range(self.N) for k in range(self.K)}    # list // Channel coefficient between base stations and users (i.e g[n,k] is between BS n and user k)
        self.B     = {(n,m):data['B'][n][m] for n in range(self.N) for m in range(self.M)}    # list // Brandwith of all RBG (i.e. BW[n,m] determines the brandwith of the RBG m at the BS n)
        self.L     = {k:data['L'][k] for k in range(self.K)}                                  # list // Amount of remained data of all users in the transimssion buffer (i.e. L[k] is the remaining data of user k)
        
        """ Variables """
        self.P     = {(n,m):(self.Pmax+self.Pmin)/2 for n in range(self.N) for m in range(self.M)}    # list // Transmission power allocation to RBG of BS (i.e. P[n,m] is the power of RBG m at BS n)
        self.alpha = {(n,m,k):1 for n in range(self.N) for m in range(self.M) for k in range(self.K)} # list // Distribution of RGB to each user (self.alpha[n,m,k] = 1 iff user k has RBG m at BS n, 0 otherwise)

    def gamma_maj(self, n : int, m : int) -> list[float]:
        return [math.log2(1+
                         sum([self.alpha[n_prime, m, k]*self.g[n_prime,k] for k in range(self.K)])/
                         sum([self.alpha[n,m,k]*self.g[n,k] for k in range(self.K)])) for n_prime in range(self.N) if n_prime != n]

    def eta(self, n : int, m : int, k : int) -> float:
        return ((self.alpha[n, m, k]*self.g[n,k]*self.P[n,m])
            /(sum([self.alpha[n_prime, m, k_prime]*self.beta[n_prime,k_prime]*self.g[n_prime,k_prime]*self.P[n_prime,m] # is there a typo on the paper?
                for n_prime in range(self.N) if n_prime != n
                for k_prime in range(self.K)])
            + self.sigma**2))

    def C(self, n : int, m : int) -> float:
        return self.B[n,m]*math.log2(1+sum([self.eta(n,m,k) for k in range(self.K)]))
    
    def R(self, n : int, m : int) -> float:
        lhs = self.C(n,m)*self.T
        rhs = sum([self.alpha[n,m,k]*self.beta[n,k]*self.L[k] for k in range(self.K)])
        return min(lhs, rhs)

    def transmissionRate(self) -> float:
        return sum([self.R(n, m) for n in range(self.N) for m in range(self.M)])

    def updateBuffer(self) -> list[bool]:
        probability = np.random.poisson(self.lamda*self.T, self.K) #math.exp(-self.lamda*self.T/1000) * (self.lamda*self.T/1000)
        ret = np.random.random(self.K) > probability
        for k in range(self.K): # "insert" new data to be transmitted
            if ret[k]:
                self.L[k] = min(self.buffSize, self.L[k]+self.bits)

            for n in range(self.N): # "send" the data of the buffer
                for m in range(self.M):
                    self.L[k] = max(0, self.alpha[n,m,k]*(self.L[k] - int(self.R(n,m))))

        return ret.tolist()

    def valid(self) -> bool:
        for n in range(self.N):
            for m in range(self.M):
                if self.Pmin <= self.P[n,m] <= self.Pmax and sum([self.alpha[n,m,k] for k in range(self.K)]) == 1:
                    continue
                return False

        return True

    def assign(self, policy=lambda:None):
        """
        Here goes the policy to solve the problem
        """
        policy()

    def gameloop(self, iter=1000):
        for _ in range(iter):
            self.updateBuffer()
            self.assign()
    
def main():
    with open('tests/data.json', 'r') as data_file: # load the data
        data = json.load(data_file)
    
    env = ENVIRONMENT(data)
    env.gameloop()

if __name__ == '__main__':
    import json
    main()