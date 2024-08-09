import numpy as np
from pyomo.environ import log
import random


""" Global variables """
DISPLAY = True
ITER    = 10
DIV = 50

class ENVIRONMENT():
    def __init__(self, data, render):
        " Save the data "
        self.data = data
        self.render = render
        self.iterations = 0

        """ Parameters """
        self.N        = range(data['N']) # int // Number of base stations
        self.K        = range(data['K']) # int // Number of users int the system
        self.M        = range(data['M']) # int // Total number of RGB x BS
        self.bits     = data['bits']     # int // We assume the pending data is queued each 'bits'
        self.buffSize = data['buffSize'] # int // The size of the transmission buffer where the pending traffic is queued

        self.Pmin  = data['Pmin']  # float // min transmition power
        self.Pmax  = data['Pmax']  # float // max transmition power
        self.T     = data['T']     # float // Length of time slot
        self.sigma = data['sigma']*1e10 # float // noise
        self.lamda = data['lamda'] # float // Poison process rate per second

        self.g     = {(n,k):data['g'][n][k]*1e10 for n in self.N for k in self.K}    # list // Channel coefficient between base stations and users (i.e g[n,k] is between BS n and user k)
        self.B     = {(n,m):data['B'][n][m] for n in self.N for m in self.M}    # list // Brandwith of all RBG (i.e. BW[n,m] determines the brandwith of the RBG m at the BS n)
        self.L     = {k:data['L'][k] for k in self.K}                           # list // Amount of remained data of all users in the transimssion buffer (i.e. L[k] is the remaining data of user k)
        
        """ Variables """
        self.P     = {(n,m):self.Pmin for n in self.N for m in self.M} # list // Transmission power allocation to RBG of BS (i.e. P[n,m] is the power of RBG m at BS n)
        self.alpha = {(n,m,k):0 for n in self.N for m in self.M for k in self.K}     # list // Distribution of RGB to each user (self.alpha[n,m,k] = 1 iff user k has RBG m at BS n, 0 otherwise)
        self.beta  = {(n,k):1 for n in self.N for k in self.K}                       # list // User distribution in BS (i.e self.beta[n,k] = 1 iff user[k] is on BS n, 0 otherwise)
    
        usr = 0
        for n in self.N:
            for m in self.M:
                for k in self.K:
                    self.alpha[n,m,k] = int(k == usr)
                usr += 1
                usr = usr %(self.K[-1] + 1)


    def gamma_maj(self, n : int, m : int) -> list[float]:
        return [log(1+
                         sum([self.alpha[n_prime, m, k]*self.beta[n_prime,k]*self.g[n_prime,k] for k in self.K])/
                         sum([self.alpha[n,m,k]*self.beta[n,k]*self.g[n,k] for k in self.K]))/log(2) for n_prime in self.N if n_prime != n]

    def numerator(self, n : int, m : int, k : int, model=None):
        if model is None: model = self
        return model.alpha[n, m, k]*model.beta[n,k]*model.g[n,k]*model.P[n,m]

    def denominator(self, n : int, m : int, model=None):
        if model is None: model = self
        return sum([model.alpha[n_prime, m, k]*model.beta[n_prime,k]*model.g[n_prime,k]*model.P[n_prime,m]
            for n_prime in model.N if n_prime != n
            for k in model.K])

    def eta(self, n : int, m : int, k : int, model=None):
        if model is None: model = self
        return self.numerator(n, m, k, model)/(self.denominator(n, m, model) + model.sigma) # If sigma is too small there are numerical errors

    def C(self, n : int, m : int, model=None):
        if model is None: model = self
        return model.B[n,m]*log(1+sum([self.eta(n,m,k,model) for k in model.K]))/log(2)

    def R(self, n : int, m : int, model=None, P=None):
        if model is None: model = self        
        if P is not None:
            P = self.P[n,m] = P
            return self.C(n,m,model)

        a = 3 # The power value that maximizes the transmission rate of RBG (n,m)
        throughput_value = 2000000 # The value of the maximum throughput that the RBG can achieve

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
        return sum([model.alpha[n,m,k]*model.beta[n,k]*self.R(n, m, model)*model.T for n in model.N for m in model.M])

    def Bits(self, k : int, model=None):
        if model is None or model == self: return min(self.lhs(k), self.rhs(k))

        """
        p = 20
        rhs = self.rhs(k, model) + 1
        lhs = self.lhs(k, model) + 1
        return rhs + lhs - (rhs**p+lhs**p)**(1/p) - 1
        """
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

        for k in self.K:
            self.L[k] = 100_000# + np.random.randint(-50_000, 50_000)
        
        return

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

    def reset(self):
        self.__init__(self.data, self.render)
        self.RqData()
        self.ori_bits = sum(self.L.values())

        new_state = []

        for n in self.N:
            for m in self.M:
                new_state += self.gamma_maj(n, m)
        
        for k in self.K:
            new_state.append(self.Bits(k))

        for k in self.K:
            new_state.append(self.L[k])

        for n in self.N:
            for m in self.M:
                new_state.append(self.P[n,m]) 
        
        new_state.append(self.iterations)
        
        return new_state
    
    def action_translator(self, action):

        N = self.N[-1] + 1
        M = self.M[-1] + 1

        BS  = int(action//(DIV*M))
        RBG = int((action%(DIV*M))//DIV)
        P   = int(action%(DIV*M)%DIV)

        return BS, RBG, P
    
    def step(self, action):
        N = self.N[-1] + 1
        M = self.M[-1] + 1
        self.iterations += 1
        if self.render:
            print(f"Iteration x:")
            print("Bits in the buffer:", end=' ')
            print(f"{self.L}")

        
        BS, RBG, P = self.action_translator(action)


        self.P[BS, RBG] = self.Pmin + (self.Pmax - self.Pmin)/(DIV-1)*P
        
        if self.render: print(f"Action taken BS {BS}, RBG {RBG} to power: {self.P[BS,RBG]}")

        if self.render:
            self.results()

        
        reward = self.transmissionBits()
        ## New state ##
        self.TxData()

        if self.render: print("Bits remaining in the buffer:", self.L)

        new_state = []

        for n in self.N:
            for m in self.M:
                new_state += self.gamma_maj(n, m)

        for k in self.K:
            new_state.append(self.Bits(k))

        for k in self.K:
            new_state.append(self.L[k])

        for n in self.N:
            for m in self.M:
                new_state.append(self.P[n,m]) 

                
        new_state.append(self.iterations)

        #print(new_state)
       
        return new_state, reward, bool(sum(self.L.values()) == 0)
    
    def results(self, time : float = 0.0):        
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

    def action_space_sample(self):
        N = self.N[-1]+1
        M = self.M[-1]+1

        return random.randint(0, DIV*N*M-1)
    
    def n_action_space(self):
        N = self.N[-1]+1
        M = self.M[-1]+1

        return DIV*N*M

    def m_state_space(self):
        N = self.N[-1]+1
        M = self.M[-1]+1
        K = self.K[-1]+1
        return N*M + 2*K + 1 + N*M*(N-1)

    """ This is a test to show if the agent learns properly in a imple scenario """

    def reset2(self):
        self.__init__(self.data, self.render)
        for k in self.K:
            self.L[k] = random.randint(0,100)

        return list(self.L.values())

    def step2(self,action):
        print(self.L)
        if self.L[int(action)] == 0:
            reward = -100
        else:
            self.L[int(action)] = 0
            reward = sum(self.L.values())

        return list(self.L.values()), reward, bool(sum(self.L.values()) == 0)

    def n_action_space2(self):
        return len(self.L)

    def m_state_space2(self):
        return len(self.L)
    
    def action_space_sample2(self):
        return random.randint(0, len(self.L)-1)

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