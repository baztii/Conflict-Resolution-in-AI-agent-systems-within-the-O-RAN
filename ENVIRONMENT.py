"""
# Environment
    This module provides the `ENVIRONMENT` class which represents the functionality of users connected to base stations.

## Classes
    - **ENVIRONMENT**: The environment itself

## Functions
    - *main*: The main function to show how should the environment work

## Author
    Miquel P. Baztan Grau

## Date
    21/08/2024

"""

import numpy as np
from pyomo.environ import log

""" Global variables """
ITER    = 500   # Number of iterations of the gameloop

class ENVIRONMENT:
    """
    The environment class representing the resource allocation and the power allocation of users connected to different base stations.

    Attributes:
        N (range): The range of base stations.
        K (range): The range of users in the system.
        M (range): The total number of RGB per BS.
        bits (int): The number of bits that each packet has.
        buffSize (int): The size of the transmission buffer where the pending traffic is queued.
        Pmin (float): The minimum transmission power.
        Pmax (float): The maximum transmission power.
        T (float): The length of time slot.
        sigma (float): The noise.
        lamda (float): The Poison process rate per second.
        g (dict): The channel coefficient between base stations and users (i.e. g[n,k] is between BS n and user k).
        B (dict): The brandwidth of all RGB (i.e. BW[n,m] determines the brandwidth of the RGB m at the BS n).
        L (dict): The amount of remained data of all users in the transmission buffer (i.e. L[k] is the remaining data of user k).
        P (dict): The transmission power allocation to RGB of BS (i.e. P[n,m] is the power of RBG m at BS n).
        alpha (dict): The distribution of RGB to each user (i.e. alpha[n,m,k] = 1 iff user k has RBG m at BS n, 0 otherwise).
        beta (dict): The user distribution in BS (i.e. beta[n,k] = 1 iff user[k] is on BS n, 0 otherwise).
    
    ## Methods
        - *gamma_maj*: Calculates the logarithmic normalized CSI
        - *eta*: Calculates the signal interference noise ratio (SINR)
        - *C*: Calculates the capacity of the system
        - *transmissionRate*: Calculates the transmission rate of the system
        - *lhs*: Calculates the amount of data remaining in the buffer
        - *rhs*: Calculates the amount of data that can be transmitted at maximum capacity
        - *Bits*: Calculates the amount of bits that the user will be able to send
        - *TransmissionBits*: Calculates the sum of bits that the users will be able to send (i.e., the sum of the Bits function)
        - *RBGs*: Calculates all the RBGs that are used by the users
        - *TxData*: Transmits the data of all users
        - *RqData*: Rquests the data of all users
        - *assign*: Applies the policy to the environment
        - *gameloop*: The gameloop of the environment
        - *results*: Prints the results of the simulation
    """

    def __init__(self, data) -> None:
        """
        Initializes the environment class representing the user network.
        
        Parameters:
            data (dict): A dictionary containing the parameters of the environment.
        
        Returns:
            None
        """

        """ Parameters """
        self.N        = range(data['N'])                                          # int // Number of base stations
        self.K        = range(data['K'])                                          # int // Number of users int the system
        self.M        = range(data['M'])                                          # int // Total number of RGB x BS
        self.bits     = data['bits']                                              # int // We assume the pending data is queued each 'bits'
        self.buffSize = data['buffSize']                                          # int // The size of the transmission buffer where the pending traffic is queued

        self.Pmin  = data['Pmin']                                                 # float // min transmition power
        self.Pmax  = data['Pmax']                                                 # float // max transmition power
        self.T     = data['T']                                                    # float // Length of time slot
        self.sigma = data['sigma']*1e10                                           # float // noise (it's scaled due to better performance)
        self.lamda = data['lamda']                                                # float // Poison process rate per second

        self.g     = {(n,k):data['g'][n][k]*1e10 for n in self.N for k in self.K} # list // Channel coefficient between base stations and users (i.e g[n,k] is between BS n and user k), it's scaled due to better performance
        self.B     = {(n,m):data['B'][n][m] for n in self.N for m in self.M}      # list // Brandwith of all RBG (i.e. BW[n,m] determines the brandwith of the RBG m at the BS n)
        self.L     = {k:data['L'][k] for k in self.K}                             # list // Amount of remained data of all users in the transimssion buffer (i.e. L[k] is the remaining data of user k)
        
        """ Variables """
        self.P     = {(n,m):self.Pmin for n in self.N for m in self.M}            # list // Transmission power allocation to RBG of BS (i.e. P[n,m] is the power of RBG m at BS n)
        self.alpha = {(n,m,k):0 for n in self.N for m in self.M for k in self.K}  # list // Distribution of RGB to each user (self.alpha[n,m,k] = 1 iff user k has RBG m at BS n, 0 otherwise)
        self.beta  = {(n,k):0 for n in self.N for k in self.K}                    # list // User distribution in BS (i.e self.beta[n,k] = 1 iff user[k] is on BS n, 0 otherwise)

    def gamma_maj(self, n : int, m : int) -> list[float]:
        """
        Calculates the logarithmic normalized CSI

        Parameters:
            n (int): The index of the base station
            m (int): The index of the RBG
            
        Returns:
            list[float]: The logarithmic normalized CSI
        
        """
        return [log(1+
                         sum([self.alpha[n_prime, m, k]*self.beta[n_prime,k]*self.g[n_prime,k] for k in self.K])/
                         sum([self.alpha[n,m,k]*self.beta[n,k]*self.g[n,k] for k in self.K]))/log(2) for n_prime in self.N if n_prime != n]

    def eta(self, n : int, m : int, k : int, model=None) -> float:
        """
        Calculates the signal interference noise ratio (SINR) between BS n and user k on RBG m

        Parameters:
            n (int): The index of the base station
            m (int): The index of the RBG
            k (int): The index of the user
            model (None/ENVIRONMENT/ConcreteModel): The environment model
        
        Returns:
            float: The signal interference noise ratio
        """

        if model is None: model = self

        numerator = model.alpha[n, m, k]*model.beta[n,k]*model.g[n,k]*model.P[n,m]
        denominator = sum([model.alpha[n_prime, m, k]*model.beta[n_prime,k]*model.g[n_prime,k]*model.P[n_prime,m]
            for n_prime in model.N if n_prime != n
            for k in model.K]) + model.sigma

        return numerator/denominator # If sigma is too small there are numerical issues

    def C(self, n : int, m : int, model=None) -> float:
        """
        Calculates the transmission capacity of BS n on RBG m

        Parameters:
            n (int): The index of the base station
            m (int): The index of the RBG
            model (None/ENVIRONMENT/ConcreteModel): The environment model
        
        Returns:
            float: The transmission capacity
        """

        if model is None: model = self
        return model.B[n,m]*log(1+sum([self.eta(n,m,k,model) for k in model.K]))/log(2)

    def transmissionRate(self, model=None) -> float:
        """
        Returns the total transmission rate of all the RBGs from all the base stations

        Parameters:
            model (None/ENVIRONMENT/ConcreteModel): The environment model
        
        Returns:
            float: The total transmission rate
        
        ## Observation
            Later on, we will select the minimum number between the bits that the user needs to send (i.e., L[k], the amount of data remaining in the buffer) and the bits that transmitting at maximum capacity would be sent.
        
        """
        if model is None: model = self
        return sum([self.C(n, m, model) for n in model.N for m in model.M])

    def lhs(self, k : int, model=None) -> int:
        """
        Calculates the amount of data remaining in the buffer

        Parameters:
            k (int): The index of the user
            model (None/ENVIRONMENT/ConcreteModel): The environment model
        
        Returns:
            int: The amount of data remaining in the buffer
        """

        if model is None: model = self
        return model.L[k]

    def rhs(self, k : int, model=None) -> float:
        """
        Calculates the amount of data that can be transmitted at maximum capacity

        Parameters:
            k (int): The index of the user
            model (None/ENVIRONMENT/ConcreteModel): The environment model
        
        Returns:
            float: The amount of data that can be transmitted at maximum capacity
        """

        if model is None: model = self
        return sum([model.alpha[n,m,k]*model.beta[n,k]*self.C(n, m, model)*model.T for n in model.N for m in model.M])

    def Bits(self, k : int, model=None) -> int:
        """
        Calculates the amount of bits that the user will be able to send

        Parameters:
            k (int): The index of the user
            model (None/ENVIRONMENT/ConcreteModel): The environment model
        
        Returns:
            int: The amount of bits that the user will be able to send
        """

        if model is None or model == self: return min(self.lhs(k), self.rhs(k))

        """
        I keep this because it is a fancy way of calculating the min(rhs, lhs).
        Unfortunately, if quite inefficient and produces aproximation errors.
            p = 20
            rhs = self.rhs(k, model) + 1
            lhs = self.lhs(k, model) + 1
            return rhs + lhs - (rhs**p+lhs**p)**(1/p) - 1
        """
        
        return model.min_bool_bits[k]*self.lhs(k, model) + (1-model.min_bool_bits[k])*self.rhs(k, model) 
    
    def transmissionBits(self, model=None) -> int:
        """
        Calculates the sum of bits that the users will be able to send (i.e., the sum of the Bits function)

        Parameters:
            model (None/ENVIRONMENT/ConcreteModel): The environment model
        
        Returns:
            int: The sum of bits that the users will be able to send
        """

        if model is None: model = self
        return sum([self.Bits(k, model) for k in model.K])

    def RBGs(self, model=None) -> int:
        """
        Calculates all the RBGs that are used by the users

        Parameters:
            model (None/ENVIRONMENT/ConcreteModel): The environment model
        
        Returns:
            int: The amount of RBGs that are used by the users
        """

        if model is None: model = self
        return sum([model.alpha[n,m,k] for n in model.N for m in model.M for k in model.K])

    def TxData(self) -> None:
        """
        Each user transmits what they can transmit (i.e., all the bits in the buffer or the bits that can be sent transmitting at maximum capacity)

        Returns:
            None
        """

        for k in self.K:
            self.L[k] = round(self.L[k] - self.Bits(k))

    def RqData(self) -> None:
        """
        The users request the data from the base stations (the request follows a poisson distribution)

        Returns:
            None
        """

        pkg = np.random.poisson(self.lamda*self.T, self.K[-1]+1) # The expected value of the random variable
        
        for k in self.K:
            self.L[k] = min(self.buffSize*self.bits, self.L[k]+self.bits*pkg[k])

    def assign(self, policy=lambda:None) -> None:
        """
        The policy is applied to the environment

        Parameters:
            policy (lambda/function): The policy to be applied to the environment
        
        Returns:
            None
        """

        policy()

    def gameloop(self, iter=ITER, policy=lambda:None) -> None:
        """
        The gameloop of the environment (each iteration is a time step). Here, the policy is applied to the environment and the users request and transmit the data

        Parameters:
            iter (int): The number of iterations
            policy (lambda/function): The policy to be applied to the environment
        
        Returns:
            None
        """

        for i in range(iter):
            print(f"Iteration {i}:")
            print("Bits in the buffer:", end=' ')
            print(f"{self.L}")
            self.assign(policy)
            self.TxData()
            self.RqData()
        
        print("Bits remaining in the buffer:", end=' ')
        print(f"{self.L}")

    def results(self, time : float = 0.0) -> None:
        """
        Prints the results of the simulation

        Parameters:
            time (float): The time that the model took to get the answer
        
        Returns:
            None
        """

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

def main():
    import json
    with open('tests/test3/data.json', 'r') as data_file: # load the data
        data = json.load(data_file)
    
    env = ENVIRONMENT(data)

    env.gameloop() # here goes your policy

if __name__ == '__main__':
    main()
