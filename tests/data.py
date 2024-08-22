"""
# Data
    This module provides the `random_data` function that generates random data for the environment.

## Functions
    - *random_data*: Generates a random data dictionary with the specified dimensions N, M, and K.
    - *upload_data*: Uploads a data dictionary to the specified directory.
    - *main*: The main function to show how should the online optimization model work

## Author
    Miquel P. Baztan Grau

## Date
    21/08/2024

"""

import json
import random
import os

N = 1
M = 1
K = 1

def random_data(N : int = N, M : int = M, K : int = K) -> dict:
    """
    Generates a random data dictionary with the specified dimensions N, M, and K. 
    If any of these dimensions are not provided, they are set to random values between 1 and 100.
    
    Args:
        N (int, optional): The number of base stations. Defaults to N.
        M (int, optional): The total number of RBGs per base station. Defaults to M.
        K (int, optional): The number of users in the system. Defaults to K.
    
    Returns:
        dict: A dictionary containing the generated data with the following keys:
            - 'N' (int): The number of base stations.
            - 'M' (int): The total number of RBGs per base station.
            - 'K' (int): The number of users in the system.
            - 'B' (list of lists): A 2D list representing the bandwidth of all RBGs.
            - 'T' (float): The length of the time slot.
            - 'Pmin' (float): The minimum transmission power.
            - 'Pmax' (float): The maximum transmission power.
            - 'sigma' (float): The noise.
            - 'buffSize' (int): The size of the transmission buffer.
            - 'g' (list of lists): A 2D list representing the channel coefficient between base stations and users.
            - 'lamda' (float): The Poison process rate per second.
            - 'bits' (int): The number of bits that each packet has.
            - 'L' (list): A list representing the amount of remained data of all users in the transmission buffer.
    """

    N = random.randint(1,10) if N == -1 else N
    M = random.randint(10,50) if M == -1 else M
    K = random.randint(10,100) if K == -1 else K

    g = [[(3+random.random())*1e-8 for __ in range(K)] for _ in range(N)]
    B = [[30e3+random.random()*1e2 for __ in range(M)] for _ in range(N)]

    T = 0.5

    Pmin = 0.0012589254117941673
    Pmax = 6.30957344480193

    sigma = 3.9810717055349695e-15 + random.random()*1e-15
    buffSize = 10_000

    lamda = 200

    bits = 3500

    L = [0 for _ in range(K)]

    data = {
        'N': N,
        'M': M,
        'K': K,
        'B': B,
        'T': T,
        'Pmin': Pmin,
        'Pmax': Pmax,
        'sigma': sigma,
        'buffSize': buffSize,
        'g': g,
        'lamda': lamda,
        'bits': bits,
        'L': L
    }

    return data

def upload_data(data : dict, where=None) -> None:
    """
    Uploads data to a JSON file in a newly created test directory if where is None, otherwise uploads it to the specified directory.

    Args:
        data (dict): The data to be uploaded.
        where (int, optional): The directory number to upload the data to. Defaults to None.

    Returns:
        None
    """

    if where is None:
        with os.scandir('.') as dir:
            n = sum(1 for file in dir if file.is_dir() and file.name.startswith('test')) + 1

        os.makedirs(f"./test{n}", exist_ok=True)
    else:
        n = where

    with open(f"./test{n}/data.json", 'w') as file:
        json.dump(data, file)
    
    print(f"Test{n} folder and data created succesfully!")

def main():
    upload_data(random_data(N=2,K=3,M=5))
    #upload_data(random_data(N=3,K=7,M=5),4)


if __name__ == '__main__':
    main()