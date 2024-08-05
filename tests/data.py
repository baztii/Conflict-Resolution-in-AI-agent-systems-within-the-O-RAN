import json
import random
import os

N = 1
M = 12
K = 10

def random_data(N : int = N, M : int = M, K : int = K) -> dict:
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

    lamda = 1000

    bits = 3824

    L = [random.randint(0,382400) for _ in range(K)]

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

def upload_data(data : dict) -> None:
    with os.scandir('.') as dir:
        n = sum(1 for file in dir if file.is_dir() and file.name.startswith('test')) + 1

    os.makedirs(f"./test{n}", exist_ok=True)

    with open(f"./test{n}/data.json", 'w') as file:
        json.dump(data, file)
    
    print(f"Test{n} folder and data created succesfully!")

def main():
    upload_data(random_data())

if __name__ == '__main__':
    main()