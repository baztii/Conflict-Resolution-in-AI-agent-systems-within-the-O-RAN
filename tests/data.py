import json
import random
import os

def random_data() -> dict:
    N = random.randint(1,10)
    M = random.randint(10,50)
    K = random.randint(10,100)

    usrBS = [random.randint(0,N-1) for _ in range(K)]
    beta = [[0 for __ in range(K)] for _ in range(N)]

    for usr, bs in enumerate(usrBS):
        beta[bs][usr] = 1     

    g = [[(3+random.random())*1e-8 for __ in range(K)] for _ in range(N)]
    B = [[20e6+random.random()*1e7 for __ in range(M)] for _ in range(N)]

    T = 0.1

    Pmin = 0.0012589254117941673
    Pmax = 6.30957344480193

    sigma = 3.9810717055349695e-15 + random.random()*1e-15
    buffSize = 10_000

    lamda = random.randint(10,20)

    bits = 1200

    L = [random.randint(0,6000) for _ in range(K)]

    data = {
        'N': N,
        'M': M,
        'K': K,
        'beta': beta,
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