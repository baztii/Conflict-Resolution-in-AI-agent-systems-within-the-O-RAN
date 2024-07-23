import json
import random

def main():
    usrBS = [random.randint(0,3) for _ in range(30)]
    beta = [[0 for __ in range(30)] for _ in range(4)]

    g = [[(3+random.random())*1e-8 for __ in range(30)] for _ in range(4)]

    for usr, bs in enumerate(usrBS):
        beta[bs][usr] = 1      

    print(g)


if __name__ == '__main__':
    main()