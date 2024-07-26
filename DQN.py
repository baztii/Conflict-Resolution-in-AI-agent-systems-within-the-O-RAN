import torch
from ENVIRONMENT import ENVIRONMENT as ENV

class DQN(ENV):
    def __init__(self,data):
        super().__init__(data)
    


def main():
    print(torch.__version__)

if __name__ == '__main__':
    main()