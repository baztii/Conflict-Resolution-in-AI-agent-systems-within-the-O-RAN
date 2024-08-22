"""
Main File
=========
    The main function to train and test the agent in the `CUSTOM_ENVIRONMENT` environment.

Usage
-----
    `$> python main.py <parameterSet> [--train]`

Arguments
---------
 - parameterSet: The name of the parameter set to use within the `hyperparameters.yml` file.
 - --train (optional): Whether to train the agent or not.

Examples
--------
 `$> python main.py test1_power_allocation --train`\n
 `$> python main.py test1_power_allocation`

Author
------
    Miquel P. Baztan Grau

Date
----
    21/08/2024
"""

from agent import Agent
from gymnasium.envs.registration import register
import argparse

def main():
    # Register environment
    register(id="ENVIRONMENT-v1", entry_point="gym_environment:CUSTOM_ENVIRONMENT")

    # Parse command line inputs
    parser = argparse.ArgumentParser(description='Train or test model.')
    parser.add_argument('hyperparameters', help='')
    parser.add_argument('--train', help='Training mode', action='store_true')
    args = parser.parse_args()

    agent = Agent(hyperparameter_set=args.hyperparameters, training=args.train)
    agent.run()

if __name__ == '__main__':
    main()
