from agent import Agent
from gymnasium.envs.registration import register
import argparse

# Register environment
register(id="ENVIRONMENT-v1", entry_point="gym_environment:CUSTOM_ENVIRONMENT")

# Parse command line inputs
parser = argparse.ArgumentParser(description='Train or test model.')
parser.add_argument('hyperparameters', help='')
parser.add_argument('--train', help='Training mode', action='store_true')
args = parser.parse_args()

agent = Agent(hyperparameter_set=args.hyperparameters, training=args.train)
agent.run()
