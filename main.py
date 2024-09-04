"""
Main File
=========
    The main function to train and test the agent in the `CUSTOM_ENVIRONMENT` environment.

Functions
---------
 - *main*: The main function to train and test the agent in the `CUSTOM_ENVIRONMENT` environment.
 - *merge_agents*: Merges two agents, a resource allocation agent and a power allocation agent, to perform actions in their shared environments.

Usage
-----
    `$> python main.py <parameterSet> [--train]` or 
    `$> python main.py merge<n>`

Arguments
---------
 - parameterSet: The name of the parameter set to use within the `hyperparameters.yml` file.
 - --train (optional): Whether to train the agent or not.
 - n: The test number to merge agents.

Examples
--------
 `$> python main.py test1_power_allocation --train`\n
 `$> python main.py test1_power_allocation`\n
 `$> python main.py merge1`

Author
------
    Miquel P. Baztan Grau

Date
----
    21/08/2024
"""

from agent import Agent, DeepQNetwork
from gymnasium.envs.registration import register
import itertools
import torch as T
import argparse

def merge_agents(agentRA : Agent, agentPA : Agent):
        """
        Merges two agents (multiple agents), a resource allocation agent and a power allocation agent, 
        to perform actions in their respective environments.

        Parameters
        ----------
        agentRA : Agent
            The resource allocation agent.
        agentPA : Agent
            The power allocation agent.

        Returns
        -------
        None
        """

        agentRA.policy_dqn = DeepQNetwork(agentRA.num_states, agentRA.num_actions, agentRA.hidden_dims).to(agentRA.device)
        agentRA.policy_dqn.load_state_dict(T.load(agentRA.MODEL_FILE))
        agentRA.policy_dqn.eval()

        agentPA.policy_dqn = DeepQNetwork(agentPA.num_states, agentPA.num_actions, agentPA.hidden_dims).to(agentPA.device)
        agentPA.policy_dqn.load_state_dict(T.load(agentPA.MODEL_FILE))
        agentPA.policy_dqn.eval()
    
        for episode in itertools.count():
            state, info = agentRA.env.reset(options={"restart_mode":"merge"})
            state = T.tensor(state, dtype=T.float32, device=agentRA.device)

            state, info = agentPA.env.reset(options={"restart_mode":"merge"})
            state = T.tensor(state, dtype=T.float32, device=agentPA.device)

            agentRA.env.unwrapped.set_state(agentPA.env.unwrapped.alpha, agentPA.env.unwrapped.beta, agentPA.env.unwrapped.P, agentPA.env.unwrapped.L)

            terminated = truncated = False
            episode_reward = 0.0

            while not truncated:
                # Select best action for the resource allocation agent
                action = agentRA.select_action(state)

                # Perform the action for the resource allocation agent
                next_state, reward, terminated, truncated, info = agentRA.env.step(action.item())
                next_state = T.tensor(next_state, dtype=T.float32, device=agentRA.device)

                # Acumulative_reward
                episode_reward += reward

                agentPA.env.unwrapped.set_state(agentRA.env.unwrapped.alpha, agentRA.env.unwrapped.beta, agentRA.env.unwrapped.P, agentRA.env.unwrapped.L)

                state = next_state

                # Select best action for the power allocation agent
                action = agentPA.select_action(state)

                # Perform the action for the power allocation agent
                next_state, reward, terminated, truncated, info = agentPA.env.step(action.item())
                next_state = T.tensor(next_state, dtype=T.float32, device=agentPA.device)

                # Acumulative_reward
                episode_reward += reward

                agentRA.env.unwrapped.set_state(agentPA.env.unwrapped.alpha, agentPA.env.unwrapped.beta, agentPA.env.unwrapped.P, agentPA.env.unwrapped.L)

                state = next_state

def main():
    # Register environment
    register(id="ENVIRONMENT-v1", entry_point="gym_environment:CUSTOM_ENVIRONMENT")

    # Parse command line inputs
    parser = argparse.ArgumentParser(description='Train or test model.')
    parser.add_argument('hyperparameters', help='')
    parser.add_argument('--train', help='Training mode', action='store_true')
    args = parser.parse_args()

    if "merge" in args.hyperparameters:
        hyperparameters = "test" + args.hyperparameters[4] + "_power_allocation"
        agentPA = Agent(hyperparameter_set=hyperparameters, training=False)
        hyperparameters = "test" + args.hyperparameters[4] + "_resource_allocation"
        agentRA = Agent(hyperparameter_set=hyperparameters, training=False)
        merge_agents(agentRA, agentPA)
    else:
        agent = Agent(hyperparameter_set=args.hyperparameters, training=args.train)
        agent.run()

if __name__ == '__main__':
    main()
