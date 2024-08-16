import matplotlib, matplotlib.pyplot as plt
from datetime import datetime
import numpy as np

import json

def save_graph(graph_file, rewards_per_episode, epsilon_history):
    # 'Agg': used to generate plots as images and save them to a file instead of rendering to screen
    matplotlib.use('Agg')
    # Save plots
    fig = plt.figure(1)

    # Plot average rewards (Y-axis) vs episodes (X-axis)
    mean_rewards = np.zeros(len(rewards_per_episode))
    for x in range(len(mean_rewards)):
        mean_rewards[x] = np.mean(rewards_per_episode[max(0, x-99):(x+1)])
    plt.subplot(121) # plot on a 1 row x 2 col grid, at cell 1
    # plt.xlabel('Episodes')
    plt.ylabel('Mean Rewards')
    plt.plot(mean_rewards)

    # Plot epsilon decay (Y-axis) vs episodes (X-axis)
    plt.subplot(122) # plot on a 1 row x 2 col grid, at cell 2
    # plt.xlabel('Time Steps')
    plt.ylabel('Epsilon Decay')
    plt.plot(epsilon_history)

    plt.subplots_adjust(wspace=1.0, hspace=1.0)

    # Save plots
    fig.savefig(graph_file)
    plt.close(fig)

def asserts(data : dict) -> None:
    assert len(data["B"])    == data["N"]
    assert len(data["B"][0]) == data["M"]
    
    assert len(data["g"])    == data["N"]
    assert len(data["g"][0]) == data["K"]

    assert len(data["L"])    == data["K"]

def load_data(n : int) -> dict:
    file = f"tests/test{n}/data.json"
    with open(file, 'r') as data_file: # load the data
        data = json.load(data_file)
    
    asserts(data)

    return data

def log_message(message : str, DATE_FORMAT : str, file : str, mode : str) -> None:
    time = datetime.now().strftime(DATE_FORMAT)
    log_message = f"{time}: {message}"
    print(log_message)

    with open(file, mode) as file:
        file.write(log_message + '\n')

