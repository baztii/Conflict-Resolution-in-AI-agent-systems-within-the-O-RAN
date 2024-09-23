"""
# Utils
    This module provides various utility functions for logging, graphing, and other miscellaneous tasks.

## Functions
    - *save_graph*:  Saves a graph of the agent's performance to a file.
    - *asserts*:     Checks if a data set is valid.
    - *load_data*:   Loads a data set from the "tests" folder.
    - *log_message*: Logs a message to a file and prints it to the console.
    - *read_file_to_string*: Reads the content of a text file and returns it as a string.
    - *plot_two_dicts*: Plots two dictionaries on the same graph and optionally saves the figure to a file.

## Author
    Miquel P. Baztan Grau

## Date
    21/08/2024
"""

import matplotlib, matplotlib.pyplot as plt
from datetime import datetime
import numpy as np

import json

def save_graph(graph_file : str, rewards_per_episode : list, epsilon_history : list) -> None:
    """
    Saves a graph of the agent's performance to a file.   
    
    Parameters:
        graph_file (str): The path to the file where the graph will be saved.
        rewards_per_episode (list): A list of rewards received by the agent in each episode.
        epsilon_history (list): A list of epsilon values used by the agent in each episode. 
    
    Returns:
        None
    """

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
    """
    Checks if the data is correct.
    
    Parameters:
        data (dict): The data to be checked.
    
    Returns:
        None
    """

    assert len(data["B"])    == data["N"]
    assert len(data["B"][0]) == data["M"]
    
    assert len(data["g"])    == data["N"]
    assert len(data["g"][0]) == data["K"]

    assert len(data["L"])    == data["K"]
    
def load_data(n : int) -> dict:
    """
    Loads a data set from the "tests" folder.
    
    Parameters:
        n (int): The number of the data set to load.
    
    Returns:
        dict: The loaded data set.
    """

    assert type(n) == int, "You must specify a data set number!"

    file = f"tests/test{n}/data.json"
    with open(file, 'r') as data_file: # load the data
        data = json.load(data_file)
    
    asserts(data) # Check if the data is correct

    return data

def log_message(message : str, DATE_FORMAT : str, file : str, mode : str) -> None:
    """
    Logs a message to a file and prints it to the console.

    Parameters:
        message (str): The message to be logged.
        DATE_FORMAT (str): The format of the date to be included in the log message.
        file (str): The file where the message will be logged.
        mode (str): The mode in which the file will be opened ('w' for write, 'a' for append).

    Returns:
        None
    """

    # Get the current date and time
    time = datetime.now().strftime(DATE_FORMAT)

    # Construct the log message
    log_message = f"{time}: {message}"

    # Print the log message to the console
    print(log_message)

    # Open the file in the specified mode and write the log message to it
    with open(file, mode) as file:
        file.write(log_message + '\n')

def read_file_to_string(file_path : str) -> str:
    """
    Reads the content of a text file and returns it as a string.
    
    Args:
        file_path (str): The path to the text file.
    
    Returns:
        str: The content of the file as a string.
    """
    
    try:
        with open(file_path, 'r') as file:
            content = file.read()
        return content
    except FileNotFoundError:
        raise FileNotFoundError("File not found.")
    except Exception as e:
        raise Exception(f"An error occurred: {str(e)}")

def plot_two_dicts(dict1 : dict, dict2 : dict, title : str = "Two Dictionaries Plot", label1 : str = "Dict1", label2 : str = "Dict2", xlabel : str = 'Keys', ylabel : str = 'Values', save_path : str = None) -> None:
    """ 
    Plots two dictionaries on the same graph and optionally saves the figure to a file.
    
    Args:
        dict1 (dict): The first dictionary (x: y values).
        dict2 (dict): The second dictionary (x: y values).
        title (str): The title of the graph.
        label1 (str): The label for the first dictionary.
        label2 (str): The label for the second dictionary.
        xlabel (str): The xlabel for the graph.
        ylabel (str): The ylabel for the graph.
        save_path (str): The file path to save the plot. If None, no file is saved.
    
    Returns:
        None
    """

    # Unpack the first dictionary (x and y values)
    x1, y1 = zip(*dict1.items())
    
    # Unpack the second dictionary (x and y values)
    x2, y2 = zip(*dict2.items())
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Plot the first dictionary
    plt.plot(x1, y1, label=label1, marker='o')
    
    # Plot the second dictionary
    plt.plot(x2, y2, label=label2, marker='s')
    
    # Add title and labels
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    # Add a legend
    plt.legend()
    
    # Show grid
    plt.grid(True)
    
    # Save the plot to a file if a path is provided
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        print("No save path provided, plot not saved.")
    
    # Close the plot to prevent memory leaks in non-interactive environments
    plt.close()
