# Conflict-Resolution-in-AI-agent-systems-within-the-O-RAN

## Introduction
In this project, we have developed a simulation environment (`ENVIRONMENT.py`) designed to optimize resource management within a network consisting of base stations, users, and Resource Block Groups (RBGs). The primary aim is to enhance the performance and efficiency of the system by strategically allocating power to RBGs and assigning users to base stations, ensuring that users receive the necessary RBGs to meet their needs.

Our environment encompasses two distinct scenarios within a gym-like simulation framework (`gym_environment.py`):

1. Power Allocation Scenario: In this scenario, the focus is on allocating power specifically to RBGs. The objective is to optimize the distribution of power to maximize the overall efficiency and performance of the RBGs.

2. Resource Allocation Scenario: This scenario involves two key tasks—first, positioning users at base stations, and second, distributing RBGs to these users. The goal here is to ensure that users are effectively served by base stations and provided with the appropriate RBGs to fulfill their requirements.

Additionally, we incorporate a Nonlinear Optimization (`ONL.py`) file that addresses the problem using advanced nonlinear optimization techniques. This file serves as a tool for solving complex optimization problems related to power and resource allocation.

To further enhance our approach, we have developed agent (`agent.py`) models that learn the optimal policies using Deep Q-Networks (DQN). These agents are trained to maximize the amount of data that users can transmit, employing the gym environment to refine their strategies for both power allocation and resource allocation scenarios.

Finally, we conduct comprehensive testing to evaluate the effectiveness of both the ONL-based solutions and the DQN-based agents, comparing their performance in solving the resource allocation and power allocation problems.

This project integrates various methodologies to optimize network performance, combining theoretical optimization techniques with practical reinforcement learning algorithms to achieve superior results.

## Installation

### 1. Set Up a Virtual Environment (Recommended)

It's a good practice to use a virtual environment to manage dependencies and avoid conflicts with other projects. Follow these steps to create and activate a virtual environment:

- **On macOS/Linux:**

  1. Create a virtual environment:
  
     ```bash
     $> python3 -m venv venv
     ```

  2. Activate the virtual environment:
  
     ```bash
     $> source venv/bin/activate
     ```

- **On Windows:**

  1. Create a virtual environment:
  
     ```bash
     $> python -m venv venv
     ```

  2. Activate the virtual environment:
  
     ```bash
     $> venv\Scripts\activate
     ```

Once activated, your terminal prompt should change to indicate that you are now working within the virtual environment. You can now proceed with installing the project dependencies.

### 2. Install Dependencies from `requirements.txt`

After setting up and activating your virtual environment, you can install the dependencies listed in `requirements.txt`. Run the following command:

```bash
$> pip install -r requirements.txt
```

### 3. Install the Projact Package
With the dependencies installed, you can now install the project package itself. Use the following command:
```bash
$> pip install -e .
````
## Project Usage Guide

This section explains how to use the different files within the project to execute the environment, agent, and experiments. 

It is divided into the following sections:

1. [ENVIRONMENT](#ENVRIONMENTpy)
2. [ONL](#ONLpy)
3. [gym_environment](#gym_environmentpy)
4. [utils](#utilspy)
5. [agent](#agentpy)
6. [main](#mainpy)

---

### ENVIRONMENT.py

The `ENVIRONMENT.py` file is responsible for creating the environment in which the game loop operates.

#### Parameters
- **ITER**: This sets the number of iterations for the game loop.

#### Configure Parameters
- Open the `ENVIRONMENT.py` file in a text editor.
- Set the `ITER` parameter to the number of iterations you want the game loop to run for.

#### Example
```
ITER = 10
```

Once you have configured the parameter you need to load the data to be used in the environment (you have and example in the ENVIRONMENT.py file).

Then you need to create an environment object using the following command:
   ```python
   env = ENVIRONMENT(data)
   ```

Finally execute the game loop by calling the `gameloop()` method with your policy function. The policy should determine how resources are allocated and the power is set during each iteration:
   ```python
   env.gameloop(policy=your_policy)
   ```

### ONL.py

This script processes data files to resolve the optimization scenarios. The key parameters to adjust in the `ONL` script are `START`, `END`, and `CONSOLE`. These parameters control the range of data files to process and the output format.

#### Parameters
- **START**: Specifies the starting data file index.
- **END**: Specifies the ending data file index.
- **CONSOLE**: Determines the output destination (console or `.txt` file).

#### Configure Parameters:

- Open the ONL script in a text editor.
- Set the `START` parameter to the index of the first test file you want to process (e.g., `START = 1` for `test1`).
- Set the `END` parameter to the index of the last test file to process (e.g., `END = 3` for `test1`, `test2`, and `test3`).
- Set the `CONSOLE` parameter to `True` if you want the output to be printed to the console, or `False` if you prefer to save it in a `.txt` file.

#### Example
```
START = 1
END = 3
CONSOLE = False
```

When you have configured the parameters, you can run the script using the following command:
```
$> python ONL.py
```


> [!NOTE]
> Ensure that the test data files are located in the `tests` directory.
> Adjust `START` and `END` according to the number of data files and the scenarios you want to process.





### gym_environment.py

The `gym_environment.py` file does not execute on its own but is responsible for creating a custom Gym environment.

#### Parameters

- **DIV**: Defines the number of divisions for power allocation.

This file is imported into other scripts, such as `agent.py`, where the Gym environment is utilized for training or testing.

#### Configure Parameters

- Open the `gym_environment.py` file in a text editor.
- Set the `DIV` parameter to the number of divisions you want for power allocation.

### utils.py

The `utils.py` file contains utility functions that are used across different parts of the project. It does not execute independently, but you can import and use the functions within this file wherever needed.

### agent.py 

The `agent.py` file is where you can control and execute the agent's behavior.

#### Parameters

- **DATE_FORMAT**: The format in which logs are saved (e.g., `"YYYY-MM-DD HH:mm:ss"`).
- **RUNS_DIR**: The directory where all logs, graphs, and other outputs will be saved.
- **SAVE_GRAPH_STEP**: The interval for updating and saving the graph. If set to 1000, the graph will be saved every 1000 iterations.

#### Configure Parameters

- Open the `agent.py` file in a text editor.
- Set the parameters as needed.

#### Example

```
DATE_FORMAT = "YYYY-MM-DD HH:mm:ss"
RUNS_DIR = "runs"
SAVE_GRAPH_STEP = 1000
```

> [!NOTE]
> This file imports and utilizes the `gym_environment.py` file. You can choose your custom Gym environment within your hyperparameter set.

To run the agent, use the following command:
   ```bash
   python agent.py <hyperparameterSet> [--train]
   ```
   - `<hyperparameterSet>`: The name of the hyperparameter set you want to use. This should include the Gym environment you wish to employ.
   - `--train` (optional): Add this flag if you want the agent to train.

For instance,

```bash
python agent.py test1_power_allocation --train
```

### main.py

The `main.py` file serves as the entry point for running the entire system. There are multiple ways to use it, depending on your needs.

#### Usage

```bash
$ python main.py <parameterSet> [--train]
$ python main.py merge<n>
```

#### Arguments

- **parameterSet**: The name of the parameter set to use, defined within the `hyperparameters.yml` file.
- **--train** (optional): Add this flag if you want the agent to train.
- **n**: The test number for merging agents.

#### Examples

1. **Running with Training**:
   ```bash
   python main.py test1_power_allocation --train
   ```
2. **Running without Training**:
   ```bash
   python main.py test1_power_allocation
   ```
3. **Merging Agents**:
   ```bash
   python main.py merge1
   ```

## Future Work

1. **Re-run the Agent Test**: Re-run the `test3_power_allocation` because the results were not saved properly during the initial run.
2. **Test Agent Merging**: Test the merging of the two agents (power allocation and resource allocation) to assess how well they work together.
3. **Performance Comparison**: Compare the results of the `ONL.py` file with the merged agents to determine which performs better.
4. **Document Results**: Create a detailed document showing the results of the above comparisons and tests.

--- 

This guide should help you understand how to use the files in this project and plan for future improvements.