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

## Usage

### ONL.py

This script processes data files to resolve the optimization scenarios. The key parameters to adjust in the `ONL` script are `START`, `END`, and `CONSOLE`. These parameters control the range of data files to process and the output format.

#### Parameters
- START: Specifies the starting data file index.
- END: Specifies the ending data file index.
- CONSOLE: Determines the output destination (console or `.txt` file).

### Configure Parameters:

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

> [!NOTE]
> Ensure that the test data files are located in the `tests` directory.
> Adjust `START` and `END` according to the number of data files and the scenarios you want to process.