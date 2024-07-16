

# Reinforcement Learning Agent in Grid World

This repository contains an implementation of a reinforcement learning agent that navigates a grid world environment to reach a goal while avoiding obstacles. The agent is trained using the Q-learning algorithm and visualized using Tkinter.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Code Structure](#code-structure)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Overview

The project consists of a grid world environment where the agent (a bee) has to navigate from the start position to the goal position (a flower) while avoiding obstacles (cats). The agent is trained using the Q-learning algorithm.

Key components of the project:
- Grid world environment
- Q-learning agent
- Visualization using Tkinter

## Installation

To run the code, you need to have Python installed. Additionally, you need to install the required libraries.

```bash
pip install numpy pillow
```

## Usage

1. Clone the repository:
    ```bash
    git clone https://github.com/your_username/rl-grid-world.git
    cd rl-grid-world
    ```

2. Place the images (`bee.png`, `flower.png`, `cat.png`) in the same directory as the script.

3. Run the script:
    ```bash
    python rl_grid_world.py
    ```

## Code Structure

- `GridWorld` class: Represents the grid world environment with methods for resetting the environment, taking steps, and rendering the grid.
- `QLearningAgent` class: Implements the Q-learning algorithm with methods for choosing actions, updating the Q-table, and training the agent.
- `RLVisualizer` class: Uses Tkinter to visualize the agent's progress in the grid world.

### Key Methods and Functions:

- `GridWorld._generate_cat_positions()`: Generates random positions for the obstacles (cats).
- `GridWorld.step(action)`: Takes a step in the environment based on the action and returns the next state, reward, and whether the episode is done.
- `QLearningAgent.choose_action(state)`: Chooses an action based on the epsilon-greedy policy.
- `QLearningAgent.update_q_table(state, action, reward, next_state)`: Updates the Q-table based on the action taken and the reward received.
- `RLVisualizer.render_grid()`: Renders the current state of the grid world using Tkinter.

## Results

The agent is trained for 10,000 episodes and the visualization shows the agent's progress in navigating the grid world. The visualization includes:

- The bee (agent)
- The flower (goal)
- The cats (obstacles)
- The current reward and cumulative reward displayed at the top of the window

### Example Output:

- **Initial Grid:**
  ```
  B . . . .
  . . C . .
  . . . . .
  . . C . .
  . . . . F
  ```

- **Final Grid after training:**
  ```
  . . . . .
  . . C . .
  . . . . .
  . . C . .
  . . . B F
  ```

## Contributing

Contributions are welcome! Please fork the repository and create a pull request with your changes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

