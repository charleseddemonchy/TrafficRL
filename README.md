# SUMO-TrafficRL

**SUMO-TrafficRL** is a production-ready reinforcement learning project for controlling traffic lights using SUMO (Simulation of Urban MObility) and a Deep Q-Network (DQN). This project demonstrates how to integrate SUMO with an RL agent to optimize traffic flow at an intersection by dynamically adjusting traffic light phases.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Hyperparameters and Configuration](#hyperparameters-and-configuration)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Overview

In this project, a DQN agent interacts with a SUMO simulation environment to control a traffic light at an intersection. The agent learns to minimize congestion by observing vehicle counts on monitored lanes and adjusting the traffic signal phases accordingly. The SUMO environment is wrapped into an OpenAI Gym-like interface, making it easy to experiment with various RL algorithms and configurations.

## Features

- **Integration with SUMO:** Uses TraCI to communicate with SUMO for real-time simulation control.
- **Gym-Compatible Environment:** The traffic light simulation is wrapped in a Gym environment for seamless RL integration.
- **Deep Q-Network (DQN):** A DQN agent with experience replay and target network updates to stabilize learning.
- **Configurable Parameters:** Easily adjust simulation settings, hyperparameters, and RL-specific parameters.
- **Logging and Checkpoints:** Detailed logging of training progress and periodic model checkpoint saving.

## Installation

### Prerequisites

- **SUMO:** Ensure that SUMO is installed and that the `SUMO_HOME` environment variable is set.  
  [SUMO Installation Instructions](https://sumo.dlr.de/docs/Installing/index.html)
- **Python 3.7+**
- **Pip Packages:** Install the required Python packages.

### Python Dependencies

Install the dependencies using pip:

```bash
pip install numpy gym torch sumolib traci
