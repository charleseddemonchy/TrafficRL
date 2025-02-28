# Traffic Light Management with Reinforcement Learning

A comprehensive reinforcement learning project to optimize traffic flow through intelligent traffic light control.

## Overview

This project implements a Deep Q-Network (DQN) agent to control traffic lights in a simulated grid of intersections. The agent learns to optimize traffic flow and reduce congestion by intelligently timing the traffic lights based on current traffic conditions.

## Features

- Customizable grid-based traffic simulation environment
- Realistic traffic patterns (uniform, rush hour, weekend)
- Advanced Deep Q-Learning implementation with:
  - Experience replay (standard and prioritized)
  - Double DQN
  - Dueling network architecture
  - Gradient clipping
  - Learning rate scheduling
- Real-time visualization of traffic and agent performance
- Ability to generate videos of trained agents in action
- Comprehensive evaluation and benchmark tools

## Project Structure

```
traffic_rl/
├── __init__.py                # Package initialization
├── main.py                    # Entry point
├── config.py                  # Configuration handling
├── environment/
│   ├── __init__.py
│   ├── traffic_simulation.py  # Environment implementation
│   └── visualization.py       # Visualization functions
├── agents/
│   ├── __init__.py
│   ├── dqn_agent.py           # DQN agent implementation
│   ├── models.py              # Neural network models
│   └── memory.py              # Experience replay buffers
├── utils/
│   ├── __init__.py
│   └── logger.py              # Logging utilities
└── runs/                      # For saving results and models
    ├── models/
    └── results/
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/username/traffic-rl.git
cd traffic-rl
```

2. Install the package:
```bash
pip install -e .
```

## Usage

### Training a New Agent

```bash
# Basic training with default parameters
traffic-rl --mode train --output runs/experiment1

# Train with specific configuration and visualization
traffic-rl --mode train --config my_config.json --grid-size 5 --visualize
```

### Testing a Trained Agent

```bash
# Test the best model with visualization
traffic-rl --mode test --model runs/experiment1/models/best_model.pth --visualize

# Test on different traffic patterns
traffic-rl --mode test --model runs/experiment1/models/best_model.pth --traffic-pattern rush_hour
```

### Recording a Video

```bash
# Record a video of a trained agent
traffic-rl --mode record --model runs/experiment1/models/best_model.pth --video-path videos/demo.mp4
```

## Configuration

The system behavior can be customized through a configuration file (JSON format). Key parameters include:

- `grid_size`: Size of the traffic grid (default: 4x4)
- `num_episodes`: Number of training episodes
- `learning_rate`: Learning rate for the optimizer
- `gamma`: Discount factor for future rewards
- `epsilon_*`: Exploration parameters
- `traffic_patterns`: Configuration for different traffic patterns

Example configuration:

```json
{
  "grid_size": 4,
  "num_episodes": 500,
  "learning_rate": 0.0003,
  "gamma": 0.99,
  "epsilon_start": 1.0,
  "epsilon_end": 0.01,
  "epsilon_decay": 0.995,
  "traffic_patterns": {
    "uniform": {
      "arrival_rate": 0.03,
      "variability": 0.01
    },
    "rush_hour": {
      "morning_peak": 0.33,
      "evening_peak": 0.71,
      "peak_intensity": 2.0,
      "base_arrival": 0.03
    }
  }
}
```

## Results

After training, you can find the following results in the output directory:

- Training metrics in CSV format
- Learning curves visualization
- Saved models at different training stages
- Test performance metrics

## Authors

- Henri Chevreux
- Charles de Monchy
- Emiliano Pizaña Vela
- Alfonso Mateos Vicente

École Polytechnique - MSc&T in AI&ViC

## License

This project is licensed under the MIT License - see the LICENSE file for details.