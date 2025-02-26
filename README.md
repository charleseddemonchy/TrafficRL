# Traffic Light Management with Reinforcement Learning

![Traffic Control System](https://img.shields.io/badge/AI-Traffic%20Control-brightgreen)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)

> A comprehensive reinforcement learning project to optimize traffic flow through intelligent traffic light control.

**Authors**
- Henri Chevreux
- Charles de Monchy
- Emiliano Pizaña Vela
- Alfonso Mateos Vicente

**Institution**: École Polytechnique - MSc&T in Artificial Intelligence and Advanced Visual Computing

## Table of Contents

- [Traffic Light Management with Reinforcement Learning](#traffic-light-management-with-reinforcement-learning)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
    - [Key Features](#key-features)
  - [Project Structure](#project-structure)
  - [Installation](#installation)
    - [Prerequisites](#prerequisites)
    - [Setup](#setup)
  - [Quick Start](#quick-start)
    - [Train a Model](#train-a-model)
    - [Visualize Trained Agent](#visualize-trained-agent)
    - [Record a Video](#record-a-video)
  - [Features](#features)
    - [Traffic Simulation](#traffic-simulation)
    - [Reinforcement Learning](#reinforcement-learning)
    - [Visualization and Analysis](#visualization-and-analysis)
  - [Technical Implementation](#technical-implementation)
    - [Environment](#environment)
      - [State Space](#state-space)
      - [Action Space](#action-space)
      - [Reward Function](#reward-function)
      - [Traffic Dynamics](#traffic-dynamics)
    - [Reinforcement Learning Algorithms](#reinforcement-learning-algorithms)
      - [Deep Q-Network (DQN)](#deep-q-network-dqn)
      - [Dueling DQN](#dueling-dqn)
      - [Double DQN](#double-dqn)
    - [Neural Network Architecture](#neural-network-architecture)
    - [Replay Buffer Mechanisms](#replay-buffer-mechanisms)
      - [Standard Experience Replay](#standard-experience-replay)
      - [Prioritized Experience Replay](#prioritized-experience-replay)
  - [Usage Guide](#usage-guide)
    - [Training a Model](#training-a-model)
    - [Testing a Model](#testing-a-model)
    - [Visualizing Traffic Flow](#visualizing-traffic-flow)
    - [Recording Videos](#recording-videos)
    - [Analyzing Agent Performance](#analyzing-agent-performance)
    - [Benchmarking](#benchmarking)
  - [Configuration](#configuration)
    - [Key Configuration Parameters](#key-configuration-parameters)
      - [Simulation Parameters](#simulation-parameters)
      - [Learning Parameters](#learning-parameters)
      - [Advanced Options](#advanced-options)
      - [Traffic Patterns](#traffic-patterns)
  - [Results and Performance](#results-and-performance)
    - [Training Performance](#training-performance)
    - [Metrics](#metrics)
    - [Policy Visualization](#policy-visualization)
  - [Extending the Project](#extending-the-project)
    - [Adding New Traffic Patterns](#adding-new-traffic-patterns)
    - [Custom Neural Network Architectures](#custom-neural-network-architectures)
    - [Alternative RL Algorithms](#alternative-rl-algorithms)
  - [Troubleshooting](#troubleshooting)
    - [Common Issues](#common-issues)
    - [Debugging](#debugging)
  - [References](#references)
    - [Papers](#papers)
    - [Books and Tutorials](#books-and-tutorials)
  - [License](#license)
  - [Citation](#citation)
  - [Acknowledgments](#acknowledgments)

## Overview

Traffic congestion is a major challenge in urban environments, leading to increased travel times, fuel consumption, and pollution. Traditional traffic light control systems use fixed timing schemes that cannot adapt to changing traffic conditions.

This project implements an intelligent traffic light control system using deep reinforcement learning that can optimize traffic flow in real-time. The system learns to make adaptive decisions based on current traffic conditions, resulting in more efficient traffic management.

### Key Features

- **Intelligent Adaptation**: The system adapts traffic light timings based on real-time traffic density
- **Grid-Based Simulation**: Models a network of interconnected intersections with realistic traffic flow
- **State-of-the-Art RL**: Implements advanced reinforcement learning techniques including Dueling DQN and Prioritized Experience Replay
- **Realistic Traffic Patterns**: Simulates different traffic scenarios including rush hour and weekend patterns
- **Comprehensive Visualization**: Tools for visualizing agent behavior and traffic flow
- **Performance Analysis**: Detailed metrics and analysis tools for evaluating system performance

## Project Structure

```
traffic_light_rl/
├── main.py                      # Main implementation file
├── run.py                       # Convenient run script wrapper
├── performance_dashboard.py     # Dashboard for visualizing results
├── config.json                  # Configuration file
├── requirements.txt             # Dependencies list
├── models/                      # Directory for saved models
│   └── best_model.pth           # Best performing model
├── results/                     # Results and metrics
│   ├── training_results.csv     # Training metrics data
│   └── training_progress.png    # Training visualization
├── videos/                      # Directory for recorded visualizations
├── analysis/                    # Analysis outputs
│   ├── agent_analysis.json      # Detailed agent analysis data
│   └── policy_visualization.png # Policy heatmap visualization
└── logs/                        # Log files
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/AlffonsoMV/TrafficRL.git
   cd TrafficRL
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

Alternatively, you can use the automatic dependency checker in the run script:
```bash
python run.py --mode train --install
```

## Quick Start

### Train a Model

```bash
python run.py --mode train --output results
```

### Visualize Trained Agent

```bash
python run.py --mode visualize --model results/models/best_model.pth
```

### Record a Video

```bash
python run.py --mode record --model results/models/best_model.pth --output videos
```

## Features

### Traffic Simulation

- **Grid-Based Network**: Configurable grid of intersections (default: 4x4)
- **Traffic Density Modeling**: Realistic traffic flow with congestion effects
- **Directional Flow**: North-South and East-West traffic streams at each intersection
- **Realistic Patterns**: Time-of-day traffic patterns including rush hour peaks
- **Traffic Flow Physics**: Inter-intersection flow with density gradient modeling

### Reinforcement Learning

- **Advanced Architectures**: 
  - Dueling DQN architecture that separates state value and action advantage
  - Double DQN to reduce overestimation bias
  
- **Experience Replay**:
  - Standard replay buffer with uniform sampling
  - Prioritized Experience Replay for more efficient learning
  
- **Learning Optimizations**:
  - Batch normalization for stable learning
  - Learning rate scheduling
  - Gradient clipping
  - Early stopping

### Visualization and Analysis

- **Real-time Visualization**: PyGame-based visualization of traffic state
- **Video Recording**: Generate videos of traffic simulation with matplotlib
- **Performance Dashboard**: Comprehensive metrics visualization
- **Policy Analysis**: Heatmap visualizations of agent decision boundaries

## Technical Implementation

### Environment

The traffic system is modeled as a grid of intersections, where each intersection has four incoming lanes (North, South, East, West). The environment is implemented as a custom Gym environment with:

#### State Space

For each intersection, the state includes:
- North-South traffic density (normalized 0-1)
- East-West traffic density (normalized 0-1)
- Current traffic light state (0: NS Green, 1: EW Green)

#### Action Space

For each intersection, the agent can set the traffic light to one of two states:
- 0: North-South Green (East-West Red)
- 1: East-West Green (North-South Red)

#### Reward Function

The reward function is designed to:
- Penalize waiting time for vehicles at red lights
- Reward throughput (number of vehicles passing through)
- Balance between minimizing congestion and maximizing flow

The reward is calculated as:
```
reward = -waiting_penalty + throughput_reward + switching_penalty
```

#### Traffic Dynamics

The simulation includes realistic traffic dynamics:
- **Congestion Effects**: Speed decreases with higher density
- **Density Propagation**: Traffic flows between adjacent intersections
- **Traffic Generation**: Time-dependent arrival patterns
- **Light Running**: Small probability for vehicles to pass through red lights

### Reinforcement Learning Algorithms

#### Deep Q-Network (DQN)

Basic DQN algorithm components:
- Neural network for Q-value approximation
- Experience replay for stable learning
- Target network for reducing correlation in target values
- Epsilon-greedy exploration policy

#### Dueling DQN

Improves upon standard DQN by:
- Separating state value and action advantage estimates
- Allowing better policy evaluation
- Reducing overestimation of action values

```python
# Dueling DQN architecture
Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
```

#### Double DQN

Reduces overestimation bias by:
- Using online network to select the action
- Using target network to evaluate the action

```python
# Standard DQN
Q_target = r + γ * max_a' Q_target(s', a')

# Double DQN
a' = argmax_a Q_online(s', a)
Q_target = r + γ * Q_target(s', a')
```

### Neural Network Architecture

The value function is approximated using a deep neural network:

```
Input Layer (state_size) 
    → FC Layer (256 units + Batch Norm + ReLU + Dropout) 
    → FC Layer (256 units + Batch Norm + ReLU + Dropout) 
    → FC Layer (128 units + Batch Norm + ReLU) 
    → Output Layer (action_size)
```

For Dueling DQN:
```
Input Layer (state_size) 
    → FC Layers (Shared Features) 
    → Value Stream (State Value) 
    → Advantage Stream (Action Advantages) 
    → Combined Output (Q-values)
```

### Replay Buffer Mechanisms

#### Standard Experience Replay

- Stores transitions (s, a, r, s', done) in a fixed-size buffer
- Randomly samples batches for training
- Breaks correlations between sequential experiences

#### Prioritized Experience Replay

- Assigns priorities to transitions based on TD-error
- Samples transitions with probability proportional to their priority
- Uses importance sampling weights to correct for bias
- Updates priorities after each learning step

## Usage Guide

### Training a Model

Train a new agent with default configuration:

```bash
python run.py --mode train
```

Training with custom parameters:

```bash
python run.py --mode train --config my_config.json --episodes 1000 --output custom_results
```

Key training parameters:
- `--episodes`: Number of training episodes
- `--output`: Directory to save results
- `--seed`: Random seed for reproducibility
- `--debug`: Enable debug logging for more verbose output

### Testing a Model

Test a trained model:

```bash
python run.py --mode test --model results/models/best_model.pth
```

Test with visualization:

```bash
python run.py --mode test --model results/models/best_model.pth --visualize
```

Test on different traffic patterns:

```bash
python run.py --mode test --model results/models/best_model.pth --traffic-pattern rush_hour
```

### Visualizing Traffic Flow

Visualize the traffic system in real-time:

```bash
python run.py --mode visualize --model results/models/best_model.pth
```

The visualization shows:
- Traffic density represented by color intensity
- Traffic light states (green/red)
- Vehicle flow and waiting times

### Recording Videos

Record a video of the traffic simulation:

```bash
python run.py --mode record --model results/models/best_model.pth --record-video videos/traffic_sim.mp4
```

Customize the video:

```bash
python run.py --mode record --model results/models/best_model.pth --traffic-pattern rush_hour --video-duration 60
```

### Analyzing Agent Performance

Run a comprehensive analysis of agent behavior:

```bash
python run.py --mode analyze --model results/models/best_model.pth --output analysis
```

This generates:
- Q-value analysis for different traffic states
- Decision boundary visualization
- Performance metrics across traffic patterns

View analysis results:

```bash
python performance_dashboard.py --results results/training_results.csv --analysis analysis/agent_analysis.json
```

### Benchmarking

Compare RL agent performance against baseline controllers:

```bash
python run.py --mode benchmark --model results/models/best_model.pth --output benchmark_results
```

## Configuration

The system is highly configurable through `config.json`:

```json
{
  "sim_time": 3600,
  "num_episodes": 500,
  "max_steps": 1000,
  "learning_rate": 0.0003,
  "gamma": 0.99,
  "epsilon_start": 1.0,
  "epsilon_end": 0.01,
  "epsilon_decay": 0.995,
  "buffer_size": 100000,
  "batch_size": 64,
  "target_update": 5,
  "eval_frequency": 20,
  "save_frequency": 25,
  "grid_size": 4,
  "max_cars": 30,
  "green_duration": 10,
  "yellow_duration": 3,
  "visualization": false,
  "device": "auto",
  "early_stopping_reward": 500,
  "checkpoint_dir": "checkpoints",
  "hidden_dim": 256,
  "weight_decay": 0.0001,
  "grad_clip": 1.0,
  "use_lr_scheduler": true,
  "lr_step_size": 100,
  "lr_decay": 0.5,
  "clip_rewards": true,
  "reward_scale": 0.1,
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
    },
    "weekend": {
      "midday_peak": 0.5,
      "peak_intensity": 1.5,
      "base_arrival": 0.02
    }
  },
  "advanced_options": {
    "prioritized_replay": true,
    "per_alpha": 0.6,
    "per_beta": 0.4,
    "dueling_network": true,
    "double_dqn": true
  }
}
```

### Key Configuration Parameters

#### Simulation Parameters
- `grid_size`: Size of the traffic grid (NxN intersections)
- `max_cars`: Maximum number of cars per lane
- `green_duration`: Duration of green light in time steps
- `yellow_duration`: Duration of yellow light in time steps

#### Learning Parameters
- `learning_rate`: Learning rate for the optimizer
- `gamma`: Discount factor for future rewards
- `epsilon_*`: Exploration parameters
- `buffer_size`: Size of the experience replay buffer
- `batch_size`: Batch size for training
- `target_update`: Frequency of target network updates

#### Advanced Options
- `prioritized_replay`: Whether to use prioritized experience replay
- `dueling_network`: Whether to use dueling DQN architecture
- `double_dqn`: Whether to use double DQN algorithm

#### Traffic Patterns
- Configure different traffic patterns with unique arrival rates and time-of-day effects

## Results and Performance

### Training Performance

The training process shows a distinctive pattern:
1. **Initial High Rewards**: The agent starts with high rewards during early exploration
2. **Quick Stabilization**: Rewards stabilize as the agent converges to a policy
3. **Evaluation Performance**: Evaluation rewards (without exploration) are significantly higher than training rewards

### Metrics

Key performance metrics:
- **Average Reward**: Measures overall policy effectiveness
- **Waiting Time**: Average time vehicles spend waiting at intersections
- **Throughput**: Number of vehicles passing through intersections
- **Traffic Density**: Overall congestion level in the system

### Policy Visualization

The agent's learned policy shows:
- Tendency to prioritize green lights for directions with higher traffic density
- Consideration of current light state when making decisions
- Coordination between adjacent intersections in grid configurations

## Extending the Project

### Adding New Traffic Patterns

Create new patterns in `config.json`:

```json
"traffic_patterns": {
  "custom_pattern": {
    "arrival_rate": 0.04,
    "variability": 0.02,
    "peak_times": [0.25, 0.75],
    "peak_intensities": [1.5, 2.0]
  }
}
```

### Custom Neural Network Architectures

Modify the `DQN` or `DuelingDQN` classes to implement custom architectures:

```python
class CustomDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(CustomDQN, self).__init__()
        # Define your custom architecture
        ...
```

### Alternative RL Algorithms

The framework can be extended to support other algorithms:

- **Policy Gradient Methods**: REINFORCE, PPO, A2C
- **Actor-Critic Methods**: DDPG, SAC, TD3
- **Multi-Agent RL**: Independent learning or centralized training with decentralized execution

## Troubleshooting

### Common Issues

**Problem**: CUDA out of memory error  
**Solution**: Reduce batch size in config.json or use CPU mode

**Problem**: Unstable learning/rewards collapse  
**Solution**: Try adjusting learning rate, use gradient clipping, enable PER

**Problem**: Visualization not working  
**Solution**: Ensure pygame is installed, try headless mode for recording

**Problem**: Training is too slow  
**Solution**: Reduce grid size, increase learning rate, use GPU if available

### Debugging

Enable debug mode for verbose logging:

```bash
python run.py --mode train --debug
```

View logs in the `logs/` directory for detailed information.

## References

### Papers

1. Mnih, V., et al. (2015). "Human-level control through deep reinforcement learning." Nature, 518(7540), 529-533.

2. Wang, Z., et al. (2016). "Dueling Network Architectures for Deep Reinforcement Learning." ICML 2016.

3. Schaul, T., et al. (2016). "Prioritized Experience Replay." ICLR 2016.

4. Van Hasselt, H., et al. (2016). "Deep Reinforcement Learning with Double Q-learning." AAAI 2016.

5. Wei, H., et al. (2018). "IntelliLight: A Reinforcement Learning Approach for Intelligent Traffic Light Control." KDD 2018.

### Books and Tutorials

- Sutton, R. S., & Barto, A. G. (2018). Reinforcement learning: An introduction. MIT press.

- Lapan, M. (2018). Deep Reinforcement Learning Hands-On. Packt Publishing.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Citation

If you use this code for your research, please cite:

```
@misc{TrafficLightRL,
  author = {[Your Name]},
  title = {Traffic Light Management with Reinforcement Learning},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/yourusername/traffic-light-rl}}
}
```

## Acknowledgments

- École Polytechnique for supporting this research
- Open source community for tools and libraries
- [Your mentors/professors] for guidance and feedback
