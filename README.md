# Traffic RL: Reinforcement Learning for Traffic Light Control

This package provides a comprehensive framework for training and evaluating reinforcement learning agents for traffic light control optimization.

## Features

- **Multiple Agent Types**: DQN, Dueling DQN, Fixed-Timing, Adaptive-Timing, and Random agents
- **Realistic Traffic Simulation**: Simulates various traffic patterns and densities
- **Comprehensive Evaluation**: Tools for benchmarking and analyzing agent performance
- **Visualization**: Extensive visualization capabilities for training metrics and traffic simulations

## Installation

### From Source

To install the package from source:

```bash
# Clone the repository
git clone https://github.com/yourusername/traffic-rl.git
cd traffic-rl

# Install the package in development mode
pip install -e .
```

This will install the `traffic_rl` command-line tool and all required dependencies.

### Requirements

- Python 3.7+
- PyTorch
- NumPy
- Matplotlib
- Gymnasium
- Pygame
- Pandas
- Seaborn

## CLI Usage

After installation, you can use the `traffic_rl` command-line tool:

### Training

```bash
# Basic training
traffic_rl train --output results/training
```

### Evaluation

```bash
# Evaluate a model on multiple traffic patterns
traffic_rl test --model results/training/best_model.pth --episodes 20 --output results/evaluation
```

### Visualization

```bash
# Visualize environment with trained model
traffic_rl visualize --model results/training/best_model.pth --traffic-pattern rush_hour --output results/visualizations

# Record a video of the environment
traffic_rl record --model results/training/best_model.pth --traffic-pattern rush_hour --record-video videos/traffic_sim.mp4 --video-duration 30
```

### Benchmarking

```bash
# Benchmark multiple agents
traffic_rl benchmark --model results/training/best_model.pth --episodes 15 --output results/benchmark
```

### Analysis

```bash
# Comprehensive analysis
traffic_rl analyze --model results/training/best_model.pth --output results/analysis --episodes 10
```

## Model Selection

During training, the system periodically evaluates the agent and saves the best-performing model as `best_model.pth`. This model is selected based on evaluation performance, not necessarily the final model from training.

When you evaluate or visualize a model, you can specify:

```bash
# Use the best model saved during training
traffic_rl test --model results/training/best_model.pth

# Use a specific checkpoint from training
traffic_rl test --model results/training/model_episode_450.pth

# Use the final model from training
traffic_rl test --model results/training/final_model.pth
```

## Advanced Features

The DQN implementation includes several advanced features that can be enabled in the configuration:

- **Prioritized Experience Replay**: Samples more important transitions more frequently
- **Dueling Network Architecture**: Separates state value and action advantage estimation
- **Double DQN**: Reduces overestimation of Q-values
- **Early Stopping**: Stops training when performance plateaus

## Configuration

You can provide a custom configuration file using the `--config` option:

```bash
traffic_rl train --config my_config.json --output results/custom_training
```

Or override specific configuration parameters via command-line arguments:

```bash
traffic_rl train --episodes 1000 --seed 42 --output results/long_training
```
