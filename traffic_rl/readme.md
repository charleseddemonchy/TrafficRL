# Traffic RL: Reinforcement Learning for Traffic Light Control

This package provides a comprehensive framework for training and evaluating reinforcement learning agents for traffic light control optimization.

## Installation

```bash
pip install -e .
```

## Features

- **Multiple Agent Types**: DQN, Dueling DQN, Fixed-Timing, Adaptive-Timing, and Random agents
- **Realistic Traffic Simulation**: Simulates various traffic patterns and densities
- **Comprehensive Evaluation**: Tools for benchmarking and analyzing agent performance
- **Visualization**: Extensive visualization capabilities for training metrics and traffic simulations

## CLI Usage

The package provides a unified command-line interface for all operations:

### Training

```bash
# Basic training
traffic-rl train --output results/training

# Training with custom configuration
traffic-rl train --config custom_config.json --episodes 500 --output results/custom_training
```

### Evaluation

```bash
# Evaluate a model on multiple traffic patterns
traffic-rl evaluate --model results/training/best_model.pth --patterns uniform,rush_hour,weekend --episodes 20 --output results/evaluation
```

### Visualization

```bash
# Visualize environment with trained model
traffic-rl visualize --type environment --model results/training/best_model.pth --pattern rush_hour --duration 60 --output results/visualizations

# Visualize training metrics
traffic-rl visualize --type metrics --metrics results/training/training_metrics.json --output results/visualizations

# Visualize traffic patterns
traffic-rl visualize --type patterns --output results/visualizations
```

### Benchmarking

```bash
# Benchmark multiple agents
traffic-rl benchmark --agents random,fixed,adaptive,dqn --model results/training/best_model.pth --patterns uniform,rush_hour,weekend --episodes 15 --output results/benchmark
```

### Analysis

```bash
# Comprehensive analysis
traffic-rl analyze --model results/training/best_model.pth --metrics results/training/training_metrics.json --benchmark-dir results/benchmark --output results/analysis --episodes 10
```

## Configuration

You can customize the behavior of the agents and environment through a configuration file. Here's an example:

```json
{
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
    "grid_size": 4,
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
        "dueling_network": true,
        "double_dqn": true
    }
}
```

## Model Selection

During training, the system periodically evaluates the agent and saves the best-performing model as `best_model.pth`. This model is selected based on evaluation performance, not necessarily the final model from training.

When you evaluate or visualize a model, you can specify:

```bash
# Use the best model saved during training
traffic-rl evaluate --model results/training/best_model.pth

# Use a specific checkpoint from training
traffic-rl evaluate --model results/training/model_episode_450.pth

# Use the final model from training
traffic-rl evaluate --model results/training/final_model.pth
```

## Advanced Features

The DQN implementation includes several advanced features that can be enabled in the configuration:

- **Prioritized Experience Replay**: Samples more important transitions more frequently
- **Dueling Network Architecture**: Separates state value and action advantage estimation
- **Double DQN**: Reduces overestimation of Q-values
- **Early Stopping**: Stops training when performance plateaus

## Dependencies

- PyTorch
- NumPy
- Matplotlib
- Gymnasium
- Pygame
- Pandas
- Seaborn
