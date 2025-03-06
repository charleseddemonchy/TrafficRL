# Traffic RL: Reinforcement Learning for Traffic Light Control

This package provides a comprehensive framework for training and evaluating reinforcement learning agents for traffic light control optimization.
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
python -m train --output results/training

```

### Evaluation

```bash
# Evaluate a model on multiple traffic patterns
python -m evaluate --model models/best_model.pth --episodes 20 --output results/evaluation
```

### Visualization

```bash
# Visualize environment with trained model
python -m  visualize --type environment --model models/best_model.pth --pattern rush_hour --duration 60 --output results/visualizations

# Visualize training metrics
python -m  visualize --type metrics --metrics models/training_metrics.json --output results/visualizations

# Visualize traffic patterns
python -m  visualize --type patterns --output results/visualizations
```

### Benchmarking

```bash
# Benchmark multiple agents
python -m  benchmark --model models/best_model.pth --episodes 15 --output results/benchmark
```

### Analysis

```bash
# Comprehensive analysis
python -m  analyze --model models/best_model.pth --metrics models/training_metrics.json --benchmark-dir results/benchmark --output results/analysis --episodes 10
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
