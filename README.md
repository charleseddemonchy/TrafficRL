# Traffic RL: Reinforcement Learning for Traffic Light Control

This package provides a comprehensive framework for training and evaluating reinforcement learning agents for traffic light control optimization.

## Features

- **Multiple Agent Types**: DQN, Dueling DQN, Fixed-Timing, Adaptive-Timing, and Random agents
- **Realistic Traffic Simulation**: Simulates various traffic patterns and densities
- **Comprehensive Evaluation**: Tools for benchmarking and analyzing agent performance
- **Visualization**: Extensive visualization capabilities for training metrics and traffic simulations

## CLI Usage

The package provides a unified command-line interface for all operations:

1.  **Install Dependencies:**

    *   It is recommended to create a virtual environment before installing the dependencies.
    *   Run `pip install -r requirements.txt` to install all required packages.
2.  **Training:**

    *   Run `python -m traffic_rl.train --output results/training` to start the training process.
3.  **Evaluation:**

    *   Run `python -m traffic_rl.evaluate --model traffic_rl/results/training/best_model.pth --episodes 20 --output results/evaluation` to evaluate a trained model.
4.  **Visualization:**

    *   Run `python -m traffic_rl.visualize --type environment --model traffic_rl/results/training/best_model.pth --pattern rush_hour --duration 60 --output results/visualizations` to visualize the environment with a trained model.
    *   Run `python -m traffic_rl.visualize --type metrics --metrics traffic_rl/results/training/training_metrics.json --output results/visualizations` to visualize training metrics.
    *   Run `python -m traffic_rl.visualize --type patterns --output results/visualizations` to visualize traffic patterns.
5.  **Benchmarking:**

    *   Run `python -m traffic_rl.benchmark --model traffic_rl/results/training/best_model.pth --episodes 15 --output results/benchmark` to benchmark multiple agents.
6.  **Analysis:**

    *   Run `python -m traffic_rl.analyze --model traffic_rl/results/training/best_model.pth --metrics traffic_rl/results/training/training_metrics.json --benchmark-dir results/benchmark --output results/analysis --episodes 10` to perform a comprehensive analysis.

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
