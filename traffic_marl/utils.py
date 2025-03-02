"""
Utility functions for traffic light control with reinforcement learning.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import logging

# Set up logging
logger = logging.getLogger("TrafficRL")

def setup_logging(debug=False):
    """Set up logging configuration."""
    try:
        os.makedirs("logs", exist_ok=True)
        log_level = logging.DEBUG if debug else logging.INFO
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler("logs/traffic_rl.log"),
                logging.StreamHandler()
            ]
        )
    except Exception as e:
        # Fallback to console-only logging
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        )
        print(f"Warning: Could not set up file logging: {e}")

def enable_debug_logging():
    """Enable debug logging level."""
    logger.setLevel(logging.DEBUG)
    for handler in logger.handlers:
        handler.setLevel(logging.DEBUG)
    logger.debug("Debug logging enabled")

def visualize_results(rewards_history, avg_rewards_history, save_path=None):
    """
    Visualize training results.
    
    Args:
        rewards_history: List of episode rewards
        avg_rewards_history: List of average rewards
        save_path: Path to save the plot
    """
    try:
        plt.figure(figsize=(12, 6))
        
        # Plot episode rewards
        plt.plot(rewards_history, alpha=0.6, label='Episode Reward')
        
        # Plot 100-episode rolling average
        plt.plot(avg_rewards_history, label='Avg Reward (100 episodes)')
        
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Training Progress')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            # Ensure directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
        
        plt.show()
    except Exception as e:
        logger.error(f"Visualization failed: {e}")
        logger.info("Saving raw data instead...")
        
        # Save raw data as CSV if plotting fails
        if save_path:
            try:
                base_path = os.path.splitext(save_path)[0]
                with open(f"{base_path}_data.csv", 'w') as f:
                    f.write("episode,reward,avg_reward\n")
                    for i, (r, ar) in enumerate(zip(rewards_history, avg_rewards_history)):
                        f.write(f"{i},{r},{ar}\n")
                logger.info(f"Raw data saved to {base_path}_data.csv")
            except Exception as e2:
                logger.error(f"Failed to save raw data: {e2}")

# Default configuration
CONFIG = {
    "sim_time": 3600,           # Simulation time in seconds
    "num_episodes": 500,        # Number of training episodes
    "max_steps": 1000,          # Maximum steps per episode
    "learning_rate": 0.0003,    # Learning rate for the optimizer
    "gamma": 0.99,              # Discount factor
    "epsilon_start": 1.0,       # Starting epsilon for exploration
    "epsilon_end": 0.01,        # Ending epsilon for exploration
    "epsilon_decay": 0.995,     # Epsilon decay rate
    "buffer_size": 100000,      # Replay buffer size
    "batch_size": 64,           # Batch size for training
    "target_update": 5,         # Target network update frequency
    "eval_frequency": 20,       # Evaluation frequency (episodes)
    "save_frequency": 25,       # Model saving frequency (episodes)
    "grid_size": 4,             # Size of the traffic grid (4x4)
    "max_cars": 30,             # Maximum number of cars per lane
    "green_duration": 10,       # Default green light duration (seconds)
    "yellow_duration": 3,       # Default yellow light duration (seconds)
    "visualization": False,     # Enable visualization during training
    "device": "auto",           # Auto-detect device (CUDA, CPU, MPS)
    "early_stopping_reward": 500,  # Reward threshold for early stopping
    "checkpoint_dir": "checkpoints",  # Directory for checkpoints
    "hidden_dim": 256,          # Hidden dimension for neural networks
    "weight_decay": 0.0001,     # L2 regularization parameter
    "grad_clip": 1.0,           # Gradient clipping value
    "use_lr_scheduler": True,   # Use learning rate scheduler
    "lr_step_size": 100,        # LR scheduler step size
    "lr_decay": 0.5,            # LR decay factor
    "clip_rewards": True,       # Whether to clip rewards
    "reward_scale": 0.1,        # Reward scaling factor
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
        "prioritized_replay": True,
        "per_alpha": 0.6,
        "per_beta": 0.4,
        "dueling_network": True,
        "double_dqn": True
    }
}

# Set seeds for reproducibility
RANDOM_SEED = 42