"""
Configuration for the Traffic Light Management with Reinforcement Learning project.
"""

import os
import json
import logging

logger = logging.getLogger("TrafficRL")

# Default configuration
DEFAULT_CONFIG = {
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
    "device": "mps",           # Auto-detect device (CUDA, CPU, MPS)
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

def load_config(config_path):
    """
    Load configuration from a JSON file, falling back to defaults if not found.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary containing the configuration
    """
    config = DEFAULT_CONFIG.copy()
    
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                loaded_config = json.load(f)
                # Update default config with loaded values
                config.update(loaded_config)
                logger.info(f"Loaded configuration from {config_path}")
        else:
            logger.warning(f"Configuration file {config_path} not found, using defaults")
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        logger.info("Continuing with default configuration")
    
    return config

def save_config(config, config_path):
    """
    Save configuration to a JSON file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save the configuration file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        
        logger.info(f"Configuration saved to {config_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to save configuration: {e}")
        return False