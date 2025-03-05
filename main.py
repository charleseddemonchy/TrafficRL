"""
Traffic Light Management with Reinforcement Learning
====================================================
A comprehensive reinforcement learning project to optimize traffic flow
through intelligent traffic light control.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
import pygame
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import random
import time
from tqdm import tqdm
import argparse
import json
import logging

# Handle missing optional dependencies
try:
    import pandas as pd
except ImportError:
    pd = None

# Set up logging
try:
    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("logs/traffic_rl.log"),
            logging.StreamHandler()
        ]
    )
except Exception as e:
    # Fallback to console-only logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )
    print(f"Warning: Could not set up file logging: {e}")

logger = logging.getLogger("TrafficRL")

# Function to enable debug logging if needed
def enable_debug_logging():
    logger.setLevel(logging.DEBUG)
    for handler in logger.handlers:
        handler.setLevel(logging.DEBUG)
    logger.debug("Debug logging enabled")

# Set seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# Config
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

# Define the TrafficSimulation environment
class TrafficSimulation(gym.Env):
    """
    Custom Gym environment for traffic simulation.
    
    Represents a grid of intersections controlled by traffic lights.
    Each intersection has four incoming lanes (North, East, South, West).
    """
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}
    
    def __init__(self, grid_size=4, max_cars=30, green_duration=10, yellow_duration=3, 
                 visualization=False, random_seed=None):
        super(TrafficSimulation, self).__init__()
        
        # Environment configuration
        self.grid_size = grid_size
        self.max_cars = max_cars
        self.green_duration = green_duration
        self.yellow_duration = yellow_duration
        self.visualization = visualization
        
        # Set random seed if provided
        if random_seed is not None:
            self.np_random = np.random.RandomState(random_seed)
        else:
            self.np_random = np.random
        
        # Number of intersections in the grid
        self.num_intersections = grid_size * grid_size
        
        # Traffic light states: 0=North-South Green, 1=East-West Green
        # Complex states could be expanded (e.g., left turn signals)
        self.action_space = spaces.Discrete(2)
        
        # Observation space: traffic density and light state for each intersection
        # For each intersection: [NS_density, EW_density, light_state]
        self.observation_space = spaces.Box(
            low=0, 
            high=1, 
            shape=(self.num_intersections, 5),
            dtype=np.float32
        )
        
        # Initialize default traffic pattern
        self.traffic_pattern = "uniform"
        self.traffic_config = CONFIG["traffic_patterns"]["uniform"]
        
        # Initialize visualization if enabled
        if self.visualization:
            try:
                self._init_visualization()
            except Exception as e:
                logger.warning(f"Could not initialize visualization: {e}")
                self.visualization = False
                
        # Reset the environment
        self.reset()
        
        # Add save_visualization method to the class
        self.save_visualization = save_visualization
    
    def reset(self, seed=None, options=None):
        """Reset the environment to initial state."""
        # Set seed if provided
        if seed is not None:
            self.np_random = np.random.RandomState(seed)
        
        # Initialize traffic densities (random initial state)
        self.traffic_density = self.np_random.uniform(
            0.0, 0.5, size=(self.num_intersections, 2)
        )
        
        # Initialize traffic light states (all start with NS green)
        self.light_states = np.zeros(self.num_intersections, dtype=int)
        
        # Initialize timers for each traffic light
        self.timers = np.zeros(self.num_intersections)
        
        # Track waiting time for cars at each intersection
        self.waiting_time = np.zeros((self.num_intersections, 2))
        
        # Track number of cars passed through each intersection
        self.cars_passed = np.zeros((self.num_intersections, 2))
        
        # Track green light durations for each direction
        self.ns_green_duration = np.zeros(self.num_intersections)
        self.ew_green_duration = np.zeros(self.num_intersections)
        self.light_switches = 0
        
        # Simulation time
        self.sim_time = 0
        
        # Generate observation
        observation = self._get_observation()
        
        # Info dictionary
        info = {}
        
        return observation, info
    
    def step(self, actions):
        """
        Take a step in the environment given the actions.
        
        Args:
            actions: Array of actions for each intersection (0=NS Green, 1=EW Green)
                    or a single action to apply to all intersections
        
        Returns:
            observation: Current observation
            reward: Reward from the action
            terminated: Whether the episode is done
            truncated: Whether the episode is truncated
            info: Additional information
        """
        try:
            # Handle both scalar and array inputs for actions
            if isinstance(actions, (int, np.integer, float, np.floating)):
                # If a single action is provided, convert to array
                actions_array = np.full(self.num_intersections, int(actions))
            elif isinstance(actions, (list, np.ndarray)):
                # If array-like with single value, convert to array of that value
                if len(actions) == 1:
                    actions_array = np.full(self.num_intersections, actions[0])
                elif len(actions) != self.num_intersections:
                    # If array with wrong length, broadcast or truncate
                    logger.warning(f"Actions array length {len(actions)} doesn't match num_intersections {self.num_intersections}")
                    actions_array = np.resize(actions, self.num_intersections)
                else:
                    # Correct length array
                    actions_array = np.array(actions)
            else:
                # Fallback for unexpected action type
                logger.warning(f"Unexpected action type: {type(actions)}, defaulting to all 0")
                actions_array = np.zeros(self.num_intersections, dtype=int)
            
            # Update traffic lights based on actions
            for i in range(self.num_intersections):
                # Only change the light if the timer has expired
                if self.timers[i] <= 0:
                    self.light_states[i] = actions_array[i]
                    self.timers[i] = self.green_duration
                else:
                    # Decrease the timer
                    self.timers[i] -= 1
                    
            # Update duration trackers
            for i in range(self.num_intersections):
                if self.light_states[i] == 0:  # NS is green
                    self.ns_green_duration[i] += 1
                    self.ew_green_duration[i] = 0
                else:  # EW is green
                    self.ew_green_duration[i] += 1
                    self.ns_green_duration[i] = 0
                
            # Simulate traffic flow
            self._update_traffic()
            
            # Calculate reward
            reward = self._calculate_reward()
            
            # Generate observation
            observation = self._get_observation()
            
            # Update simulation time
            self.sim_time += 1
            
            # Check if episode is done
            terminated = False
            truncated = False
            
            # Additional info
            info = {
                'average_waiting_time': np.mean(self.waiting_time),
                'total_cars_passed': np.sum(self.cars_passed),
                'traffic_density': np.mean(self.traffic_density)
            }
            
            # Render if visualization is enabled
            if self.visualization:
                self.render()
            
            return observation, reward, terminated, truncated, info
            
        except Exception as e:
            logger.error(f"Error in environment step: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            # Return a safe fallback state
            fallback_obs = self._get_observation()
            return fallback_obs, 0.0, True, False, {"error": str(e)}
    
    def _update_traffic(self):
        """Simulate traffic flow and update densities with more realistic behavior."""
        try:
            # Track pre-update densities
            prev_density = np.copy(self.traffic_density)
            
            # Define speed factors based on density (congestion effects)
            # Higher density = slower traffic flow
            speed_factor_ns = 1.0 - 0.7 * prev_density[:, 0]  # Speed factor for NS direction
            speed_factor_ew = 1.0 - 0.7 * prev_density[:, 1]  # Speed factor for EW direction
            
            # Base flow rates
            base_flow_rate = 0.1  # Base flow rate with green light
            red_light_flow = 0.01  # Small flow even with red light (running red)
            
            for i in range(self.num_intersections):
                # Get current light state (0=NS Green, 1=EW Green)
                light = self.light_states[i]
                
                # Calculate flow rates with congestion effects
                if light == 0:  # NS green
                    # Green light flow rate affected by congestion
                    ns_flow_rate = base_flow_rate * speed_factor_ns[i]
                    # Small flow through red light (some cars run the red)
                    ew_flow_rate = red_light_flow * speed_factor_ew[i]
                else:  # EW green
                    # Small flow through red light
                    ns_flow_rate = red_light_flow * speed_factor_ns[i]
                    # Green light flow rate
                    ew_flow_rate = base_flow_rate * speed_factor_ew[i]
                
                # Calculate actual flow based on current density
                ns_cars_flow = min(self.traffic_density[i, 0], ns_flow_rate)
                ew_cars_flow = min(self.traffic_density[i, 1], ew_flow_rate)
                
                # Update densities and stats
                self.traffic_density[i, 0] -= ns_cars_flow
                self.traffic_density[i, 1] -= ew_cars_flow
                
                # Track cars that passed through
                self.cars_passed[i, 0] += ns_cars_flow * self.max_cars
                self.cars_passed[i, 1] += ew_cars_flow * self.max_cars
                
                # Calculate waiting time based on density and whether light is red
                if light == 0:  # NS Green
                    # Cars wait in EW direction
                    self.waiting_time[i, 1] += self.traffic_density[i, 1]
                else:  # EW Green
                    # Cars wait in NS direction
                    self.waiting_time[i, 0] += self.traffic_density[i, 0]
            
            # Simulate new cars arriving with daily patterns
            # Time of day effect (0=midnight, 0.5=noon, 1.0=midnight again)
            time_of_day = (self.sim_time % 1440) / 1440.0  # Normalize to [0,1]
            
            # Get traffic pattern configuration
            if self.traffic_pattern == "rush_hour":
                # Morning rush hour around 8am (time_of_day ~= 0.33)
                # Evening rush hour around 5pm (time_of_day ~= 0.71)
                morning_peak = self.traffic_config.get("morning_peak", 0.33)
                evening_peak = self.traffic_config.get("evening_peak", 0.71)
                peak_intensity = self.traffic_config.get("peak_intensity", 2.0)
                base_arrival = self.traffic_config.get("base_arrival", 0.03)
                
                rush_hour_factor = peak_intensity * (
                    np.exp(-20 * (time_of_day - morning_peak)**2) +  # Morning peak
                    np.exp(-20 * (time_of_day - evening_peak)**2)    # Evening peak
                )
            elif self.traffic_pattern == "weekend":
                # Weekend pattern: one peak around noon
                midday_peak = self.traffic_config.get("midday_peak", 0.5)
                peak_intensity = self.traffic_config.get("peak_intensity", 1.5)
                base_arrival = self.traffic_config.get("base_arrival", 0.02)
                
                rush_hour_factor = peak_intensity * np.exp(-10 * (time_of_day - midday_peak)**2)
            else:  # uniform pattern
                base_arrival = self.traffic_config.get("arrival_rate", 0.03)
                variability = self.traffic_config.get("variability", 0.01)
                rush_hour_factor = 0
            
            # Add randomness to arrival patterns
            for i in range(self.num_intersections):
                # New cars arrive from each direction with pattern effects
                arrival_factor = base_arrival * (1 + rush_hour_factor)
                ns_arrivals = arrival_factor * self.np_random.uniform(0.5, 1.5)
                ew_arrivals = arrival_factor * self.np_random.uniform(0.5, 1.5)
                
                # Add new cars (ensure density doesn't exceed 1.0)
                self.traffic_density[i, 0] = min(1.0, self.traffic_density[i, 0] + ns_arrivals)
                self.traffic_density[i, 1] = min(1.0, self.traffic_density[i, 1] + ew_arrivals)
            
            # Simulate traffic flow between adjacent intersections with directional flow
            if self.grid_size > 1:
                # Create a copy of current densities after individual intersection updates
                new_density = np.copy(self.traffic_density)
                
                # Calculate flow between intersections based on density gradients
                flow_between = 0.05  # Base rate of flow between intersections
                
                for i in range(self.grid_size):
                    for j in range(self.grid_size):
                        idx = i * self.grid_size + j
                        
                        # For each direction, flow depends on density gradient
                        # Flow from high density to low density
                        
                        # North neighbor (i-1, j)
                        if i > 0:
                            north_idx = (i-1) * self.grid_size + j
                            # NS flow from current to north (if current has higher density)
                            density_diff = self.traffic_density[idx, 0] - self.traffic_density[north_idx, 0]
                            if density_diff > 0:
                                flow = flow_between * density_diff
                                actual_flow = min(flow, self.traffic_density[idx, 0] * 0.3)  # Limit to 30%
                                new_density[idx, 0] -= actual_flow
                                new_density[north_idx, 0] += actual_flow
                        
                        # South neighbor (i+1, j)
                        if i < self.grid_size - 1:
                            south_idx = (i+1) * self.grid_size + j
                            # NS flow from current to south
                            density_diff = self.traffic_density[idx, 0] - self.traffic_density[south_idx, 0]
                            if density_diff > 0:
                                flow = flow_between * density_diff
                                actual_flow = min(flow, self.traffic_density[idx, 0] * 0.3)
                                new_density[idx, 0] -= actual_flow
                                new_density[south_idx, 0] += actual_flow
                        
                        # West neighbor (i, j-1)
                        if j > 0:
                            west_idx = i * self.grid_size + (j-1)
                            # EW flow from current to west
                            density_diff = self.traffic_density[idx, 1] - self.traffic_density[west_idx, 1]
                            if density_diff > 0:
                                flow = flow_between * density_diff
                                actual_flow = min(flow, self.traffic_density[idx, 1] * 0.3)
                                new_density[idx, 1] -= actual_flow
                                new_density[west_idx, 1] += actual_flow
                        
                        # East neighbor (i, j+1)
                        if j < self.grid_size - 1:
                            east_idx = i * self.grid_size + (j+1)
                            # EW flow from current to east
                            density_diff = self.traffic_density[idx, 1] - self.traffic_density[east_idx, 1]
                            if density_diff > 0:
                                flow = flow_between * density_diff
                                actual_flow = min(flow, self.traffic_density[idx, 1] * 0.3)
                                new_density[idx, 1] -= actual_flow
                                new_density[east_idx, 1] += actual_flow
                
                # Update traffic density with new values, ensuring it stays in [0,1]
                self.traffic_density = np.clip(new_density, 0.0, 1.0)
                
        except Exception as e:
            logger.error(f"Error in traffic update: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def _calculate_reward(self):
        """
        Calculate reward based on traffic flow efficiency.
        
        Reward components:
        1. Negative reward for waiting cars (weighted by density)
        2. Positive reward for cars passing through
        3. Penalty for switching lights too frequently
        
        Returns:
            float: The calculated reward
        """
        try:
            # Calculate waiting time penalty (scaled down to prevent extreme negative values)
            waiting_penalty = -np.sum(self.waiting_time) * 0.05
            throughput_reward = np.sum(self.cars_passed) * 0.05
            
            # Add fairness component - penalize uneven queues
            ns_density_avg = np.mean(self.traffic_density[:, 0])
            ew_density_avg = np.mean(self.traffic_density[:, 1])
            fairness_penalty = -abs(ns_density_avg - ew_density_avg) * 10.0
            
            # Add switching penalty to prevent rapid oscillation
            switching_penalty = -self.light_switches * 0.01
            
            return waiting_penalty + throughput_reward + fairness_penalty + switching_penalty
            
            return total_reward
            
        except Exception as e:
            logger.error(f"Error calculating reward: {e}")
            return 0.0  # Safe default
    
    def _get_observation(self):
        """
        Construct the observation from the current state.
        
        For each intersection, the observation includes:
        - NS traffic density (normalized)
        - EW traffic density (normalized)
        - Traffic light state (0 for NS green, 1 for EW green)
        """
        observation = np.zeros((self.num_intersections, 5), dtype=np.float32)
        
        for i in range(self.num_intersections):
            # Traffic density for NS and EW
            observation[i, 0] = self.traffic_density[i, 0]
            observation[i, 1] = self.traffic_density[i, 1]
            
            # Traffic light state
            observation[i, 2] = self.light_states[i]
            
            # Add waiting time information 
            observation[i, 3] = self.waiting_time[i, 0] / 10.0  # Normalized NS waiting
            observation[i, 4] = self.waiting_time[i, 1] / 10.0  # Normalized EW waiting
        
        return observation
    
    def _init_visualization(self):
        """Initialize pygame for visualization."""
        try:
            pygame.init()
            self.screen_width = 800
            self.screen_height = 800
            # Try to set display mode, fall back to headless if needed
            try:
                self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            except pygame.error:
                os.environ['SDL_VIDEODRIVER'] = 'dummy'
                self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
                logger.warning("Using dummy video driver for headless rendering")
                
            pygame.display.set_caption("Traffic Simulation")
            self.clock = pygame.time.Clock()
        except Exception as e:
            logger.error(f"Failed to initialize visualization: {e}")
            raise
    
    def render(self, mode='human'):
        """Render the environment."""
        if not self.visualization:
            return None
        
        try:
            # Fill background
            self.screen.fill((255, 255, 255))
            
            cell_width = self.screen_width // self.grid_size
            cell_height = self.screen_height // self.grid_size
            
            # Draw grid and traffic lights
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    idx = i * self.grid_size + j
                    
                    # Calculate position
                    x = j * cell_width
                    y = i * cell_height
                    
                    # Draw intersection
                    pygame.draw.rect(self.screen, (200, 200, 200), 
                                    (x, y, cell_width, cell_height))
                    
                    # Draw roads
                    road_width = min(cell_width, cell_height) // 4
                    
                    # NS road
                    pygame.draw.rect(self.screen, (100, 100, 100),
                                    (x + cell_width//2 - road_width//2, y, 
                                     road_width, cell_height))
                    
                    # EW road
                    pygame.draw.rect(self.screen, (100, 100, 100),
                                    (x, y + cell_height//2 - road_width//2, 
                                     cell_width, road_width))
                    
                    # Draw traffic light
                    light_radius = road_width // 2
                    light_x = x + cell_width // 2
                    light_y = y + cell_height // 2
                    
                    if self.light_states[idx] == 0:  # NS Green
                        # NS light green
                        pygame.draw.circle(self.screen, (0, 255, 0), 
                                          (light_x, light_y - light_radius), light_radius // 2)
                        # EW light red
                        pygame.draw.circle(self.screen, (255, 0, 0), 
                                          (light_x + light_radius, light_y), light_radius // 2)
                    else:  # EW Green
                        # NS light red
                        pygame.draw.circle(self.screen, (255, 0, 0), 
                                          (light_x, light_y - light_radius), light_radius // 2)
                        # EW light green
                        pygame.draw.circle(self.screen, (0, 255, 0), 
                                          (light_x + light_radius, light_y), light_radius // 2)
                    
                    # Display traffic density as text
                    try:
                        font = pygame.font.Font(None, 24)
                        ns_text = font.render(f"NS: {self.traffic_density[idx, 0]:.2f}", True, (0, 0, 0))
                        ew_text = font.render(f"EW: {self.traffic_density[idx, 1]:.2f}", True, (0, 0, 0))
                        
                        self.screen.blit(ns_text, (x + 10, y + 10))
                        self.screen.blit(ew_text, (x + 10, y + 30))
                    except Exception as e:
                        # Continue without text if font rendering fails
                        logger.warning(f"Font rendering failed: {e}")
            
            pygame.display.flip()
            self.clock.tick(self.metadata['render_fps'])
            
            if mode == 'rgb_array':
                try:
                    return np.transpose(
                        np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
                    )
                except Exception:
                    return None
                    
        except Exception as e:
            logger.error(f"Render failed: {e}")
            self.visualization = False  # Disable visualization after error
            return None
    
    def close(self):
        """Close the environment."""
        if self.visualization:
            pygame.quit()

# Create a visualization video of the traffic simulation
def enhanced_save_visualization(self, filename="traffic_simulation.mp4", fps=30, duration=30):
    """
    Save an enhanced video visualization of the traffic simulation.
    
    Args:
        filename: Output video file name
        fps: Frames per second
        duration: Duration of video in seconds
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        import matplotlib.animation as animation
        from matplotlib import pyplot as plt
        from matplotlib.gridspec import GridSpec
        from matplotlib.patches import Rectangle, Circle, Polygon, PathPatch
        from matplotlib.collections import PatchCollection
        import matplotlib.patches as mpatches
        import matplotlib.colors as mcolors
        import numpy as np
        import os
        
        # Make sure visualization is enabled
        old_viz_state = self.visualization
        self.visualization = True
        
        # Reset environment
        self.reset()
        
        # Set up metrics tracking
        metrics_history = {
            'time': [],
            'avg_density': [],
            'waiting_time': [],
            'throughput': [],
            'avg_queue_length': []
        }
        
        # Initialize simulation metrics
        total_waiting_time = 0
        total_cars_passed = 0
        
        # Setup figure with grid layout
        fig = plt.figure(figsize=(16, 10), facecolor='#f8f9fa')
        gs = GridSpec(3, 4, figure=fig, height_ratios=[1, 5, 1])
        
        # Header area
        ax_header = fig.add_subplot(gs[0, :])
        ax_header.axis('off')
        
        # Main simulation view
        ax_main = fig.add_subplot(gs[1, :])
        
        # Statistics panels
        ax_stats1 = fig.add_subplot(gs[2, 0])
        ax_stats2 = fig.add_subplot(gs[2, 1])
        ax_stats3 = fig.add_subplot(gs[2, 2])
        ax_stats4 = fig.add_subplot(gs[2, 3])
        
        # Turn off axes for stat panels
        for ax in [ax_stats1, ax_stats2, ax_stats3, ax_stats4]:
            ax.axis('off')
        
        # Create custom colormaps for traffic density
        # Green (low) to Red (high) for traffic density
        density_cmap = mcolors.LinearSegmentedColormap.from_list(
            'density', [(0, '#00cc00'), (0.5, '#ffcc00'), (1, '#cc0000')]
        )
        
        # Blue-based colormap for North-South traffic
        ns_cmap = mcolors.LinearSegmentedColormap.from_list(
            'ns_traffic', [(0, '#e6f2ff'), (0.5, '#4d94ff'), (1, '#0047b3')]
        )
        
        # Orange-based colormap for East-West traffic
        ew_cmap = mcolors.LinearSegmentedColormap.from_list(
            'ew_traffic', [(0, '#fff2e6'), (0.5, '#ffad33'), (1, '#cc7000')]
        )
        
        # Car shapes for visualization
        def get_car_shape(x, y, direction, size=0.04):
            """Create a car shape at (x,y) pointing in the given direction."""
            if direction == 'NS':  # North-South direction (vertical)
                car_points = [
                    (x - size/2, y),            # Front middle
                    (x - size/3, y - size/2),   # Front right
                    (x + size/3, y - size/2),   # Front left
                    (x + size/2, y),            # Rear middle
                    (x + size/3, y + size/2),   # Rear left
                    (x - size/3, y + size/2),   # Rear right
                ]
            else:  # East-West direction (horizontal)
                car_points = [
                    (x, y - size/2),            # Front middle
                    (x + size/2, y - size/3),   # Front right
                    (x + size/2, y + size/3),   # Front left
                    (x, y + size/2),            # Rear middle
                    (x - size/2, y + size/3),   # Rear left
                    (x - size/2, y - size/3),   # Rear right
                ]
            return Polygon(car_points, closed=True)
        
        # Function to draw individual cars based on density
        def draw_cars(ax, i, j, direction, density, color):
            """Draw individual cars along a road segment based on density."""
            # Calculate number of cars to show (proportional to density)
            max_cars_to_show = 12  # Maximum cars to show on a road segment
            num_cars = int(density * max_cars_to_show)
            
            car_patches = []
            if direction == 'NS':
                # North-South road - cars positioned along the y-axis
                road_center_x = j + 0.5
                road_width = 0.1
                car_width = min(road_width * 0.7, 0.04)
                
                spacing = 1.0 / max(1, max_cars_to_show + 1)
                
                for k in range(num_cars):
                    car_y = i + (k + 1) * spacing
                    if car_y < i + 1:  # Ensure the car is within the road segment
                        car = get_car_shape(road_center_x, car_y, 'NS', car_width)
                        car_patches.append(car)
            else:
                # East-West road - cars positioned along the x-axis
                road_center_y = i + 0.5
                road_width = 0.1
                car_width = min(road_width * 0.7, 0.04)
                
                spacing = 1.0 / max(1, max_cars_to_show + 1)
                
                for k in range(num_cars):
                    car_x = j + (k + 1) * spacing
                    if car_x < j + 1:  # Ensure the car is within the road segment
                        car = get_car_shape(car_x, road_center_y, 'EW', car_width)
                        car_patches.append(car)
            
            # Add all cars to the plot
            if car_patches:
                car_collection = PatchCollection(car_patches, facecolor=color, edgecolor='black', 
                                                linewidth=0.5, alpha=0.85)
                ax.add_collection(car_collection)
        
        # Function to update the main visualization
        def update_main_visualization(frame):
            ax_main.clear()
            
            # Set axis properties
            ax_main.set_xlim(0, self.grid_size)
            ax_main.set_ylim(0, self.grid_size)
            ax_main.set_facecolor('#eef7ed')  # Light green background for environment
            ax_main.set_aspect('equal')
            ax_main.set_xticks([])
            ax_main.set_yticks([])
            
            # Calculate grid cell size
            cell_width = 1.0
            cell_height = 1.0
            
            # Draw background "terrain" - grass/land areas
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    # Draw grass/land background except where roads will be
                    land = Rectangle(
                        (j, i), cell_width, cell_height,
                        facecolor='#c9e5bc',  # Light green for grass/land
                        edgecolor='none',
                        alpha=0.5
                    )
                    ax_main.add_patch(land)
            
            # Draw city blocks (buildings)
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    # Draw building blocks in each cell (except where roads intersect)
                    building_margin = 0.15  # Margin from road
                    building_x = j + building_margin
                    building_y = i + building_margin
                    building_width = cell_width - 2 * building_margin
                    building_height = cell_height - 2 * building_margin
                    
                    # Randomize building colors slightly for variation
                    r = np.random.uniform(0.65, 0.75)
                    g = np.random.uniform(0.65, 0.75)
                    b = np.random.uniform(0.65, 0.75)
                    
                    building = Rectangle(
                        (building_x, building_y), building_width, building_height,
                        facecolor=(r, g, b),  # Gray with slight variation
                        edgecolor='#404040',
                        linewidth=0.5,
                        alpha=0.9
                    )
                    ax_main.add_patch(building)
            
            # Draw roads and traffic
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    idx = i * self.grid_size + j
                    
                    # Get traffic densities
                    ns_density = self.traffic_density[idx, 0]
                    ew_density = self.traffic_density[idx, 1]
                    
                    # Determine colors for roads and traffic based on density
                    ns_road_color = '#333333'  # Dark gray for base road
                    ew_road_color = '#333333'  # Dark gray for base road
                    
                    ns_traffic_color = ns_cmap(ns_density)
                    ew_traffic_color = ew_cmap(ew_density)
                    
                    # Calculate road dimensions
                    road_width = 0.1
                    ns_road_x = j + 0.5 - road_width/2
                    ew_road_y = i + 0.5 - road_width/2
                    
                    # Draw roads first (as base)
                    
                    # NS road (vertical)
                    ns_road = Rectangle(
                        (ns_road_x, i), road_width, cell_height,
                        facecolor=ns_road_color,
                        edgecolor='none',
                        alpha=0.9
                    )
                    ax_main.add_patch(ns_road)
                    
                    # EW road (horizontal)
                    ew_road = Rectangle(
                        (j, ew_road_y), cell_width, road_width,
                        facecolor=ew_road_color,
                        edgecolor='none',
                        alpha=0.9
                    )
                    ax_main.add_patch(ew_road)
                    
                    # Draw lane markings
                    def draw_lane_markings(x, y, is_vertical, length):
                        """Draw dashed lane markings on roads."""
                        if is_vertical:  # NS road
                            lane_x = x + road_width/2
                            num_dashes = int(length / 0.1)
                            for k in range(num_dashes):
                                if k % 2 == 0:  # Skip every other dash for dashed line
                                    dash_y = y + k * 0.1
                                    dash = Rectangle(
                                        (lane_x - 0.005, dash_y), 0.01, 0.05,
                                        facecolor='#ffffff',
                                        edgecolor='none',
                                        alpha=0.7
                                    )
                                    ax_main.add_patch(dash)
                        else:  # EW road
                            lane_y = y + road_width/2
                            num_dashes = int(length / 0.1)
                            for k in range(num_dashes):
                                if k % 2 == 0:  # Skip every other dash for dashed line
                                    dash_x = x + k * 0.1
                                    dash = Rectangle(
                                        (dash_x, lane_y - 0.005), 0.05, 0.01,
                                        facecolor='#ffffff',
                                        edgecolor='none',
                                        alpha=0.7
                                    )
                                    ax_main.add_patch(dash)
                    
                    # Add lane markings
                    draw_lane_markings(ns_road_x, i, True, cell_height)
                    draw_lane_markings(j, ew_road_y, False, cell_width)
                    
                    # Draw individual cars instead of just colored rectangles
                    draw_cars(ax_main, i, j, 'NS', ns_density, ns_traffic_color)
                    draw_cars(ax_main, i, j, 'EW', ew_density, ew_traffic_color)
                    
                    # Draw the intersection
                    intersection = Rectangle(
                        (ns_road_x, ew_road_y), road_width, road_width,
                        facecolor='#3a3a3a',  # Darker gray for intersection
                        edgecolor='#222222',
                        linewidth=0.5,
                        alpha=1.0
                    )
                    ax_main.add_patch(intersection)
                    
                    # Draw traffic light housing
                    light_housing_size = 0.04
                    light_housing = Rectangle(
                        (j + 0.5 - light_housing_size/2, i + 0.5 - light_housing_size/2), 
                        light_housing_size, light_housing_size,
                        facecolor='#222222',
                        edgecolor='#000000',
                        linewidth=0.5,
                        alpha=0.9
                    )
                    ax_main.add_patch(light_housing)
                    
                    # Draw traffic lights
                    light_size = 0.015
                    
                    if self.light_states[idx] == 0:  # NS Green
                        ns_color = 'green'
                        ew_color = 'red'
                    else:  # EW Green
                        ns_color = 'red'
                        ew_color = 'green'
                    
                    # NS traffic light (vertical orientation)
                    ns_light_x = j + 0.5
                    ns_light_y = i + 0.5 - light_housing_size/4
                    ns_light = Circle((ns_light_x, ns_light_y), light_size, color=ns_color, alpha=0.9)
                    ax_main.add_patch(ns_light)
                    
                    # EW traffic light (horizontal orientation)
                    ew_light_x = j + 0.5 - light_housing_size/4
                    ew_light_y = i + 0.5
                    ew_light = Circle((ew_light_x, ew_light_y), light_size, color=ew_color, alpha=0.9)
                    ax_main.add_patch(ew_light)
                    
                    # Add intersection ID
                    ax_main.text(j + 0.5, i + 0.5, f"{idx}",
                                fontsize=8, ha='center', va='center',
                                color='white', fontweight='bold')
            
            # Add scale/legend
            legend_x = 0.02
            legend_y = 0.02
            legend_width = 0.2
            legend_height = 0.08
            
            # Legend background
            legend_bg = Rectangle(
                (legend_x, legend_y), legend_width, legend_height,
                facecolor='white', alpha=0.7, transform=ax_main.transAxes
            )
            ax_main.add_patch(legend_bg)
            
            # Add legend entries
            ns_legend = Rectangle((0, 0), 1, 1, facecolor=ns_cmap(0.7), alpha=0.7)
            ew_legend = Rectangle((0, 0), 1, 1, facecolor=ew_cmap(0.7), alpha=0.7)
            ax_main.legend([ns_legend, ew_legend], ["NS Traffic", "EW Traffic"],
                          loc='lower left', bbox_to_anchor=(legend_x + 0.01, legend_y + 0.01),
                          frameon=False, fontsize=8)
            
            # Return the axis for animation
            return ax_main
        
        # Function to update the header with simulation time and status
        def update_header(frame, sim_time):
            ax_header.clear()
            ax_header.axis('off')
            
            # Calculate time metrics
            sim_seconds = sim_time
            sim_hours = int(sim_seconds / 3600)
            sim_minutes = int((sim_seconds % 3600) / 60)
            sim_seconds = int(sim_seconds % 60)
            
            # Calculate simulated time of day (0-24h) with simulation starting at 6:00 AM
            start_hour = 6  # 6 AM
            current_hour = (start_hour + sim_hours) % 24
            am_pm = "AM" if current_hour < 12 else "PM"
            display_hour = current_hour if current_hour <= 12 else current_hour - 12
            if display_hour == 0:
                display_hour = 12
            
            time_str = f"{display_hour:02d}:{sim_minutes:02d}:{sim_seconds:02d} {am_pm}"
            
            # Add time of day indicator
            time_of_day_width = 0.3
            ax_header.text(0.5, 0.7, f"Simulation Time: {time_str}", 
                         fontsize=14, fontweight='bold', ha='center', va='center')
            
            # Add day/night indicator based on time
            is_daytime = 6 <= current_hour < 18  # 6 AM to 6 PM is day
            day_night_status = "Daytime" if is_daytime else "Nighttime"
            day_night_color = '#4d94ff' if is_daytime else '#1a1a4d'
            
            day_night_indicator = Rectangle(
                (0.5 - time_of_day_width/2, 0.3), time_of_day_width, 0.2,
                facecolor=day_night_color, alpha=0.7, transform=ax_header.transAxes
            )
            ax_header.add_patch(day_night_indicator)
            
            # Add sun/moon icon
            if is_daytime:
                # Sun
                sun = Circle((0.5, 0.4), 0.03, facecolor='#ffcc00', edgecolor='#ff9900',
                           alpha=0.9, transform=ax_header.transAxes)
                ax_header.add_patch(sun)
            else:
                # Moon
                moon = Circle((0.5, 0.4), 0.03, facecolor='#f0f0f0', edgecolor='#d0d0d0',
                            alpha=0.9, transform=ax_header.transAxes)
                ax_header.add_patch(moon)
            
            # Add day/night text
            ax_header.text(0.5, 0.4, "", fontsize=8, ha='center', va='center',
                         transform=ax_header.transAxes)
            
            # Add simulation title
            ax_header.text(0.05, 0.5, "Traffic Light Management", 
                         fontsize=18, fontweight='bold', ha='left', va='center')
            
            # Add simulation parameters
            config_text = f"Grid: {self.grid_size}×{self.grid_size} | Pattern: {self.traffic_pattern}"
            ax_header.text(0.95, 0.5, config_text, 
                         fontsize=10, ha='right', va='center')
            
            return ax_header
        
        # Function to update statistics panels
        def update_stats(frame, metrics_history):
            # Calculate current metrics
            avg_density = np.mean(self.traffic_density)
            waiting_time = np.mean(self.waiting_time) if hasattr(self, 'waiting_time') else 0
            queue_lengths = np.sum(self.traffic_density > 0.5) / self.traffic_density.size
            
            # Update metrics history
            metrics_history['time'].append(frame/fps)
            metrics_history['avg_density'].append(avg_density)
            metrics_history['waiting_time'].append(waiting_time)
            metrics_history['avg_queue_length'].append(queue_lengths)
            
            # Keep only the most recent data points for plotting
            window_size = 100
            if len(metrics_history['time']) > window_size:
                for key in metrics_history:
                    metrics_history[key] = metrics_history[key][-window_size:]
            
            # Update panel 1: Overall Traffic Density
            ax_stats1.clear()
            ax_stats1.set_xlim(0, 1)
            ax_stats1.set_ylim(0, 1)
            
            # Create a density gauge
            gauge_height = 0.3
            gauge_background = Rectangle((0.1, 0.35), 0.8, gauge_height, 
                                       facecolor='#f0f0f0', edgecolor='#333333', linewidth=1)
            ax_stats1.add_patch(gauge_background)
            
            # Fill gauge based on average density
            gauge_fill = Rectangle((0.1, 0.35), 0.8 * avg_density, gauge_height, 
                                 facecolor=density_cmap(avg_density))
            ax_stats1.add_patch(gauge_fill)
            
            # Add tick marks
            for i in range(11):
                x_pos = 0.1 + i * 0.08
                ax_stats1.plot([x_pos, x_pos], [0.32, 0.35], 'k-', linewidth=1)
                if i % 2 == 0:  # Label every other tick
                    ax_stats1.text(x_pos, 0.27, f"{i*10}%", ha='center', va='top', fontsize=8)
            
            # Add panel title and value
            ax_stats1.text(0.5, 0.8, "Traffic Density", ha='center', va='center', fontsize=10, fontweight='bold')
            ax_stats1.text(0.5, 0.15, f"{avg_density*100:.1f}%", ha='center', va='center', 
                         fontsize=12, fontweight='bold', color=density_cmap(avg_density))
            
            # Update panel 2: Average Waiting Time
            ax_stats2.clear()
            ax_stats2.set_xlim(0, 1)
            ax_stats2.set_ylim(0, 1)
            
            # Plot waiting time history if we have enough data
            if len(metrics_history['time']) > 1:
                time_data = metrics_history['time']
                wait_data = metrics_history['waiting_time']
                
                # Normalize data for plotting
                wait_data_norm = np.array(wait_data) / max(1, max(wait_data))
                
                # Plot area
                ax_stats2.fill_between(
                    np.linspace(0.1, 0.9, len(time_data)), 
                    0.3, 
                    0.3 + 0.4 * wait_data_norm, 
                    color='#ff9966', alpha=0.7
                )
                
                # Add grid lines
                for y_pos in [0.3, 0.5, 0.7]:
                    ax_stats2.plot([0.1, 0.9], [y_pos, y_pos], 'k--', alpha=0.3)
            
            # Add panel title and current value
            ax_stats2.text(0.5, 0.8, "Waiting Time", ha='center', va='center', fontsize=10, fontweight='bold')
            ax_stats2.text(0.5, 0.15, f"{waiting_time:.2f}", ha='center', va='center', 
                         fontsize=12, fontweight='bold', color='#cc5200')
            
            # Update panel 3: Traffic Flow / Throughput
            ax_stats3.clear()
            ax_stats3.set_xlim(0, 1)
            ax_stats3.set_ylim(0, 1)
            
            # Calculate throughput as number of cars that passed
            if hasattr(self, 'cars_passed'):
                throughput = np.sum(self.cars_passed)
                metrics_history['throughput'].append(throughput)
            else:
                throughput = 0
                metrics_history['throughput'].append(0)
            
            # Plot throughput history if we have enough data
            if len(metrics_history['time']) > 1:
                time_data = metrics_history['time']
                throughput_data = metrics_history['throughput']
                
                # Get differences for instantaneous throughput
                if len(throughput_data) > 1:
                    instant_throughput = [throughput_data[i] - throughput_data[i-1] for i in range(1, len(throughput_data))]
                    instant_throughput = [0] + instant_throughput  # Add initial value
                    
                    # Normalize for plotting
                    max_throughput = max(1, max(instant_throughput))
                    throughput_norm = np.array(instant_throughput) / max_throughput
                    
                    # Create bar chart effect
                    bar_width = 0.8 / len(time_data)
                    for i, (t, tp) in enumerate(zip(time_data, throughput_norm)):
                        x_pos = 0.1 + i * bar_width
                        ax_stats3.add_patch(Rectangle(
                            (x_pos, 0.3), bar_width * 0.8, 0.4 * tp,
                            facecolor='#66cc99', edgecolor='none', alpha=0.7
                        ))
            
            # Add panel title and current value
            ax_stats3.text(0.5, 0.8, "Traffic Flow", ha='center', va='center', fontsize=10, fontweight='bold')
            ax_stats3.text(0.5, 0.15, f"{throughput:.0f} cars", ha='center', va='center', 
                         fontsize=12, fontweight='bold', color='#339966')
            
            # Update panel 4: Queue Lengths
            ax_stats4.clear()
            ax_stats4.set_xlim(0, 1)
            ax_stats4.set_ylim(0, 1)
            
            # Create a simple heatmap showing queue status at each intersection
            queue_heatmap_size = min(self.grid_size * 0.08, 0.3)  # Scale based on grid size
            cell_size = queue_heatmap_size / self.grid_size
            heatmap_x = 0.5 - queue_heatmap_size / 2
            heatmap_y = 0.4
            
            # Draw heatmap background
            heatmap_bg = Rectangle(
                (heatmap_x, heatmap_y), queue_heatmap_size, queue_heatmap_size,
                facecolor='#f0f0f0', edgecolor='#333333', linewidth=1
            )
            ax_stats4.add_patch(heatmap_bg)
            
            # Draw cells representing intersections
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    idx = i * self.grid_size + j
                    # Calculate average queue at this intersection
                    cell_queue = (self.traffic_density[idx, 0] + self.traffic_density[idx, 1]) / 2
                    
                    # Draw cell with color based on queue length
                    cell_x = heatmap_x + j * cell_size
                    cell_y = heatmap_y + (self.grid_size - i - 1) * cell_size  # Flip y-axis for correct orientation
                    
                    queue_cell = Rectangle(
                        (cell_x, cell_y), cell_size, cell_size,
                        facecolor=density_cmap(cell_queue), edgecolor='none'
                    )
                    ax_stats4.add_patch(queue_cell)
            
            # Add panel title and average value
            ax_stats4.text(0.5, 0.8, "Queue Status", ha='center', va='center', fontsize=10, fontweight='bold')
            ax_stats4.text(0.5, 0.15, f"Avg: {queue_lengths*100:.1f}%", ha='center', va='center', 
                         fontsize=12, fontweight='bold')
            
            return [ax_stats1, ax_stats2, ax_stats3, ax_stats4]
        
        # Main update function for animation
        def update(frame):
            # Take action every few frames to allow smoother animation
            if frame % 3 == 0:
                # Get current simulation time
                sim_time = frame / fps
                
                # Use agent if available, otherwise use intelligent/random actions
                if hasattr(self, 'recording_agent') and self.recording_agent is not None:
                    # Use trained agent
                    state = self._get_observation().flatten()
                    action = self.recording_agent.act(state, eval_mode=True)
                    _, _, _, _, info = self.step(action)
                elif frame % 15 == 0:
                    # Intelligent action: more green time for direction with higher density
                    actions = []
                    for i in range(self.num_intersections):
                        ns_density = self.traffic_density[i, 0]
                        ew_density = self.traffic_density[i, 1]
                        action = 0 if ns_density > ew_density else 1
                        actions.append(action)
                    _, _, _, _, info = self.step(actions)
                else:
                    # Random actions
                    action = np.random.randint(0, 2, size=self.num_intersections)
                    _, _, _, _, info = self.step(action)
            
            # Update all parts of the visualization
            update_main_visualization(frame)
            update_header(frame, frame / fps)
            update_stats(frame, metrics_history)
            
            return fig
        
        # Create the animation
        ani = animation.FuncAnimation(fig, update, frames=duration*fps, 
                                    interval=1000/fps, blit=False)
        
        # Save the animation
        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
        
        # Try to save with ffmpeg first
        try:
            from matplotlib.animation import FFMpegWriter
            writer = FFMpegWriter(fps=fps, metadata=dict(title='Traffic Simulation', artist='RL Agent'), bitrate=5000)
            ani.save(filename, writer=writer)
            logger.info(f"Animation saved to {filename} with FFMpegWriter")
        except Exception as e:
            logger.warning(f"FFmpeg writer failed: {e}. Trying a different approach.")
            try:
                # Try alternative save method
                ani.save(filename, fps=fps, dpi=120)
                logger.info(f"Animation saved to {filename} with alternative method")
            except Exception as e2:
                logger.error(f"Failed to save animation: {e2}")
                # Save individual frames
                frames_dir = os.path.join(os.path.dirname(filename), "frames")
                os.makedirs(frames_dir, exist_ok=True)
                
                logger.info(f"Saving individual frames to {frames_dir}...")
                for i in range(min(300, duration*fps)):  # Limit to 300 frames to avoid too many files
                    # Update figure
                    update(i)
                    # Save frame
                    frame_path = f"{frames_dir}/frame_{i:04d}.png"
                    plt.savefig(frame_path, dpi=120)
                    if i % 10 == 0:
                        logger.info(f"Saved frame {i}/{duration*fps}")
                
                logger.info(f"Frames saved. Please use an external tool to combine them into a video.")
                return False
        
        # Close figure
        plt.close(fig)
        
        # Restore original visualization state
        self.visualization = old_viz_state
        
        logger.info(f"Enhanced visualization saved to {filename}")
        return True
        
    except Exception as e:
        logger.error(f"Enhanced visualization failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

# Replace the original save_visualization with our enhanced version
def save_visualization(self, filename="traffic_simulation.mp4", fps=30, duration=30):
    """
    Save an enhanced video visualization of the traffic simulation.
    
    Args:
        filename: Output video file name
        fps: Frames per second
        duration: Duration of video in seconds
        
    Returns:
        bool: True if successful, False otherwise
    """
    return self.enhanced_save_visualization(filename, fps, duration)

# Define the DQN model
class DQN(nn.Module):
    """
    Deep Q-Network for traffic light control.
    """
    def __init__(self, input_dim, output_dim, hidden_dim=256):
        super(DQN, self).__init__()
        
        # Input layer to first hidden layer
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # Add batch normalization
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        # Add dropout for regularization
        self.dropout1 = nn.Dropout(0.2)
        
        # First hidden layer to second hidden layer
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # Add batch normalization
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        # Add dropout
        self.dropout2 = nn.Dropout(0.2)
        
        # Second hidden layer to third hidden layer (smaller)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        # Add batch normalization
        self.bn3 = nn.BatchNorm1d(hidden_dim // 2)
        
        # Output layer
        self.fc4 = nn.Linear(hidden_dim // 2, output_dim)
        
        # Initialize weights using He initialization
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc3.weight, nonlinearity='relu')
        nn.init.xavier_uniform_(self.fc4.weight)
        
    def forward(self, x):
        # Check if we need to handle batch size of 1
        if x.dim() == 1 or x.size(0) == 1:
            # Process through network without batch norm for single samples
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            x = self.fc4(x)
        else:
            # Normal batch processing with batch normalization and dropout
            x = F.relu(self.bn1(self.fc1(x)))
            x = self.dropout1(x)
            x = F.relu(self.bn2(self.fc2(x)))
            x = self.dropout2(x)
            x = F.relu(self.bn3(self.fc3(x)))
            x = self.fc4(x)
            
        return x

# Define Dueling DQN model for advanced performance
class DuelingDQN(nn.Module):
    """
    Dueling Deep Q-Network for traffic light control.
    Separates state value and action advantage for better performance.
    """
    def __init__(self, input_dim, output_dim, hidden_dim=256):
        super(DuelingDQN, self).__init__()
        
        # Shared feature network
        self.feature_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Value stream (estimates state value)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Advantage stream (estimates action advantages)
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        # Initialize weights
        for layer in self.feature_layer:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
        
        for layer in self.value_stream:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                
        for layer in self.advantage_stream:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
    
    def forward(self, x):
        # Handle different input dimensions
        is_single = (x.dim() == 1)
        if is_single:
            x = x.unsqueeze(0)
            
        # Process shared features
        if x.size(0) == 1:
            # Skip batch norm for single samples
            features = self._forward_features_single(x)
        else:
            features = self.feature_layer(x)
        
        # Get value and advantage estimates
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # Combine value and advantage using the dueling formula
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
        q_values = values + (advantages - advantages.mean(dim=1, keepdim=True))
        
        if is_single:
            q_values = q_values.squeeze(0)
            
        return q_values
    
    def _forward_features_single(self, x):
        """Process features for a single sample without batch norm."""
        x = F.relu(self.feature_layer[0](x))
        x = F.relu(self.feature_layer[4](x))
        return x

# Define standard experience replay buffer
class ReplayBuffer:
    """
    Experience replay buffer to store and sample transitions.
    """
    def __init__(self, buffer_size, batch_size):
        self.batch_size = batch_size
        self.buffer = deque(maxlen=buffer_size)
        self.experience = namedtuple("Experience", 
                                     field_names=["state", "action", "reward", "next_state", "done"])
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to the buffer."""
        experience = self.experience(state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self):
        """Sample a batch of experiences from the buffer."""
        try:
            experiences = random.sample(self.buffer, k=min(self.batch_size, len(self.buffer)))
            
            # Ensure all elements are properly processed
            valid_experiences = []
            for e in experiences:
                if e is not None:
                    valid_experiences.append(e)
            
            if not valid_experiences:
                # Return None if no valid experiences
                return None
            
            # Convert to numpy arrays first to handle different shapes
            states_np = np.array([e.state for e in valid_experiences])
            # Ensure actions are reshaped properly for gather operation
            actions_np = np.array([e.action for e in valid_experiences])
            if actions_np.ndim == 3:  # If already has shape [batch, 1, 1]
                actions_np = actions_np.squeeze(2)  # Convert to [batch, 1]
            elif actions_np.ndim == 1:  # If shape is [batch]
                actions_np = np.expand_dims(actions_np, 1)  # Convert to [batch, 1]
                
            rewards_np = np.array([e.reward for e in valid_experiences])
            if rewards_np.ndim == 3:  # If already has shape [batch, 1, 1]
                rewards_np = rewards_np.squeeze(2)  # Convert to [batch, 1]
            elif rewards_np.ndim == 1:  # If shape is [batch]
                rewards_np = np.expand_dims(rewards_np, 1)  # Convert to [batch, 1]
                
            next_states_np = np.array([e.next_state for e in valid_experiences])
            
            dones_np = np.array([e.done for e in valid_experiences])
            if dones_np.ndim == 3:  # If already has shape [batch, 1, 1]
                dones_np = dones_np.squeeze(2)  # Convert to [batch, 1]
            elif dones_np.ndim == 1:  # If shape is [batch]
                dones_np = np.expand_dims(dones_np, 1)  # Convert to [batch, 1]
            
            # Convert to torch tensors
            states = torch.tensor(states_np, dtype=torch.float32)
            actions = torch.tensor(actions_np, dtype=torch.long)
            rewards = torch.tensor(rewards_np, dtype=torch.float32)
            next_states = torch.tensor(next_states_np, dtype=torch.float32)
            dones = torch.tensor(dones_np, dtype=torch.float32)
            
            # Log shapes for debugging
            # logger.debug(f"Sampled batch shapes: states={states.shape}, actions={actions.shape}, "
            #             f"rewards={rewards.shape}, next_states={next_states.shape}, dones={dones.shape}")
            
            return (states, actions, rewards, next_states, dones)
        
        except Exception as e:
            logger.error(f"Error sampling from replay buffer: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def __len__(self):
        """Return the current size of the buffer."""
        return len(self.buffer)

# Define Prioritized Experience Replay
class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay buffer to store and sample transitions
    based on their TD error priority.
    """
    def __init__(self, buffer_size, batch_size, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.alpha = alpha  # Priority exponent
        self.beta = beta    # Importance sampling weight
        self.beta_increment = beta_increment
        self.buffer = deque(maxlen=buffer_size)
        self.priorities = deque(maxlen=buffer_size)
        self.experience = namedtuple("Experience", 
                                    field_names=["state", "action", "reward", "next_state", "done"])
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to the buffer with maximum priority."""
        try:
            experience = self.experience(state, action, reward, next_state, done)
            
            # New experiences get maximum priority
            max_priority = max(self.priorities, default=1.0)
            
            self.buffer.append(experience)
            self.priorities.append(max_priority)
        except Exception as e:
            logger.error(f"Error adding to prioritized buffer: {e}")
    
    def sample(self):
        """Sample a batch of experiences based on priorities."""
        try:
            if len(self.buffer) == 0:
                return None
                
            # Convert priorities to probabilities
            priorities = np.array(self.priorities)
            probabilities = priorities ** self.alpha
            probabilities /= probabilities.sum()
            
            # Sample indices based on priorities
            indices = np.random.choice(
                len(self.buffer), 
                min(self.batch_size, len(self.buffer)), 
                replace=False, 
                p=probabilities
            )
            
            # Calculate importance sampling weights
            weights = (len(self.buffer) * probabilities[indices]) ** -self.beta
            weights /= weights.max()  # Normalize weights
            
            # Increment beta
            self.beta = min(1.0, self.beta + self.beta_increment)
            
            # Get experiences
            experiences = [self.buffer[idx] for idx in indices]
            
            # Ensure consistent shapes
            states_np = np.array([e.state for e in experiences])
            actions_np = np.array([e.action for e in experiences])
            if actions_np.ndim == 3:
                actions_np = actions_np.squeeze(2)
            elif actions_np.ndim == 1:
                actions_np = np.expand_dims(actions_np, 1)
                
            rewards_np = np.array([e.reward for e in experiences])
            if rewards_np.ndim == 3:
                rewards_np = rewards_np.squeeze(2)
            elif rewards_np.ndim == 1:
                rewards_np = np.expand_dims(rewards_np, 1)
                
            next_states_np = np.array([e.next_state for e in experiences])
            
            dones_np = np.array([e.done for e in experiences])
            if dones_np.ndim == 3:
                dones_np = dones_np.squeeze(2)
            elif dones_np.ndim == 1:
                dones_np = np.expand_dims(dones_np, 1)
            
            # Process experiences into tensors
            states = torch.tensor(states_np, dtype=torch.float32)
            actions = torch.tensor(actions_np, dtype=torch.long)
            rewards = torch.tensor(rewards_np, dtype=torch.float32)
            next_states = torch.tensor(next_states_np, dtype=torch.float32)
            dones = torch.tensor(dones_np, dtype=torch.float32)
            weights = torch.tensor(weights, dtype=torch.float32)
            
            return (states, actions, rewards, next_states, dones, weights, indices)
        
        except Exception as e:
            logger.error(f"Error sampling from prioritized buffer: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def update_priorities(self, indices, td_errors):
        """Update priorities based on TD errors."""
        try:
            for idx, error in zip(indices, td_errors):
                # Add small constant to ensure non-zero probability
                self.priorities[idx] = abs(float(error)) + 1e-5
        except Exception as e:
            logger.error(f"Error updating priorities: {e}")
    
    def __len__(self):
        """Return the current size of the buffer."""
        return len(self.buffer)

# Define the DQNAgent with advanced features
class DQNAgent:
    """
    DQN Agent for traffic light control.
    
    This agent implements Deep Q-Learning with experience replay and target network.
    """
    def __init__(self, state_size, action_size, config):
        """Initialize the agent."""
        self.state_size = state_size
        self.action_size = action_size
        self.config = config
        
        # Get device - auto-detect if set to 'auto'
        if config["device"] == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config["device"])
        logger.info(f"Using device: {self.device}")
        
        # Q-Networks - select based on config
        if config.get("advanced_options", {}).get("dueling_network", False):
            logger.info("Using Dueling DQN architecture")
            self.local_network = DuelingDQN(state_size, action_size, hidden_dim=config.get("hidden_dim", 256)).to(self.device)
            self.target_network = DuelingDQN(state_size, action_size, hidden_dim=config.get("hidden_dim", 256)).to(self.device)
        else:
            logger.info("Using standard DQN architecture")
            self.local_network = DQN(state_size, action_size, hidden_dim=config.get("hidden_dim", 256)).to(self.device)
            self.target_network = DQN(state_size, action_size, hidden_dim=config.get("hidden_dim", 256)).to(self.device)
        
        # Optimizer with learning rate
        self.optimizer = optim.Adam(
            self.local_network.parameters(), 
            lr=config["learning_rate"],
            weight_decay=config.get("weight_decay", 0)  # L2 regularization
        )
        
        # Learning rate scheduler for stability
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=config.get("lr_step_size", 100),
            gamma=config.get("lr_decay", 0.5)
        )
        
        # Replay buffer - standard or prioritized
        if config.get("advanced_options", {}).get("prioritized_replay", False):
            logger.info("Using Prioritized Experience Replay")
            self.memory = PrioritizedReplayBuffer(
                config["buffer_size"], 
                config["batch_size"],
                alpha=config.get("per_alpha", 0.6),
                beta=config.get("per_beta", 0.4)
            )
            self.use_prioritized = True
        else:
            logger.info("Using standard Experience Replay")
            self.memory = ReplayBuffer(config["buffer_size"], config["batch_size"])
            self.use_prioritized = False
        
        # Epsilon for exploration
        self.epsilon = config["epsilon_start"]
        self.epsilon_end = config["epsilon_end"]
        self.epsilon_decay = config["epsilon_decay"]
        
        # Gradient clipping value
        self.grad_clip = config.get("grad_clip", 1.0)
        
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        
        # Training metrics
        self.loss_history = []
    
    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and learn if it's time."""
        try:
            # Convert action to scalar if it's a single-item array or tensor
            if isinstance(action, (np.ndarray, list, torch.Tensor)):
                if hasattr(action, 'item'):
                    action = action.item()  # For PyTorch tensors
                elif isinstance(action, np.ndarray) and action.size == 1:
                    action = action.item()  # For NumPy arrays
                elif len(action) == 1:
                    action = action[0]  # For lists
            
            # Convert to numpy arrays with consistent shapes
            state_np = np.array(state, dtype=np.float32)
            action_np = np.array([[action]], dtype=np.int64)  # Shape [1, 1]
            reward_np = np.array([[reward]], dtype=np.float32)  # Shape [1, 1]
            next_state_np = np.array(next_state, dtype=np.float32)
            done_np = np.array([[done]], dtype=np.float32)  # Shape [1, 1]
            
            # Save experience in replay memory
            self.memory.add(state_np, action_np, reward_np, next_state_np, done_np)
            
            # Increment the time step
            self.t_step += 1
            
            # Check if enough samples are available in memory
            if len(self.memory) > self.config["batch_size"]:
                # If enough samples, learn every UPDATE_EVERY time steps
                if self.t_step % self.config["target_update"] == 0:
                    experiences = self.memory.sample()
                    self.learn(experiences)
        except Exception as e:
            logger.error(f"Error in step() method: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # Continue without breaking training
    
    def act(self, state, eval_mode=False):
        """
        Choose an action based on the current state.
        
        Args:
            state: Current state
            eval_mode: If True, greedy policy is used
        
        Returns:
            Selected action
        """
        try:
            # Handle different input types for state
            if isinstance(state, torch.Tensor):
                # Already a tensor
                state_tensor = state.float()
                if state_tensor.dim() == 1:
                    state_tensor = state_tensor.unsqueeze(0)
            else:
                # Convert to numpy array first, ensuring correct dtype
                try:
                    # Try direct conversion if already numpy-like
                    np_state = np.array(state, dtype=np.float32)
                    state_tensor = torch.tensor(np_state, dtype=torch.float32).unsqueeze(0)
                except Exception as e:
                    logger.warning(f"Error converting state to tensor: {e}")
                    # Fallback method
                    state_tensor = torch.FloatTensor([state]).unsqueeze(0)
            
            # Move to device
            state_tensor = state_tensor.to(self.device)
            
            # Set to evaluation mode
            self.local_network.eval()
            
            with torch.no_grad():
                action_values = self.local_network(state_tensor)
            
            # Set back to training mode
            self.local_network.train()
            
            # Epsilon-greedy action selection
            if not eval_mode and random.random() < self.epsilon:
                return int(random.randrange(self.action_size))
            else:
                # Make sure to return a plain Python int, not a numpy or torch type
                return int(np.argmax(action_values.cpu().data.numpy()))
                
        except Exception as e:
            logger.error(f"Error in act() method: {e}")
            # Return random action as fallback
            return int(random.randrange(self.action_size))
    
    def learn(self, experiences):
        """
        Update value parameters using given batch of experience tuples.
        
        Args:
            experiences: Tuple of (s, a, r, s', done) tuples and possibly weights and indices
        """
        # Check if experiences is None (could happen if there was an error in sampling)
        if experiences is None:
            logger.warning("No valid experiences to learn from")
            return
            
        try:
            # Handle different formats for prioritized vs standard replay
            if self.use_prioritized:
                states, actions, rewards, next_states, dones, weights, indices = experiences
            else:
                states, actions, rewards, next_states, dones = experiences
                weights = torch.ones_like(rewards)  # Uniform weights
            
            # Debug tensor shapes
            logger.debug(f"Tensor shapes: states={states.shape}, actions={actions.shape}, "
                       f"rewards={rewards.shape}, next_states={next_states.shape}, dones={dones.shape}")
            
            # Ensure actions tensor has the right shape for gather operation
            if actions.dim() != 2 or actions.size(1) != 1:
                logger.warning(f"Reshaping actions tensor from {actions.shape}")
                # If actions is [batch], reshape to [batch, 1]
                if actions.dim() == 1:
                    actions = actions.unsqueeze(1)
                # If actions is [batch, action_dim, 1], reshape to [batch, 1]
                elif actions.dim() == 3:
                    actions = actions.squeeze(2)
            
            # Move to device
            states = states.to(self.device)
            actions = actions.to(self.device)
            rewards = rewards.to(self.device)
            next_states = next_states.to(self.device)
            dones = dones.to(self.device)
            weights = weights.to(self.device)
            
            # Double DQN: Use online network to select action, target network to evaluate
            if self.config.get("advanced_options", {}).get("double_dqn", False):
                with torch.no_grad():
                    # Get action selection from online network
                    next_actions = self.local_network(next_states).detach().max(1)[1].unsqueeze(1)
                    # Get Q values from target network for selected actions
                    Q_targets_next = self.target_network(next_states).gather(1, next_actions)
            else:
                # Standard DQN: Use max Q value from target network
                with torch.no_grad():
                    Q_targets_next = self.target_network(next_states).detach().max(1)[0].unsqueeze(1)
            
            # Compute Q targets for current states
            Q_targets = rewards + (self.config["gamma"] * Q_targets_next * (1 - dones))
            
            # Get Q values from local model for all actions
            q_values = self.local_network(states)
            
            # Get expected Q values for selected actions
            Q_expected = q_values.gather(1, actions)
            
            # Compute loss with importance sampling weights for prioritized replay
            td_errors = Q_targets - Q_expected
            loss = (weights * td_errors.pow(2)).mean()
            
            # Store loss for monitoring
            self.loss_history.append(loss.item())
            
            # Minimize the loss
            self.optimizer.zero_grad()
            loss.backward()
            
            # Clip gradients to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.local_network.parameters(), self.grad_clip)
                
            self.optimizer.step()
            
            # Update learning rate if scheduler is enabled
            if self.config.get("use_lr_scheduler", False):
                self.scheduler.step()
            
            # Update priorities in prioritized replay buffer
            if self.use_prioritized:
                # Convert TD errors to numpy and update priorities
                td_errors_np = td_errors.detach().cpu().numpy()
                self.memory.update_priorities(indices, td_errors_np)
            
            # Update target network
            if self.t_step % self.config["target_update"] == 0:
                self.target_network.load_state_dict(self.local_network.state_dict())
            
            # Update epsilon
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        except Exception as e:
            logger.error(f"Error in learn() method: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def save(self, filename):
        """Save the model."""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            
            torch.save({
                'local_network_state_dict': self.local_network.state_dict(),
                'target_network_state_dict': self.target_network.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict() if hasattr(self, 'scheduler') else None,
                'epsilon': self.epsilon,
                'loss_history': self.loss_history
            }, filename)
            logger.info(f"Model saved to {filename}")
            return True
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
    
    def load(self, filename):
        """Load the model."""
        if not os.path.isfile(filename):
            logger.warning(f"Model file {filename} not found")
            return False
        
        try:
            # Try to load with regular torch.load
            checkpoint = torch.load(filename)
        except Exception as e:
            logger.warning(f"Failed to load model with regular torch.load: {e}")
            try:
                # Try loading with CPU map_location for models saved on different devices
                checkpoint = torch.load(filename, map_location=torch.device('cpu'))
            except Exception as e2:
                logger.error(f"Failed to load model: {e2}")
                return False
        
        try:
            # Load model components
            self.local_network.load_state_dict(checkpoint['local_network_state_dict'])
            self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Load scheduler if available
            if hasattr(self, 'scheduler') and checkpoint['scheduler_state_dict'] is not None:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            # Load epsilon but ensure it's within bounds
            self.epsilon = max(
                min(checkpoint['epsilon'], self.config["epsilon_start"]), 
                self.config["epsilon_end"]
            )
            
            # Load loss history if available
            if 'loss_history' in checkpoint:
                self.loss_history = checkpoint['loss_history']
            
            logger.info(f"Model loaded successfully from {filename}")
            return True
        except Exception as e:
            logger.error(f"Error loading model components: {e}")
            return False

# Improved training function with monitoring, early stopping, and auto-tuning
def train(config, model_dir="models"):
    """
    Train the agent with improved monitoring and stability features.
    
    Args:
        config: Configuration dict
        model_dir: Directory to save models
    
    Returns:
        Dictionary of training history and metrics
    """
    # Create model directory if not exists
    try:
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
    except Exception as e:
        logger.error(f"Failed to create model directory: {e}")
        # Fallback to current directory
        model_dir = "."
    
    try:
        # Initialize environment
        env = TrafficSimulation(
            grid_size=config["grid_size"],
            max_cars=config["max_cars"],
            green_duration=config["green_duration"],
            yellow_duration=config["yellow_duration"],
            visualization=config["visualization"],
            random_seed=RANDOM_SEED
        )
        
        # Get state and action sizes
        state_size = env.observation_space.shape[0] * env.observation_space.shape[1]
        action_size = env.action_space.n
        
        # Initialize agent
        agent = DQNAgent(state_size, action_size, config)
        
        # Initialize training metrics
        metrics = {
            "rewards": [],
            "avg_rewards": [],
            "eval_rewards": [],
            "loss_values": [],
            "epsilon_values": [],
            "learning_rates": [],
            "waiting_times": [],
            "throughput": [],
            "training_time": 0
        }
        
        # Initialize early stopping variables
        best_eval_reward = -float('inf')
        patience = config.get("early_stopping_patience", 100)
        patience_counter = 0
        
        # Adaptive learning rate variables
        lr_scheduler_enabled = config.get("use_lr_scheduler", False)
        
        # Initialize dynamic traffic pattern
        current_pattern = "uniform"  # Start with uniform pattern
        pattern_schedule = {
            0: "uniform",         # Start with uniform
            100: "rush_hour",     # Switch to rush hour after 100 episodes
            200: "weekend",       # Switch to weekend after 200 episodes
            300: "uniform"        # Back to uniform after 300 episodes
        }
        
        # Training progress tracking
        progress_bar = None
        try:
            from tqdm import tqdm
            progress_bar = tqdm(total=config["num_episodes"], desc="Training Progress", ncols=100)
        except (ImportError, Exception) as e:
            logger.warning(f"Could not initialize progress bar: {e}")
        
        # Record training start time
        start_time = time.time()
        
        # Training loop
        for episode in range(1, config["num_episodes"] + 1):
            # Check if we need to switch traffic pattern
            if episode in pattern_schedule:
                current_pattern = pattern_schedule[episode]
                pattern_config = config["traffic_patterns"].get(current_pattern, config["traffic_patterns"]["uniform"])
                logger.info(f"Switching to {current_pattern} traffic pattern at episode {episode}")
                env.traffic_pattern = current_pattern
                env.traffic_config = pattern_config
            
            # Reset environment
            state, _ = env.reset()
            state = state.flatten()  # Flatten for NN input
            
            # Initialize episode variables
            total_reward = 0
            episode_steps = 0
            waiting_time = 0
            throughput = 0
            
            # Episode loop
            for step in range(config["max_steps"]):
                # Select action
                action = agent.act(state)
                
                # Take action in environment
                next_state, reward, terminated, truncated, info = env.step(action)
                next_state = next_state.flatten()  # Flatten for NN input
                
                # Apply reward clipping if enabled
                if config.get("clip_rewards", False):
                    reward = np.clip(reward, -10.0, 10.0)
                
                # Apply reward scaling if specified
                if "reward_scale" in config:
                    reward *= config["reward_scale"]
                
                # Store experience
                agent.step(state, action, reward, next_state, terminated)
                
                # Update state and stats
                state = next_state
                total_reward += reward
                episode_steps += 1
                waiting_time += info.get('average_waiting_time', 0)
                throughput += info.get('total_cars_passed', 0)
                
                # Check if episode is done
                if terminated or truncated:
                    break
            
            # Store rewards and compute averages
            metrics["rewards"].append(total_reward)
            
            # Calculate average reward over last 100 episodes (or fewer if we don't have 100 yet)
            window_size = min(100, len(metrics["rewards"]))
            avg_reward = np.mean(metrics["rewards"][-window_size:])
            metrics["avg_rewards"].append(avg_reward)
            
            # Calculate average waiting time and throughput for this episode
            avg_waiting_time = waiting_time / episode_steps if episode_steps > 0 else 0
            avg_throughput = throughput / episode_steps if episode_steps > 0 else 0
            metrics["waiting_times"].append(avg_waiting_time)
            metrics["throughput"].append(avg_throughput)
            
            # Record epsilon and learning rate
            metrics["epsilon_values"].append(agent.epsilon)
            current_lr = agent.optimizer.param_groups[0]['lr']
            metrics["learning_rates"].append(current_lr)
            
            # Record loss values if available
            if hasattr(agent, 'loss_history') and agent.loss_history:
                # Get average loss over this episode
                metrics["loss_values"].append(np.mean(agent.loss_history[-episode_steps:]) if episode_steps > 0 else 0)
            
            # Log progress
            logger.info(f"Episode {episode}/{config['num_episodes']} - "
                       f"Reward: {total_reward:.2f}, Avg Reward: {avg_reward:.2f}, "
                       f"Epsilon: {agent.epsilon:.4f}, LR: {current_lr:.6f}, "
                       f"Traffic: {current_pattern}")
            
            # Update progress bar
            if progress_bar is not None:
                progress_bar.update(1)
                progress_bar.set_postfix({
                    'reward': f"{total_reward:.2f}",
                    'avg': f"{avg_reward:.2f}",
                    'eps': f"{agent.epsilon:.2f}",
                    'pattern': current_pattern
                })
            
            # Evaluate the agent periodically
            if episode % config["eval_frequency"] == 0:
                logger.info(f"Evaluating agent at episode {episode}...")
                eval_reward = evaluate(agent, env, num_episodes=5)
                metrics["eval_rewards"].append(eval_reward)
                logger.info(f"Evaluation - Avg Reward: {eval_reward:.2f}")
                
                # Check for improvement and save model if improved
                if eval_reward > best_eval_reward:
                    best_eval_reward = eval_reward
                    patience_counter = 0
                    try:
                        model_path = os.path.join(model_dir, "best_model.pth")
                        agent.save(model_path)
                        logger.info(f"New best model saved with reward: {best_eval_reward:.2f}")
                    except Exception as e:
                        logger.error(f"Failed to save best model: {e}")
                else:
                    patience_counter += 1
                    logger.info(f"No improvement for {patience_counter} evaluations")
                    
                    # Apply early stopping if patience is exceeded
                    if patience_counter >= patience:
                        logger.info(f"Early stopping triggered after {patience} evaluations without improvement")
                        break
            
            # Save model periodically
            if episode % config["save_frequency"] == 0:
                try:
                    model_path = os.path.join(model_dir, f"model_episode_{episode}.pth")
                    agent.save(model_path)
                    logger.info(f"Model checkpoint saved at episode {episode}")
                except Exception as e:
                    logger.error(f"Failed to save model checkpoint: {e}")
            
            # Early stopping if we've reached a good performance
            if avg_reward > config.get("early_stopping_reward", float('inf')):
                logger.info(f"Early stopping at episode {episode} - Reached target performance")
                break
        
        # Close progress bar
        if progress_bar is not None:
            progress_bar.close()
        
        # Save final model
        try:
            model_path = os.path.join(model_dir, "final_model.pth")
            agent.save(model_path)
            logger.info("Final model saved successfully")
        except Exception as e:
            logger.error(f"Failed to save final model: {e}")
        
        # Record total training time
        metrics["training_time"] = time.time() - start_time
        logger.info(f"Total training time: {metrics['training_time']:.2f} seconds")
        
        # Close environment
        env.close()
        
        # Return metrics
        return metrics
    
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            "rewards": [],
            "avg_rewards": [],
            "eval_rewards": [],
            "error": str(e)
        }

# Evaluation function
def evaluate(agent, env, num_episodes=10):
    """
    Evaluate the agent without exploration.
    
    Args:
        agent: The agent to evaluate
        env: The environment
        num_episodes: Number of episodes to evaluate
    
    Returns:
        Average reward over episodes
    """
    rewards = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        state = state.flatten()  # Flatten for NN input
        total_reward = 0
        
        for step in range(1000):  # Max steps
            action = agent.act(state, eval_mode=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = next_state.flatten()  # Flatten for NN input
            
            state = next_state
            total_reward += reward
            
            if terminated or truncated:
                break
        
        rewards.append(total_reward)
    
    return np.mean(rewards)

# Visualization function
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

# Comparative analysis
def comparative_analysis(env, agents, labels, num_episodes=10):
    """
    Compare different agents on the same environment.
    
    Args:
        env: The environment
        agents: List of agents to compare
        labels: Labels for each agent
        num_episodes: Number of episodes to evaluate
    
    Returns:
        Dictionary of results
    """
    results = {label: [] for label in labels}
    
    try:
        for i, agent in enumerate(agents):
            label = labels[i]
            logger.info(f"Evaluating agent: {label}")
            
            # Evaluate agent
            for episode in range(num_episodes):
                state, _ = env.reset()
                state = state.flatten()
                total_reward = 0
                
                for step in range(1000):  # Max steps
                    action = agent.act(state, eval_mode=True)
                    next_state, reward, terminated, truncated, info = env.step(action)
                    next_state = next_state.flatten()
                    
                    state = next_state
                    total_reward += reward
                    
                    if terminated or truncated:
                        break
                
                # Create result entry
                entry = {
                    'reward': total_reward,
                }
                
                # Add additional metrics if available
                if 'average_waiting_time' in info:
                    entry['avg_waiting_time'] = info['average_waiting_time']
                if 'total_cars_passed' in info:
                    entry['total_cars_passed'] = info['total_cars_passed']
                if 'traffic_density' in info:
                    entry['traffic_density'] = info['traffic_density']
                
                results[label].append(entry)
                
                logger.info(f"Episode {episode+1}/{num_episodes} - Reward: {total_reward:.2f}")
        
        # Calculate summary statistics
        summary = {}
        for label in labels:
            summary[label] = {
                'avg_reward': np.mean([r['reward'] for r in results[label]]),
                'std_reward': np.std([r['reward'] for r in results[label]]),
                'min_reward': min([r['reward'] for r in results[label]]),
                'max_reward': max([r['reward'] for r in results[label]]),
            }
        
        results['summary'] = summary
        
    except Exception as e:
        logger.error(f"Error in comparative analysis: {e}")
    
    return results

# Main function
def main():
    """Main function."""
    TrafficSimulation.enhanced_save_visualization = enhanced_save_visualization
    TrafficSimulation.save_visualization = save_visualization

    parser = argparse.ArgumentParser(description='Traffic Light Control with RL')
    parser.add_argument('--mode', type=str, default='train', 
                        choices=['train', 'test', 'visualize', 'benchmark', 'record', 'analyze'],
                        help='Mode to run (train, test, visualize, benchmark, record, analyze)')
    parser.add_argument('--model', type=str, default='models/best_model.pth',
                        help='Path to model file for test/visualize/analyze modes')
    parser.add_argument('--config', type=str, default='config.json',
                        help='Path to configuration file')
    parser.add_argument('--episodes', type=int, default=None,
                        help='Number of episodes (overrides config)')
    parser.add_argument('--visualize', action='store_true',
                        help='Enable visualization')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed')
    parser.add_argument('--output', type=str, default='results',
                        help='Output directory')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging')
    parser.add_argument('--record-video', type=str, default=None,
                        help='Record a video of the environment (in record mode)')
    parser.add_argument('--video-duration', type=int, default=30,
                        help='Duration of recorded video in seconds')
    parser.add_argument('--traffic-pattern', type=str, default='uniform',
                        choices=['uniform', 'rush_hour', 'weekend', 'random'],
                        help='Traffic pattern to use for testing and visualization')
    
    args = parser.parse_args()
    
    # Enable debug logging if requested
    if args.debug:
        enable_debug_logging()
        logger.debug("Debug mode enabled")
    
    # Create output directory
    try:
        os.makedirs(args.output, exist_ok=True)
    except Exception as e:
        logger.warning(f"Could not create output directory: {e}")
    
    # Set random seed if provided
    if args.seed is not None:
        global RANDOM_SEED
        RANDOM_SEED = args.seed
        np.random.seed(RANDOM_SEED)
        torch.manual_seed(RANDOM_SEED)
        random.seed(RANDOM_SEED)
        logger.info(f"Using random seed: {RANDOM_SEED}")
    
    # Load configuration
    try:
        if os.path.exists(args.config):
            with open(args.config, 'r') as f:
                loaded_config = json.load(f)
                # Update default config with loaded values
                CONFIG.update(loaded_config)
                logger.info(f"Loaded configuration from {args.config}")
        else:
            logger.warning(f"Configuration file {args.config} not found, using defaults")
            # Save default config for reference
            with open(os.path.join(args.output, 'default_config.json'), 'w') as f:
                json.dump(CONFIG, f, indent=4)
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        logger.info("Continuing with default configuration")
    
    # Override config with command line arguments
    if args.episodes:
        CONFIG["num_episodes"] = args.episodes
    
    if args.visualize:
        CONFIG["visualization"] = True
    
    # Log configuration
    logger.info(f"Running with configuration: {CONFIG}")
    
    try:
        if args.mode == 'train':
            # Train agent
            logger.info("Starting training...")
            metrics = train(CONFIG, model_dir=os.path.join(args.output, 'models'))
            
            # Visualize results
            logger.info("Training complete, visualizing results...")
            visualize_results(metrics["rewards"], metrics["avg_rewards"], 
                            save_path=os.path.join(args.output, "training_progress.png"))
            
            # Save results as csv
            try:
                results_path = os.path.join(args.output, "training_results.csv")
                with open(results_path, 'w') as f:
                    f.write("episode,reward,avg_reward")
                    if "loss_values" in metrics and metrics["loss_values"]:
                        f.write(",loss")
                    if "epsilon_values" in metrics:
                        f.write(",epsilon")
                    if "learning_rates" in metrics:
                        f.write(",learning_rate")
                    f.write("\n")
                    
                    for i in range(len(metrics["rewards"])):
                        line = f"{i},{metrics['rewards'][i]},{metrics['avg_rewards'][i]}"
                        if "loss_values" in metrics and i < len(metrics["loss_values"]):
                            line += f",{metrics['loss_values'][i]}"
                        if "epsilon_values" in metrics and i < len(metrics["epsilon_values"]):
                            line += f",{metrics['epsilon_values'][i]}"
                        if "learning_rates" in metrics and i < len(metrics["learning_rates"]):
                            line += f",{metrics['learning_rates'][i]}"
                        f.write(line + "\n")
                logger.info(f"Training results saved to {results_path}")
            except Exception as e:
                logger.error(f"Failed to save training results: {e}")
        
        elif args.mode == 'test':
            # Initialize environment
            logger.info("Initializing environment for testing...")
            
            # Configure traffic pattern
            traffic_config = CONFIG["traffic_patterns"].get(args.traffic_pattern, CONFIG["traffic_patterns"]["uniform"])
            logger.info(f"Using traffic pattern: {args.traffic_pattern}")
            
            env = TrafficSimulation(
                grid_size=CONFIG["grid_size"],
                max_cars=CONFIG["max_cars"],
                green_duration=CONFIG["green_duration"],
                yellow_duration=CONFIG["yellow_duration"],
                visualization=CONFIG["visualization"],
                random_seed=RANDOM_SEED
            )
            
            # Set the traffic pattern
            env.traffic_pattern = args.traffic_pattern
            env.traffic_config = traffic_config
            
            # Get state and action sizes
            state_size = env.observation_space.shape[0] * env.observation_space.shape[1]
            action_size = env.action_space.n
            
            # Initialize agent
            agent = DQNAgent(state_size, action_size, CONFIG)
            
            # Load model
            logger.info(f"Loading model from {args.model}...")
            agent.load(args.model)
            
            # Evaluate agent
            logger.info("Evaluating agent...")
            reward = evaluate(agent, env, num_episodes=10)
            logger.info(f"Evaluation - Avg Reward: {reward:.2f}")
            
            # Save evaluation results
            try:
                with open(os.path.join(args.output, "evaluation_results.json"), 'w') as f:
                    json.dump({
                        "average_reward": float(reward),
                        "traffic_pattern": args.traffic_pattern,
                        "model_path": args.model
                    }, f, indent=4)
            except Exception as e:
                logger.error(f"Failed to save evaluation results: {e}")
            
            # Close environment
            env.close()
        
        elif args.mode == 'visualize':
            # Initialize environment with visualization
            logger.info("Initializing environment for visualization...")
            CONFIG["visualization"] = True
            
            # Configure traffic pattern
            traffic_config = CONFIG["traffic_patterns"].get(args.traffic_pattern, CONFIG["traffic_patterns"]["uniform"])
            logger.info(f"Using traffic pattern: {args.traffic_pattern}")
            
            try:
                env = TrafficSimulation(
                    grid_size=CONFIG["grid_size"],
                    max_cars=CONFIG["max_cars"],
                    green_duration=CONFIG["green_duration"],
                    yellow_duration=CONFIG["yellow_duration"],
                    visualization=True,
                    random_seed=RANDOM_SEED
                )
                
                # Set the traffic pattern
                env.traffic_pattern = args.traffic_pattern
                env.traffic_config = traffic_config
                
                # Get state and action sizes
                state_size = env.observation_space.shape[0] * env.observation_space.shape[1]
                action_size = env.action_space.n
                
                # Initialize agent
                agent = DQNAgent(state_size, action_size, CONFIG)
                
                # Load model
                logger.info(f"Loading model from {args.model}...")
                agent.load(args.model)
                
                # Run visualization
                logger.info("Starting visualization...")
                state, _ = env.reset()
                state = state.flatten()
                
                total_reward = 0
                for step in range(1000):
                    action = agent.act(state, eval_mode=True)
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    next_state = next_state.flatten()
                    
                    state = next_state
                    total_reward += reward
                    
                    # Log status periodically
                    if step % 100 == 0:
                        logger.info(f"Step {step}/1000 - Total reward: {total_reward:.2f}")
                    
                    # Add delay for visualization
                    try:
                        time.sleep(0.1)
                    except KeyboardInterrupt:
                        logger.info("Visualization interrupted by user")
                        break
                    
                    if terminated or truncated:
                        break
                
                logger.info(f"Visualization complete - Total reward: {total_reward:.2f}")
                
                # Close environment
                env.close()
            except Exception as e:
                logger.error(f"Visualization failed: {e}")
        
        elif args.mode == 'benchmark':
            # Benchmark different configurations
            logger.info("Running benchmark mode...")
            
            # Create baseline agent (fixed timing)
            class FixedTimingAgent:
                def __init__(self, action_size):
                    self.action_size = action_size
                    self.current_phase = 0
                    self.phase_duration = 30  # Fixed phase duration
                    self.timer = 0
                
                def act(self, state, eval_mode=False):
                    # Change phase every phase_duration steps
                    if self.timer >= self.phase_duration:
                        self.current_phase = (self.current_phase + 1) % self.action_size
                        self.timer = 0
                    
                    self.timer += 1
                    return self.current_phase
            
            # Initialize environment
            env = TrafficSimulation(
                grid_size=CONFIG["grid_size"],
                max_cars=CONFIG["max_cars"],
                green_duration=CONFIG["green_duration"],
                yellow_duration=CONFIG["yellow_duration"],
                visualization=False,
                random_seed=RANDOM_SEED
            )
            
            # Get state and action sizes
            state_size = env.observation_space.shape[0] * env.observation_space.shape[1]
            action_size = env.action_space.n
            
            # Initialize RL agent
            rl_agent = DQNAgent(state_size, action_size, CONFIG)
            
            # Load model if exists
            if os.path.exists(args.model):
                logger.info(f"Loading model from {args.model}...")
                rl_agent.load(args.model)
            else:
                logger.warning(f"Model {args.model} not found, using untrained agent")
            
            # Initialize fixed timing agent
            fixed_agent = FixedTimingAgent(action_size)
            
            # Benchmark agents
            agents = [rl_agent, fixed_agent]
            labels = ["RL Agent", "Fixed Timing"]
            
            logger.info("Starting comparative analysis...")
            results = comparative_analysis(env, agents, labels, num_episodes=20)
            
            # Save benchmark results
            try:
                with open(os.path.join(args.output, "benchmark_results.json"), 'w') as f:
                    # Convert numpy values to native Python types for JSON serialization
                    clean_results = {}
                    for k, v in results.items():
                        if k == 'summary':
                            clean_summary = {}
                            for agent, metrics in v.items():
                                clean_metrics = {mk: float(mv) for mk, mv in metrics.items()}
                                clean_summary[agent] = clean_metrics
                            clean_results[k] = clean_summary
                        else:
                            clean_results[k] = v
                    
                    json.dump(clean_results, f, indent=4)
                logger.info(f"Benchmark results saved to {os.path.join(args.output, 'benchmark_results.json')}")
            except Exception as e:
                logger.error(f"Failed to save benchmark results: {e}")
            
            # Log summary results
            logger.info("Benchmark Summary:")
            for label, metrics in results['summary'].items():
                logger.info(f"  {label}:")
                for metric, value in metrics.items():
                    logger.info(f"    {metric}: {value:.4f}")
            
            # Close environment
            env.close()
        
        elif args.mode == 'record':
            # Video recording mode - create a video of the environment
            logger.info("Initializing environment for video recording...")
            
            try:
                # Configure environment based on selected traffic pattern
                traffic_config = CONFIG["traffic_patterns"].get(args.traffic_pattern, CONFIG["traffic_patterns"]["uniform"])
                logger.info(f"Using traffic pattern: {args.traffic_pattern}")
                
                # Initialize the environment
                env = TrafficSimulation(
                    grid_size=CONFIG["grid_size"],
                    max_cars=CONFIG["max_cars"],
                    green_duration=CONFIG["green_duration"],
                    yellow_duration=CONFIG["yellow_duration"],
                    visualization=False,  # Don't need pygame visualization for our enhanced version
                    random_seed=RANDOM_SEED
                )
                
                # Set the traffic pattern for the environment
                env.traffic_pattern = args.traffic_pattern
                env.traffic_config = traffic_config
                
                # Create output directory if it doesn't exist
                os.makedirs(args.output, exist_ok=True)
                
                # Determine the video filename
                video_file = args.record_video if args.record_video else f"{args.output}/traffic_simulation_{args.traffic_pattern}.mp4"
                
                # Create a video with either random or trained agent actions
                if os.path.exists(args.model):
                    logger.info(f"Loading model from {args.model} for recording...")
                    
                    # Get state and action sizes
                    state_size = env.observation_space.shape[0] * env.observation_space.shape[1]
                    action_size = env.action_space.n
                    
                    # Initialize agent
                    agent = DQNAgent(state_size, action_size, CONFIG)
                    
                    # Load model
                    if agent.load(args.model):
                        logger.info(f"Recording video with trained agent...")
                        # Add agent as a parameter to the environment for video generation
                        env.recording_agent = agent
                    else:
                        logger.warning(f"Failed to load model, will use random actions instead")
                        env.recording_agent = None
                else:
                    logger.info(f"No model specified, recording video with random actions...")
                    env.recording_agent = None
                
                # Save the video with enhanced visualization
                duration = args.video_duration if args.video_duration else 30
                success = save_visualization(
                    env,
                    filename=video_file,
                    fps=30,
                    duration=duration
                )
                
                if success:
                    logger.info(f"Enhanced video saved to {video_file}")
                else:
                    logger.error(f"Failed to create video")
                
                # Close the environment
                env.close()
                
            except Exception as e:
                logger.error(f"Error recording video: {e}")
                import traceback
                logger.error(traceback.format_exc())
                
        elif args.mode == 'analyze':
            # Analysis mode - analyze agent behavior and performance
            logger.info("Starting analysis mode...")
            
            try:
                # Create output directory if it doesn't exist
                os.makedirs(args.output, exist_ok=True)
                
                # Initialize environment
                env = TrafficSimulation(
                    grid_size=CONFIG["grid_size"],
                    max_cars=CONFIG["max_cars"],
                    green_duration=CONFIG["green_duration"],
                    yellow_duration=CONFIG["yellow_duration"],
                    visualization=True,
                    random_seed=RANDOM_SEED
                )
                
                # Get state and action sizes
                state_size = env.observation_space.shape[0] * env.observation_space.shape[1]
                action_size = env.action_space.n
                
                # Initialize agent
                agent = DQNAgent(state_size, action_size, CONFIG)
                
                # Initialize analysis results
                analysis_results = {
                    "model_path": args.model,
                    "config": CONFIG,
                    "state_action_values": {},
                    "performance_metrics": {},
                    "decision_boundaries": {}
                }
                
                # Load model
                if not os.path.exists(args.model):
                    logger.error(f"Model file {args.model} not found")
                    return
                
                logger.info(f"Loading model from {args.model}...")
                agent.load(args.model)
                
                # Analyze trained policy
                logger.info("Analyzing trained policy...")
                
                # 1. Generate a grid of sample states to analyze
                logger.info("Generating sample states for analysis...")
                sample_states = []
                
                # Create states with varying traffic densities
                density_values = [0.1, 0.3, 0.5, 0.7, 0.9]
                for ns_density in density_values:
                    for ew_density in density_values:
                        for light_state in [0, 1]:  # Current light state
                            # Create a state with uniform density across all intersections
                            state = np.zeros((env.num_intersections, 3), dtype=np.float32)
                            state[:, 0] = ns_density  # NS density
                            state[:, 1] = ew_density  # EW density
                            state[:, 2] = light_state  # Light state
                            
                            sample_states.append((state, f"NS={ns_density:.1f},EW={ew_density:.1f},Light={light_state}"))
                
                # 2. Evaluate Q-values for each state
                logger.info("Evaluating Q-values for sample states...")
                q_values = {}
                
                for state, state_desc in sample_states:
                    # Flatten and convert state to tensor
                    state_tensor = torch.tensor(state.flatten(), dtype=torch.float32).to(agent.device)
                    
                    # Get Q-values from local network
                    agent.local_network.eval()
                    with torch.no_grad():
                        q_vals = agent.local_network(state_tensor).cpu().numpy()
                    
                    # Store Q-values and resulting actions
                    q_values[state_desc] = {
                        'q_values': q_vals.tolist(),
                        'action': int(np.argmax(q_vals)),
                        'q_diff': float(q_vals[1] - q_vals[0])  # Difference between actions
                    }
                
                analysis_results["state_action_values"] = q_values
                
                # 3. Test agent performance on different traffic patterns
                logger.info("Evaluating agent performance on different traffic patterns...")
                performance = {}
                
                for pattern in ["uniform", "rush_hour", "weekend"]:
                    if pattern in CONFIG["traffic_patterns"]:
                        logger.info(f"Testing on {pattern} traffic pattern...")
                        
                        # Set the traffic pattern
                        env.traffic_pattern = pattern
                        env.traffic_config = CONFIG["traffic_patterns"][pattern]
                        
                        # Run evaluation episodes
                        rewards = []
                        waiting_times = []
                        cars_passed = []
                        avg_densities = []
                        
                        for episode in range(10):  # Run 10 episodes per pattern
                            state, _ = env.reset()
                            state = state.flatten()
                            episode_reward = 0
                            episode_waiting = 0
                            episode_cars = 0
                            episode_density = []
                            
                            for step in range(100):  # Run 100 steps per episode
                                action = agent.act(state, eval_mode=True)
                                next_state, reward, terminated, truncated, info = env.step(action)
                                next_state = next_state.flatten()
                                
                                state = next_state
                                episode_reward += reward
                                episode_waiting += info['average_waiting_time']
                                episode_cars += info['total_cars_passed']
                                episode_density.append(info['traffic_density'])
                                
                                if terminated or truncated:
                                    break
                            
                            rewards.append(episode_reward)
                            waiting_times.append(episode_waiting / (step + 1))  # Average per step
                            cars_passed.append(episode_cars)
                            avg_densities.append(np.mean(episode_density))
                        
                        # Store results
                        performance[pattern] = {
                            'avg_reward': float(np.mean(rewards)),
                            'std_reward': float(np.std(rewards)),
                            'avg_waiting_time': float(np.mean(waiting_times)),
                            'avg_cars_passed': float(np.mean(cars_passed)),
                            'avg_density': float(np.mean(avg_densities))
                        }
                
                analysis_results["performance_metrics"] = performance
                
                # 4. Generate decision boundary data
                logger.info("Generating decision boundary analysis...")
                
                # Create a grid of NS vs EW densities
                ns_densities = np.linspace(0, 1, 20)
                ew_densities = np.linspace(0, 1, 20)
                
                # For each light state
                for light_state in [0, 1]:
                    decision_data = []
                    
                    for ns in ns_densities:
                        for ew in ew_densities:
                            # Create a simple state with one intersection
                            state = np.zeros((1, 3), dtype=np.float32)
                            state[0, 0] = ns  # NS density
                            state[0, 1] = ew  # EW density
                            state[0, 2] = light_state  # Light state
                            
                            # Get action from agent
                            flat_state = state.flatten()
                            action = agent.act(flat_state, eval_mode=True)
                            
                            # Store result
                            decision_data.append({
                                'ns_density': float(ns),
                                'ew_density': float(ew),
                                'action': int(action)
                            })
                    
                    analysis_results["decision_boundaries"][f"light_{light_state}"] = decision_data
                
                # 5. Save analysis results
                logger.info("Saving analysis results...")
                results_file = f"{args.output}/agent_analysis.json"
                
                with open(results_file, 'w') as f:
                    json.dump(analysis_results, f, indent=2)
                
                logger.info(f"Analysis results saved to {results_file}")
                
                # 6. Generate policy visualization
                try:
                    # Create NS vs EW density heatmaps for decision boundaries
                    logger.info("Generating policy visualization...")
                    import matplotlib.pyplot as plt
                    from matplotlib.colors import ListedColormap
                    
                    # Set up figure with subplots for each light state
                    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
                    
                    # Custom colormap for the two actions
                    cmap = ListedColormap(['green', 'red'])
                    
                    for i, light_state in enumerate([0, 1]):
                        # Extract data for this light state
                        data = analysis_results["decision_boundaries"][f"light_{light_state}"]
                        
                        # Reshape data into a grid
                        grid_size = int(np.sqrt(len(data)))
                        action_grid = np.zeros((grid_size, grid_size))
                        
                        for idx, point in enumerate(data):
                            row = idx // grid_size
                            col = idx % grid_size
                            action_grid[row, col] = point['action']
                        
                        # Plot heatmap
                        im = axes[i].imshow(
                            action_grid, 
                            origin='lower', 
                            cmap=cmap,
                            extent=[0, 1, 0, 1],
                            aspect='auto'
                        )
                        
                        # Add labels and title
                        axes[i].set_xlabel('East-West Density')
                        axes[i].set_ylabel('North-South Density')
                        axes[i].set_title(f'Current Light State: {"NS Green" if light_state == 0 else "EW Green"}')
                        
                        # Add grid
                        axes[i].grid(color='black', linestyle='--', linewidth=0.5, alpha=0.3)
                    
                    # Add colorbar with labels
                    cbar = fig.colorbar(im, ax=axes, ticks=[0.25, 0.75])
                    cbar.ax.set_yticklabels(['NS Green', 'EW Green'])
                    
                    # Add overall title
                    fig.suptitle('Traffic Light Decision Policy (Action by NS vs EW Density)', fontsize=16)
                    
                    # Adjust layout
                    plt.tight_layout(rect=[0, 0, 1, 0.95])
                    
                    # Save figure
                    policy_file = f"{args.output}/policy_visualization.png"
                    plt.savefig(policy_file, dpi=150)
                    plt.close()
                    
                    logger.info(f"Policy visualization saved to {policy_file}")
                    
                except Exception as e:
                    logger.error(f"Failed to generate policy visualization: {e}")
                
                # Close environment
                env.close()
                
            except Exception as e:
                logger.error(f"Error in analysis mode: {e}")
                import traceback
                logger.error(traceback.format_exc())
    
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()