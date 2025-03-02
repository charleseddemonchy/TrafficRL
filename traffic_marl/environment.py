"""
Custom Gym environment for traffic simulation.
"""
import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame
import time
import logging

logger = logging.getLogger("TrafficRL.Environment")

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
        
        # Initialize visualization if enabled
        if self.visualization:
            try:
                self._init_visualization()
            except Exception as e:
                logger.warning(f"Could not initialize visualization: {e}")
                self.visualization = False
        
        # Reset the environment
        self.reset()
    
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
                morning_peak = 0.33
                evening_peak = 0.71
                peak_intensity = 2.0
                base_arrival = 0.03
                
                rush_hour_factor = peak_intensity * (
                    np.exp(-20 * (time_of_day - morning_peak)**2) +  # Morning peak
                    np.exp(-20 * (time_of_day - evening_peak)**2)    # Evening peak
                )
            elif self.traffic_pattern == "weekend":
                # Weekend pattern: one peak around noon
                midday_peak = 0.5
                peak_intensity = 1.5
                base_arrival = 0.02
                
                rush_hour_factor = peak_intensity * np.exp(-10 * (time_of_day - midday_peak)**2)
            else:  # uniform pattern
                base_arrival = 0.03
                variability = 0.01
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
            
            total_reward = waiting_penalty + throughput_reward + fairness_penalty + switching_penalty
            
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

    ##Change save_visualization
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
        try:
            import cv2
            if not self.visualization:
                self._init_visualization()
            
            # Set up video writer
            frame_size = (self.screen_width, self.screen_height)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(filename, fourcc, fps, frame_size)
            
            # Reset environment
            state, _ = self.reset()
            total_frames = int(fps * duration)
            
            # Recording loop
            for frame in range(total_frames):
                # Render frame
                rgb_frame = self.render(mode='rgb_array')
                if rgb_frame is None:
                    logger.error("Failed to render frame")
                    out.release()
                    return False
                    
                # Convert RGB to BGR for OpenCV
                bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
                out.write(bgr_frame)
                
                # Take action (use recording_agent if available, else random)
                if hasattr(self, 'recording_agent') and self.recording_agent:
                    action = self.recording_agent.act(state.flatten(), eval_mode=True)
                else:
                    action = self.action_space.sample()
                    
                # Step environment
                state, _, terminated, truncated, _ = self.step(action)
                state = state.flatten()
                
                if terminated or truncated:
                    state, _ = self.reset()
            
            # Cleanup
            out.release()
            logger.info(f"Successfully recorded {total_frames} frames")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save visualization: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False