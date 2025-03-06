"""
Fixed Timing Agent
================
Baseline agent using fixed-timing traffic light control.
"""

import numpy as np
import logging

logger = logging.getLogger("TrafficRL.Agents.FixedTiming")

class FixedTimingAgent:
    """
    Fixed timing agent for traffic light control.
    
    This is a simple baseline agent that changes traffic light phases
    based on a fixed schedule, regardless of traffic conditions.
    """
    def __init__(self, action_size, phase_duration=30):
        """
        Initialize the fixed timing agent.
        
        Args:
            action_size: Size of the action space
            phase_duration: Duration of each phase in time steps
        """
        self.action_size = action_size
        self.phase_duration = phase_duration
        self.current_phase = 0
        self.timer = 0
        
        logger.info(f"Fixed timing agent initialized with phase duration: {phase_duration}")
    
    def act(self, state, eval_mode=False):
        """
        Choose an action based on fixed timing.
        
        Args:
            state: Current state (ignored, as this is fixed timing)
            eval_mode: Evaluation mode flag (ignored)
        
        Returns:
            Selected action
        """
        # Change phase every phase_duration steps
        if self.timer >= self.phase_duration:
            self.current_phase = (self.current_phase + 1) % self.action_size
            self.timer = 0
        
        self.timer += 1
        return self.current_phase
    
    def reset(self):
        """Reset the agent to initial state."""
        self.current_phase = 0
        self.timer = 0


class AdaptiveTimingAgent:
    """
    Adaptive timing agent for traffic light control.
    
    This agent adjusts green phase duration based on traffic density,
    but still follows a fixed cycle for phase order.
    """
    def __init__(self, action_size, min_phase_duration=10, max_phase_duration=60):
        """
        Initialize the adaptive timing agent.
        
        Args:
            action_size: Size of the action space
            min_phase_duration: Minimum duration of each phase in time steps
            max_phase_duration: Maximum duration of each phase in time steps
        """
        self.action_size = action_size
        self.min_phase_duration = min_phase_duration
        self.max_phase_duration = max_phase_duration
        self.current_phase = 0
        self.timer = 0
        self.current_duration = min_phase_duration
        
        logger.info(f"Adaptive timing agent initialized with phase duration range: "
                   f"{min_phase_duration}-{max_phase_duration}")
    
    def act(self, state, eval_mode=False):
        """
        Choose an action based on adaptive timing.
        
        Args:
            state: Current state (traffic densities)
            eval_mode: Evaluation mode flag (ignored)
        
        Returns:
            Selected action
        """
        # Check if timer has expired
        if self.timer >= self.current_duration:
            # Change to next phase
            self.current_phase = (self.current_phase + 1) % self.action_size
            self.timer = 0
            
            # Calculate new phase duration based on traffic density
            try:
                # Reshape state to extract traffic densities
                # Assuming state format matches environment observation
                state_array = np.array(state)
                
                # If the state is flattened, reshape it
                if state_array.ndim == 1:
                    # Estimate number of intersections from state length
                    # State format per intersection is [NS_density, EW_density, light_state, ...]
                    num_features_per_intersection = 5  # Typical value in our environment
                    num_intersections = len(state_array) // num_features_per_intersection
                    
                    # Reshape to [num_intersections, features]
                    state_array = state_array.reshape(num_intersections, -1)
                
                # Calculate average density for current direction
                if self.current_phase == 0:  # NS Green phase
                    avg_density = np.mean(state_array[:, 0])  # NS density
                else:  # EW Green phase
                    avg_density = np.mean(state_array[:, 1])  # EW density
                
                # Scale duration based on density: higher density = longer duration
                # Map density [0,1] to duration [min,max]
                self.current_duration = int(
                    self.min_phase_duration + 
                    (self.max_phase_duration - self.min_phase_duration) * avg_density
                )
                
                # Ensure duration is within bounds
                self.current_duration = max(self.min_phase_duration, 
                                           min(self.max_phase_duration, self.current_duration))
                
            except Exception as e:
                logger.warning(f"Error calculating adaptive duration: {e}. Using default.")
                self.current_duration = self.min_phase_duration
        
        self.timer += 1
        return self.current_phase
    
    def reset(self):
        """Reset the agent to initial state."""
        self.current_phase = 0
        self.timer = 0
        self.current_duration = self.min_phase_duration