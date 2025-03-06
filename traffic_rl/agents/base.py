"""
Base Agent
=========
Base class for all reinforcement learning agents.
"""

import numpy as np
import logging

logger = logging.getLogger("TrafficRL.Agents.Base")

class BaseAgent:
    """
    Base class for all reinforcement learning agents.
    
    This class defines the interface that all agent implementations must follow.
    """
    def __init__(self, state_size, action_size):
        """
        Initialize the base agent.
        
        Args:
            state_size: Dimensionality of the state space
            action_size: Dimensionality of the action space
        """
        self.state_size = state_size
        self.action_size = action_size
        logger.info(f"BaseAgent initialized with state_size={state_size}, action_size={action_size}")
        
    def act(self, state, eval_mode=False):
        """
        Choose an action based on the current state.
        
        Args:
            state: Current state
            eval_mode: If True, use evaluation (greedy) policy
        
        Returns:
            Selected action
        """
        raise NotImplementedError("Subclasses must implement abstract method")
    
    def step(self, state, action, reward, next_state, done):
        """
        Process a step of experience.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
        """
        pass  # Optional method, not all agents need to implement
    
    def learn(self, experiences):
        """
        Update the agent's knowledge based on experiences.
        
        Args:
            experiences: Batch of experience tuples
        """
        pass  # Optional method, not all agents need to implement
    
    def save(self, filepath):
        """
        Save the agent's model to a file.
        
        Args:
            filepath: Path to save the model
            
        Returns:
            Whether the save was successful
        """
        return False  # Default implementation does nothing
    
    def load(self, filepath):
        """
        Load the agent's model from a file.
        
        Args:
            filepath: Path to load the model from
            
        Returns:
            Whether the load was successful
        """
        return False  # Default implementation does nothing
    
    def reset(self):
        """
        Reset the agent's internal state.
        """
        pass  # Optional method, not all agents need to implement


class RandomAgent(BaseAgent):
    """
    Random agent that selects random actions.
    
    This can be used as a baseline for comparison.
    """
    def __init__(self, state_size, action_size, seed=None):
        """
        Initialize the random agent.
        
        Args:
            state_size: Dimensionality of the state space
            action_size: Dimensionality of the action space
            seed: Random seed for reproducibility
        """
        super(RandomAgent, self).__init__(state_size, action_size)
        self.np_random = np.random.RandomState(seed)
        logger.info(f"RandomAgent initialized with action space size: {action_size}")
    
    def act(self, state, eval_mode=False):
        """
        Choose a random action.
        
        Args:
            state: Current state (ignored)
            eval_mode: Evaluation mode flag (ignored)
        
        Returns:
            Random action
        """
        return self.np_random.randint(self.action_size)