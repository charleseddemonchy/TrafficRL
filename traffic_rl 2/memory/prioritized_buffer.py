"""
Prioritized Experience Replay Buffer
==================================
Prioritized replay memory for storing and sampling experiences based on TD-error.
"""

import numpy as np
import torch
import random
from collections import namedtuple, deque
import logging

logger = logging.getLogger("TrafficRL.Memory")

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