"""
Reinforcement learning agents for traffic light control.
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import random
import logging

# Import models
from .models import DQN, DuelingDQN

logger = logging.getLogger("TrafficRL.Agents")

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
            
            return (states, actions, rewards, next_states, dones)
        
        except Exception as e:
            logger.error(f"Error sampling from replay buffer: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def __len__(self):
        """Return the current size of the buffer."""
        return len(self.buffer)

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

class FixedTimingAgent:
    """Agent with fixed timing strategy for traffic lights."""
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