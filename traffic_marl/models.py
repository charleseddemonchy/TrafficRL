"""
Neural network models for reinforcement learning.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger("TrafficRL.Models")

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