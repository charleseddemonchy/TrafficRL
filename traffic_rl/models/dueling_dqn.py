"""
Dueling Deep Q-Network Model
===========================
Dueling architecture for improved Deep Q-Learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger("TrafficRL.Models")

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