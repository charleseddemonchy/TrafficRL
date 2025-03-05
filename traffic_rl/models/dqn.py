"""
Deep Q-Network Models
====================
Neural network models for Deep Q-Learning.
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