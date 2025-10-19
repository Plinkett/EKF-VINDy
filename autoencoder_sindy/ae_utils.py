import torch
import torch.nn as nn

"""
Utility class for autoencoders
"""

def initializate_weights(module: nn.Module):
    """
    Just use He initialization, since we will use ReLU activations.
    To avoid breaking things in the future, we only initialize the linear modules, biases to zero.
    """
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, nonlinearity='relu')    
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)