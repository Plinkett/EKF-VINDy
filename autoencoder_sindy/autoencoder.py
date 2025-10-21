import torch
import torch.nn as nn
from typing import List
from autoencoder_sindy.ae_utils import initializate_weights

"""
Autoencoder to be used with SINDy, unclear how to actually include the SINDy library here.

TODO: Does He work here? I think it must change a little bit since you are compressing? Does the game with the variance hold?
"""
class Encoder(nn.Module):
    """
    Encoder class, pretty standard.
    """
    
    def __init__(self, input_dim: int, latent_dim: int, 
                 init_scheme = 'he', activation = 'relu'):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, 16) # We will use the first 32 POD modes
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, latent_dim)
        self.init_scheme = init_scheme
        self.activation = activation

        self.apply(initializate_weights, init_scheme, activation)   # Initialize with He scheme

    def forward(self, x):
        """
        Forward pass with ReLU activations, for encoder. Leave last layer alone.
        Forward pass with chosen activation function.
        """
        if self.activation == 'relu':
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
        elif self.activation == 'tanh':
            x = torch.tanh(self.fc1(x))
            x = torch.tanh(self.fc2(x))
        else:
            raise ValueError("Activation function not supported: " + str(self.activation))
        x = self.fc3(x)
        
        return x
    
class Decoder(nn.Module):
    """
    Decoder class, pretty standard.
    """
    
    def __init__(self, input_dim: int, latent_dim: int, init_scheme = 'He'):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 8) # We will use the first 32 POD modes
        self.fc2 = nn.Linear(8, 16)
        self.fc3 = nn.Linear(16, input_dim)
        self.apply(initializate_weights, init_scheme)    # Initialize with He scheme
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        
class Autoencoder(nn.Module):
    def __init__(self, input_dim = 32, latent_dim = 2):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(input_dim, latent_dim)

