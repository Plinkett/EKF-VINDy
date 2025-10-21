import torch
import torch.nn as nn
import numpy as np
from typing import List
import autoencoder_sindy.ae_utils as util
"""
SINDy layer that provides the standard SINDy libraries, differentiable via PyTorch.

Do you actually need that layer? After all, should it not be inside the loss? Do you actually need to backpropagate through that?
No, you define this "fake" layer that is the Theta * Xi. So let's define that.

Can you recycle this in the loss? I think you can just shortcircuit this into the loss, can't you? No need to recompute it.

TODO: Add Fourier features as well...not straight forward to mix parameters (input formatting)
TODO: Have a config class to avoid passing too much stuff
"""

class SINDyLayer(nn.Module):

    def __init__(self, latent_dim: int, n_parameters: int, poly_order: int, parameter_names: List[str],
                 big_xi_initialization = 'uniform'):
        super(SINDyLayer, self).__init__()
        self.latent_dim = latent_dim
        self.n_parameters = n_parameters
        self.n_variables = self.latent_dim + self.n_parameters
        
        # Generate variable names, and lambda functions (for library terms)
        self.var_names = util.generate_variable_names(self.latent_dim, self.n_parameters, parameter_names)
        self.var_symbols, \
        self.library_symbols, \
        self.lambdified_library = util.generate_library_torch(self.var_names, poly_order)
        
        # sindy coeffs initialized with either one or uniformly, unlike network weights
        # network weights initialized with He scheme, or depending on whatever non-linearlity you are using
        
        # self.library_funcs = 
        # # Initialize SINDy coefficients (they are weights just like any other network weight)
        # # this will be reshaped into (n_latents + n_parameters , n_library_terms)
        # self.sindy_coefficients = 
    
    def _initialize_SINDy_coefficients(self):
        pass
    
    def evaluate_theta(self, z: torch.Tensor, betas: torch.Tensor):
        """
        Evaluate library terms with latents (and parameters, called "betas" here). 
        Both z and betas are column vectors. Output tensor of the same type and device as the input.
        """

        # We momentarily call "state" the concatenation of latent variables and parameters of the dynamical systme
        state = torch.cat([z, betas], dim = 0)
        
        # The theta vector is of size (p, 1)
        theta = [f(state) for f in self.lambdified_library] 
        theta = torch.tensor(theta, dtype = z.dtype, device = z.device).reshape(-1, 1)

        return theta
            
        
    def forward(z: torch.Tensor):
        pass


# For stupid tests
if __name__ == '__main__':
    # some test for the SINDy lib
    z = torch.tensor([2.5, 5.3, 6.1])
    sl = SINDyLayer(latent_dim=3, n_parameters=2, poly_order=3, parameter_names=None)
    sl.evaluate_theta(z, torch.tensor([0.4, 2.2]))
