import torch
import torch.nn as nn
import numpy as np
from typing import List
import aesindy.utils as util
"""
SINDy layer that provides the standard SINDy libraries, differentiable via PyTorch.

Do you actually need that layer? After all, should it not be inside the loss? Do you actually need to backpropagate through that?
No, you define this "fake" layer that is the Theta * Xi. So let's define that.

Can you recycle this in the loss? I think you can just shortcircuit this into the loss, can't you? No need to recompute it.
We don't actually optimize with PySINDy, we just use PySINDy to get the library terms... 

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
    
    def evaluate_theta(self, z: torch.Tensor, betas: torch.Tensor | None = None):
        """
        Evaluate library terms with latents (and parameters, called "betas" here). We assume the batch size to be the same for both z and betas (no input verification here).
        A single theta vector is of size (1, p), where p is the number of library terms. In general, accounting for batch size, it will be (batch_size, p).
        Recall that p is the number of library terms.

        Parameters
        ----------
        z : torch.Tensor
            Latent variables, shape (batch_size, latent_dim)
        betas : torch.Tensor
            Parameters of the dynamical system, shape (batch_size,  )
        Returns
        -------
        theta : torch.Tensor
            Evaluated library terms, shape (batch_size, p)
        """
        
        # We call "states" the concatenation of latent variables and parameters of the dynamical systme
        if betas is None:
            states = z
        else:
            states = torch.cat([z, betas], dim = 1)
            
        theta_list = [f(states) for f in self.lambdified_library]  # list of (batch_size,1) tensors
        theta = torch.cat(theta_list, dim=1)  

        return theta
        
    def forward(z: torch.Tensor):
        pass


# For stupid tests
if __name__ == '__main__':
    # some test for the SINDy lib
    # z = torch.tensor([2.5, 5.3]).unsqueeze(0)
    z = torch.tensor([[2.5, 5.3], [0.1, -3.3]])
    betas = torch.tensor([[0.4], [0.1]])
    sl = SINDyLayer(latent_dim=2, n_parameters=1, poly_order=2, parameter_names=None)
    library_terms = sl.library_symbols
    print(library_terms)
    print(sl.evaluate_theta(z, betas))