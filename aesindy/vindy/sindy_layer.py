import torch
import torch.nn as nn
import numpy as np
from aesindy import torch_config 
from typing import List
import aesindy.utils as util
"""
SINDy Torch layer. Instead of using STLSQ (sequential thresholding least squares) we use gradient descent with L1 regularization and
thresholding. The float threshold is a hyperparameter and attribute of the class, however, 

SINDy layer that provides the standard SINDy libraries, differentiable via PyTorch.

Do you actually need that layer? After all, should it not be inside the loss? Do you actually need to backpropagate through that?
No, you define this "fake" layer that is the Theta * Xi. So let's define that.

Can you recycle this in the loss? I think you can just shortcircuit this into the loss, can't you? No need to recompute it.
We don't actually optimize with PySINDy, we just use PySINDy to get the library terms... 

TODO: Add Fourier features as well...not straight forward to mix parameters (input formatting)
TODO: Have a config class to avoid passing too much stuff
TODO: Add mask to train certain parameters.
"""

class SINDyLayer(nn.Module):
    """
    SINDy layer class
    
    Attributes
    ----------
    latent_dim : int
        Dimension of the latent space
    n_parameters : int
        Number of parameters of the dynamical system
    n_variables : int
        Total number of variables (latent + parameters)
    threshold : float
        Hyperparameter to be used during training for defining the pruned SINDy coefficients (i.e., Xi matrix).
    var_names : List[str]
        Names of the variables (latent and parameters)
    var_symbols : List[sympy.Symbol]
        SymPy symbols for the variables
    library_symbols : List[sympy.Symbol]
        Symbols for the library terms
    lambdified_library : List[Callable]
        List of torch functions to compute library terms
    p : int
        Number of library terms
    big_xi : torch.nn.Parameter
        SINDy coefficients to be trained. Shape (p, n_variables)
    """
    def __init__(self, latent_dim: int, n_parameters: int, poly_order: int, parameter_names: List[str],
                 prune_threshold = 5e-2, big_xi_initialization = 'zeros'):
        super(SINDyLayer, self).__init__()
        self.latent_dim = latent_dim
        self.n_parameters = n_parameters
        self.n_variables = self.latent_dim + self.n_parameters
        self.prune_threshold = prune_threshold
        
        # Generate variable names, and lambda functions (for library terms)
        self.var_names = util.generate_variable_names(self.latent_dim, self.n_parameters, parameter_names)
        self.var_symbols, \
        self.library_symbols, \
        self.lambdified_library = util.generate_library_torch(self.var_names, poly_order)
        
        # Number of library terms. We will prune many terms so after training this should change I assume... maybe through some variable p_pruned
        self.p = len(self.library_symbols)
        
        # Initialize SINDy coefficients
        self.big_xi = nn.Parameter(torch.empty(self.p, self.n_variables, device=torch_config.device, dtype=torch_config.dtype))        
        self.mask = torch.ones(self.p, self.n_variables, device=torch_config.device, dtype=torch_config.dtype)
        self._initialize_SINDy_coefficients(big_xi_initialization)
    
    def _initialize_SINDy_coefficients(self, init_scheme: str):
        if init_scheme == 'uniform':
            nn.init.uniform_(self.big_xi, -0.05, 0.05)
        elif init_scheme == 'normal':
            nn.init.normal_(self.big_xi, 0.0, 1.0)
        elif init_scheme == 'zeros':
            nn.init.zeros_(self.big_xi)
        elif init_scheme == 'ones':
            nn.init.ones_(self.big_xi)
        else:
            raise ValueError("Unknown initialization scheme.")
    
    def _evaluate_theta(self, z: torch.Tensor, betas: torch.Tensor | None = None):
        """
        Evaluate library terms with latents (and parameters, called "betas" here). We assume the batch size to be the same for both z and betas (no input verification here).
        A single theta vector is of size (1, p), where p is the number of library terms. In general, accounting for batch size, it will be (batch_size, p).

        Parameters
        ----------
        z : torch.Tensor
            Latent variables, shape (batch_size, latent_dim)
        betas : torch.Tensor
            Parameters of the dynamical system, shape (batch_size, n_parameters)
            
        Returns
        -------
        theta : torch.Tensor
            Evaluated library terms, shape (batch_size, p)
        """
        
        # We call "states" the concatenation of latent variables and parameters of the dynamical system
        if betas is None:
            states = z
        else:
            states = torch.cat([z, betas], dim = 1)
            
        theta_list = [f(states) for f in self.lambdified_library]  # list of (batch_size,1) tensors
        theta = torch.cat(theta_list, dim=1)  

        return theta
    
    def _apply_mask(self):
        """
        Implements sequential thresholding, ideally at the end of some epochs, to enforce sparsity. It relies on a binary mask.
        So this should be called outside the class during the training loop.
        """
        self.mask = (self.big_xi.abs() >= self.prune_threshold).float()
        
        with torch.no_grad():
            self.big_xi *= self.mask
        
    def forward(self, z: torch.Tensor, betas: torch.Tensor | None = None):
        """
        Evaluate ODE, z and betas are of shape (batch_size, n_variables).
        Recall that big_xi is of size (p, n_variables), and _evaluate_theta outputs a shape (batch_size, n_variables). 
        """
        return self._evaluate_theta(z, betas) @ self.big_xi


# # For stupid tests
# if __name__ == '__main__':
#     # some test for the SINDy lib
#     # z = torch.tensor([2.5, 5.3]).unsqueeze(0)
#     torch_config.setup_device_and_type()
    
#     z = torch.tensor([[2.5, 5.3], [0.1, -3.3]])
#     betas = torch.tensor([[0.4], [0.1]])
#     sl = SINDyLayer(latent_dim=2, n_parameters=1, poly_order=2, parameter_names=None)
#     library_terms = sl.library_symbols
#     print(library_terms)
#     print(sl._evaluate_theta(z, betas))
#     print(sl.big_xi)
#     print(sl(z, betas))
