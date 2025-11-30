import torch
import torch.nn as nn
import numpy as np
from ekf_vindy.vindy import torch_config 
from typing import List
import ekf_vindy.vindy.utils as util
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
TODO: Add mask to train certain parameters.4
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
    var_symbols : Tuple[sympy.Symbol]
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
        self._init_tensor(self.big_xi, big_xi_initialization)
    
    def _init_tensor(self, tensor: torch.Tensor, init_scheme: str):
        if init_scheme == 'uniform':
            nn.init.uniform_(tensor, -1, 1)
        elif init_scheme == 'normal':
            nn.init.normal_(tensor, 0.0, 1.0)
        elif init_scheme == 'zeros':
            nn.init.zeros_(tensor)
        elif init_scheme == 'ones':
            nn.init.ones_(tensor)
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
        Recall that big_xi is of size (p, n_variables), and _evaluate_theta outputs a shape (batch_size, p). 
        """
        theta = self._evaluate_theta(z, betas)
        big_xi_masked = self.big_xi * self.mask
        return theta @ big_xi_masked

    def print_model(self, include_pruned=False):
        """
        Print the current masked SINDy library in readable form, like:
        (z0)' = 0.92 1 + -0.1 z0 + -1.0 z0 z1^2

        Parameters
        ----------
        include_pruned : bool
            If True, prints all terms, including those pruned (with zeroed weight).
            If False, prints only non-pruned terms (mask > 0).
        """
        with torch.no_grad():
            big_xi_masked = self.big_xi * self.mask
            for var_idx, var_name in enumerate(self.var_names):
                eq_terms = []
                for term_idx, term in enumerate(self.library_symbols):
                    weight = big_xi_masked[term_idx, var_idx].item()
                    if not include_pruned and abs(weight) < 1e-12:
                        continue
                    
                    # Convert sympy term to readable string
                    term_str = str(term)
                    # Replace ** with ^ and * with space
                    term_str = term_str.replace('**', '^').replace('*', ' ')
                    eq_terms.append(f"{weight:.3g} {term_str}")
                
                if eq_terms:
                    eq_str = " + ".join(eq_terms)
                else:
                    eq_str = "0"
                print(f"({var_name})' = {eq_str}")
