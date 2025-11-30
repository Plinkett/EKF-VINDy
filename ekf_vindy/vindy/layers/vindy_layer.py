import torch
import torch.nn as nn
import numpy as np
from ekf_vindy.vindy.distributions.laplace import Laplace
from ekf_vindy.vindy import torch_config 
from typing import List
import ekf_vindy.vindy.utils as util

"""
VINDy layer that implements a variational inference version of SINDy. Instead of regularizing with the L1 loss and the sequential thresholding,
we directly fit distributions at each step, via the the standard MSE loss and the KL divergence. To enforce sparsity, we rely on Laplace priors.
"""

class VINDyLayer(nn.Module):
    """
        Attributes
        ----------
        latent_dim : int
            Dimension of the latent latent space.
        n_parameters : int
            Number of known parameters of the dynamical system.
        n_variables : int
            Total number of variables (latent + parameters).
        var_names : List[str]
            Names of all variables (latent + parameters).
        var_symbols : List[sympy.Symbol]
            Symbolic representation of each variable.
        library_symbols : List[sympy.Symbol]
            Symbolic representation of library terms used in SINDy.
        lambdified_library : List[Callable]
            List of callable functions that compute each library term given input tensors.
        p : int
            Number of library terms.
        big_xi : torch.nn.Parameter
            Trainable SINDy coefficients (shape: [p, n_variables]).
        big_xi_scales : torch.nn.Parameter
            Trainable scale parameters for the Laplace distributions (shape: [p, n_variables]).
        big_xi_distribution : Laplace
            Laplace distribution over SINDy coefficients for variational inference.
        laplace_prior : Laplace
            Laplace prior used for KL divergence regularization.
        mask : torch.Tensor
            Binary mask used for pruning coefficients during training.
    """
    def __init__(self, latent_dim: int, n_parameters: int, poly_order: int, parameter_names: List[str],
                 distribution_initialization = 'uniform', prior_loc = 0.0, prior_log_scale = 0.0):
        super(VINDyLayer, self).__init__()
        self.latent_dim = latent_dim
        self.n_parameters = n_parameters
        self.n_variables = self.latent_dim + self.n_parameters
        
        # Generate variable names, and lambda functions (for library terms)
        self.var_names = util.generate_variable_names(self.latent_dim, self.n_parameters, parameter_names)
        self.var_symbols, \
        self.library_symbols, \
        self.lambdified_library = util.generate_library_torch(self.var_names, poly_order)
        
        # Number of library terms. We will prune many terms so after training this should change I assume... maybe through some variable p_pruned
        self.p = len(self.library_symbols)
        
        # Initialize SINDy coefficients distributions
        self.big_xi = nn.Parameter(torch.empty(self.p, self.n_variables, device=torch_config.device, dtype=torch_config.dtype))        
        self.big_xi_scales = nn.Parameter(torch.empty(self.p, self.n_variables, device=torch_config.device, dtype=torch_config.dtype)) 
        
        self.mask = torch.ones(self.p, self.n_variables, device=torch_config.device, dtype=torch_config.dtype)
        
        self._initialize_distribution(distribution_initialization)
        self.big_xi_distribution = Laplace(self.big_xi, self.big_xi_scales)
        self.laplace_prior = self._build_prior(prior_loc, prior_log_scale)
    
    def _initialize_distribution(self, init_scheme: str):
        super()._init_tensor(self.big_xi, init_scheme)
        super()._init_tensor(self.big_xi_scales, init_scheme)
        
    def _build_prior(self, loc: float, log_scale: float):
        """
        Given the prior loc and log_scale, we instantiate a Laplace object against which we will compare during training i.e., we will 
        compute the KL divergence.
        """
        prior_loc_tensor = torch.full_like(self.big_xi, loc)
        prior_log_scale_tensor = torch.full_like(self.big_xi_scales, log_scale)
        return Laplace(prior_loc_tensor, prior_log_scale_tensor)
     
    def forward(self, z: torch.Tensor, betas: torch.Tensor | None = None):
        """
        Evaluate ODE, z and betas are of shape (batch_size, n_variables).
        Recall that big_xi is of size (p, n_variables), and _evaluate_theta outputs a shape (batch_size, p). 
        
        We may change this if we want to have sample-based UQ.
        """
        theta = super()._evaluate_theta(z, betas)
        big_xi_masked = self.big_xi * self.mask
        return theta @ big_xi_masked