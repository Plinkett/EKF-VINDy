import torch
import torch.nn as nn
import numpy as np
import ekf_vindy.vindy.utils as util
from ekf_vindy.vindy.distributions.laplace import Laplace
from ekf_vindy.vindy import torch_config 
from typing import List

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
        big_xi_log_scales : torch.nn.Parameter
            Trainable log-scale parameters for the Laplace distributions (shape: [p, n_variables]).
        big_xi_distribution : Laplace
            Laplace distribution over SINDy coefficients for variational inference.
        laplace_prior : Laplace
            Laplace prior used for KL divergence regularization.
        mask : torch.Tensor
            Binary mask used for pruning coefficients during training.
    """
    def __init__(self, latent_dim: int, n_parameters: int, poly_order: int, parameter_names: List[str],
                 distribution_initialization = 'uniform', prior_loc = 0.0, prior_scale = 1.0):
        super().__init__() 
        
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
        self.big_xi_log_scales = nn.Parameter(torch.empty(self.p, self.n_variables, device=torch_config.device, dtype=torch_config.dtype)) 
        
        # Initialize mask (all ones at the beginning)
        self.mask = torch.ones(self.p, self.n_variables, device=torch_config.device, dtype=torch_config.dtype)
        
        # Initialize distributions and prior
        self._initialize_distribution(distribution_initialization)
        self.big_xi_distribution = Laplace(self.big_xi, self.big_xi_log_scales)
        self.laplace_prior = self._build_prior(prior_loc, prior_scale)
    
    def _initialize_distribution(self, init_scheme: str):
        self._init_tensor(self.big_xi, init_scheme)
        self._init_tensor(self.big_xi_log_scales, init_scheme)
        
    def _build_prior(self, prior_loc: float, prior_scale: float):
        """
        Given the prior loc and log_scale, we instantiate a Laplace object against which we will compare during training i.e., we will 
        compute the KL divergence.
        
        In general, we may use different priors for each coefficient. Indeed the Laplace class accepts tensors for loc and log_scale.
        """
        prior_loc_tensor = torch.full_like(self.big_xi, prior_loc)
        prior_log_scale_tensor = torch.full_like(self.big_xi_log_scales, np.log(prior_scale))
        return Laplace(prior_loc_tensor, prior_log_scale_tensor)
     
    def forward(self, z: torch.Tensor, betas: torch.Tensor | None = None):
        """
        Evaluate ODE, z and betas are of shape (batch_size, n_variables).
        Recall that big_xi is of size (p, n_variables), and _evaluate_theta outputs a shape (batch_size, p). 
        
        We may change this if we want to have sample-based UQ.
        """
        theta = self._evaluate_theta(z, betas)
        big_xi_masked = self.big_xi * self.mask
        return theta @ big_xi_masked
    
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


