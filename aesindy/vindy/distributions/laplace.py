import numpy as np
import torch
from aesindy.vindy.distributions.base_distribution import BaseDistribution

class Laplace(BaseDistribution):
    """
    Class handling the sampling and managing of a Laplace distribution. We can treat the weights of the Xi matrix (i.e., SINDy coefficients)
    as Laplace random variables, thus promoting sparsity. 
    
    Two main attributes, the loc and the scale parameter (usually denoted with "b"). The variance is 2b^2, and b is \sqrt(0.5 * Var).
    
    Parameters
    ----------
    loc : torch.Tensor
        The location parameter of the Laplace distribution. Mean, mode and median.
    log_scale : torch.Tensor
        The log of the scale parameter of the Laplace distribution.
    """
    
    def __init__(self, loc: torch.Tensor, log_scale: torch.Tensor):
        super().__init__()
        self._loc = loc
        self._log_scale = log_scale    
    
    def forward(self):
        return self.sample(self._loc, self._log_scale)

    def sample(self, loc: torch.Tensor, log_scale: torch.Tensor):
        """
        Use reparametrization trick to differentiate through sampling. Sample from "standard" Laplace and then scale and shift.
        When this is not used for this object's own loc and scale, we assume the device and dtype are the same as ours.
        
        We sample a Laplacian of dimensions loc.shape[0], and a batch of size loc.shape[1] if applicable.
        """
        dim = loc.shape[1]
        batch = loc.shape[0]
        
        # Sample standard Laplace
        distribution = torch.distributions.Laplace(
            torch.tensor(0.0, device=loc.device, dtype=loc.dtype),
            torch.tensor(1.0, device=loc.device, dtype=loc.dtype)
        )
        epsilon = distribution.sample((batch, dim))

        # Reparameterization trick i.e., X = loc + scale * epsilon
        return loc + torch.exp(log_scale) * epsilon

    def kl_divergence(self, to_compare: 'Laplace'):
        """
        Let's recall given L_1(loc_1, scale_1) and L_2(loc_2, scale_2) the KL divergence is:
          KL(L_1, L_2) = log(scale_2 / scale_2) 
                         + (scale_1 * {exp[- (loc_1 - loc_2) / scale_1]} + |loc_1 - loc_2| / scale_2) 
                         - 1

        Output Torch scalar with KL divergence value.
        """
        loc_1 = self.loc
        loc_2 = to_compare.loc
        scale_1 = torch.exp(self.log_scale)
        scale_2 = torch.exp(to_compare.log_scale)
        
        kl_divergence = (
            torch.log(scale_2 / scale_1) 
            + (scale_1 * torch.exp(- torch.abs(loc_1 - loc_2) / scale_1) + torch.abs(loc_1 - loc_2)) / scale_2
            - 1
        )
        return kl_divergence

    @property
    def loc(self):
        return self._loc
    
    @property
    def log_scale(self):
        return self._log_scale
    

locs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
log_scales = torch.log(torch.tensor([[0.5, 1.0], [1.5, 2.0]]))
laplace_batch = Laplace(locs, log_scales)