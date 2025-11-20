import numpy as np
import torch
from aesindy.vindy.distributions.base_distribution import BaseDistribution


# TODO: I think the way they meant this to work was as a Laplace representing the entire Xi matrix? 
# not super clear...

class Laplace(BaseDistribution):
    """
    Class handling the sampling and managing of a Laplace distribution. We can treat the weights of the Xi matrix (i.e., SINDy coefficients)
    as Laplace random variables, thus promoting sparsity. 
    
    Two main attributes, the loc and the scale parameter (usually denoted with "b"). The variance is 2b^2, and b is \sqrt(0.5 * Var).
    
    Parameters
    ----------
    loc : torch.Tensor
        The location parameter of the Laplace distribution. Mean, mode and median. Shape (batch_size, dim)
    log_scale : torch.Tensor
        The log of the scale parameter of the Laplace distribution. Shape (batch_size, dim)
    """
    
    def __init__(self, loc: torch.Tensor, log_scale: torch.Tensor):
        super().__init__()
        self._loc = loc
        self._log_scale = log_scale    
    
    def forward(self):
        """
        Sample from the Laplace distribution. Output is of size (batch_size, dim).
        """
        return self.sample(self._loc, self._log_scale)

    def sample(self, loc: torch.Tensor, log_scale: torch.Tensor):
        """
        We sample in batches. Inputs are of shape (batch_size, dim). We use the reparameterization trick to sample from the Laplace distribution.
        Sample from a standard Laplace distribution (0, 1) and scale and shift element-wise, to support batches.
        """
        batch = loc.shape[0]
        dim = loc.shape[1]
        
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
        Compute KL divergence element-wise.
        
        Let's recall given L_1(loc_1, scale_1) and L_2(loc_2, scale_2) the KL divergence is:
          KL(L_1, L_2) = log(scale_2 / scale_2) 
                         + (scale_1 * {exp[- (loc_1 - loc_2) / scale_1]} + |loc_1 - loc_2| / scale_2) 
                         - 1

        Output Torch scalar with KL divergence value, broadcasted if we have batches.
        If we only compare 2 distributions (batch_size = 1), then we expect both distributions to have statistics (loc, log_scale)
        of size (1, dim).
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