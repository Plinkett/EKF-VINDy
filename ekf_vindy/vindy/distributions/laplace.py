import numpy as np
import torch
from ekf_vindy.vindy.distributions.base_distribution import BaseDistribution

class Laplace(BaseDistribution):
    """
    Class handling the sampling and managing of a Laplace distribution. We can treat the weights of the Xi matrix (i.e., SINDy coefficients)
    as Laplace random variables, thus promoting sparsity. 
    
    Two main attributes, the loc and the scale parameter (usually denoted with "b"). The variance is 2b^2, and b is \sqrt(0.5 * Var).
    
    Recall that that "batch_size" and "dim" are just names. For VINDy, for example, "batch_dim" will be equal to the number of library terms "p", whereas
    "dim" will still indicate the number of states in our dynamical system (or more in general, variables).
    
    Parameters
    ----------
    loc : torch.Tensor
        The location parameter of the Laplace distribution. Mean, mode and median. Shape (batch_size, dim)
    log_scale : torch.Tensor
        The log of the scale parameter of the Laplace distribution. Shape (batch_size, dim)
    """
    
    def __init__(self, loc: torch.Tensor, log_scale: torch.Tensor):
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
        Compute the per-dimension KL divergence between two Laplace distributions.

        For two Laplace distributions

            L1 = Laplace(loc_1, scale_1)
            L2 = Laplace(loc_2, scale_2),

        the KL divergence in a single dimension has the closed-form expression:

            KL(L1 || L2)
                = log(scale_2 / scale_1)
                + (scale_1 * exp(-|loc_1 - loc_2| / scale_1) + |loc_1 - loc_2|) / scale_2
                - 1

        This method applies the formula element-wise across all dimensions, producing
        a tensor of shape (batch_size, dim). No summation over dimensions is performed.

        Parameters
        ----------
        to_compare : Laplace
            The second Laplace distribution L2.

        Returns
        -------
        kl : torch.Tensor
            Tensor of KL divergences with shape (batch_size, dim), containing the
            per-dimension KL values for each distribution in the batch.
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
    
    @property
    def variance(self):
        """
        Returns the variance of each Laplace distribution that we are holding. 
        So the resulting tensor is of size (batch_size, dim).
        """
        return 2 * torch.exp(self.log_scale) ** 2
            
    def evaluate_pdf(self, granularity = 3000, x_range = 10.0):
        """
        Returns PDF evaluations of shape (batch_size, dim, granularity)
        using a shared x-grid across all batches and dimensions, centered at zero.

        Returns:
            x: np.ndarray, shape (B, D, N)
            pdf: np.ndarray, shape (B, D, N)
        """
        loc_np = self.loc.detach().cpu().numpy() # shape (B, D)
        scale_np = torch.exp(self.log_scale).detach().cpu().numpy()  # shape (B, D)
        B, D = loc_np.shape

        # Shared x-grid for all batches/dims, shape (N,)
        x_base = np.linspace(-x_range, x_range, granularity)

        # Broadcast x_base to shape (B, D, N)
        x = np.broadcast_to(x_base, (B, D, granularity))

        # Compute PDF with broadcasting
        pdf = (1 / (2 * scale_np[:, :, None])) * np.exp(-np.abs(x - loc_np[:, :, None]) / scale_np[:, :, None])

        return x, pdf