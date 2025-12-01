import numpy as np
import torch
from torch import nn
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
        Compute KL divergence element-wise.
        
        Let's recall given L_1(loc_1, scale_1) and L_2(loc_2, scale_2) the KL divergence is:
          KL(L_1, L_2) = log(scale_2 / scale_2) 
                         + (scale_1 * {exp[- (loc_1 - loc_2) / scale_1]} + |loc_1 - loc_2| / scale_2) 
                         - 1

        Output Torch scalar with KL divergence value, broadcasted if we have batches.
        If we only compare 2 distributions (batch_size = 1), then we expect both distributions to have statistics (loc, log_scale)
        of size (1, dim).
        
        The output of size (batch_size, 1). If used exclusively with VINDy (no VAE), then you only compare with whatever prior you are using.
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
        
# locs_1 = torch.tensor([[1.0, 2.0, 2.3], [3.0, 4.0, 2.3]])
# log_scales_1 = torch.log(torch.tensor([[0.5, 1.0, 2.3], [1.5, 2.0, 2.3]]))
# locs_2 = torch.tensor([[56.0, 22.0], [31.0, 14.0]])
# log_scales_2 = torch.log(torch.tensor([[3.5, 7.0], [2.5, 1.0]]))
# laplace_batch = Laplace(locs_1, log_scales_1)
# laplace_batch2 = Laplace(locs_2, log_scales_2)

# from ekf_vindy.plotting.plotter import plot_pdf
# import matplotlib.pyplot as plt


# from sympy import symbols

# # 3 library terms (columns)
# z0, z1, z2 = symbols("z_0 z_1 z_2")
# library_symbols = [z0, z1, z2]  # each term is a single SymPy symbol

# # 2 batches (rows)
# var_symbols = (z0, z1)  # matches B=2 from laplace_batch

# fig, axes = plot_pdf(
#     *laplace_batch.evaluate_pdf(),
#     batch_labels=var_symbols,  # rows
#     dim_labels=library_symbols,  # columns
#     ylim=(0, 1.5),
#     palette="magma",
#     label_size="large"
# )
# plt.show()