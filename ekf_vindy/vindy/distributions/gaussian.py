# import numpy as np
# import torch 
# from aesindy.vindy.distributions.base_distribution import BaseDistribution

# class Gaussian(BaseDistribution):
#     """
#     Gaussian distribution class, to be used as prior in VINDy.
#     We only treat scalar Gaussians. The VINDy objective only consider sums of individual Gaussians, so it makes sense.
#     """
    
#     def __init__(self, mean: torch.Tensor, log_var: torch.Tensor):
#         super(Gaussian, self).__init__()
#         self.mean = mean
#         self.log_var = log_var
    
#     def forward(self):
#         """
#         Sample from the Gaussian distribution using the reparameterization trick.
#         """
#         return self.sample(self.mean, self.log_var)
    
#     def sample(self, mean: torch.Tensor, log_var: torch.Tensor):
#         """
#         Sample from the Gaussian distribution for given mean and log-variance using the reparameterization trick.
#         Accepts both numpy arrays and torch tensors. Uses the device/dtype of self.mean.
#         """
#         std = torch.exp(0.5 * log_var, dtype=self.mean.dtype, device=self.mean.device)
#         eps = torch.randn_like(std, dtype=self.mean.dtype, device=self.mean.device)
        
#         return mean + eps * std
    
#     def kl_divergence(self, to_compare: 'Gaussian'):
#         """
#         Compute the Kullback-Leibler divergence between this Gaussian and another Gaussian distribution.
#         """
        
#         mu1, std1 = self.mean, self.std
#         mu2, std2 = to_compare.mean, to_compare.std
        
#         kl_divergence = (
#             torch.log(std2 / std1) 
#             + (std1 ** 2 + (mu2 - mu1) ** 2) / (2 * std2 ** 2)
#             - 0.5
#         ) 
        
#         return kl_divergence
    
#     @property
#     def variance(self):
#         """
#         Compute the variance of the Gaussian distribution scaled by a factor.
#         """
#         return torch.exp(self.log_var)
    
#     @property
#     def std(self):
#         """
#         Compute standard deviation of Gaussian distribution, we do it on-the-fly from the log-var attribute.
#         """
#         return torch.exp(0.5 * self.log_var)
    
#     @property
#     def mean(self):
#         """
#         Return mean of Gaussian distribution.
#         """
#         return self.mean


# dummy_gaussian = Gaussian(0, 1)
# print(dummy_gaussian())