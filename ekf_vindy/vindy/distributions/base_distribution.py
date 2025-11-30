import torch
from abc import ABC, abstractmethod

class BaseDistribution(torch.nn.Module, ABC):
    """
    Base class for probability distributions.
    """
    
    @abstractmethod
    def forward(self):
        """
        Sample from the distribution in a differentiable manner via the reparametrization trick.
        """
        raise NotImplementedError
    
    @abstractmethod
    def kl_divergence(self, distribution: 'BaseDistribution'):
        """
        Compute the Kullback-Leibler divergence between this distribution and another distribution of the 
        same family.
        """
        raise NotImplementedError
    
    @abstractmethod
    def variance(self):
        """
        Compute variance of distribution
        """
        raise NotImplementedError
        
    @abstractmethod
    def evaluate_pdf(self, granularity = 3000):
        """
        Compute discrete evaluations of the probability density function (PDF) 
        across the support of the distribution.

        Parameters
        ----------
        granularity : int
            The number of points across the support where the PDF is evaluated.

        Returns
        -------
        np.ndarray
            Array of PDF values corresponding to evenly spaced points across the support.

        Notes
        -----
        This does not return a callable function, but a discretized approximation
        of the PDF over the support of the distribution.
        """
        raise NotImplementedError