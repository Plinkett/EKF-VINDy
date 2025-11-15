import matplotlib.pyplot as plt
import numpy as np
import torch
from abc import ABC, abstractmethod

class BaseDistribution(torch.nn.Module, ABC):
    """
    Base class for probability distributions i.e., priors to be used with VINDy.
    We outsource the definitions of "mean" or "loc". We only care that they implement the KL divergence.
    """
    
    @abstractmethod
    def forward(self):
        """
        Sample from the distribution.
        """
        pass
    
    @abstractmethod
    def kl_divergence(self, distribution: 'BaseDistribution'):
        """
        Compute the Kullback-Leibler divergence between this distribution and another distribution of the 
        same family.
        """
        pass