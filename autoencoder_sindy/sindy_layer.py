import torch
import torch.nn as nn
import sympy as sp
import pysindy as ps
import numpy as np
from typing import List
from autoencoder_sindy.ae_utils import initializate_weights


"""
SINDy layer that provides the standard SINDy libraries, differentiable via PyTorch.
TODO: Add Fourier features as well...not straight forward to mix parameters (input formatting)
TODO: Have a config class to avoid passing too much stuff
"""
def torchify_library(variables: List[str], library_terms: List[str]):
    """ Convert library to PyTorch functions, for backpropagation """
    funcs = [sp.lambdify([variables], term, "torch") for term in library_terms]
    return funcs

def create_library(n_variables: int, poly_order: int, var_names: List[str]):
    """
    We must fit a dummy model to get the library terms in string format. Parameter naming and ordering must be managed by the user.
    Outputs library terms already in SymPy format
    TODO: Fourier library
    """

    poly_lib = ps.PolynomialLibrary(degree = poly_order).fit(np.zeros((1, n_variables)))
    if var_names:
        poly_terms = poly_lib.get_feature_names(var_names)
    else:
        poly_terms = poly_lib.get_feature_names()
    
    return poly_terms

def generate_names(latent_dim: int, n_parameters: int, parameter_names: List[str]):
    """
    Generate list of names (of both latent variables and parameters) in string format.
    """
    latent_names = [f'z_{i}' for i in range(latent_dim)]
    if parameter_names is None:
        parameter_names_def = [f'beta_{i}' for i in range(n_parameters)]
    else:
        parameter_names_def = parameter_names
    
    return latent_names + parameter_names_def

class SINDyLayer(nn.Module):

    def __init__(self, latent_dim: int, n_parameters: int, poly_order: int, parameter_names: List[str]):
        super(SINDyLayer, self).__init__()
        self.latent_dim = latent_dim
        self.n_parameters = n_parameters
        self.n_variables = self.latent_dim + self.n_parameters

        variables = generate_names(self.latent_dim, self,n_parameters, parameter_names)
        library_terms = create_library(self.n_variables)
        


