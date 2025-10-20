import numpy as np
import pysindy as ps
from typing import List
from pysindy.feature_library import PolynomialLibrary, FourierLibrary

n_input = 2
poly_lib = PolynomialLibrary(degree=3).fit(np.zeros((1, n_input)))
print(np.zeros((1, n_input)))
print(poly_lib.get_feature_names())

def generate_names(latent_dim: int, n_parameters: int, 
                   parameter_names: List[str] = None):
    """
    Generate list of names (of both latent variables and parameters) in string format.
    """
    latent_names = [f'z_{i}' for i in range(latent_dim)]
    if parameter_names is None:
        parameter_names_def = [f'beta_{i}' for i in range(n_parameters)]
    else:
        parameter_names_def = parameter_names
    
    return latent_names + parameter_names_def


print(generate_names(2, 3))