import sympy as sp
import torch.nn as nn
import pysindy as ps
import numpy as np
from typing import List

"""
Utility class for autoencoders
"""

def initializate_weights(module: nn.Module):
    """
    Just use He initialization, since we will use ReLU activations.
    To avoid breaking things in the future, we only initialize the linear modules, biases to zero.
    """
    
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, nonlinearity='relu')    
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
            
def generate_library_torch(variables: List[str], poly_order: int): 
    """ Generate lambdified library terms in torch """
    
    # Generate library in symbol format first
    var_symbols, library_symbols = generate_library_symbols(variables, poly_order)
    lamdbified_terms = [sp.lambdify([variables], term, "torch") for term in library_symbols]
    
    return var_symbols, library_symbols, lamdbified_terms

def generate_library_symbols(variables: List[str], poly_order: int): 
    """ 
    Given the variable names, and the order of the polynomial, generate the library terms from
    PySINDy and cast them into SymPy symbols to be lambdified later (into PyTorch).
    
    P.S. We have a dummy fit with PySINDy, only way to get the library term in string format.
    """
    
    n_variables = len(variables)
    
    poly_lib = ps.PolynomialLibrary(degree = poly_order).fit(np.zeros((1, n_variables)))
    poly_terms = poly_lib.get_feature_names(variables)
    
    # Handling of variables in string and SymPy format
    var_symbols = sp.symbols(' '.join(variables))
    locals_dict = {name: symbol for name, symbol in zip(variables, var_symbols)}
    
    # Properly format library terms for exponentiation and multiplication
    library_terms = [term.replace('^', '**').replace(' ', '*') for term in poly_terms]
    library_symbols = [sp.sympify(term, locals=locals_dict) for term in library_terms]

    return var_symbols, library_symbols


def generate_variable_names(latent_dim: int, n_parameters: int, 
                   parameter_names: List[str] = None):
    """
    Generate list of names (of both latent variables and parameters) in string format.
    """
    
    latent_names = [f'z_{i}' for i in range(latent_dim)]
    if parameter_names is None:
        parameter_names_def = [f'b_{i}' for i in range(n_parameters)]
    else:
        parameter_names_def = parameter_names
    
    return latent_names + parameter_names_def