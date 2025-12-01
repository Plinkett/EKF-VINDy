import sympy as sp
import torch
import torch.nn as nn
import pysindy as ps
import numpy as np
from typing import List

def add_lognormal_noise(trajectory: np.ndarray, sigma: float):
    noise = np.random.lognormal(mean=0, sigma=sigma, size=trajectory.shape)
    return trajectory * noise, noise

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
    """
    Generate lambdified library terms compatible with PyTorch batch inputs.

    Recall that the torchified lambdas are vectorized by default, so we just need to pass the right arguments.
    Furthermore, SymPy automatically has keyword arguments when lambdifying, so we can pass a dictionary of variable values.

    For example, consider a dummy f_mixed_squared = z_1^2 * z_2^2, this takes as arguments (by passing directly the dictionary)
        f_single(
            z_0 = tensor([1., 2., 3.]),
            z_1 = tensor([10., 20., 30.])
        )
    The output shape of each term is (batch_size, 1). So each f_batch gives us a column vector.
    """
    var_symbols, library_symbols = generate_library_symbols(variables, poly_order)
    lambdified_terms = []

    for term in library_symbols:
        # A lambda function that takes ONE state and outputs a scalar corresponding to the term
        
        f_single = sp.lambdify(var_symbols, term, "torch")
        
        # we "batchify" f_single
        def f_batch(x_batch, f_single=f_single):
            vals = {str(sym): x_batch[:, i] for i, sym in enumerate(var_symbols)}
            out = f_single(**vals)
        
            # After unpacking dictionary, make sure the output is a tensor of shape (batch_size, 1)
            if not isinstance(out, torch.Tensor):
                out = torch.tensor([out] * x_batch.shape[0], dtype=x_batch.dtype, device=x_batch.device)

            return out.reshape(-1, 1)

        lambdified_terms.append(f_batch)
    
    return var_symbols, library_symbols, lambdified_terms

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