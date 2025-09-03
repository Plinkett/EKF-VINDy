""" 
Utils for converting strings (usually obtain from PySINDy library) into SymPy symbols for 
symbolic differentiation, and easy Jacobian computation for EKF. 

For this project we will rely on dense matrices, sparse matrix optimization from scipy.sparse
can be taken into consideration if we actually deal with high-dimensional systems and huge libraries... since for this project
we work on a small latent space it's ok. For example, big_xi_t would be a csr_matrix in an efficient implementation.

In such a scenario, you would use something like this to identify the indices of entries:

rows = [i for i, col_list in enumerate(to_lambdify) for _ in col_list]
cols = [c for col_list in to_lambdify for c in col_list]
"""

from typing import List, Optional
import ekf_vindy.utils as utils
import sympy as sp
import numpy as np

def sympify_str(variables: List[str], library_terms: List[str]):
    """
    Gets library terms (for example, from PySINDy) and transform them into SymPy symbols for
    easy handling of derivatives.
    List of library terms can be obtained from model.get_feature_names()
    List of variables can be obtained with model.feature_names
    """
    
    library_symbols = []
    var_symbols = sp.symbols(' '.join(variables))
    
    #annoying handling of symbols and their corresponding string names
    locals_dict = {name: symbol for name, symbol in zip(variables, var_symbols)} 
    
    # By default, SINDy uses other symbols for exponentiation and multiplication
    library_terms = [term.replace('^', '**').replace(' ', '*') for term in library_terms]
    library_symbols = [sp.sympify(term, locals=locals_dict) for term in library_terms]

    return var_symbols, library_symbols

def differentiate_library(variables: List[sp.Symbol], 
                          library: List[sp.Symbol], 
                          to_lambdify: Optional[List[List[int]]] = None):
    """  
    Returns a list of partial derivatives, lamdbified and in symbolic format.
    Differentiates each library term w.r.t. each variable. 
    to_lamdbify says which partial derivatives to compute per row (outer list is for the N equations, inner for the p paramters)
    """
    symbolic_derivatives = []
    lambdified_derivatives = []

    # Not the best since you effectevily iterate over twice. Easy to read though, and we are dealing with 
    # a small number of items anyway
    if not to_lambdify:
        symbolic_derivatives = [[sp.diff(term, var) for term in library] for var in variables]
        lambdified_derivatives = [[sp.lambdify(variables, dterm) for dterm in sym_row] for sym_row in symbolic_derivatives]
    else:
        symbolic_derivatives = [[sp.diff(term, var) for term in partial] for var, partial in zip(variables, to_lambdify)]
        lambdified_derivatives = [[sp.lambdify(variables, dterm) for dterm in sym_row] for sym_row in symbolic_derivatives]

    # list of lists
    return lambdified_derivatives, symbolic_derivatives

def lambdify_library(variables: List[sp.Symbol], 
                     library: List[sp.Symbol]):
    """Return lambdified library functions that take a state vector x"""
    funcs = [sp.lambdify(variables, term, "numpy") for term in library]
    # wrap so each f takes a single array-like x
    # may NOT work if library terms have unordered variables (apparently an issue for PDE libraries)
    wrapped_funcs = [lambda x, f=f: f(*x) for f in funcs]
    return wrapped_funcs

def lambdified_jacobian_blocks(variables: List[str], 
                               library_terms: List[str],
                               tracked_terms: List[List[int]],
                               coeffs: np.ndarray):
    """ 
    We compute the Jacobian in blocks. The upper right block is the Jacobian w.r.t. the original state x. The left upper block is the Jacobian w.r.t. the coefficients xi.
    tracked_terms defines a list, one per equation of the ODE system, of indices of coefficients to track. We return lamdbdified versions of those blocks.
    """

    # Gotta do this for terms that are not activated across equations, for the moment just ignore this and brute-force it to see if it works
    # # Compute only derivatives for the non-zero coefficients or the tracked coefficients.
    # non_zero_coeffs = utils.find_non_zero(coeffs)
    # # list of indices for which to evaluate the partial derivative i.e., the non-zero coefficients and the tracked terms 
    # to_lambdify = [sorted(list(set(nz).union(set(tr)))) for nz, tr in zip(non_zero_coeffs, tracked_terms)]
    
    # turn into symbols (we're turning the entire library into symbols, could create issues?)
    var_symbols, library_symbols = sympify_str(variables, library_terms)
    
    # we differentiate only the selected entries of the matrix, this is essentially dTheta/dt
    _, lambdified_derivatives = differentiate_library(var_symbols, library_symbols) # , to_lambdify)

    # now we create the upper right block, we lambdified the entire library and select the necessary entries at runtime (from tracked_terms)
    right_block_lambdas = lambdify_library(var_symbols, library_symbols)

    return lambdified_derivatives, right_block_lambdas