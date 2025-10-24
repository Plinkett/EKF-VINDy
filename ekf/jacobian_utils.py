""" 
Utils for converting strings (usually obtained from PySINDy library) into SymPy symbols for 
symbolic differentiation, and easy Jacobian computation for EKF. 
"""

from typing import List, Optional
import ekf.utils as utils
import sympy as sp
import numpy as np

def sympify_str(variables: List[str], library_terms: List[str]):
    """
    Gets library terms (for example, from PySINDy) and transform them into SymPy symbols for
    easy handling of derivatives.
    List of library terms can be obtained from model.get_feature_names()
    List of variables can be obtained with model.feature_names
    """
    var_symbols = sp.symbols(' '.join(variables))
    
    # annoying handling of symbols and their corresponding string names
    locals_dict = {name: symbol for name, symbol in zip(variables, var_symbols)} 
    
    # by default, SINDy uses other symbols for exponentiation and multiplication
    library_terms = [term.replace('^', '**').replace(' ', '*') for term in library_terms]
    library_symbols = [sp.sympify(term, locals=locals_dict) for term in library_terms]

    return var_symbols, library_symbols

def differentiate_library(variables: List[sp.Symbol], 
                          library: List[sp.Expr], 
                          to_lambdify: Optional[List[int]] = None):
    """
    Returns lambdified and symbolic partial derivatives.

    - If `to_lambdify` is None: full Jacobian (all variables x all library terms).
    - If `to_lambdify` is a list of indices: Jacobian restricted to those terms, still rectangular obviously.
    """

    # select terms to differentiate
    terms = library if to_lambdify is None else [library[i] for i in to_lambdify]

    # symbolic Jacobian (list of lists)
    symbolic_derivatives = [
        [sp.diff(term, var) for term in terms]
        for var in variables
    ]

    # lambdified Jacobian (list of lists)
    lambdified_derivatives = [
        [sp.lambdify([variables], dterm) for dterm in row]
        for row in symbolic_derivatives
    ]

    return lambdified_derivatives, symbolic_derivatives

def lambdify_library(variables: List[sp.Symbol], 
                     library: List[sp.Symbol]):
    """Return lambdified library functions that take a state vector x, because of wrapping "variables" in []"""
    funcs = [sp.lambdify([variables], term, "numpy") for term in library]
    return funcs

def lambdified_jacobian_blocks(variables: List[str], 
                               library_terms: List[str],
                               tracked_terms: List[List[int]],
                               coeffs: np.ndarray):
    """ 
    We compute the Jacobian in blocks. The upper right block is the Jacobian w.r.t. the original state x. The left upper block is the Jacobian w.r.t. the coefficients xi.
    tracked_terms defines a list of list of terms who may evolve. 

    We only consider, for all computations, the library terms that are either active in at least on equation (non-zero) or the ones corresponding to tracked terms (column-wise)
    """

    # Find all columns (library terms) that are active at least once, and the columnns corresponding to tracked terms.
    non_zero_coeffs = utils.non_zero_columns(coeffs)
    tracked_columns = {col for row in tracked_terms for col in row}
    to_lambdify = sorted(set(non_zero_coeffs) | tracked_columns) # terms of interest, indices w.r.t. the entire library

    # turn into symbols (we're turning the entire library into symbols, could create issues?)
    var_symbols, library_symbols = sympify_str(variables, library_terms)
    
    # we differentiate only the selected entries of the matrix, this is essentially dTheta/dt (a block)
    lambdified_derivatives, symbolic_derivatives = differentiate_library(var_symbols, library_symbols, to_lambdify)

    # now we create the upper right block, we lambdified the entire library and select the necessary entries at runtime (from tracked_terms)
    lambdified_library = lambdify_library(var_symbols, library_symbols)

    return lambdified_derivatives, lambdified_library, library_symbols, to_lambdify, symbolic_derivatives