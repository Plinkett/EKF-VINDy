""" 
Utils for converting strings (usually obtain from PySINDy library) into SymPy symbols for 
symbolic differentiation, and easy Jacobian computation for EKF. 
"""
from typing import List
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

def differentiate_library(variables: List[sp.Symbol], library: List[sp.Symbol]):
    """  
    Returns a matrix of partial derivatives, lamdbified and in symbolic format.
    Differentiates each library term w.r.t. each variable.
    """
    symbolic_derivatives = []
    lambdified_derivatives = []

    # Not the best since you effectevily iterate over twice. Easy to read though, and we are dealing with 
    # a small number of items anyway
    symbolic_derivatives = [[sp.diff(term, var) for term in library] for var in variables]
    lambdified_derivatives = [[sp.lambdify(variables, dterm) for dterm in sym_row] for sym_row in symbolic_derivatives]

    return lambdified_derivatives, symbolic_derivatives

def lambdify_library(variables: List[sp.Symbol], library: List[sp.Symbol]):
    """ Returns lambdified version of library """
    return [sp.lambdify(variables, term) for term in library]

def compute_jacobian(variables: List[str], 
                     library_terms: List[str],
                     tracked_terms: List[List[int]],
                     coeffs: np.ndarray):
    """ 
    tracked_terms define a list, one per equation of the ODE system, of
    indices of coefficients to track. 
 
    We compute the Jacobian in blocks. 
    The upper right block is the Jacobian w.r.t. the original state x. 
    The left upper block is the Jacobian w.r.t. the coefficients xi.
    """
    
    """
    Compute only derivatives for the non-zero coefficients or
    the tracked coefficients.
    """
    
    non_zero_coeffs = utils.find_non_zero(coeffs)
    
    # list of indices for which to evaluate the partial derivative (avoid duplicates)
    to_lambdify = [sorted(list(set(nz).union(set(tr)))) for nz, tr in zip(non_zero_coeffs, tracked_terms)]
    
    
    # indices partial derivatives to lambdify (avoid duplicates)
    #to_lambdify = sorted(list(set().union(*to_evaluate)))
    # but the actual partial derivative must be multiplied by the parameter
    # you have that...
        
    # sympify variables and library terms
    # just 
    var_symbols, library_symbols = sympify_str(variables, library_terms[to_lambdify])
    
    return None

