""" 
Utils for converting strings (usually obtain from PySINDy library) into SymPy symbols for 
symbolic differentiation, and easy Jacobian computation for EKF. 
"""
from typing import List, Callable
import numpy as np
import sympy as sp

def sympify_str(variables: List[str], library_terms: List[str]):
    """
    Gets library terms (for example, from PySINDy) and transform them into SymPy symbols.
    List of library terms can be obtained from model.get_feature_names()
    List of variables can be obtained with model.feature_names
    """
    
    feature_symbols = []
    var_symbols = sp.symbols('_'.join(variables))

    #annoying handling of symbols and their corresponding string names
    locals_dict = {name: symbol for name, symbol in zip(variables, var_symbols)} 
    
    # By default, SINDy uses other symbols for exponentiation and multiplication
    library_terms = [term.replace('^', '**').replace(' ', '*') for term in library_terms]
    
    feature_symbols = [sp.sympify(term, locals=locals_dict) for term in library_terms]

    return var_symbols, feature_symbols

def differentiate_library(variables: List[sp.Symbol], library: List[sp.Symbol]):
    """  
    Returns a matrix of partial derivatives, lamdbified and in symbolic format.
    Differentiates each library term w.r.t. each variable.
    """
    symbolic_derivatives = []
    lambdified_derivatives = []

    # Not the best since you effectevily iterate over twice. Cleaner though, and we are dealing with 
    # a small number of items anyway
    symbolic_derivatives = [[sp.diff(term, var) for term in library] for var in variables]
    lambdified_derivatives = [[sp.lambdify(variables, dterm) for dterm in sym_row] for sym_row in symbolic_derivatives]

    return lambdified_derivatives, symbolic_derivatives

def lambdify_library(variables: List[sp.Symbol], library: List[sp.Symbol]):
    """ Returns lambdified version of library """
    return [sp.lambdify(variables, term) for term in library]


    # for term in library_terms:
    #     # By default, SINDy uses other symbols for exponentiation and multiplication
    #     term = term.replace('^', '**').replace(' ', '*')
    #     try:
    #         feature_symbols.append(sp.sympify(term, locals={}))
    #     except SyntaxError as e:
    #         print(f"Error parsing term '{term}': {e}")
