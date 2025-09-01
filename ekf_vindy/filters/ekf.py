

# TODO: Input Jacobian, obtained somewhere else I guess, with the help of SINDy library.
# TODO: Tomorrow maybe play with the SINDy library to understand, as they do in the Paolo's repo, how to compute the derivatives in a straight-forward manner.
import numpy as np
from typing import List, Callable
from state import State, StateHistory
from jacobian_utils import lambdified_jacobian_blocks

class EKF:

    def __init__(self, 
                 variables: List[str], 
                 library_terms: List[str], 
                 tracked_terms: List[List[int]], 
                 coeffs: np.ndarray):
        
        self.states = StateHistory()
        self.tracked_terms = tracked_terms
        self.coeffs = coeffs
        self.p = len(library_terms) 
        self.n = len(variables) 

        # get sparse lambdified blocks for Jacobian computation
        self.lambdified_derivatives, self.right_block_lambdas = lambdified_jacobian_blocks(variables, library_terms, tracked_terms, coeffs)  
        
    def jacobian_f(self, state: State, 
                big_xi_t: np.ndarray):
        """ Note, what I call "big_xi_t" is Xi transpose, so an (n x p) matrix """
        # compute \Theta_t
        big_theta_t = np.zeros((self.n, self.p))
        for i, (row_lambdas, row_indices) in enumerate(zip(self.lambdified_derivatives, self.tracked_terms)):
            for lam, col in zip(row_lambdas, row_indices):
                big_theta_t[i, col] = lam(state.x[i])  # or lam(x) if lambda expects full x
        
        left_upper_block = big_xi_t @ big_theta_t
        right_upper_block = np.zeros((self.n, len(self.tracked_terms)))
        ### the right upper block fuuuuuuuuck right this as a tensor product or sth...
        # very inefficient... but it should do for our easy enough systems...

        return None