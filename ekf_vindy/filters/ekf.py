

# TODO: Input Jacobian, obtained somewhere else I guess, with the help of SINDy library.
# TODO: Tomorrow maybe play with the SINDy library to understand, as they do in the Paolo's repo, how to compute the derivatives in a straight-forward manner.
import numpy as np
from typing import List, Callable
from state import State, StateHistory
from jacobian_utils import lambdified_jacobian_blocks, lambdify_library

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
        """ Also, yes, this can be greatly optimized if using sparse matrices and/or vectorized operations """
        # compute \Theta_t
        big_theta_t = np.zeros((self.n, self.p))

        # for sparse systems, must fix tracked terms, they identify COLUMNS, equation-wise if it's not active 
        # and not tracked then you drop it... not like you are doing now.

        # for i, (row_lambdas, row_indices) in enumerate(zip(self.lambdified_derivatives, self.tracked_terms)):
        #     for lam, j in zip(row_lambdas, row_indices):
        #         big_theta_t[i, j] = lam(state.x[i])  
        
        # fill partial derivative matrix
        
        # matrix of coefficients (found by SINDy) times partial derivative matrix
        left_upper_block = big_xi_t @ big_theta_t
        right_upper_block = np.zeros((self.n, len(self.tracked_terms)))

        # Fill right upper block with entries, this is essentially a tensor product
        for i, row in enumerate(self.tracked_terms):
                for j, col in enumerate(row):
                    right_upper_block[i, j] = self.right_block_lambdas[col](state)



        return None