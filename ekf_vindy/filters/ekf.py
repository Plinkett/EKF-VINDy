
# TODO: Have a callable object that is like a rhs of an equation, to feed directly into SciPy solvers. You don't need that, just one integration step really
#       Euler or RK, even probabilistic solver step lol.

# TODO: Properly format xi_t, write "RK" or "Euler" step prediction.

import numpy as np
from typing import List
from state import State, StateHistory
from jacobian_utils import lambdified_jacobian_blocks

class EKF:

    def __init__(self, 
                 variables: List[str], 
                 library_terms: List[str], 
                 tracked_terms: List[List[int]], 
                 coeffs: np.ndarray):
        
        self.states = StateHistory()
        self.n = len(variables) 

        # get lambdified sparse system terms and its derivative, symbolic partial derivatives, and the indices of terms of interest (w.r.t. the whole library)
        self.lambdified_derivatives, \
        self.lambdified_library, \
        self.system_terms, \
        self.symbolic_derivatives = lambdified_jacobian_blocks(variables, library_terms, tracked_terms, coeffs)

        """ tracked_terms with new indices, after compressing sparse system e.g. tracked = [[2, 3], [3, 5]] with to_lambdify = [1, 2, 3, 5](active terms) will become tracked = [[1, 2], [2, 3]],
            if it was to_lambdify = [2, 3, 5] it would be tracked = [[0, 1], [1, 2]], and we select only the library terms we care about. """
        self.tracked_terms = [[{col: i for i, col in enumerate(self.system_terms)}[col] for col in row] for row in tracked_terms]
        self.lambdified_library = [self.lambdified_library[idx] for idx in self.system_terms]
        self.n_tracked_terms = sum(len(inner) for inner in self.tracked_terms)

    def _xi_t(self):
         # here you should build the matrix of terms, or get it directly from your state, should be saved like that as well...
         # you don't really need to keep it vectorized
         return None
    
    def jacobian_f(self, state: State, 
                big_xi_t: np.ndarray):
        """ Note, what I call "big_xi_t" is Xi transpose, so an (n x p) matrix """
        
        # compute dTheta_t/dx (p x n matrix), here Theta is not the entire library, just the terms of interest (from self.system_terms)
        p = len(self.system_terms)
        big_theta_t = np.zeros((self.n, p))

        for i, row in enumerate(self.lambdified_derivatives):
            for j, f_j in enumerate(row):
                big_theta_t[i, j] = f_j(state.x[i])

        left_upper_block = big_xi_t @ big_theta_t
        right_upper_block = np.zeros((self.n, len(self.n_tracked_terms))) 

        # Fill right upper block with entries, this is essentially a tensor product (as shown in the paper)
        for i, row in enumerate(self.tracked_terms):
            for j in row:
                right_upper_block[i, j] = self.lambdified_library[j](state)

        jacobian = np.hstack([left_upper_block, right_upper_block])
        jacobian = np.vstack((jacobian), np.zeros((p, self.n + self.n_tracked_terms))) # pad with zeroes

        return jacobian