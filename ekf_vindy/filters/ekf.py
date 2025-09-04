# TODO: Have a callable object that is like a rhs of an equation, to feed directly into SciPy solvers. You don't need that, just one integration step really
#       Euler or RK, even probabilistic solver step lol.
# TODO: Properly format xi_t, write "RK" or "Euler" step prediction.

import numpy as np
from typing import List
from state import State, StateHistory
from jacobian_utils import lambdified_jacobian_blocks

class EKF:

    """
    Extended Kalman Filter for systems identified with SINDy or VINDy. The state is augmented to include both the original state variables and the coefficients of the identified system.
    Attributes
    ----------
    states : StateHistory
        History of states we infer.
    n : int
        Dimension of the original state space (no parameters accounted for).
    lambdified_derivatives : List[List[Callable]]
        Lambdified functions for the derivatives of the library terms with respect to the state variables. We only consider the terms "of interest". 
        A term of interest is either given as non-zero by SINDy (on any equation) or is being tracked (again, you consider the entire column corresponding to that term).
        This is a rectangular (or square) "matrix of callables".
    lambdified_library : List[Callable]
        Lambdified functions for the library terms of interest.
    terms_of_interest : List[int]
        Indices of the library terms that are part of the identified system, such indices refer to the entire library (not sparse one).
    symbolic_derivatives : List[List[sympy.Expr]]
        Symbolic expressions for the derivatives of the library terms with respect to the state variables.
    tracked_terms : List[List[int]]
        Indices of the library terms that are being tracked (i.e., whose coefficients are included in the state).
    coeffs : np.ndarray
        Coefficient matrix of the identified system, compressed to only the terms of interest. At runtime we simply replace the coefficients of the tracked terms with their current values from the state.
    library_symbols : List[sympy.Symbol]
        Symbolic representations of the library terms of interest.
    n_tracked_terms : int
        Total number of tracked terms.
    """
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
        self.library_symbols, \
        self.terms_of_interest, \
        self.symbolic_derivatives = lambdified_jacobian_blocks(variables, library_terms, tracked_terms, coeffs)

        
        """ tracked_terms with new indices, after compressing sparse system e.g. tracked = [[2, 3], [3, 5]] with to_lambdify = [1, 2, 3, 5](active terms) will become tracked = [[1, 2], [2, 3]],
            if it was to_lambdify = [2, 3, 5] it wouldd be tracked = [[0, 1], [1, 2]], and we select only the library terms we care about. """
        self.tracked_terms = [[{col: i for i, col in enumerate(self.terms_of_interest)}[col] for col in row] for row in tracked_terms]
        self.coeffs = coeffs[:, self.terms_of_interest] # compress the coeffs matrix to only the terms of interest
     
        """ we have a vector of the lambdified library terms "of interest", indices therefore correspond to compressed system
            we can use the same indices to access the tracked terms, they all indicate the same term in the sparse library """
        self.lambdified_library = [self.lambdified_library[idx] for idx in self.terms_of_interest]
        self.library_symbols = [self.library_symbols[idx] for idx in self.terms_of_interest]
        self.n_tracked_terms = sum(len(inner) for inner in self.tracked_terms)
        self.p = len(self.terms_of_interest) + self.n_tracked_terms
    
    def evaluate_f(self, x: np.ndarray):
        """ Evaluate the dynamics f at the given state. This is Theta(x), we don't account for the coefficients xi here. """
        theta_xt = np.array([f(x) for f in self.lambdified_library])
        return theta_xt

    def big_xi(self, tracked_xi: np.ndarray):
        """ Compute the coefficient matrix, replacing the tracked terms with their current values from the state. """
        # we ASSUME that state.xi has the same ordering as self.tracked_terms i.e. from left to right, top to bottom
        # CHECK THIS
        big_xi = np.array(self.coeffs, copy=True)
        k = 0
        for i, row in enumerate(self.tracked_terms):
            for col in row:
                big_xi[i, col] = tracked_xi[k]
                k += 1
        return big_xi
    
    def jacobian_f(self, state: State, 
                big_xi_t: np.ndarray):
        """ Note, what I call "big_xi_t" is Xi transpose, so an (n x p) matrix """
        
        # compute dTheta_t/dx (p x n matrix), here Theta is not the entire library, just the terms of interest )
        big_theta_t = np.zeros((self.n, self.p))

        for i, row in enumerate(self.lambdified_derivatives):
            for j, f_j in enumerate(row):
                big_theta_t[i, j] = f_j(state.x[i])

        left_upper_block = big_xi_t @ big_theta_t
        right_upper_block = np.zeros((self.n, len(self.n_tracked_terms))) 

        # fill right upper block with entries, this is essentially a tensor product (as shown in the paper)
        for i, row in enumerate(self.tracked_terms):
            for j in row:
                right_upper_block[i, j] = self.lambdified_library[j](state)

        # pad with zeroes
        jacobian = np.hstack([left_upper_block, right_upper_block])
        jacobian = np.vstack((jacobian), np.zeros((self.p, self.n + self.n_tracked_terms))) 

        return jacobian