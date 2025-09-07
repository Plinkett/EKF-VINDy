# TODO: Make implementation more stable. Use Cholesky, square root, QR and some other stuff (?)
# TODO: Many things can be vectorized... but they compromise readability (and as a consequence ease of debugging)
# TODO: Maybe use a configuration class (@dataclass or something)
# TODO: Once you verify it works make things private
# TODO: Check that the right upper block is correct!!!
# TODO: You can remove the attribute "variables", can be inferred from generically from the dimension

import numpy as np
from typing import List, Callable, Iterable
from ekf_vindy.filters.state import State, StateHistory
from ekf_vindy.jacobian_utils import lambdified_jacobian_blocks
from tqdm import tqdm

class EKF:
    """
    Extended Kalman Filter for systems identified with SINDy or VINDy. The state is augmented to include both the original state variables and the coefficients of the identified system.
    We assume the process noise R and measurement noise Q to be constant. 
    Attributes
    ----------

    states : StateHistory
        History of states we infer.
    n : int
        Dimension of the original state space (no parameters accounted for).
    Q : np.ndarray
        Process noise covariance.
    R : np.ndarray
        Measurement / observation noise covariance.
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
        Indices of the library terms that are being tracked (i.e., whose coefficients are included in the state), again, these indices refer to the entire library (not sparse identified by SINDy).
    coeffs : np.ndarray
        Coefficient matrix of the identified system, compressed to only the terms of interest (either non zero or tracked, both equation-wise). At runtime we simply replace the coefficients of the tracked terms with their current values from the state.
    library_symbols : List[sympy.Symbol]
        Symbolic representations of the library terms of interest.
    n_tracked_terms : int
        Total number of tracked terms.
    H : np.ndarray
        Observation model, here we just select the original state from the augmented one. In the future, we can consider non-trivial observation models (which also need to be linearized).
    I : np.ndarray
        Identity matrix of size (self.n + self.n_tracked_terms), just as a utility for the update step.
    integration_rule : str
        What integration rule to use during the prediction step, either Euler or RK4 step.
    """
    def __init__(self, 
                 x0 : np.ndarray,
                 p0 : np.ndarray,
                 Q: np.ndarray,
                 R: np.ndarray,
                 variables: List[str], 
                 library_terms: List[str], 
                 tracked_terms: List[List[int]], 
                 initial_coeffs: np.ndarray,
                 t0 = 0.0,
                 integration_rule = 'Euler'):
        
        self._states = StateHistory()
        
        # set process and observation noise
        self.Q = Q
        self.R = R
        self.n = len(variables) 

        # get lambdified sparse system terms and its derivative, symbolic partial derivatives, and the indices of terms of interest (w.r.t. the whole library)
        self.lambdified_derivatives, \
        self.lambdified_library, \
        self.library_symbols, \
        self.terms_of_interest, \
        self.symbolic_derivatives = lambdified_jacobian_blocks(variables, library_terms, tracked_terms, initial_coeffs)
        
        """ 
        tracked_terms with new indices, after compressing sparse system e.g. tracked = [[2, 3], [3, 5]] with to_lambdify = [1, 2, 3, 5](active terms) will become tracked = [[1, 2], [2, 3]],
        if it was to_lambdify = [2, 3, 5] it would be tracked = [[0, 1], [1, 2]], and we select only the library terms we care about. 
        """
        self.tracked_terms = [[{col: i for i, col in enumerate(self.terms_of_interest)}[col] for col in row] for row in tracked_terms]
        self.coeffs = initial_coeffs[:, self.terms_of_interest] # compress the coeffs matrix to only the terms of interest
        # print(f'self.tracked_terms (compressed): {self.tracked_terms}')
        """ 
        we have a vector of the lambdified library terms "of interest", indices therefore correspond to compressed system
        we can use the same indices to access the tracked terms, they all indicate the same term in the sparse library 
        """
        self.lambdified_library = [self.lambdified_library[idx] for idx in self.terms_of_interest]
        self.library_symbols = [self.library_symbols[idx] for idx in self.terms_of_interest]
        
        self.n_tracked_terms = sum(len(inner) for inner in self.tracked_terms)
        self.p = len(self.terms_of_interest)
        self.H = np.eye(self.n, self.n + self.n_tracked_terms)
        self.I = np.eye(self.n + self.n_tracked_terms)
        self.integration_rule = integration_rule

        # save initial state (after extracting initial value of tracked terms, from initial_coeffs)
        xi0 = np.array([initial_coeffs[i, col] for i, row in enumerate(tracked_terms) for col in row])
        self._states.append(State(t0, x0, xi0, p0))
    
    def _evaluate_theta(self, x: np.ndarray):
        """ 
        Evaluate the sparse library at the given state. This is Theta(x), we don't account for the coefficients xi here. 
        This returns a vector of size (p, 1)
        """
        # annoying lambda functions return scalars, or (1, 1) vectors... .item() should fix that
        theta_xt = np.array([f(x).item() if isinstance(f(x), np.ndarray) else f(x) 
                     for f in self.lambdified_library])
        
        return theta_xt.reshape(-1, 1)
    
    def _evaluate_f(self, x: np.ndarray, xi: np.ndarray):
        """
        Evaluate the right-hand side of the dynamical system
        """
        theta = self._evaluate_theta(x) 
        xi_t = self._big_xi_t(xi)

        return xi_t @ theta

    def _predict(self, state: State, dt: float):
        """ Predict state (non-augmented, so no xi_tilde) and covariance (for augmented state) through numerical integration """
        
        evaluate_f = lambda x: self._evaluate_f(x, state.xi)
        x_pred = self._integration_step(state.x, evaluate_f, dt, self.integration_rule).reshape(-1, 1)

        # Jacobian computed at predicted mean
        jacobian = self._jacobian_f(x_pred, state.xi)

        # This is F * P + P * F^T + Q in the paper, it's just an anonymous function to be passed to our integrator.
        covariance_f = lambda p: jacobian @ p + p @ jacobian.T + self.Q
        p_pred = self._integration_step(state.cov, covariance_f, dt, self.integration_rule)
        # p_pred = jacobian @ state.cov @ jacobian.T + self.Q? Assuming a very small dt... better for discrete time SSMs, or use a probabilistic ODE solver lol.

        return x_pred, p_pred

    def _update(self, x_pred: np.ndarray, p_pred: np.ndarray, xi: np.ndarray, observation: np.ndarray):
        """
        Compute updated estimate of state and covariance. For the moment, we consider a trivial observation model that selects a few state components.
        xi is just a vectorized version (size n_tracked_terms x 1) of the tracked_terms
        """

        # Kalman gain (use square root implementation)
        g = p_pred @ self.H.T @ np.linalg.inv(self.H @ p_pred @ self.H.T + self.R)
        innovation = observation - x_pred 

        # predicted augmented state (caligraphic x in the paper)
        cal_x_pred = np.vstack([x_pred, xi])

        # updated state and covariance (with Joseph update, look it up)
        cal_x_updt = cal_x_pred + g @ innovation
        p_updt = (self.I - g @ self.H) @ p_pred @ (self.I - g @ self.H).T + g @ self.R @ g.T
        x_updt, xi_updt = cal_x_updt[:self.n], cal_x_updt[-self.n_tracked_terms:]
        
        return x_updt, xi_updt, p_updt

    def _step(self, curr_state: State, dt: float, observation: np.ndarray):
        """ Stitch prediction and update steps"""
        x_pred, p_pred = self._predict(curr_state, dt)
        x_updt, xi_updt, p_updt = self._update(x_pred, p_pred, curr_state.xi, observation)
        
        state_upd = State(curr_state.t + dt, x_updt, xi_updt, p_updt)
        return state_upd
    
    def run_filter(self, dts: Iterable[np.ndarray], observations: Iterable[np.ndarray], online = False):
        """ 
        Main call to this class. We assume dt and observations to be of the same length.
        We take either the whole vector of (dt, observations) or do it in an online fashion with yield
        """
        for dt, observation in tqdm(zip(dts, observations), total=len(dts), desc="Processing"):
            previous_state = self._states.last
            state_upd = self._step(previous_state, dt, observation.reshape(-1, 1))
            self._states.append(state_upd)
            
    @property
    def states(self):
        return self._states
    
    def _integration_step(self, y: np.ndarray, f: Callable, dt: float, method='Euler'):
        """
        A simple integration step (Euler or RK4). We assume an autonomous ODE i.e., no explicit time-dependence.
        This is for either evolving the state and/or solving the Lyapunov equation to obtain the predicted covariance.
        It is worth noting that we are assuming our linearization (through the Jacobian) holds for the intermediate steps of the RK4. 
        Ideally, you would compute the Jacobian at the intermediate states as well.
        """
        if method in ('Euler', 'FE', 'EF'):
            y_new = y + dt * f(y)
        elif method == 'RK4':
            y_k1 = f(y)
            y_k2 = f(y + dt/2 * y_k1)
            y_k3 = f(y + dt/2 * y_k2)
            y_k4 = f(y + dt * y_k3)
            y_new = y + dt/6 * (y_k1 + 2 * y_k2 + 2 * y_k3 + y_k4)
        else:
            raise ValueError("Integration method not supported: " + str(method))
        return y_new
    
    def _big_xi_t(self, tracked_xi: np.ndarray):
        """ 
        Compute the coefficient matrix, Xi transposed, replacing the tracked terms with their current values from the state. 
        We assume that the vector state.xi are the tracked terms in the Xi matrix, ordered sequentially from left to right, and from top to bottom.
        """
        big_xi = np.array(self.coeffs, copy=True)
        xi = tracked_xi.ravel() # flatten just in case
        k = 0
        for i, row in enumerate(self.tracked_terms):
            for col in row:
                big_xi[i, col] = xi[k]
                k += 1
        return big_xi
    
    def _jacobian_f(self, x_pred: np.ndarray, xi: np.ndarray):
        """ 
        We evaluate the Jacobian at the predicted mean. Note that "xi" is actually "xi_pred", but since it does not evolve we just pass what we already had.
        What I call "big_xi_t" is Xi transpose, so an (n x p) matrix 
        """

        # compute dTheta_t/dx (p x n matrix), here Theta is not the entire library, just the terms of interest
        dt_theta_t = np.zeros((self.n, self.p))

        for i, row in enumerate(self.lambdified_derivatives):
            for j, f_j in enumerate(row):
                dt_theta_t[i, j] = f_j(x_pred)
        
        left_upper_block = self._big_xi_t(xi) @ dt_theta_t.T
        right_upper_block = np.zeros((self.n, self.n_tracked_terms)) 

        # fill right upper block with entries, this is essentially a tensor product (as shown in the paper)    
        k = 0
        for i, row in enumerate(self.tracked_terms):
            for j in row:
                right_upper_block[i, k] = self.lambdified_library[j](x_pred)
                k += 1

        # pad with zeroes
        jacobian_top = np.hstack([left_upper_block, right_upper_block])
        jacobian = np.vstack([jacobian_top, np.zeros((self.n_tracked_terms, self.n + self.n_tracked_terms))])
        

        return jacobian

    def _tuning(self):
        # Must be done at some point if covariance is not provided...
        # May adapt Q and R at runtime?
        # See: Adaptive Adjustment of Noise Covariance in Kalman Filter for Dynamic State Estimation by Akhlaghi et. al. (2017)
        raise NotImplementedError