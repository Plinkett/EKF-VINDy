# TODO: Once you verify it works make things private
# TODO: You could work with the square root (and make it more robust) if you properly discretize the Lyapunov equation (Van-Loan formula)

import numpy as np
from tqdm import tqdm
from typing import Iterable
from scipy.linalg import cho_factor, cho_solve
from ekf_vindy.filters.state import State, StateHistory
from ekf_vindy.filters.constraints import Constraint
from ekf_vindy.filters.config import DynamicsConfig
from ekf_vindy.jacobian_utils import lambdified_jacobian_blocks
from ekf_vindy.utils import integration_step

class EKF:
    """
    Extended Kalman Filter for systems identified with SINDy or VINDy. The state is augmented to include both the original state variables and the coefficients of the identified system.
    We assume the process noise R and measurement noise Q to be constant. 
    Attributes
    ----------

    states : StateHistory
        History of states we infer.
    observations : List[np.ndarray]
        List of observations we receive.
    Q : np.ndarray
        Process noise covariance.
    R : np.ndarray
        Measurement / observation noise covariance.
    n : int
        Dimension of the original state space (no parameters accounted for).
    lambdified_derivatives : List[List[Callable]]
        Lambdified functions for the derivatives of the library terms with respect to the state variables. We only consider the terms "of interest". A term of interest is either given as non-zero by SINDy (on any equation) or is being tracked (again, you consider the entire column corresponding to that term).
    lambdified_library : List[Callable]
        Lambdified functions for the library terms of interest.
    library_symbols : List[sympy.Symbol]
        Symbolic representations of the library terms of interest.
    terms_of_interest : List[int]
        Indices of the library terms that are part of the identified system, such indices refer to the entire library (not sparse one).
    symbolic_derivatives : List[List[sympy.Expr]]
        Symbolic expressions for the derivatives of the library terms with respect to the state variables.
    tracked_terms : List[List[int]]
        Indices of the library terms that are being tracked (i.e., whose coefficients are included in the state), again, these indices refer to the entire library (not sparse identified by SINDy).
    coeffs : np.ndarray
        Coefficient matrix of the identified system, compressed to only the terms of interest (either non zero or tracked, both equation-wise). At runtime we simply replace the coefficients of the tracked terms with their current values from the state.
    n_tracked_terms : int
        Total number of tracked terms.
    p : int
        The number of library terms we are considering, the usual p you see in the Xi matrix.
    H : np.ndarray
        Observation model, here we just select the original state from the augmented one. In the future, we can consider non-trivial observation models (which also need to be linearized).
    I : np.ndarray
        Identity matrix of size (self.n + self.n_tracked_terms), just as a utility for the update step.
    integration_rule : str
        What integration rule to use during the prediction step, either Euler or RK4 step.
    """
    def __init__(self, x0 : np.ndarray, p0 : np.ndarray, t0: float = 0.0, *, config: DynamicsConfig, integration_rule = 'Euler'):
        self._states = StateHistory()
        self._observations = [x0.reshape(-1, 1)]  
        
        # set process and observation noise
        self.Q = config.Q
        self.R = config.R
        self.n = len(config.variables) 

        # get lambdified sparse system terms and its derivative, symbolic partial derivatives, and the indices of terms of interest (w.r.t. the whole library)
        self.lambdified_derivatives, \
        self.lambdified_library, \
        self.library_symbols, \
        self.terms_of_interest, \
        self.symbolic_derivatives = lambdified_jacobian_blocks(config.variables, config.library_terms, config.tracked_terms, config.initial_coeffs)
        
        """ 
        tracked_terms with new indices, after compressing sparse system e.g. tracked = [[2, 3], [3, 5]] with to_lambdify = [1, 2, 3, 5](active terms) will become tracked = [[1, 2], [2, 3]],
        if it was to_lambdify = [2, 3, 5] it would be tracked = [[0, 1], [1, 2]], and we select only the library terms we care about. 
        """
        self.tracked_terms = [[{col: i for i, col in enumerate(self.terms_of_interest)}[col] for col in row] for row in config.tracked_terms]
        self.coeffs = config.initial_coeffs[:, self.terms_of_interest] 

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
        xi_tilde_0 = np.array([config.initial_coeffs[i, col] for i, row in enumerate(config.tracked_terms) for col in row])
        self._states.append(State(t0, x0, xi_tilde_0, p0))
    
    def run_filter(self, dts: Iterable[np.ndarray], observations: Iterable[np.ndarray], constraint: Constraint | None = None):  
        """ 
        Main call to this class. We assume dt and observations to be of the same length.
        We take either the whole vector of (dt, observations) or do it in an online fashion with yield
        """
        for dt, observation in tqdm(zip(dts, observations), total=len(dts), desc="Processing"):
            previous_state = self._states.last
            state_upd = self._step(previous_state, dt, observation.reshape(-1, 1), constraint)
            self._states.append(state_upd)

    def _evaluate_theta(self, x: np.ndarray):
        """ 
        Evaluate the sparse library at the given state. This is Theta(x), we don't account for the coefficients xi here. 
        This returns a vector of size (p, 1)
        """
        # annoying lambda functions return scalars, or (1, 1) vectors... .item() should fix that
        theta_xt = np.array([f(x).item() if isinstance(f(x), np.ndarray) else f(x) 
                     for f in self.lambdified_library])
        
        return theta_xt.reshape(-1, 1)

    def _evaluate_f(self, x: np.ndarray, big_xi_t: np.ndarray):
        """
        Evaluate the right-hand side of the dynamical system
        """
        theta = self._evaluate_theta(x) 
        
        return big_xi_t @ theta

    def _predict(self, state: State, dt: float):
        """ Predict state (non-augmented, so no xi_tilde) and covariance (for augmented state) through numerical integration """
        big_xi_t = self._big_xi_t(state.xi_tilde)
        evaluate_f = lambda x: self._evaluate_f(x, big_xi_t)
        x_pred = integration_step(state.x, evaluate_f, dt, self.integration_rule).reshape(-1, 1)

        # Jacobian computed at predicted mean
        jacobian = self._jacobian_f(x_pred, big_xi_t)

        # this is F @ P + P @ F.T + Q in the paper, it's just an anonymous function to be passed to our integrator. Enforce symmetry with 1/2 * (P + P.T)
        covariance_f = lambda p: jacobian @ p + p @ jacobian.T + self.Q
        p_pred = integration_step(state.cov, covariance_f, dt, self.integration_rule)
        p_pred = 0.5 * (p_pred + p_pred.T)

        return x_pred, p_pred
        
    def _update(self, x_pred: np.ndarray, p_pred: np.ndarray, xi_tilde: np.ndarray, observation: np.ndarray):
        """
        Compute updated estimate of state and covariance. For the moment, we consider a trivial observation model that selects a few state components.
        xi is just a vectorized version of the tracked coefficients, size (n_tracked_terms x 1) 
        
        We need many weird numerical tricks here to ensure numerical stability... especially because of the energy constraints...
        """
        # Innovation covariance 
        s = self.H @ p_pred @ self.H.T + self.R

        # Compute Kalman gain safely
        cho, lower = cho_factor(s)
        gain = cho_solve((cho, lower), (p_pred @ self.H.T).T).T

        # Innovation
        innovation = observation - x_pred

        # Update full state (stacked with parameters)
        cal_x_pred = np.vstack([x_pred, xi_tilde])
        cal_x_updt = cal_x_pred + gain @ innovation
        x_updt, xi_tilde_updt = cal_x_updt[:self.n], cal_x_updt[-self.n_tracked_terms:]

        # Joseph-form covariance update + symmetry
        id_minus_KH = self.I - gain @ self.H
        p_updt = id_minus_KH @ p_pred @ id_minus_KH.T + gain @ self.R @ gain.T
        p_updt = 0.5 * (p_updt + p_updt.T)

        return x_updt, xi_tilde_updt, p_updt
    
    def update_constraint(self, x_cal_uc: np.ndarray, p_uc: np.ndarray, constr: Constraint, 
                          x_cal_pred: np.ndarray, obs: np.ndarray, dt: float, only_x = False):
        x_cal_prev = self._states.last.x_cal
        jacobian_h = constr.jacobian(x_cal_pred, dt)
        
        # innovation covariance
        s = jacobian_h @ p_uc @ jacobian_h.T + constr.R

        cho, lower = cho_factor(s)
        gain = cho_solve((cho, lower), (p_uc @ jacobian_h.T).T).T

        # innovation
        innovation = constr.innovation(x_cal_prev, self._observations[-1], obs, dt)
        
        # update mean and extract x and xi_tilde
        cal_x_constr = x_cal_uc + gain @ innovation
        x_constr = cal_x_constr[:self.n]
        xi_tilde_constr = cal_x_constr[-self.n_tracked_terms:]

        # update covariance under constraint
        id_minus_KH = self.I - gain @ jacobian_h
        p_constr = id_minus_KH @ p_uc @ id_minus_KH.T + gain @ constr.R @ gain.T
        p_constr = 0.5 * (p_constr + p_constr.T)
       
        return x_constr, xi_tilde_constr, p_constr

    # def _update_constraint(self, x_uc: np.ndarray, p_uc: np.ndarray,
    #                    xi_tilde_uc: np.ndarray, constr: Constraint, observation: np.ndarray, dt: float, 
    #                    x_pred: np.ndarray, xi_pred: np.ndarray, only_x = False):
    #     """
    #     Pseudo-observation update that only corrects the dynamical system state x,
    #     leaving parameters xi_tilde and their covariance untouched.
    #     """

    #     # energy difference estimated from current and previous observation 
    #     loss_obs = constr.aux_function(observation, self.observations[-1]) 
        
    #     # predicted energy difference from model 
    #     prev_state = self._states.last
    #     loss_predicted = constr.innovation(prev_state.x, prev_state.xi_tilde, dt)

    #     eigvals, eigvecs = np.linalg.eigh(p_uc)
    #     eigvals = np.clip(eigvals, 1e-12, None)
    #     p_uc_copy = eigvecs @ np.diag(eigvals) @ eigvecs.T

    #     # h_j = constr.jacobian(x_uc)  
    #     h_j = constr.jacobian(x_pred.squeeze(), xi_pred.squeeze(), dt)

    #     # --- extract state block ---
    #     if only_x:
    #         p_uc_copy = p_uc[:self.n, :self.n]
    #         h_j = h_j[:, :self.n]

    #     # innovation covariance
    #     s = h_j @ p_uc_copy @ h_j.T + constr.R
    #     s += 1e-12 * np.eye(s.shape[0])
  
    #     # Kalman gain restricted to x-block
    #     try:
    #         cho, lower = cho_factor(s)
    #         gain = cho_solve((cho, lower), (p_uc_copy @ h_j.T).T).T
    #     except np.linalg.LinAlgError:
    #         gain = p_uc_copy @ h_j.T @ np.linalg.pinv(s)

    #     innovation = (loss_obs - loss_predicted).reshape(-1, 1)
    #     # print(f'Innovation (constraint): {innovation.ravel()}')
    #     # update whole state or only x and covariance blocks related to x
    #     if only_x:
    #         x_constr = x_uc + gain @ innovation 
    #         xi_tilde_constr = xi_tilde_uc
    #         id_minus_KH = np.eye(self.n) - gain @ h_j
    #         p_xx_constr = id_minus_KH @ p_uc_copy @ id_minus_KH.T + gain @ constr.R @ gain.T
    #         p_constr = p_uc.copy()
    #         p_constr[:self.n, :self.n] = p_xx_constr
    #     else:
    #         cal_x_uc = np.vstack([x_uc, xi_tilde_uc])
    #         cal_x_constr = cal_x_uc + gain @ innovation 
    #         x_constr = cal_x_constr[:self.n]
    #         xi_tilde_constr = cal_x_constr[-self.n_tracked_terms:]
    #         id_minus_KH = self.I - gain @ h_j
    #         p_constr = id_minus_KH @ p_uc_copy @ id_minus_KH.T + gain @ constr.R @ gain.T

    #     p_constr = 0.5 * (p_constr + p_constr.T)
   
    #     return x_constr, xi_tilde_constr, p_constr
    
    def _step(self, curr_state: State, dt: float, observation: np.ndarray, constr: Constraint | None = None):
        """ Stitch prediction and update steps"""
        
        x_pred, p_pred = self._predict(curr_state, dt)
        x_updt, xi_tilde_updt, p_updt = self._update(x_pred, p_pred, curr_state.xi_tilde, observation)
        
        if constr: 
            x_updt, xi_tilde_updt, p_updt = self._update_constraint(x_updt, p_updt, xi_tilde_updt, constr, observation, dt, x_pred, curr_state.xi_tilde, False)

        state_upd = State(curr_state.t + dt, x_updt, xi_tilde_updt, p_updt)
        self._observations.append(observation)
        
        return state_upd
    
    def _big_xi_t(self, xi_tilde: np.ndarray):
        """ 
        Compute the coefficient matrix, Xi transposed, replacing the tracked terms with their current values from the state. 
        We assume that the vector state.xi are the tracked terms in the Xi matrix, ordered sequentially from left to right, and from top to bottom.
        """
        big_xi = np.array(self.coeffs, copy=True)
        xi_tilde_flat = xi_tilde.ravel() # flatten just in case
        k = 0
        for i, row in enumerate(self.tracked_terms):
            for col in row:
                big_xi[i, col] = xi_tilde_flat[k]
                k += 1
        return big_xi
    
    def _jacobian_f(self, x_pred: np.ndarray, big_xi_t: np.ndarray):
        """ 
        We evaluate the Jacobian at the predicted mean. 
        What we call "big_xi_t" is Xi transpose, so an (n x p) matrix, and it uses the current value of the tracked coefficients. 
        """

        # compute dTheta_t/dx (p x n matrix), here Theta is not the entire library, just the terms of interest
        dt_theta_t = np.zeros((self.n, self.p))

        for i, row in enumerate(self.lambdified_derivatives):
            for j, f_j in enumerate(row):
                dt_theta_t[i, j] = f_j(x_pred)
        
        left_upper_block = big_xi_t @ dt_theta_t.T
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
    
    @property
    def states(self):
        return self._states
    
    @property
    def observations(self):
        return self._observations
    
    def _tuning(self):
        # Must be done at some point if covariance is not provided...
        # May adapt Q and R at runtime?
        # See: Adaptive Adjustment of Noise Covariance in Kalman Filter for Dynamic State Estimation by Akhlaghi et. al. (2017)
        raise NotImplementedError