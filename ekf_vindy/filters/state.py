""" 
Base classes for the augmented state, as described in the "Online learning in bifurcating dynamic systems via SINDy
and Kalman filtering" paper
"""
import numpy as np

class State:
    """
    State base class, contains the original state x and the SINDy or VINDy coefficients phi. For the moment everything is treated
    as a NumPy array, we don't really need to track gradients when the network has already been trained (in this case). 

    Attributes
    ----------
    t : float
        Time instant to which this state refers.
    x : np.ndarray
        Actual vector in R^n representing the state of the original dynamical system, either in latent space (with VAE) or the original one.
    xi_tilde : np.ndarray
        Vector of coefficients, obtained with VINDy or SINDy, that we are tracking. Ordered sequentially from left to right, top to bottom looking at the ODE system.
    x_cal : np.ndarray
        Augmented state, i.e., the concatenation of x and xi_tilde.
    sdev : np.ndarray
        Standard deviation of each component of the state.
    state_with_sdev : np.ndarray
        Augmented state concatenated with the standard deviations of each component.
    cov: np.ndarray
        Covariance matrix of x and phi.
    """
    def __init__(self, t: float, x: np.ndarray, xi_tilde: np.ndarray, cov: np.ndarray):
        self._t = t
        self._x = x.reshape(-1, 1)
        self._xi_tilde = xi_tilde.reshape(-1, 1)
        self._cov = cov.squeeze()

    @property
    def x(self) -> np.ndarray:
        return self._x
    
    @property 
    def xi_tilde(self) -> np.ndarray:
        return self._xi_tilde
    
    @property
    def x_cal(self) -> np.ndarray:
        """
        This is the "caligraphic x" from the paper i.e., the augmented state. Useful in the updated step.
        """
        return np.vstack([self._x, self._xi_tilde])

    @property
    def t(self) -> float:
        return self._t

    @property
    def sdev(self) -> np.ndarray:
        return np.sqrt(np.diag(self._cov)).reshape(-1, 1)

    @property
    def state_with_sdev(self) -> np.ndarray:
        return np.hstack([self.x_cal, self.sdev])

    @property
    def cov(self) -> np.ndarray:
        """
        Note that, in general, the entire covariance matrix P will be dense, despite the fact that initially we can have a diagonal covariance.
        """
        return self._cov
    

class StateHistory:
    """ 
    Utility class to store the history of states, for easy fetching.
    """
    def __init__(self):
        """
        Initialize the state history.
        """
        self._states = []

    def append(self, state: State):
        if state is None:
            raise ValueError("State cannot be None.")        
        self._states.append(state)

    @property
    def x_states(self) -> np.ndarray: 
        """ 
        The states of the dynamical system, without coefficients, in an array.
        """
        self._assert_not_empty()
        # Squeeze to get output of shape (T, n) instead of (T, n). Where T are the time instances.
        return np.stack([state.x.squeeze() for state in self._states])

    @property 
    def xi_tilde_states(self) -> np.ndarray:
        """ 
        History of coefficient evolution. Currently assume it is flattened.
        """
        self._assert_not_empty()
        return np.stack([state.xi_tilde.squeeze() for state in self._states])
    @property 
    def xcal_states(self) -> np.ndarray:
        self._assert_not_empty()
        return np.stack([state.x_cal.squeeze() for state in self._states])

    @property 
    def sdev_states(self) -> np.ndarray:
        self._assert_not_empty()
        return np.stack([state.sdev.squeeze() for state in self._states])

    @property
    def xcals_sdevs(self) -> np.ndarray:
        self._assert_not_empty()
        return np.stack([state.state_with_sdev for state in self._states])

    @property
    def cov_states(self) -> np.ndarray:
        self._assert_not_empty()
        return np.stack([state.cov for state in self._states])

    @property
    def length(self):
        """
        The length of the state history.
        """
        return len(self._states)
    
    @property
    def last(self) -> State:
        """
        The last state of the history (i.e., the previous state, if you are filtering)
        """
        self._assert_not_empty()
        return self._states[-1] if self._states else None
    
    def _assert_not_empty(self):
        """
        Assert that the state history is not empty.
        """
        if not self._states:
            raise ValueError("State history is empty.")