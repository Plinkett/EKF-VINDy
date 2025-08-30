""" 
Base classes for the augmented state, as described in the EKF-SINDy paper by Rosafalco et al. (2024)
TODO: Structure of weights phi is missing, currently assumed to be handled flattened. Correct this.
TODO: Take into account the boolean map B, as shown in the paper.
"""
import numpy as np

class State:
    """
    State base class, contains the original state x and the SINDy or VINDy coefficients phi. For the moment everything is treated
    as a NumPy array, we don't really need to track gradients when the network has already been trained (in this case). 

    Attributes
    ----------
    x : np.ndarray
        Actual vector in R^n representing the state of the original dynamical system, either in latent space (with VAE) or the original one.
    phi : np.ndarray
        Vector of coefficients, obtained with VINDy or SINDy, that we are tracking. 
        Also called \xi in the "Online learning in bifurcating dynamic systems via SINDy and Kalman filtering" paper.
    cov: np.ndarray
        Covariance matrix of x and phi. I can't recall exactly right now, but I think it has a block structure, in that case it would be better
        to save the 2 blocks separately. 
    """
    def __init__(self, x: np.ndarray, phi: np.ndarray, cov: np.ndarray):
        self.x = x
        self.phi = phi
        self.cov = cov
        self.dim_x = x.size
        self.dim_phi = phi.size # assumed to be flattened dimension, in principle this is a matrix

    @property
    def x(self) -> np.ndarray:
        return self.x
    
    @property 
    def phi(self) -> np.ndarray:
        return self.phi
    
    @property
    def cov(self) -> np.ndarray:
        return self.cov
    
    @property
    def dim_x(self) -> int:
        return self.dim_x
    
    @property
    def dim_phi(self) -> int:
        return self.dim_phi
    
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
        """
        Append a state to the history.

        Parameters
        ----------
        state
            The state to be appended.
        """
        if state is None:
            raise ValueError("State cannot be None.")
        
        self._states.append(state)

    @property
    def x_states(self) -> np.ndarray: 
        """ 
        The states of the dynamical system, without coefficients, in an array.
        """
        self._assert_not_empty()
        # Squeeze to get output of shape (n, d) instead of (n, d, 1), assuming x is of shape (n,)
        return np.stack(state.x.squeeze() for state in self._states)

    @property 
    def phi_states(self) -> np.ndarray:
        """ 
        History of coefficient evolution, however they should be in the matrix structure... to correct later.
        Currently assume it is flattened.
        """
        self._assert_not_empty()
        # assume it is flattened...
        return np.stack(state.phi.squeeze() for state in self._states)

    @property
    def length(self):
        """
        The length of the state history.
        """
        return len(self._states)
    
    @property
    def last(self) -> State:
        """
        The last state of the history.
        """
        self._assert_not_empty()
        return self._states[-1] if self._states else None
    
    def _assert_not_empty(self):
        """
        Assert that the state history is not empty.
        """
        if not self._states:
            raise ValueError("State history is empty.")