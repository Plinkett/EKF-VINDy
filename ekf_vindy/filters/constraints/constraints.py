import numpy as np
from abc import ABC, abstractmethod

"""
Base Constraint class. You pass this object to the filter and the constraint application is handled internally.
We compute the Jacobian symbolically, in the future maybe use autodiff.

In this ca
"""

class Constraint(ABC):  
    
    def __init__(self, R: np.ndarray):
        self._R = R
        self._assert_valid()
    
    @abstractmethod
    def innovation(self, x: np.ndarray):
        """ In the subclasses you will define the behaviour of this constraint, same goes for self.jacobian() and self.obs_processing """
        pass

    @abstractmethod
    def jacobian(self, x: np.ndarray):
        """ Jacobian of the constraint with respect to the augmented state."""
        pass
    
    @property
    def R(self):
        return self._R
    
    def _assert_valid(self):
        if self._R.ndim != 2 or (self._R.shape[0] != self._R.shape[1]):
            raise ValueError('R must be a square matrix')
        
class ConservationDuffingCubic(Constraint):
    """
    Energy conservation constraint in Duffing oscillator with linear and cubic terms.
    Oscillator of the form:
    
    dot{x_0} = x_1
    dot{x_1} = -\alpha x_0 - \beta x_0^3
    
    All annoying shaping is done here... we give everything correctly formatted to the filter.
    Parameters
    ----------
    R : np.ndarray
        Observation noise covariance matrix.
    full_dimension : int
        Dimension of the augmented state (for padding the Jacobian).
    x0 : np.ndarray
        Initial state (for computing the initial energy level), non augmented.
    """
    def __init__(self, R: np.ndarray, x0: np.ndarray, full_dimension: int):
        self._full_dimension = full_dimension
        self._x0 = x0
        super().__init__(R)

    def _energy(self, x_cal: np.ndarray):
        """
        Energy associated to the current state (Hamiltonian?), given the parameters of the system.
        """
        kinetic = 0.5 * x_cal[1]**2 
        potential = 0.5 * self._alpha * x_cal[0]**2 + 0.25 * self._beta * x_cal[0]**4
        
        return kinetic + potential
    
    def innovation(self, x: np.ndarray):
        """ Innovation, by comparing with the initial energy level, pseudo-observation is zero with very low variance (hyperameter)."""
        energy_0 = self._energy(self._x0)
        energy_t = self._energy(x)
        
        return np.array([energy_0 - energy_t]).reshape(-1, 1)

    def jacobian(self, x_cal: np.ndarray):
        """
        Jacobian of the constraint with respect to the augmented state.
        """
        # for readability
        alpha = x_cal[2]
        beta = x_cal[3]
        
        dH_dx = np.array([alpha * x_cal[0] + beta * x_cal[0]**3, x_cal[1]])
        dH_dx = dH_dx.reshape(1, -1)
        dH_dx = np.hstack((dH_dx, np.zeros((1, self._full_dimension - 2)))) # -2 for the non-augmented state dimension
    
        return dH_dx

class LossDuffingCubic(Constraint):
    def __init__(self, R: np.ndarray, x0: np.ndarray, full_dimension: int):
        self._full_dimension = full_dimension
        self._x0 = x0
        
        """ Alpha and beta may also vary slightl, although we consider any loss of energy associated to that to be negligible per step, """
        self._last_alpha = None
        self._last_beta = None
        super().__init__(R)

    def _energy(self, x_cal: np.ndarray):
        """ Energy associated to the current state (Hamiltonian?), given the parameters of the system. """
        kinetic = 0.5 * x_cal[1]**2
        potential = 0.5 * x_cal[2] * x_cal[0]**2 + 0.25 * x_cal[3] * x_cal[0]**4

        return kinetic + potential

    def _loss_from_obs(self, obs_curr: np.ndarray, obs_prev: np.ndarray):
        """ Loss as the difference in energy between two consecutive observations. """
        return self._energy(obs_curr) - self._energy(obs_prev)
    
    def innovation(self, x_cal, ):
    
    

    