import numpy as np
from abc import ABC, abstractmethod

"""
Base Constraint class. You pass this object to the filter and the constraint application is handled internally.
We compute the Jacobian symbolically, in the future maybe use autodiff.

In this case we assume :math:`\alpha` and :math:`\beta` to be fixed, since otherwise our energy constraint must account for them as well.
Thus, the only (SINDy) library tracked terms are the linear and cubic damping terms.
"""

class Constraint(ABC):  
    
    def __init__(self, R: np.ndarray):
        self._R = R
        self._assert_valid()
    
    @abstractmethod
    def innovation(self, x_cal_prev: np.ndarray, x_cal_curr: np.ndarray, 
                   obs_prev: np.ndarray, obs_curr: np.ndarray, dt: float):
        """ In the subclasses you will define the behaviour of this constraint, same goes for self.jacobian() and self.obs_processing """
        pass

    @abstractmethod
    def jacobian(self, x: np.ndarray, dt: float):
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
    We pass the observations as well to be consistent with the general interface, but we do not use them here.

    Oscillator of the form:
   
    .. math::
        \dot{x_0} = x_1 \\
        \dot{x_1} = -\alpha x_0 - \beta x_0^3
    
    All annoying shaping is done here... we give everything correctly formatted to the filter.
    You may pass the non-augmented or augmented one, in either case we know the indices, and they are both nd.arrays.
    
    Parameters
    ----------
    R : np.ndarray
        Observation noise covariance matrix.
    full_dimension : int
        Dimension of the augmented state (for padding the Jacobian).
    x0 : np.ndarray
        Initial state (for computing the initial energy level), non augmented.
    """
    def __init__(self, alpha: float, beta: float, R: np.ndarray, x0: np.ndarray, full_dimension: int):
        self._alpha = alpha
        self._beta = beta
        self._x0 = x0
        self._full_dimension = full_dimension

        super().__init__(R)

    def _energy(self, x: np.ndarray):
        """
        Energy associated to the current state (Hamiltonian?), given the parameters of the system.
        """
        kinetic = 0.5 * x[1]**2 
        potential = 0.5 * self._alpha * x[0]**2 + 0.25 * self._beta * x[0]**4
        
        return kinetic + potential
    
    def innovation(self, x_cal_prev: np.ndarray, x_cal_curr: np.ndarray, obs_prev: np.ndarray, obs_curr: np.ndarray, dt: float):
        """ Innovation, by comparing with the initial energy level, pseudo-observation is zero with very low variance (hyperameter)."""
        energy_0 = self._energy(self._x0)
        energy_t = self._energy(x_cal_curr)
        
        return np.array([energy_0 - energy_t]).reshape(-1, 1)

    def jacobian(self, x_curr: np.ndarray, dt: float):
        """
        Jacobian of the constraint with respect to the augmented state.
        """
        
        dH_dx = np.array([self._alpha * x_curr[0] + self._beta * x_curr[0]**3, x_curr[1]])
        dH_dx = dH_dx.reshape(1, -1)
        dH_dx = np.hstack((dH_dx, np.zeros((1, self._full_dimension - 2)))) # -2 for the non-augmented state dimension
    
        return dH_dx

class LossDuffingCubic(Constraint):
    """
    We assume the damping coefficients to be part of the augmente state, and for them to be indexed as follows:
        ``x_cal[0] = position``
        ``x_cal[1] = velocity``
        ``x_cal[2] = linear damping coeff``
        ``x_cal[3] = cubic damping coeff``
    
    And nothing else is tracked, so the augmented state is 4-dimensional.
    And the model is (converted to first-order system):
    .. math::
        \dot{x_0} = x_1 \\
        \dot{x_1} = -\alpha x_0 - \beta x_0^3 - \delta_1 x_1 - \delta_2 x_1^3

    P.S. This is not really a pseudo-observation, nor a constraint, just a particular observation model.
    """
    def __init__(self, alpha: float, beta: float, R: np.ndarray, x0: np.ndarray, full_dimension: int):
        self._alpha = alpha
        self._beta = beta
        self._x0 = x0
        self._full_dimension = full_dimension

        super().__init__(R)

    def _energy(self, x_cal: np.ndarray):
        """ Energy associated to the current state (Hamiltonian?), given the parameters of the system. """
        kinetic = 0.5 * x_cal[1]**2
        potential = 0.5 * self._alpha * x_cal[0]**2 + 0.25 * self._beta * x_cal[0]**4

        return kinetic + potential

    def _loss_from_obs(self, obs_curr: np.ndarray, obs_prev: np.ndarray):
        """ Loss as the difference in energy between two consecutive observations. """
        return self._energy(obs_curr) - self._energy(obs_prev)
    
    def jacobian(self, x_cal: np.ndarray, dt: float):
        """ 
        Jacobian of the constraint with respect to the augmented state. 
        For readability, we rename some variables. Also we pad the Jacobian with zeros to match the full augmented state dimension.
        """
        v = x_cal[1]
        lin_damp = x_cal[2]
        cubic_damp = x_cal[3]

        dL_dx0 = 0.0
        dL_dx1 = -(lin_damp * 2 * v + cubic_damp * 4 * v**3) * dt
        dL_dx4 = -v**2 * dt
        dL_dx6 = -v**4 * dt
        
        return np.array([dL_dx0, dL_dx1, dL_dx4, dL_dx6]).reshape(1, -1)
        
        # TODO: manipulate the library better e.g, (only quadratic terms!!!)
        # actually it was already correctly only tracking the linear and cubic damping terms
        # womp womp, so "uncorrect" this


    def innovation(self, x_cal_prev: np.ndarray, x_cal_curr: np.ndarray, 
                   obs_prev: np.ndarray, obs_curr: np.ndarray, dt: float):
        """ Innovation as the loss computed from the current and previous observation. """
        loss_from_obs = self._loss_from_obs(obs_curr, obs_prev)
        
        """ This is a Euler step prediction for the loss of energy, this is what we computa the Jacobian from"""
        loss_predicted = -(x_cal_prev[2] * x_cal_prev[1] ** 2 + x_cal_prev[3] * x_cal_prev[0] ** 4) * dt  
        innovation = loss_from_obs - loss_predicted

        return np.array([innovation]).reshape(-1, 1)
    

    