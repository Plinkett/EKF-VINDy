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
    def __init__(self, alpha: float, beta: float, R: np.ndarray, x_init: np.ndarray, full_dimension: int):
        self._alpha = alpha
        self._beta = beta
        self.x_init = x_init
        self._full_dimension = full_dimension

        super().__init__(R)

    def _energy(self, x: np.ndarray):
        """
        Energy associated to the current state, given the parameters of the system.
        """
        kinetic = 0.5 * x[1]**2 
        potential = 0.5 * self._alpha * x[0]**2 + 0.25 * self._beta * x[0]**4
        
        return kinetic + potential
    
    def innovation(self, x_cal_prev: np.ndarray, x_cal_curr: np.ndarray, obs_prev: np.ndarray, obs_curr: np.ndarray, dt: float):
        """ Innovation, by comparing with the initial energy level, pseudo-observation is zero with very low variance (hyperameter)."""
        energy_0 = self._energy(self.x_init)
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
    def __init__(self, alpha: float, beta: float, R: np.ndarray, full_dimension: int):
        self._alpha = alpha
        self._beta = beta
        self._full_dimension = full_dimension

        super().__init__(R)

    def _energy(self, x_cal: np.ndarray):
        """ Energy associated to the current state (Hamiltonian?), given the parameters of the system. """
        kinetic = 0.5 * x_cal[1]**2
        potential = 0.5 * self._alpha * x_cal[0]**2 + 0.25 * self._beta * x_cal[0]**4

        return kinetic + potential

    def _loss_from_obs(self, obs_prev: np.ndarray, obs_curr: np.ndarray):
        """ Loss as the difference in energy between two consecutive observations. """
        # is sign correct?
        return self._energy(obs_curr) - self._energy(obs_prev)

    # def jacobian(self, x_cal: np.ndarray, dt: float):
    #     """ 
    #     Jacobian of the constraint with respect to the augmented state. 
    #     For readability, we rename some variables. Also we pad the Jacobian with zeros to match the full augmented state dimension.
    #     Lastly, dL_x0 is an np.array for consistency, even if it is zero (otherwise it throws an "inhomogeneous" error).
    #     """
    #     v = x_cal[1]
    #     lin_damp = x_cal[2]
    #     cubic_damp = x_cal[3]

    #     dL_dx0 = np.array([0.0])
    #     dL_dx1 = -(lin_damp * 2 * v + cubic_damp * 4 * v**3) * dt
    #     dL_dx2 = -v**2 * dt
    #     dL_dx3 = -v**4 * dt
    #     return np.array([dL_dx0, dL_dx1, dL_dx2, dL_dx3]).reshape(1, -1)
    
    def jacobian(self, x_cal: np.ndarray, dt: float):
        """Jacobian of the RK4 loss prediction w.r.t. the augmented state."""
        x, v = x_cal[0], x_cal[1]
        lin_damp = x_cal[2]
        cubic_damp = x_cal[3]
        alpha, beta = self._alpha, self._beta

        def accel(x, v):
            return -alpha * x - beta * x**3 - lin_damp * v - cubic_damp * v**3

        # --- RK4 stage velocities ---
        k1x = v
        k1v = accel(x, v)
        v1 = v

        k2x = v + 0.5 * dt * k1v
        k2v = accel(x + 0.5 * dt * k1x, v + 0.5 * dt * k1v)
        v2 = v + 0.5 * dt * k1v

        k3x = v + 0.5 * dt * k2v
        k3v = accel(x + 0.5 * dt * k2x, v + 0.5 * dt * k2v)
        v3 = v + 0.5 * dt * k2v

        k4x = v + dt * k3v
        k4v = accel(x + dt * k3x, v + dt * k3v)
        v4 = v + dt * k3v

        # --- Weighted derivatives ---
        dL_dx0 = np.array([0.0])  # No direct dependency on position

        dL_dx1 = -(dt / 6.0) * (
            (2 * lin_damp * v1 + 4 * cubic_damp * v1**3)
            + 2 * (2 * lin_damp * v2 + 4 * cubic_damp * v2**3)
            + 2 * (2 * lin_damp * v3 + 4 * cubic_damp * v3**3)
            + (2 * lin_damp * v4 + 4 * cubic_damp * v4**3)
        )

        dL_dx2 = -(dt / 6.0) * (v1**2 + 2 * v2**2 + 2 * v3**2 + v4**2)
        dL_dx3 = -(dt / 6.0) * (v1**4 + 2 * v2**4 + 2 * v3**4 + v4**4)

        return np.array([dL_dx0, dL_dx1, dL_dx2, dL_dx3]).reshape(1, -1)
        # TODO: manipulate the library better e.g, (only quadratic terms!!!)


    # def innovation(self, x_cal_prev: np.ndarray, x_cal_curr: np.ndarray, 
    #                obs_prev: np.ndarray, obs_curr: np.ndarray, dt: float):
    #     """ Innovation as the loss computed from the current and previous observation. """
    #     loss_from_obs = self._loss_from_obs(obs_prev, obs_curr)

    #     """ This is a Euler step prediction for the loss of energy, this is what we computa the Jacobian from"""
    #     loss_predicted = -(x_cal_prev[2] * x_cal_prev[1] ** 2 + x_cal_prev[3] * x_cal_prev[1] ** 4) * dt  
    #     innovation = loss_from_obs - loss_predicted

    #     return np.array([innovation]).reshape(-1, 1)
    def innovation(self, x_cal_prev: np.ndarray, x_cal_curr: np.ndarray, 
                obs_prev: np.ndarray, obs_curr: np.ndarray, dt: float):
        """ Innovation as the loss computed from the current and previous observation (RK4 version). """

        # Energy loss observed from real data
        loss_from_obs = self._loss_from_obs(obs_prev, obs_curr)

        # Extract parameters
        x, v = x_cal_prev[0], x_cal_prev[1]
        lin_damp = x_cal_prev[2]
        cubic_damp = x_cal_prev[3]
        alpha, beta = self._alpha, self._beta

        # Helper functions
        def accel(x, v):
            return -alpha * x - beta * x**3 - lin_damp * v - cubic_damp * v**3

        def fE(v):
            return -(lin_damp * v**2 + cubic_damp * v**4)

        # --- RK4 stages ---
        k1x = v
        k1v = accel(x, v)
        v1 = v

        k2x = v + 0.5 * dt * k1v
        k2v = accel(x + 0.5 * dt * k1x, v + 0.5 * dt * k1v)
        v2 = v + 0.5 * dt * k1v

        k3x = v + 0.5 * dt * k2v
        k3v = accel(x + 0.5 * dt * k2x, v + 0.5 * dt * k2v)
        v3 = v + 0.5 * dt * k2v

        k4x = v + dt * k3v
        k4v = accel(x + dt * k3x, v + dt * k3v)
        v4 = v + dt * k3v

        # RK4 weighted average of fE(v)
        loss_predicted = (dt / 6.0) * (fE(v1) + 2*fE(v2) + 2*fE(v3) + fE(v4))

        # Innovation
        innovation = loss_from_obs - loss_predicted
        return np.array([innovation]).reshape(-1, 1)