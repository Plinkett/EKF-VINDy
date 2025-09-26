import numpy as np
from typing import Callable

"""
Simple tuple class for constraints (and their Jacobians). May be expanded with autodiff... 
Auxiliary function is optional, for stuff like computation of energy difference.
"""

class Constraint:
    
    def __init__(self, constraint: Callable, jacobian: Callable, R: np.ndarray, aux_function: Callable = None):
        self._constraint = constraint
        self._jacobian = jacobian
        self._R = R
        self._aux_function = aux_function
        self._assert_valid()
    
    @property
    def constraint(self):
        return self._constraint
    
    @property 
    def jacobian(self):
        return self._jacobian
    
    @property
    def R(self):
        return self._R
    
    @property
    def aux_function(self):
        return self._aux_function
    
    def _assert_valid(self):
        if self._constraint is None or self._jacobian is None:
            raise ValueError('Constraint or jacobian are empty')
        if self._R.ndim != 2 or (self._R.shape[0] != self._R.shape[1]):
            raise ValueError('R must be a square matrix')