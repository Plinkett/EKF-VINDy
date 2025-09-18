import numpy as np
from typing import Callable

"""
Simple tuple class for constraints (and their Jacobians). May be expanded with autodiff... 
"""

class Constraint:
    
    def __init__(self, constraint: Callable, jacobian: Callable, R: np.ndarray):
        self._constraint = constraint
        self._jacobian = jacobian
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
    
    def _assert_valid(self):
        if self._constraint is None or self._jacobian is None:
            raise ValueError('Constraint or jacobian are empty')
        if self.R.ndim != 2 or (self.R.shape[0] != self.R.shape[1]):
            raise ValueError('R must be a square matrix')